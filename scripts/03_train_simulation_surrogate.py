"""
ICU-DT :: Simulation Engine surrogate (FR-5) — NOT used for SHAP explanation
------------------------------------------------------------------------------
Explanation vs. simulation are now split, deliberately:

  - EXPLANATION ("why is this patient's risk high") uses shap.TreeExplainer
    directly on the production XGBoost model (see 05_train_production_xgboost.py)
    — EXACT Shapley values, not a surrogate, so there's no agreement gap to
    manage for that part at all.

  - SIMULATION ("what if MAP were higher right now") needs to run inside the
    browser with no backend, on every slider drag. Embedding a 600-tree
    XGBoost ensemble client-side isn't practical (multi-MB of tree JSON per
    task). So simulation uses a small LINEAR model instead, distilled from
    the real production XGBoost (soft targets = XGBoost's predicted
    probability, not ground truth — same distillation principle as before,
    now correctly targeting whichever model is actually in production).

Verification mechanism (the requested "do the two models agree" check,
scoped correctly to the piece that still uses an approximation): fidelity of
the linear simulator against XGBoost is reported below (Pearson r, Spearman
r, binary agreement rate, MAE) and surfaced in the dashboard next to the
simulator so nobody mistakes it for an exact readout.

Feature-space note: the surrogate is trained on the SAME 215 features
XGBoost uses (41 single-hour + 174 window-summary). The dashboard's sliders
only let a clinician vary the ~10 single-hour vitals "right now" — the 24h
window-summary features (mean/std/min/max/trend) stay fixed at the patient's
actually-observed history, since moving a slider doesn't rewrite the past.
"""
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
train_mod = import_module("02_train_lstm")

DATA_PATH = str(BASE_DIR / "data" / "processed_patient_hours.parquet")
MODEL_DIR = str(BASE_DIR / "model")

CONDITIONS = train_mod.CONDITIONS

with open(f"{MODEL_DIR}/splits.json") as f:
    SPLITS = json.load(f)
with open(f"{MODEL_DIR}/production_models.pkl", "rb") as f:
    PRODUCTION = pickle.load(f)


class LinearSurrogate(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_distilled(Xtr_raw, soft_tr, mean, std, epochs=400, lr=0.05):
    Xz = (Xtr_raw - mean) / std
    Xt = torch.tensor(Xz, dtype=torch.float32)
    yt = torch.tensor(soft_tr, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    model = LinearSurrogate(Xz.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    return model.linear.weight.detach().numpy().ravel(), float(model.linear.bias.item())


def main():
    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    df = train_mod.build_multi_task_labels(df)

    train_ids, test_ids = set(SPLITS["train"]), set(SPLITS["test"])
    train_df = df[df.patient_id.isin(train_ids)]
    test_df = df[df.patient_id.isin(test_ids)]

    surrogates = {}
    for cond in CONDITIONS:
        task = f"{cond}_6h"
        entry = PRODUCTION[task]
        prod_model, prod_mean, prod_std, feature_cols = (
            entry["model"], entry["mean"], entry["std"], entry["feature_cols"]
        )

        tr = train_df.dropna(subset=[task])
        te = test_df.dropna(subset=[task])
        Xtr_raw = tr[feature_cols].values.astype(float)
        Xte_raw = te[feature_cols].values.astype(float)

        soft_tr = prod_model.predict_proba((Xtr_raw - prod_mean) / prod_std)[:, 1]
        soft_te = prod_model.predict_proba((Xte_raw - prod_mean) / prod_std)[:, 1]

        mean = Xtr_raw.mean(axis=0)
        std = Xtr_raw.std(axis=0)
        std[std == 0] = 1
        coef, intercept = train_distilled(Xtr_raw, soft_tr, mean, std)

        def predict(X):
            z = (X - mean) / std
            return 1 / (1 + np.exp(-(z @ coef + intercept)))

        surrogate_te = predict(Xte_raw)

        pearson_r, _ = pearsonr(surrogate_te, soft_te)
        spearman_r, _ = spearmanr(surrogate_te, soft_te)
        agree_rate = float(np.mean((surrogate_te >= 0.5) == (soft_te >= 0.5)))
        mae = float(np.mean(np.abs(surrogate_te - soft_te)))

        truth = te[task].values
        valid = ~np.isnan(truth)
        surrogate_auroc = production_auroc = None
        if valid.sum() > 10 and len(np.unique(truth[valid])) > 1:
            surrogate_auroc = round(float(roc_auc_score(truth[valid], surrogate_te[valid])), 4)
            production_auroc = round(float(roc_auc_score(truth[valid], soft_te[valid])), 4)

        surrogates[cond] = {
            "feature_cols": feature_cols,
            "mean": mean.tolist(), "std": std.tolist(),
            "coef": coef.tolist(), "intercept": intercept,
            "distilled_from_model": "xgboost",
            "purpose": "LIVE WHAT-IF SIMULATION ONLY — main SHAP explanation uses exact TreeExplainer, "
                       "not this surrogate (see shap_explainers.pkl / dashboard_data.json).",
            "fidelity_to_production_model": {
                "pearson_r": round(float(pearson_r), 4), "spearman_r": round(float(spearman_r), 4),
                "binary_agreement_rate_at_0.5": round(agree_rate, 4), "mean_absolute_error": round(mae, 4),
            },
            "auroc_vs_ground_truth": {"surrogate": surrogate_auroc, "production_model": production_auroc},
        }
        print(f"[SIM-SURROGATE] {cond}: fidelity to XGBoost — Pearson r={pearson_r:.3f}, "
              f"agreement@0.5={agree_rate:.3f}, MAE={mae:.4f} | "
              f"AUROC vs truth: surrogate={surrogate_auroc} xgboost={production_auroc}")

    with open(f"{MODEL_DIR}/simulation_surrogates.json", "w") as f:
        json.dump(surrogates, f, indent=2)
    print(f"[SIM-SURROGATE] Saved -> {MODEL_DIR}/simulation_surrogates.json")


if __name__ == "__main__":
    main()
