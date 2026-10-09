"""
ICU-DT :: Ensemble Training — XGBoost + Random Forest with Soft Voting
-----------------------------------------------------------------------
Trains a Random Forest alongside the existing XGBoost, then learns a
validation-optimal blending weight alpha per condition:
    p_ensemble = alpha * p_xgb + (1 - alpha) * p_rf

If the ensemble gains are marginal (<0.5 pp AUROC), XGBoost alone
remains the production default.

Input:  data/publication_patient_hours.parquet (from script 12)
Output: model/ensemble_models.pkl, model/ensemble_metrics.json
"""
import json, pickle, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "publication_patient_hours.parquet")
MODEL_DIR = str(BASE_DIR / "model")
SPLITS_PATH = str(BASE_DIR / "model" / "splits.json")
OUT_MODELS = str(BASE_DIR / "model" / "ensemble_models.pkl")
OUT_METRICS = str(BASE_DIR / "model" / "ensemble_metrics.json")

CONDITIONS = ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]
HORIZONS = [6, 12, 24]
RNG = np.random.RandomState(42)

XGB_SEARCH_SPACE = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.08],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.7, 0.8],
    "min_child_weight": [1, 5],
    "gamma": [0, 0.1],
}
RF_SEARCH_SPACE = {
    "n_estimators": [200, 300],
    "max_depth": [8, 12, 16],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [5, 10],
    "max_features": ["sqrt", "log2"],
}
N_TRIALS = 5
ALPHA_GRID = np.linspace(0, 1, 21)  # 0.00, 0.05, ..., 1.00


def sample_xgb_config():
    return {k: RNG.choice(v).item() for k, v in XGB_SEARCH_SPACE.items()}

def sample_rf_config():
    cfg = {}
    for k, v in RF_SEARCH_SPACE.items():
        candidates = [x for x in v if x is not None]
        include_none = None in v
        if include_none and RNG.random() < 1.0 / (len(candidates) + 1):
            val = None
        else:
            val = candidates[RNG.randint(len(candidates))]
            if isinstance(val, (np.integer, int)):
                val = int(val)
            elif isinstance(val, (np.floating, float)):
                val = float(val)
            elif isinstance(val, (np.str_, str)):
                val = str(val)
        cfg[k] = val
    return cfg


def fit_xgb(Xtr, ytr, Xva, yva, config):
    scale_pos = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=600, **config, scale_pos_weight=min(scale_pos, 15),
        eval_metric="auc", random_state=42, n_jobs=-1, early_stopping_rounds=30,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf


def fit_rf(Xtr, ytr, config):
    scale_pos = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    clf = RandomForestClassifier(
        **config, class_weight={0: 1, 1: min(scale_pos, 15)},
        random_state=42, n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    return clf


def platt_fit(p, y):
    lr = LogisticRegression()
    lr.fit(p.reshape(-1, 1), y)
    return lr


def find_best_alpha(p_xgb_val, p_rf_val, y_val):
    best_alpha, best_auroc = 0.5, -1
    for a in ALPHA_GRID:
        p_ens = a * p_xgb_val + (1 - a) * p_rf_val
        try:
            auroc = roc_auc_score(y_val, p_ens)
        except ValueError:
            continue
        if auroc > best_auroc:
            best_auroc, best_alpha = auroc, a
    return float(round(best_alpha, 2)), float(round(best_auroc, 4))


def main():
    print("=" * 70)
    print("Ensemble Training: XGBoost + Random Forest")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    label_cols = [f"{c}_{h}h" for c in CONDITIONS for h in HORIZONS]
    miss_cols = [c for c in df.columns if c.startswith("miss_")]
    exclude = set(["patient_id", "hour", "ICULOS", "SepsisLabel",
                   "ARF_proxy", "Cardiac_proxy", "AKI_proxy", "Coagulopathy_proxy"]
                  + label_cols)
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("miss_")]
    feature_cols = feature_cols + miss_cols
    feature_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8]]
    print(f"[ENS] Using {len(feature_cols)} features (including {len(miss_cols)} missingness indicators)")

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])
    train_df = df[df.patient_id.isin(train_ids)]
    val_df = df[df.patient_id.isin(val_ids)]
    test_df = df[df.patient_id.isin(test_ids)]

    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)

    all_models = {}
    all_metrics = {}

    for cond in CONDITIONS:
        rep_task = f"{cond}_6h"
        tr = train_df.dropna(subset=[rep_task])
        va = val_df.dropna(subset=[rep_task])
        Xtr = ((tr[feature_cols] - mean) / std).values.astype(np.float32)
        ytr = tr[rep_task].values.astype(int)
        Xva = ((va[feature_cols] - mean) / std).values.astype(np.float32)
        yva = va[rep_task].values.astype(int)

        # ── XGBoost hyperparameter search ──
        print(f"\n[ENS] === {cond}: XGBoost search ({N_TRIALS} trials) ===")
        best_xgb_auroc, best_xgb_config = -1, None
        t0 = time.time()
        for trial in range(N_TRIALS):
            cfg = sample_xgb_config()
            clf = fit_xgb(Xtr, ytr, Xva, yva, cfg)
            p = clf.predict_proba(Xva)[:, 1]
            auroc = roc_auc_score(yva, p)
            if auroc > best_xgb_auroc:
                best_xgb_auroc, best_xgb_config = auroc, cfg
        print(f"  XGBoost best val AUROC={best_xgb_auroc:.4f} ({time.time()-t0:.0f}s)")

        # ── Random Forest hyperparameter search ──
        print(f"[ENS] === {cond}: Random Forest search ({N_TRIALS} trials) ===")
        best_rf_auroc, best_rf_config = -1, None
        t0 = time.time()
        for trial in range(N_TRIALS):
            cfg = sample_rf_config()
            clf = fit_rf(Xtr, ytr, cfg)
            p = clf.predict_proba(Xva)[:, 1]
            auroc = roc_auc_score(yva, p)
            if auroc > best_rf_auroc:
                best_rf_auroc, best_rf_config = auroc, cfg
        print(f"  RF best val AUROC={best_rf_auroc:.4f} ({time.time()-t0:.0f}s)")

        # ── Train final models per horizon ──
        for h in HORIZONS:
            task = f"{cond}_{h}h"
            tr = train_df.dropna(subset=[task])
            va = val_df.dropna(subset=[task])
            te = test_df.dropna(subset=[task])
            Xtr = ((tr[feature_cols] - mean) / std).values.astype(np.float32)
            ytr = tr[task].values.astype(int)
            Xva = ((va[feature_cols] - mean) / std).values.astype(np.float32)
            yva = va[task].values.astype(int)
            Xte = ((te[feature_cols] - mean) / std).values.astype(np.float32)
            yte = te[task].values.astype(int)

            clf_xgb = fit_xgb(Xtr, ytr, Xva, yva, best_xgb_config)
            clf_rf = fit_rf(Xtr, ytr, best_rf_config)

            p_xgb_val = clf_xgb.predict_proba(Xva)[:, 1]
            p_rf_val = clf_rf.predict_proba(Xva)[:, 1]
            p_xgb_test = clf_xgb.predict_proba(Xte)[:, 1]
            p_rf_test = clf_rf.predict_proba(Xte)[:, 1]

            alpha, val_ens_auroc = find_best_alpha(p_xgb_val, p_rf_val, yva)
            p_ens_test = alpha * p_xgb_test + (1 - alpha) * p_rf_test

            auroc_xgb = roc_auc_score(yte, p_xgb_test)
            auroc_rf = roc_auc_score(yte, p_rf_test)
            auroc_ens = roc_auc_score(yte, p_ens_test)
            auprc_xgb = average_precision_score(yte, p_xgb_test)
            auprc_rf = average_precision_score(yte, p_rf_test)
            auprc_ens = average_precision_score(yte, p_ens_test)

            # Platt scaling on ensemble
            platt = platt_fit(
                alpha * p_xgb_val + (1 - alpha) * p_rf_val, yva
            )

            ens_gain = auroc_ens - auroc_xgb
            production_model = "ensemble" if ens_gain >= 0.005 else "xgboost"

            all_models[task] = {
                "xgboost": clf_xgb,
                "random_forest": clf_rf,
                "alpha": alpha,
                "production_model": production_model,
                "mean": mean.values,
                "std": std.values,
                "feature_cols": feature_cols,
                "platt_calibrator": platt,
            }

            all_metrics[task] = {
                "xgboost": {"test_AUROC": round(auroc_xgb, 4), "test_AUPRC": round(auprc_xgb, 4)},
                "random_forest": {"test_AUROC": round(auroc_rf, 4), "test_AUPRC": round(auprc_rf, 4)},
                "ensemble": {
                    "test_AUROC": round(auroc_ens, 4), "test_AUPRC": round(auprc_ens, 4),
                    "alpha": alpha,
                    "gain_over_xgb_pp": round(ens_gain * 100, 2),
                },
                "production_model": production_model,
                "support": {
                    "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
                    "prevalence": round(float(yte.mean()), 4),
                },
            }

            print(f"[ENS] {task}: XGB={auroc_xgb:.4f} RF={auroc_rf:.4f} "
                  f"ENS={auroc_ens:.4f} (alpha={alpha}) "
                  f"gain={ens_gain*100:+.2f}pp -> {production_model}")

    all_metrics["_meta"] = {
        "ensemble_method": "soft_voting: p = alpha * p_xgb + (1-alpha) * p_rf",
        "alpha_selection": "grid search on validation AUROC (21 steps from 0 to 1)",
        "production_threshold": "ensemble adopted only when gain >= 0.5pp AUROC over XGBoost alone",
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "n_features": len(feature_cols),
        "xgb_search_trials": N_TRIALS,
        "rf_search_trials": N_TRIALS,
    }

    with open(OUT_MODELS, "wb") as f:
        pickle.dump(all_models, f)
    with open(OUT_METRICS, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[ENS] Saved {len(all_models)} ensemble model bundles -> {OUT_MODELS}")
    print(f"[ENS] Saved metrics -> {OUT_METRICS}")

    # Summary
    print("\n" + "=" * 70)
    print("Ensemble Summary")
    print("=" * 70)
    for task, m in all_metrics.items():
        if task.startswith("_"):
            continue
        e = m["ensemble"]
        print(f"  {task:20s}  XGB={m['xgboost']['test_AUROC']:.4f}  "
              f"RF={m['random_forest']['test_AUROC']:.4f}  "
              f"ENS={e['test_AUROC']:.4f}  alpha={e['alpha']}  "
              f"gain={e['gain_over_xgb_pp']:+.2f}pp  "
              f"prod={m['production_model']}")


if __name__ == "__main__":
    main()
