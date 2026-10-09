"""
ICU-DT :: Fair Baseline Comparison (clinical rule / LogReg / RF / XGBoost / LSTM)
--------------------------------------------------------------------------------
NOTE ON PROVENANCE: this script reconstructs one that was run earlier in
development and then accidentally deleted when the pipeline was rebuilt
around XGBoost as the production model. The numbers currently sitting in
model/model_comparison.json came from that original run; this script is
functionally equivalent (same features, same model families, same fair
24h-matched comparison) and is meant to reproduce them on rerun, but
individual runs will differ slightly due to RF/XGBoost's own internal
randomness even with random_state set, and because it depends on whichever
LSTM checkpoint exists in model/lstm_metrics.json at the time it's run.

This is a COMPARISON-ONLY script. It does NOT touch production_models.pkl —
that file is owned by 05_train_production_xgboost.py, which trains XGBoost
with its own tuned, early-stopped, per-condition hyperparameters (better
than the fixed-hyperparameter XGBoost trained here for comparison purposes).
Run this BEFORE 05 if you want model_comparison.json to include a genuine,
untuned "off the shelf" XGBoost data point alongside the other baselines,
distinct from the tuned production model.

Fairness note: every model here sees the SAME 24h of context as the LSTM —
41 single-hour features + 174 window-summary features (mean/std/min/max/
6h-trend/24h-trend per vital/lab over the trailing 24h, computed in
01_etl_pipeline.py) = 215 features total for the tabular models, vs. the
LSTM's raw 24-step sequence of the 41 single-hour features.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
train_mod = import_module("02_train_lstm")

DATA_PATH = str(BASE_DIR / "data" / "processed_patient_hours.parquet")
MODEL_DIR = str(BASE_DIR / "model")

FEATURE_COLS = train_mod.FEATURE_COLS
TASK_NAMES = train_mod.TASK_NAMES  # all 15 (condition x horizon) tasks

with open(f"{MODEL_DIR}/splits.json") as f:
    SPLITS = json.load(f)

# Prefer the LSTM's own comprehensive metrics file; fall back to whatever
# metrics.json currently holds (e.g. if it hasn't been regenerated yet).
lstm_metrics_path = Path(MODEL_DIR) / "lstm_metrics.json"
if not lstm_metrics_path.exists():
    lstm_metrics_path = Path(MODEL_DIR) / "metrics.json"
    print(f"[BASELINES] WARNING: lstm_metrics.json not found, falling back to {lstm_metrics_path} "
          f"— make sure this actually holds LSTM numbers, not XGBoost's.")
with open(lstm_metrics_path) as f:
    LSTM_METRICS = json.load(f)


def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / len(y_prob)) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(total)


def clinical_rule_score(df):
    """qSOFA-style: 0-3 rule score, rescaled to [0,1]. No training — pure clinical heuristic baseline."""
    return df["qSOFA_proxy"].values / 3.0


def main():
    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    df = train_mod.build_multi_task_labels(df)

    window_cols = sorted([c for c in df.columns if c.startswith("W_")])
    tabular_feature_cols = FEATURE_COLS + window_cols
    print(f"[BASELINES] {len(tabular_feature_cols)} features "
          f"({len(FEATURE_COLS)} single-hour + {len(window_cols)} window-summary, 24h lookback) "
          f"— matches the LSTM's 24h context exactly.")

    train_ids, test_ids = set(SPLITS["train"]), set(SPLITS["test"])
    train_df = df[df.patient_id.isin(train_ids)]
    test_df = df[df.patient_id.isin(test_ids)]

    mean = train_df[tabular_feature_cols].mean()
    std = train_df[tabular_feature_cols].std().replace(0, 1)

    comparison = {}
    for task in TASK_NAMES:
        tr = train_df.dropna(subset=[task])
        te = test_df.dropna(subset=[task])
        ytr = tr[task].values
        yte = te[task].values
        Xtr = ((tr[tabular_feature_cols] - mean) / std).values
        Xte = ((te[tabular_feature_cols] - mean) / std).values

        results = {}

        # 1) clinical rule (no training, evaluated directly on test)
        rule_scores = clinical_rule_score(te)
        results["clinical_rule_qSOFA"] = {
            "AUROC": round(float(roc_auc_score(yte, rule_scores)), 4) if len(np.unique(yte)) > 1 else None,
            "AUPRC": round(float(average_precision_score(yte, rule_scores)), 4) if len(np.unique(yte)) > 1 else None,
            "ECE": round(ece(yte, rule_scores), 4),
        }

        # 2) Logistic Regression
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
        lr.fit(Xtr, ytr)
        p = lr.predict_proba(Xte)[:, 1]
        results["logistic_regression"] = {
            "AUROC": round(float(roc_auc_score(yte, p)), 4),
            "AUPRC": round(float(average_precision_score(yte, p)), 4),
            "ECE": round(ece(yte, p), 4),
        }

        # 3) Random Forest
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=42)
        rf.fit(Xtr, ytr)
        p = rf.predict_proba(Xte)[:, 1]
        results["random_forest"] = {
            "AUROC": round(float(roc_auc_score(yte, p)), 4),
            "AUPRC": round(float(average_precision_score(yte, p)), 4),
            "ECE": round(ece(yte, p), 4),
        }

        # 4) XGBoost (fixed, untuned hyperparameters — for comparison purposes only;
        #    see 05_train_production_xgboost.py for the actually-tuned production model)
        scale_pos = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=min(scale_pos, 15), eval_metric="auc",
            random_state=42, n_jobs=-1,
        )
        xgb_clf.fit(Xtr, ytr)
        p = xgb_clf.predict_proba(Xte)[:, 1]
        results["xgboost"] = {
            "AUROC": round(float(roc_auc_score(yte, p)), 4),
            "AUPRC": round(float(average_precision_score(yte, p)), 4),
            "ECE": round(ece(yte, p), 4),
        }

        # 5) BiLSTM (from its own comprehensive metrics file — trained separately, once,
        #    on the full 24h sequence; not retrained here)
        lstm_m = LSTM_METRICS.get(task, {})
        results["lstm_multitask_bilstm"] = {
            "AUROC": lstm_m.get("test_AUROC"), "AUPRC": lstm_m.get("test_AUPRC"),
            "ECE": lstm_m.get("calibration", {}).get("ece_after_platt_scaling"),
        }

        valid_results = [(k, v["AUROC"]) for k, v in results.items() if v.get("AUROC") is not None]
        best_model = max(valid_results, key=lambda kv: kv[1])[0] if valid_results else None
        comparison[task] = {"results": results, "best_by_AUROC": best_model}
        print(f"[BASELINES] {task}: " + " | ".join(
            f"{k}={v['AUROC']}" for k, v in results.items() if v.get("AUROC") is not None
        ) + f"  -> best: {best_model}")

    comparison["_meta"] = {
        "note": "Fair, information-matched comparison: every model sees the same 24h of context "
                "(LSTM as a raw sequence, tabular models as 215 engineered single-hour + window-summary "
                "features). XGBoost here uses FIXED hyperparameters for a clean baseline comparison — "
                "see 05_train_production_xgboost.py for the tuned production model actually served.",
        "lstm_metrics_source": str(lstm_metrics_path.name),
    }

    with open(f"{MODEL_DIR}/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[BASELINES] Saved -> {MODEL_DIR}/model_comparison.json")


if __name__ == "__main__":
    main()
