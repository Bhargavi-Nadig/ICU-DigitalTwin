"""
ICU-DT :: Production Model Training — XGBoost (chosen after the fair,
information-matched comparison in model_comparison.json: XGBoost won 9/15
tasks outright and was within ~1pt AUROC on the rest; adopted uniformly
across all 15 tasks for a single, simple, consistent serving architecture).

This script:
  1) Runs a small per-CONDITION randomized hyperparameter search (using the
     6h-horizon task as the representative difficulty for that condition,
     train-fit / val-select — never touches test) with early stopping.
  2) Trains final models for all 3 horizons of each condition with the
     tuned config + early stopping on val.
  3) Evaluates comprehensively on test (AUROC/AUPRC/precision/recall/F1/
     confusion matrix/calibration/ROC&PR curves/subgroup-by-age) — same
     depth of reporting as the LSTM got.
  4) Builds an EXACT SHAP explainer (TreeExplainer) per condition — not a
     surrogate. TreeExplainer computes exact Shapley values for tree
     ensembles in polynomial time, so there is no fidelity gap to manage:
     the explanation IS the model, not an approximation of it.
"""
import json
import pickle
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, precision_score, recall_score, f1_score,
)
from sklearn.linear_model import LogisticRegression
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
train_mod = import_module("02_train_lstm")

DATA_PATH = str(BASE_DIR / "data" / "processed_patient_hours.parquet")
MODEL_DIR = str(BASE_DIR / "model")

FEATURE_COLS = train_mod.FEATURE_COLS
CONDITIONS = train_mod.CONDITIONS
HORIZONS = train_mod.HORIZONS

with open(f"{MODEL_DIR}/splits.json") as f:
    SPLITS = json.load(f)

RNG = np.random.RandomState(42)

SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.3],
}
N_SEARCH_TRIALS = 10


def sample_config():
    return {k: RNG.choice(v).item() for k, v in SEARCH_SPACE.items()}


def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total, reliability = 0.0, []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf, acc = y_prob[mask].mean(), y_true[mask].mean()
        total += (mask.sum() / len(y_prob)) * abs(conf - acc)
        reliability.append({"bin_lo": round(float(lo), 2), "bin_hi": round(float(hi), 2),
                             "confidence": round(float(conf), 4), "observed": round(float(acc), 4),
                             "n": int(mask.sum())})
    return float(total), reliability


def platt_fit(p, y):
    lr = LogisticRegression()
    lr.fit(p.reshape(-1, 1), y)
    return lr


def subsample_curve(x, y, n=25):
    if len(x) <= n:
        return list(zip([round(float(v), 4) for v in x], [round(float(v), 4) for v in y]))
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return list(zip([round(float(x[i]), 4) for i in idx], [round(float(y[i]), 4) for i in idx]))


def fit_xgb(Xtr, ytr, Xval, yval, config, n_estimators=600):
    scale_pos = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, **config, scale_pos_weight=min(scale_pos, 15),
        eval_metric="auc", random_state=42, n_jobs=-1, early_stopping_rounds=30,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return clf


def main():
    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    df = train_mod.build_multi_task_labels(df)

    window_cols = sorted([c for c in df.columns if c.startswith("W_")])
    tabular_feature_cols = FEATURE_COLS + window_cols
    print(f"[PROD] Training on {len(tabular_feature_cols)} features "
          f"({len(FEATURE_COLS)} single-hour + {len(window_cols)} window-summary, 24h lookback).")

    train_ids, val_ids, test_ids = set(SPLITS["train"]), set(SPLITS["val"]), set(SPLITS["test"])
    train_df = df[df.patient_id.isin(train_ids)]
    val_df = df[df.patient_id.isin(val_ids)]
    test_df = df[df.patient_id.isin(test_ids)]

    mean = train_df[tabular_feature_cols].mean()
    std = train_df[tabular_feature_cols].std().replace(0, 1)

    metrics = {}
    production_models = {}
    explainers = {}
    tuned_configs = {}

    for cond in CONDITIONS:
        rep_task = f"{cond}_6h"
        tr = train_df.dropna(subset=[rep_task])
        va = val_df.dropna(subset=[rep_task])
        Xtr = ((tr[tabular_feature_cols] - mean) / std).values
        ytr = tr[rep_task].values
        Xva = ((va[tabular_feature_cols] - mean) / std).values
        yva = va[rep_task].values

        print(f"[PROD] === {cond}: hyperparameter search ({N_SEARCH_TRIALS} trials, "
              f"tuned on {rep_task}, train-fit/val-select) ===")
        best_auroc, best_config = -1, None
        t0 = time.time()
        for trial in range(N_SEARCH_TRIALS):
            cfg = sample_config()
            clf = fit_xgb(Xtr, ytr, Xva, yva, cfg, n_estimators=400)
            p = clf.predict_proba(Xva)[:, 1]
            auroc = roc_auc_score(yva, p)
            if auroc > best_auroc:
                best_auroc, best_config = auroc, cfg
        print(f"[PROD] {cond}: best val AUROC={best_auroc:.4f} config={best_config} "
              f"({time.time()-t0:.0f}s)")
        tuned_configs[cond] = {"config": best_config, "val_AUROC_at_search": round(float(best_auroc), 4)}

        for h in HORIZONS:
            task = f"{cond}_{h}h"
            tr = train_df.dropna(subset=[task])
            va = val_df.dropna(subset=[task])
            te = test_df.dropna(subset=[task])
            Xtr = ((tr[tabular_feature_cols] - mean) / std).values
            ytr = tr[task].values
            Xva = ((va[tabular_feature_cols] - mean) / std).values
            yva = va[task].values
            Xte = ((te[tabular_feature_cols] - mean) / std).values
            yte = te[task].values

            clf = fit_xgb(Xtr, ytr, Xva, yva, best_config, n_estimators=600)
            p_test = clf.predict_proba(Xte)[:, 1]
            p_val = clf.predict_proba(Xva)[:, 1]

            auroc = roc_auc_score(yte, p_test)
            auprc = average_precision_score(yte, p_test)
            fpr, tpr, thr = roc_curve(yva, p_val)
            threshold = float(thr[np.argmax(tpr - fpr)])
            bin_preds = (p_test >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(yte, bin_preds).ravel()
            precision = precision_score(yte, bin_preds, zero_division=0)
            recall = recall_score(yte, bin_preds, zero_division=0)
            f1 = f1_score(yte, bin_preds, zero_division=0)

            ece_before, rel_before = ece(yte, p_test)
            platt = platt_fit(p_val, yva)
            p_test_cal = platt.predict_proba(p_test.reshape(-1, 1))[:, 1]
            ece_after, rel_after = ece(yte, p_test_cal)

            fpr_t, tpr_t, _ = roc_curve(yte, p_test)
            prec_t, rec_t, _ = precision_recall_curve(yte, p_test)

            ages = te["Age"].values
            subgroup = {}
            for lo, hi, label in [(0, 40, "<40"), (40, 60, "40-60"), (60, 80, "60-80"), (80, 200, "80+")]:
                m = (ages >= lo) & (ages < hi)
                if m.sum() < 20 or len(np.unique(yte[m])) < 2:
                    continue
                subgroup[label] = {"n": int(m.sum()), "AUROC": round(float(roc_auc_score(yte[m], p_test[m])), 4),
                                    "prevalence": round(float(yte[m].mean()), 4)}

            pids = te["patient_id"].values
            hours = te["hour"].values
            examples = {}
            for kind, mask in [
                ("true_positive", (yte == 1) & (bin_preds == 1)),
                ("false_positive", (yte == 0) & (bin_preds == 1)),
                ("false_negative", (yte == 1) & (bin_preds == 0)),
                ("true_negative", (yte == 0) & (bin_preds == 0)),
            ]:
                idxs = np.where(mask)[0][:3]
                examples[kind] = [{"patient_id": pids[j], "hour": int(hours[j]),
                                    "predicted_risk": round(float(p_test[j]), 3)} for j in idxs]

            metrics[task] = {
                "test_AUROC": round(float(auroc), 4), "test_AUPRC": round(float(auprc), 4),
                "operating_threshold": round(threshold, 4),
                "precision": round(float(precision), 4), "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
                "support": {"n_positive": int(yte.sum()), "n_negative": int(len(yte) - yte.sum()),
                            "prevalence": round(float(yte.mean()), 4)},
                "calibration": {"ece_before": round(ece_before, 4), "ece_after_platt_scaling": round(ece_after, 4),
                                "reliability_before": rel_before, "reliability_after": rel_after},
                "roc_curve": subsample_curve(fpr_t, tpr_t), "pr_curve": subsample_curve(rec_t, prec_t),
                "subgroup_by_age": subgroup, "failure_examples": examples,
                "n_estimators_used": clf.best_iteration + 1 if hasattr(clf, "best_iteration") else None,
            }
            production_models[task] = {
                "model": clf, "mean": mean.values, "std": std.values,
                "feature_cols": tabular_feature_cols, "model_type": "xgboost",
                "platt_calibrator": platt,
            }
            print(f"[PROD] {task}: AUROC={auroc:.4f} AUPRC={auprc:.4f} F1={f1:.3f} "
                  f"ECE {ece_before:.3f}->{ece_after:.3f} trees={metrics[task]['n_estimators_used']}")

        # exact SHAP TreeExplainer for this condition's 6h model (used in dashboard explanations)
        clf_6h = production_models[f"{cond}_6h"]["model"]
        explainers[cond] = shap.TreeExplainer(clf_6h)

    metrics["_meta"] = {
        "n_train_patients": len(train_ids), "n_val_patients": len(val_ids), "n_test_patients": len(test_ids),
        "patient_level_split_disjoint": True,
        "architecture": "XGBoost (per-condition tuned hyperparameters, early stopping on val)",
        "hyperparameter_search": tuned_configs,
        "note": "Sepsis uses PhysioNet-adjudicated labels. ARF/Cardiac/AKI/Coagulopathy use rule-based "
                "clinical proxy labels — not adjudicated ground truth. All models (incl. XGBoost) see the "
                "same 24h window of context as the LSTM (single-hour + 24h window-summary features).",
    }
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[PROD] Saved comprehensive metrics -> {MODEL_DIR}/metrics.json")

    with open(f"{MODEL_DIR}/production_models.pkl", "wb") as f:
        pickle.dump(production_models, f)
    with open(f"{MODEL_DIR}/shap_explainers.pkl", "wb") as f:
        pickle.dump(explainers, f)
    print(f"[PROD] Saved {len(production_models)} production XGBoost models + "
          f"{len(explainers)} exact TreeExplainers -> {MODEL_DIR}/")


if __name__ == "__main__":
    main()
