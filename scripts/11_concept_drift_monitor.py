"""
ICU-DT :: Concept Drift + Continual Learning Monitor
------------------------------------------------------
Detects data/model drift by comparing feature distributions and model
predictions between the training set and recent data windows.

Method:
  1. Compute reference statistics from the training set (mean, std,
     quantiles for each feature; prediction distribution).
  2. Simulate time-based drift by splitting the test set into temporal
     windows and comparing each window against the reference.
  3. Per-feature drift detection using Population Stability Index (PSI).
  4. Prediction drift using KL divergence on binned prediction distributions.
  5. Flag features and time windows where retraining may be needed.

Output: model/drift_report.json
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "processed_patient_hours.parquet")
MODEL_DIR = str(BASE_DIR / "model")

FEATURE_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2",
    "BaseExcess", "HCO3", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine", "Glucose", "Lactate", "Magnesium",
    "Phosphate", "Potassium", "Bilirubin_total", "Hct", "Hgb", "WBC",
    "Platelets", "Age", "Gender",
    "ShockIndex", "PulsePressure", "PF_ratio",
    "Delta_HR", "Delta_MAP", "Delta_Resp", "Delta_Temp",
    "Delta_Lactate", "Delta_Creatinine", "qSOFA_proxy",
]
CONDITIONS = ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]

with open(f"{MODEL_DIR}/splits.json") as f:
    SPLITS = json.load(f)
with open(f"{MODEL_DIR}/production_models.pkl", "rb") as f:
    PRODUCTION = pickle.load(f)


def compute_psi(reference, current, n_bins=10):
    """Population Stability Index between two distributions."""
    bins = np.linspace(min(reference.min(), current.min()) - 1e-8,
                       max(reference.max(), current.max()) + 1e-8, n_bins + 1)
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    ref_pct = (ref_hist + 1) / (len(reference) + n_bins)
    cur_pct = (cur_hist + 1) / (len(current) + n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def classify_drift(psi):
    if psi < 0.1:
        return "no_drift"
    elif psi < 0.2:
        return "moderate_drift"
    else:
        return "significant_drift"


def compute_reference_stats(X_train, feature_names):
    stats = {}
    for i, fname in enumerate(feature_names):
        col = X_train[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) < 10:
            continue
        stats[fname] = {
            "mean": round(float(np.mean(valid)), 4),
            "std": round(float(np.std(valid)), 4),
            "p5": round(float(np.percentile(valid, 5)), 4),
            "p25": round(float(np.percentile(valid, 25)), 4),
            "p50": round(float(np.percentile(valid, 50)), 4),
            "p75": round(float(np.percentile(valid, 75)), 4),
            "p95": round(float(np.percentile(valid, 95)), 4),
        }
    return stats


def main():
    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)

    feature_cols = PRODUCTION["Sepsis_6h"]["feature_cols"]

    train_ids = set(SPLITS["train"])
    test_ids = set(SPLITS["test"])
    train_df = df[df.patient_id.isin(train_ids)]
    test_df = df[df.patient_id.isin(test_ids)]

    X_train = train_df[feature_cols].values.astype(float)
    X_test = test_df[feature_cols].values.astype(float)

    print("[DRIFT] Computing reference statistics from training data...")
    ref_stats = compute_reference_stats(X_train, feature_cols)

    print("[DRIFT] Computing feature-level drift (PSI)...")
    feature_drift = {}
    for i, fc in enumerate(feature_cols):
        train_col = X_train[:, i]
        test_col = X_test[:, i]
        train_valid = train_col[~np.isnan(train_col)]
        test_valid = test_col[~np.isnan(test_col)]
        if len(train_valid) < 10 or len(test_valid) < 10:
            continue

        psi = compute_psi(train_valid, test_valid)
        mean_shift = abs(np.mean(test_valid) - np.mean(train_valid)) / (np.std(train_valid) + 1e-8)

        feature_drift[fc] = {
            "psi": round(psi, 4),
            "drift_status": classify_drift(psi),
            "mean_shift_std": round(float(mean_shift), 4),
            "train_mean": round(float(np.mean(train_valid)), 4),
            "test_mean": round(float(np.mean(test_valid)), 4),
        }

    drifted_features = [fc for fc, d in feature_drift.items() if d["drift_status"] != "no_drift"]
    print(f"[DRIFT] {len(drifted_features)}/{len(feature_drift)} features show drift")

    print("[DRIFT] Computing prediction drift per condition...")
    prediction_drift = {}
    for cond in CONDITIONS:
        task = f"{cond}_6h"
        entry = PRODUCTION[task]
        mean, std_v = entry["mean"], entry["std"]

        train_preds = entry["model"].predict_proba((X_train - mean) / std_v)[:, 1]
        test_preds = entry["model"].predict_proba((X_test - mean) / std_v)[:, 1]

        pred_psi = compute_psi(train_preds, test_preds)

        train_high = float(np.mean(train_preds > 0.5))
        test_high = float(np.mean(test_preds > 0.5))

        prediction_drift[cond] = {
            "prediction_psi": round(pred_psi, 4),
            "drift_status": classify_drift(pred_psi),
            "train_mean_risk": round(float(np.mean(train_preds)), 4),
            "test_mean_risk": round(float(np.mean(test_preds)), 4),
            "train_high_risk_pct": round(train_high * 100, 2),
            "test_high_risk_pct": round(test_high * 100, 2),
        }
        print(f"[DRIFT] {cond}: PSI={pred_psi:.4f} ({classify_drift(pred_psi)}), "
              f"mean risk train={np.mean(train_preds):.4f} test={np.mean(test_preds):.4f}")

    print("[DRIFT] Computing temporal drift windows...")
    test_patients = sorted(test_df.patient_id.unique())
    n_windows = 4
    window_size = len(test_patients) // n_windows
    temporal_windows = []

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else len(test_patients)
        window_pids = set(test_patients[start:end])
        window_df = test_df[test_df.patient_id.isin(window_pids)]
        X_window = window_df[feature_cols].values.astype(float)

        window_drift = {}
        n_drifted = 0
        for i, fc in enumerate(feature_cols[:41]):
            train_col = X_train[:, i][~np.isnan(X_train[:, i])]
            win_col = X_window[:, i][~np.isnan(X_window[:, i])]
            if len(win_col) < 10:
                continue
            psi = compute_psi(train_col, win_col)
            if psi >= 0.1:
                window_drift[fc] = round(psi, 4)
                n_drifted += 1

        window_preds = {}
        for cond in CONDITIONS:
            task = f"{cond}_6h"
            entry = PRODUCTION[task]
            preds = entry["model"].predict_proba(
                (X_window - entry["mean"]) / entry["std"])[:, 1]
            window_preds[cond] = round(float(np.mean(preds)), 4)

        temporal_windows.append({
            "window": w + 1,
            "n_patients": len(window_pids),
            "n_features_drifted": n_drifted,
            "drifted_features": window_drift,
            "mean_risk_by_condition": window_preds,
        })

    any_significant = any(d["drift_status"] == "significant_drift" for d in prediction_drift.values())
    many_features_drifted = len(drifted_features) > len(feature_cols) * 0.1

    output = {
        "reference_stats": ref_stats,
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "temporal_windows": temporal_windows,
        "summary": {
            "n_features_total": len(feature_cols),
            "n_features_drifted": len(drifted_features),
            "drifted_feature_names": drifted_features[:20],
            "retraining_recommended": any_significant or many_features_drifted,
            "reason": (
                "Significant prediction drift detected" if any_significant
                else f"{len(drifted_features)} features show distributional drift" if many_features_drifted
                else "No significant drift detected"
            ),
        },
    }

    out_path = f"{MODEL_DIR}/drift_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[DRIFT] Saved -> {out_path}")


if __name__ == "__main__":
    main()
