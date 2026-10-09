"""
ICU-DT :: Anomaly / Out-of-Distribution Detection
---------------------------------------------------
Detects patients whose input features are very different from the training
distribution, and flags predictions that may be unreliable as a result.

Method:
  1. Fit a Mahalanobis distance model on the training set's z-normalized
     feature vectors (using a regularized covariance estimate).
  2. Compute per-feature z-scores to identify which specific features
     are outliers.
  3. For each dashboard patient, compute an OOD score and identify the
     specific features driving the anomaly.

Output: model/ood_detector.json — contains training statistics and
        per-patient OOD assessments.
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


def fit_ood_model(X_train):
    """Fit a regularized Mahalanobis distance model."""
    mean = X_train.mean(axis=0)
    X_centered = X_train - mean
    cov = np.cov(X_centered, rowvar=False)
    reg = 1e-4 * np.eye(cov.shape[0])
    cov_reg = cov + reg
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)

    train_distances = np.array([
        np.sqrt(np.dot(np.dot(x, cov_inv), x)) for x in X_centered
    ])

    return {
        "mean": mean,
        "cov_inv": cov_inv,
        "std": X_train.std(axis=0),
        "percentiles": {
            "p50": float(np.percentile(train_distances, 50)),
            "p90": float(np.percentile(train_distances, 90)),
            "p95": float(np.percentile(train_distances, 95)),
            "p99": float(np.percentile(train_distances, 99)),
        },
    }


def compute_ood_score(x, ood_model):
    """Compute Mahalanobis distance for a single sample."""
    diff = x - ood_model["mean"]
    dist = np.sqrt(np.dot(np.dot(diff, ood_model["cov_inv"]), diff))
    return float(dist)


def get_ood_features(x, ood_model, top_n=5):
    """Identify features with the largest z-scores."""
    z_scores = np.abs((x - ood_model["mean"]) / (ood_model["std"] + 1e-8))
    top_indices = np.argsort(-z_scores)[:top_n]
    return [
        {"feature": FEATURE_COLS[i] if i < len(FEATURE_COLS) else f"W_feat_{i}",
         "z_score": round(float(z_scores[i]), 2),
         "value": round(float(x[i]), 2),
         "train_mean": round(float(ood_model["mean"][i]), 2)}
        for i in top_indices if z_scores[i] > 2.0
    ]


def classify_ood(distance, percentiles):
    if distance <= percentiles["p90"]:
        return "in_distribution"
    elif distance <= percentiles["p95"]:
        return "borderline"
    elif distance <= percentiles["p99"]:
        return "out_of_distribution"
    else:
        return "extreme_ood"


def main():
    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)

    feature_cols = PRODUCTION["Sepsis_6h"]["feature_cols"]
    prod_mean = PRODUCTION["Sepsis_6h"]["mean"]
    prod_std = PRODUCTION["Sepsis_6h"]["std"]

    train_ids = set(SPLITS["train"])
    test_ids = set(SPLITS["test"])
    train_df = df[df.patient_id.isin(train_ids)]
    test_df = df[df.patient_id.isin(test_ids)]

    X_train = train_df[feature_cols].values.astype(float)
    X_train_z = (X_train - prod_mean) / prod_std

    print("[OOD] Fitting Mahalanobis distance model on training data...")
    ood_model = fit_ood_model(X_train_z)
    print(f"[OOD] Training distribution — "
          f"p50={ood_model['percentiles']['p50']:.2f}, "
          f"p90={ood_model['percentiles']['p90']:.2f}, "
          f"p95={ood_model['percentiles']['p95']:.2f}, "
          f"p99={ood_model['percentiles']['p99']:.2f}")

    with open(str(BASE_DIR / "artifacts" / "dashboard_data.json")) as f:
        dashboard = json.load(f)
    dashboard_pids = [p["patientId"] for p in dashboard["patients"]]

    patient_ood = {}
    for pid in dashboard_pids:
        g = test_df[test_df.patient_id == pid].sort_values("hour")
        if len(g) == 0:
            continue
        last_row = g.iloc[-1][feature_cols].values.astype(float)
        x_z = (last_row - prod_mean) / prod_std

        distance = compute_ood_score(x_z, ood_model)
        status = classify_ood(distance, ood_model["percentiles"])
        anomalous_features = get_ood_features(x_z, ood_model)

        hourly_ood = []
        for _, row in g.iterrows():
            x_h = row[feature_cols].values.astype(float)
            x_h_z = (x_h - prod_mean) / prod_std
            d = compute_ood_score(x_h_z, ood_model)
            hourly_ood.append({
                "hour": int(row["hour"]),
                "distance": round(d, 2),
                "status": classify_ood(d, ood_model["percentiles"]),
            })

        patient_ood[pid] = {
            "mahalanobis_distance": round(distance, 2),
            "status": status,
            "anomalous_features": anomalous_features,
            "hourly_ood": hourly_ood,
        }

        print(f"[OOD] {pid}: distance={distance:.2f}, status={status}, "
              f"anomalous_features={len(anomalous_features)}")

    all_test_distances = []
    for pid in test_ids:
        g = test_df[test_df.patient_id == pid].sort_values("hour")
        if len(g) == 0:
            continue
        last = g.iloc[-1][feature_cols].values.astype(float)
        x_z = (last - prod_mean) / prod_std
        d = compute_ood_score(x_z, ood_model)
        all_test_distances.append(d)

    output = {
        "training_stats": {
            "n_train_samples": int(len(X_train_z)),
            "n_features": len(feature_cols),
            "percentiles": ood_model["percentiles"],
        },
        "test_distribution": {
            "n_patients": len(all_test_distances),
            "mean_distance": round(float(np.mean(all_test_distances)), 2),
            "std_distance": round(float(np.std(all_test_distances)), 2),
            "pct_borderline": round(float(np.mean(
                [d > ood_model["percentiles"]["p90"] for d in all_test_distances]
            )) * 100, 1),
            "pct_ood": round(float(np.mean(
                [d > ood_model["percentiles"]["p95"] for d in all_test_distances]
            )) * 100, 1),
        },
        "patient_ood": patient_ood,
    }

    out_path = f"{MODEL_DIR}/ood_detector.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[OOD] Saved -> {out_path}")


if __name__ == "__main__":
    main()
