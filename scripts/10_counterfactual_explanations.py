"""
ICU-DT :: Counterfactual Explanations / Actionable Recourse
-------------------------------------------------------------
For each dashboard patient x condition, finds the smallest plausible
feature changes that would move the predicted risk below a target
threshold. Only modifiable clinical features are changed (not age,
gender, or derived/window features).

Method:
  1. Start from the patient's current feature vector.
  2. Use gradient-free local search (coordinate descent on modifiable
     features) to find a nearby point where XGBoost predicts lower risk.
  3. Constrain changes to clinically plausible ranges (based on training
     set percentiles) and limit the number of features changed.

Output: model/counterfactuals.json
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "processed_patient_hours.parquet")
MODEL_DIR = str(BASE_DIR / "model")

CONDITIONS = ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]

MODIFIABLE_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2",
    "BaseExcess", "HCO3", "pH", "PaCO2", "SaO2",
    "Calcium", "Chloride", "Glucose", "Lactate", "Magnesium",
    "Phosphate", "Potassium",
]

FEATURE_DIRECTIONS = {
    "HR": "lower_is_better", "SBP": "higher_is_better", "MAP": "higher_is_better",
    "DBP": "higher_is_better", "O2Sat": "higher_is_better", "Resp": "lower_is_better",
    "Lactate": "lower_is_better", "Creatinine": "lower_is_better",
    "pH": "normalize", "Temp": "normalize", "Glucose": "normalize",
    "Potassium": "normalize", "FiO2": "lower_is_better",
}

with open(f"{MODEL_DIR}/splits.json") as f:
    SPLITS = json.load(f)
with open(f"{MODEL_DIR}/production_models.pkl", "rb") as f:
    PRODUCTION = pickle.load(f)


def compute_feature_ranges(train_df, feature_cols):
    """Compute clinically plausible ranges from training data."""
    ranges = {}
    for fc in feature_cols:
        if fc in train_df.columns:
            vals = train_df[fc].dropna().values
            ranges[fc] = {
                "p5": float(np.percentile(vals, 5)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p95": float(np.percentile(vals, 95)),
            }
    return ranges


def find_counterfactual(x_raw, model_entry, feature_cols, feat_ranges,
                        target_risk=0.3, max_features_changed=4, n_steps=200):
    """Find minimal feature changes to reduce risk below target."""
    mean, std = model_entry["mean"], model_entry["std"]
    clf = model_entry["model"]

    x_cf = x_raw.copy()
    current_risk = clf.predict_proba(((x_raw.reshape(1, -1) - mean) / std))[:, 1][0]

    if current_risk <= target_risk:
        return None

    modifiable_indices = []
    for fi, fc in enumerate(feature_cols):
        base_name = fc.split("_", 2)[-1] if fc.startswith("W_") else fc
        if base_name in MODIFIABLE_FEATURES or fc in MODIFIABLE_FEATURES:
            if not fc.startswith("W_") and fc not in ("Age", "Gender"):
                modifiable_indices.append(fi)

    if not modifiable_indices:
        return None

    feature_sensitivity = np.zeros(len(modifiable_indices))
    for j, fi in enumerate(modifiable_indices):
        x_up = x_cf.copy()
        x_down = x_cf.copy()
        delta = (feat_ranges.get(feature_cols[fi], {}).get("p75", x_cf[fi] + 1) -
                 feat_ranges.get(feature_cols[fi], {}).get("p25", x_cf[fi] - 1)) * 0.1
        if delta == 0:
            delta = 0.1
        x_up[fi] += delta
        x_down[fi] -= delta
        r_up = clf.predict_proba(((x_up.reshape(1, -1) - mean) / std))[:, 1][0]
        r_down = clf.predict_proba(((x_down.reshape(1, -1) - mean) / std))[:, 1][0]
        feature_sensitivity[j] = (r_up - r_down) / (2 * delta)

    top_sensitive = np.argsort(-np.abs(feature_sensitivity))[:max_features_changed]
    active_indices = [modifiable_indices[j] for j in top_sensitive]

    best_cf = x_cf.copy()
    best_risk = current_risk
    for step in range(n_steps):
        for fi in active_indices:
            fc = feature_cols[fi]
            fr = feat_ranges.get(fc, {})
            lo = fr.get("p5", x_raw[fi] - 3 * (model_entry["std"][fi] + 1e-8))
            hi = fr.get("p95", x_raw[fi] + 3 * (model_entry["std"][fi] + 1e-8))

            for trial_val in np.linspace(lo, hi, 20):
                x_trial = best_cf.copy()
                x_trial[fi] = trial_val
                r = clf.predict_proba(((x_trial.reshape(1, -1) - mean) / std))[:, 1][0]
                if r < best_risk:
                    best_risk = r
                    best_cf = x_trial.copy()

        if best_risk <= target_risk:
            break

    changes = []
    for fi in range(len(feature_cols)):
        if abs(best_cf[fi] - x_raw[fi]) > 1e-6:
            fc = feature_cols[fi]
            direction = FEATURE_DIRECTIONS.get(fc, "unspecified")
            changes.append({
                "feature": fc,
                "current_value": round(float(x_raw[fi]), 2),
                "suggested_value": round(float(best_cf[fi]), 2),
                "change": round(float(best_cf[fi] - x_raw[fi]), 2),
                "direction_hint": direction,
            })

    if not changes:
        return None

    return {
        "original_risk": round(float(current_risk), 4),
        "counterfactual_risk": round(float(best_risk), 4),
        "risk_reduction": round(float(current_risk - best_risk), 4),
        "target_achieved": bool(best_risk <= target_risk),
        "n_features_changed": len(changes),
        "changes": changes,
    }


def main():
    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)

    train_ids = set(SPLITS["train"])
    test_ids = set(SPLITS["test"])
    train_df = df[df.patient_id.isin(train_ids)]
    test_df = df[df.patient_id.isin(test_ids)]

    feature_cols = PRODUCTION["Sepsis_6h"]["feature_cols"]
    feat_ranges = compute_feature_ranges(train_df, feature_cols)

    with open(str(BASE_DIR / "artifacts" / "dashboard_data.json")) as f:
        dashboard = json.load(f)
    dashboard_pids = [p["patientId"] for p in dashboard["patients"]]

    counterfactuals = {}
    for pid in dashboard_pids:
        g = test_df[test_df.patient_id == pid].sort_values("hour")
        if len(g) == 0:
            continue
        last_row = g.iloc[-1][feature_cols].values.astype(float)

        patient_cfs = {}
        for cond in CONDITIONS:
            task = f"{cond}_6h"
            entry = PRODUCTION[task]
            current_risk = entry["model"].predict_proba(
                ((last_row.reshape(1, -1) - entry["mean"]) / entry["std"]))[:, 1][0]

            if current_risk < 0.15:
                continue

            target = max(0.1, current_risk * 0.5)
            cf = find_counterfactual(last_row, entry, feature_cols, feat_ranges,
                                    target_risk=target, max_features_changed=4)
            if cf is not None:
                patient_cfs[cond] = cf

        if patient_cfs:
            counterfactuals[pid] = patient_cfs
            for cond, cf in patient_cfs.items():
                print(f"[CF] {pid}/{cond}: risk {cf['original_risk']:.3f} -> "
                      f"{cf['counterfactual_risk']:.3f} "
                      f"({cf['n_features_changed']} features changed, "
                      f"target={'achieved' if cf['target_achieved'] else 'not reached'})")

    output = {
        "modifiable_features": MODIFIABLE_FEATURES,
        "feature_ranges": {k: v for k, v in feat_ranges.items() if k in MODIFIABLE_FEATURES},
        "patient_counterfactuals": counterfactuals,
    }

    out_path = f"{MODEL_DIR}/counterfactuals.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[CF] Saved {len(counterfactuals)} patient counterfactuals -> {out_path}")


if __name__ == "__main__":
    main()
