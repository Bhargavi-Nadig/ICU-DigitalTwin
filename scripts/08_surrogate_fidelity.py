"""
ICU-DT :: Surrogate Fidelity & Uncertainty
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

with open(f"{MODEL_DIR}/splits.json") as f:
    SPLITS = json.load(f)
with open(f"{MODEL_DIR}/production_models.pkl", "rb") as f:
    PRODUCTION = pickle.load(f)
with open(f"{MODEL_DIR}/simulation_surrogates.json") as f:
    SURROGATES = json.load(f)


def surrogate_predict(X_raw, surrogate):
    mean = np.array(surrogate["mean"])
    std = np.array(surrogate["std"])
    coef = np.array(surrogate["coef"])
    intercept = surrogate["intercept"]
    z = (X_raw - mean) / std
    return 1 / (1 + np.exp(-(z @ coef + intercept)))


def safe_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-8:
        return 0.0
    xm, ym = x - x.mean(), y - y.mean()
    return float(np.dot(xm, ym) / (np.sqrt(np.dot(xm, xm) * np.dot(ym, ym)) + 1e-12))


def main():
    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    test_ids = set(SPLITS["test"])
    test_df = df[df.patient_id.isin(test_ids)]

    with open(str(BASE_DIR / "artifacts" / "dashboard_data.json")) as f:
        dashboard = json.load(f)
    dashboard_pids = {p["patientId"] for p in dashboard["patients"]}

    fidelity_report = {}
    patient_fidelity = {}

    for cond in CONDITIONS:
        task = f"{cond}_6h"
        entry = PRODUCTION[task]
        feature_cols = entry["feature_cols"]
        surr = SURROGATES[cond]

        X_raw = np.nan_to_num(test_df[feature_cols].values.astype(float), nan=0.0)
        Xz = (X_raw - entry["mean"]) / entry["std"]
        xgb_probs = entry["model"].predict_proba(Xz)[:, 1]
        surr_probs = surrogate_predict(X_raw, surr)

        overall_mae = float(np.mean(np.abs(xgb_probs - surr_probs)))
        overall_corr = safe_corr(xgb_probs, surr_probs)
        overall_agree = float(np.mean((xgb_probs >= 0.5) == (surr_probs >= 0.5)))
        max_error = float(np.max(np.abs(xgb_probs - surr_probs)))

        residuals = np.abs(xgb_probs - surr_probs)
        error_thresholds = {
            "low": round(float(np.percentile(residuals, 50)), 4),
            "medium": round(float(np.percentile(residuals, 75)), 4),
            "high": round(float(np.percentile(residuals, 90)), 4),
        }

        bins = np.linspace(0, 1, 11)
        binned = []
        for i in range(10):
            lo, hi = bins[i], bins[i + 1]
            mask = (xgb_probs >= lo) & (xgb_probs < (hi + 0.01 if i == 9 else hi))
            if mask.sum() < 10:
                continue
            binned.append({
                "bin_lo": round(float(lo), 2), "bin_hi": round(float(hi), 2),
                "n": int(mask.sum()),
                "mae": round(float(np.mean(np.abs(xgb_probs[mask] - surr_probs[mask]))), 4),
                "agreement": round(float(np.mean((xgb_probs[mask] >= 0.5) == (surr_probs[mask] >= 0.5))), 4),
            })

        high_residual_mask = residuals > np.percentile(residuals, 90)
        regions = []
        for fi in range(min(41, len(feature_cols))):
            vals_high = X_raw[high_residual_mask, fi]
            vals_all = X_raw[:, fi]
            if len(vals_high) < 5:
                continue
            shift = abs(np.mean(vals_high) - np.mean(vals_all)) / (np.std(vals_all) + 1e-8)
            if shift > 0.3:
                regions.append({"feature": feature_cols[fi], "divergence_score": round(float(shift), 3)})
        regions.sort(key=lambda r: -r["divergence_score"])

        fidelity_report[cond] = {
            "overall": {
                "mae": round(overall_mae, 4), "pearson_r": round(overall_corr, 4),
                "agreement_at_0.5": round(overall_agree, 4), "max_error": round(max_error, 4),
                "n_samples": int(len(xgb_probs)),
            },
            "binned_fidelity": binned,
            "unreliable_regions": regions[:5],
            "error_thresholds": error_thresholds,
        }
        print(f"[FIDELITY] {cond}: MAE={overall_mae:.4f}, r={overall_corr:.4f}, agree={overall_agree:.4f}")

        for pid in dashboard_pids:
            g = test_df[test_df.patient_id == pid].sort_values("hour")
            if len(g) == 0:
                continue
            last_row = np.nan_to_num(g.iloc[-1][feature_cols].values.astype(float), nan=0.0).reshape(1, -1)
            xgb_p = entry["model"].predict_proba((last_row - entry["mean"]) / entry["std"])[:, 1][0]
            surr_p = surrogate_predict(last_row, surr)[0]
            error = abs(xgb_p - surr_p)

            thresholds = error_thresholds
            if error <= thresholds["low"]:
                reliability = "high"
            elif error <= thresholds["medium"]:
                reliability = "medium"
            elif error <= thresholds["high"]:
                reliability = "low"
            else:
                reliability = "very_low"

            if pid not in patient_fidelity:
                patient_fidelity[pid] = {}
            patient_fidelity[pid][cond] = {
                "xgb_risk": round(float(xgb_p), 4),
                "surrogate_risk": round(float(surr_p), 4),
                "absolute_error": round(float(error), 4),
                "reliability": reliability,
            }

    output = {"condition_fidelity": fidelity_report, "patient_fidelity": patient_fidelity}
    out_path = f"{MODEL_DIR}/surrogate_fidelity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[FIDELITY] Saved -> {out_path}")


if __name__ == "__main__":
    main()
