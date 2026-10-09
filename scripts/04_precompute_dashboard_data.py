"""
ICU-DT :: Digital Twin Engine (FR-4) + XAI (FR-6) + Cohort Similarity (FR-4.4)
---------------------------------------------------------------------------------
Production model is XGBoost (see 05_train_production_xgboost.py — chosen after
a fair, information-matched comparison against LSTM/RF/LogReg/clinical-rule).
Because XGBoost predicts from a single feature row (no sequence windowing),
computing a hour-by-hour risk trajectory for a patient is just predict_proba
on every row of their stay — no LSTM inference loop needed here at all.

For each of ~40 representative patients, this script builds:
  - hourly risk trajectories (3 conditions... 5 conditions x 3 horizons, main
    line = 6h horizon) from the real production XGBoost models
  - EXACT SHAP attributions (TreeExplainer) at each condition's peak-risk hour
  - a cohort similarity index (PCA over the standardized 215-feature tabular
    vector — no LSTM dependency, consistent with XGBoost being production)
  - a full baseline-features vector per patient (single-hour + window-summary)
    for the dashboard's live what-if simulator
"""
import json
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONDITIONS, HORIZONS, build_multi_task_labels

DATA_PATH = str(BASE_DIR / "data" / "processed_patient_hours.parquet")
MODEL_DIR = str(BASE_DIR / "model")
OUT_PATH = str(BASE_DIR / "artifacts" / "dashboard_data.json")
N_PER_CATEGORY = 4
MAX_COHORT_INDEX = 300
LATENT_DIMS = 8

with open(f"{MODEL_DIR}/splits.json") as f:
    SPLITS = json.load(f)
with open(f"{MODEL_DIR}/production_models.pkl", "rb") as f:
    PRODUCTION = pickle.load(f)
with open(f"{MODEL_DIR}/shap_explainers.pkl", "rb") as f:
    SHAP_EXPLAINERS = pickle.load(f)
with open(f"{MODEL_DIR}/simulation_surrogates.json") as f:
    SIM_SURROGATES = json.load(f)
with open(f"{MODEL_DIR}/metrics.json") as f:
    PROD_METRICS = json.load(f)
with open(f"{MODEL_DIR}/model_comparison.json") as f:
    MODEL_COMPARISON = json.load(f)


def slim_metrics_for_dashboard(metrics):
    """Dashboard only needs headline numbers + a compact calibration/ROC summary,
    not the full failure-analysis/curve detail (that stays in model/metrics.json
    for anyone inspecting the repo directly)."""
    slim = {}
    for task, m in metrics.items():
        if task.startswith("_") or "note" in m:
            slim[task] = m
            continue
        slim[task] = {
            "test_AUROC": m["test_AUROC"], "test_AUPRC": m["test_AUPRC"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "confusion_matrix": m["confusion_matrix"], "support": m["support"],
            "calibration": {"ece_before": m["calibration"]["ece_before"],
                             "ece_after_platt_scaling": m["calibration"]["ece_after_platt_scaling"],
                             "reliability_after": m["calibration"]["reliability_after"]},
            "roc_curve": m["roc_curve"][::2], "pr_curve": m["pr_curve"][::2],
            "subgroup_by_age": m["subgroup_by_age"],
        }
    return slim

DISPLAY_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Lactate", "WBC", "Creatinine"]

FEATURE_LABELS = {
    "HR": "Heart Rate", "O2Sat": "Oxygen Saturation", "Temp": "Temperature",
    "SBP": "Systolic BP", "MAP": "Mean Arterial Pressure", "DBP": "Diastolic BP",
    "Resp": "Respiratory Rate", "FiO2": "FiO2", "BaseExcess": "Base Excess",
    "HCO3": "Bicarbonate", "pH": "Blood pH", "PaCO2": "PaCO2", "SaO2": "SaO2",
    "AST": "AST (liver)", "BUN": "Blood Urea Nitrogen", "Alkalinephos": "Alkaline Phosphatase",
    "Calcium": "Calcium", "Chloride": "Chloride", "Creatinine": "Creatinine",
    "Glucose": "Glucose", "Lactate": "Lactate", "Magnesium": "Magnesium",
    "Phosphate": "Phosphate", "Potassium": "Potassium", "Bilirubin_total": "Total Bilirubin",
    "Hct": "Hematocrit", "Hgb": "Hemoglobin", "WBC": "White Blood Cell Count",
    "Platelets": "Platelets", "Age": "Age", "Gender": "Gender",
    "ShockIndex": "Shock Index (HR/SBP)", "PulsePressure": "Pulse Pressure",
    "PF_ratio": "PaO2/FiO2 Ratio", "Delta_HR": "Heart Rate Trend (1h)",
    "Delta_MAP": "MAP Trend (1h)", "Delta_Resp": "Respiratory Rate Trend (1h)",
    "Delta_Temp": "Temperature Trend (1h)", "Delta_Lactate": "Lactate Trend (1h)",
    "Delta_Creatinine": "Creatinine Trend (1h)", "qSOFA_proxy": "qSOFA (proxy) Score",
}


def window_label(col):
    """W_mean_HR -> 'Heart Rate (24h mean)' etc."""
    if not col.startswith("W_"):
        return FEATURE_LABELS.get(col, col)
    parts = col.split("_", 2)
    stat, base = parts[1], parts[2]
    base_label = FEATURE_LABELS.get(base, base)
    stat_label = {"mean": "24h mean", "std": "24h std dev", "min": "24h min", "max": "24h max",
                  "diff6h": "6h trend", "diff24h": "24h trend"}.get(stat, stat)
    return f"{base_label} ({stat_label})"


CONDITION_LABELS = {
    "Sepsis": "Sepsis (adjudicated label)",
    "ARF": "Acute Respiratory Failure (clinical proxy)",
    "Cardiac": "Cardiac / Hemodynamic Collapse (clinical proxy)",
    "AKI": "Acute Kidney Injury (clinical proxy, KDIGO-inspired)",
    "Coagulopathy": "Coagulopathy / DIC risk (clinical proxy)",
}


def narrative_summary(cond, top_features):
    parts = []
    for feat, contrib in top_features[:3]:
        label = window_label(feat)
        direction = "elevated" if contrib > 0 else "reduced"
        parts.append(f"{direction} {label}")
    return f"{cond} risk driven primarily by " + ", ".join(parts) + "."


def main():
    os.makedirs(BASE_DIR / "artifacts", exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    df = build_multi_task_labels(df)

    test_ids = SPLITS["test"]
    test_df = df[df.patient_id.isin(test_ids)]

    ever = test_df.groupby("patient_id").agg(
        everSepsis=("SepsisLabel", "max"), everARF=("ARF_proxy", "max"),
        everCardiac=("Cardiac_proxy", "max"), everAKI=("AKI_proxy", "max"),
        everCoagulopathy=("Coagulopathy_proxy", "max"),
    )
    los = test_df.groupby("patient_id").size()
    eligible = los[(los >= 15) & (los <= 35)].index

    rng = np.random.RandomState(7)

    def sample(lst, n):
        lst = list(lst)
        return list(rng.choice(lst, size=min(n, len(lst)), replace=False)) if lst else []

    cond_cols = ["everSepsis", "everARF", "everCardiac", "everAKI", "everCoagulopathy"]
    cat_any_severe = [p for p in eligible if ever.loc[p, cond_cols].sum() >= 2]
    cat_sepsis = [p for p in eligible if ever.loc[p, "everSepsis"] == 1]
    cat_arf = [p for p in eligible if ever.loc[p, "everARF"] == 1 and ever.loc[p, "everSepsis"] == 0]
    cat_cardiac = [p for p in eligible if ever.loc[p, "everCardiac"] == 1 and ever.loc[p, cond_cols].sum() == 1]
    cat_aki = [p for p in eligible if ever.loc[p, "everAKI"] == 1 and ever.loc[p, cond_cols].sum() == 1]
    cat_coag = [p for p in eligible if ever.loc[p, "everCoagulopathy"] == 1 and ever.loc[p, cond_cols].sum() == 1]
    cat_healthy = [p for p in eligible if ever.loc[p, cond_cols].sum() == 0]

    chosen = list(dict.fromkeys(
        sample(cat_sepsis, N_PER_CATEGORY) + sample(cat_arf, N_PER_CATEGORY) +
        sample(cat_cardiac, N_PER_CATEGORY) + sample(cat_aki, N_PER_CATEGORY) +
        sample(cat_coag, N_PER_CATEGORY) + sample(cat_any_severe, N_PER_CATEGORY) +
        sample(cat_healthy, N_PER_CATEGORY)
    ))
    print(f"[PRECOMPUTE] category pools: sepsis={len(cat_sepsis)} arf={len(cat_arf)} "
          f"cardiac={len(cat_cardiac)} aki={len(cat_aki)} coag={len(cat_coag)} "
          f"multi={len(cat_any_severe)} healthy={len(cat_healthy)}")
    print(f"[PRECOMPUTE] chosen for full detail: {len(chosen)} patients")

    tabular_feature_cols = PRODUCTION["Sepsis_6h"]["feature_cols"]  # same 215 cols for every task

    patients_out = []
    all_latent = {}

    for pid in eligible:
        g = test_df[test_df.patient_id == pid].sort_values("hour").reset_index(drop=True)
        X_all = g[tabular_feature_cols].values.astype(float)

        # standardized last-hour vector, used for BOTH cohort-similarity PCA and (kept out of loop) below
        last_mean = PRODUCTION["Sepsis_6h"]["mean"]
        last_std = PRODUCTION["Sepsis_6h"]["std"]
        z_last = (X_all[-1] - last_mean) / last_std
        all_latent[pid] = z_last

        if pid not in chosen:
            continue

        last_row = g.iloc[-1]
        conditions_out = {}
        risk6_by_cond = {}
        for cond in CONDITIONS:
            risk_by_horizon_traj = {}
            for h in HORIZONS:
                task = f"{cond}_{h}h"
                entry = PRODUCTION[task]
                Xz = (X_all - entry["mean"]) / entry["std"]
                probs = entry["model"].predict_proba(Xz)[:, 1]
                risk_by_horizon_traj[h] = probs
            risk6 = risk_by_horizon_traj[6]
            risk6_by_cond[cond] = risk6
            peak_idx = int(np.argmax(risk6))
            peak_row_X = X_all[peak_idx]

            explainer = SHAP_EXPLAINERS[cond]
            entry6 = PRODUCTION[f"{cond}_6h"]
            Xz_peak = ((peak_row_X - entry6["mean"]) / entry6["std"]).reshape(1, -1)
            shap_vals = explainer.shap_values(Xz_peak)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[-1]
            shap_vals = np.array(shap_vals).ravel()
            order = np.argsort(-np.abs(shap_vals))[:6]
            top_feats = [(tabular_feature_cols[i], float(shap_vals[i])) for i in order]

            conditions_out[cond] = {
                "label": CONDITION_LABELS[cond],
                "peakRisk": round(float(risk6[peak_idx]), 4),
                "peakHour": int(g.iloc[peak_idx]["hour"]),
                "currentRisk6h": round(float(risk6[-1]), 4),
                "riskByHorizon": {f"{h}h": round(float(risk_by_horizon_traj[h][-1]), 4) for h in HORIZONS},
                "topFeatures": [
                    {"feature": f, "label": window_label(f), "shap": round(c, 4), "isWindowFeature": f.startswith("W_")}
                    for f, c in top_feats
                ],
                "narrative": narrative_summary(cond, top_feats),
            }

        hourly = {"hour": [int(h) for h in g["hour"].tolist()]}
        for cond in CONDITIONS:
            hourly[f"risk{cond}"] = [round(float(x), 3) for x in risk6_by_cond[cond]]
        for v in DISPLAY_VITALS:
            hourly[v] = [None if pd.isna(x) else round(float(x), 2) for x in g[v].tolist()]

        baseline_features = [
            (None if pd.isna(last_row[fc]) else round(float(last_row[fc]), 4)) for fc in tabular_feature_cols
        ]
        composite_risk = max(conditions_out[c]["currentRisk6h"] for c in CONDITIONS)

        patients_out.append({
            "patientId": pid, "age": int(last_row["Age"]),
            "gender": "Male" if int(last_row["Gender"]) == 1 else "Female",
            "everSepsis": bool(ever.loc[pid, "everSepsis"]), "everARF": bool(ever.loc[pid, "everARF"]),
            "everCardiac": bool(ever.loc[pid, "everCardiac"]), "everAKI": bool(ever.loc[pid, "everAKI"]),
            "everCoagulopathy": bool(ever.loc[pid, "everCoagulopathy"]),
            "losHours": int(len(g)), "compositeRisk": round(float(composite_risk), 4),
            "conditions": conditions_out, "hourly": hourly, "baselineFeatures": baseline_features,
        })

    patients_out.sort(key=lambda p: -p["compositeRisk"])

    # ---- cohort similarity index: PCA over standardized 215-feature last-hour vectors ----
    pids_all = list(all_latent.keys())
    H = np.stack([all_latent[p] for p in pids_all])
    Hc = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    latent = Hc @ Vt[:LATENT_DIMS].T
    latent = latent / (latent.std(axis=0, keepdims=True) + 1e-8)

    rest = [p for p in pids_all if p not in chosen]
    keep_rest = list(rng.choice(rest, size=min(MAX_COHORT_INDEX - len(chosen), len(rest)), replace=False))
    keep_set = set(chosen) | set(keep_rest)
    keep_idx = [i for i, p in enumerate(pids_all) if p in keep_set]

    cohort_index = {"patientId": [], "age": [], "gender": [], "everSepsis": [], "everARF": [],
                     "everCardiac": [], "everAKI": [], "everCoagulopathy": [], "hasFullDetail": [], "latent": []}
    for i in keep_idx:
        pid = pids_all[i]
        row = ever.loc[pid]
        g_last = test_df[test_df.patient_id == pid].sort_values("hour").iloc[-1]
        cohort_index["patientId"].append(pid)
        cohort_index["age"].append(int(g_last["Age"]))
        cohort_index["gender"].append("M" if int(g_last["Gender"]) == 1 else "F")
        cohort_index["everSepsis"].append(bool(row.everSepsis))
        cohort_index["everARF"].append(bool(row.everARF))
        cohort_index["everCardiac"].append(bool(row.everCardiac))
        cohort_index["everAKI"].append(bool(row.everAKI))
        cohort_index["everCoagulopathy"].append(bool(row.everCoagulopathy))
        cohort_index["hasFullDetail"].append(pid in chosen)
        cohort_index["latent"].append([round(float(x), 3) for x in latent[i]])

    # ---- simulation surrogate config for JS: only single-hour features are user-adjustable ----
    # feature_cols is identical across all 5 conditions (same tabular feature set) — store once.
    surrogates_for_js = {}
    for cond in CONDITIONS:
        s = SIM_SURROGATES[cond]
        surrogates_for_js[cond] = {
            "mean": s["mean"], "std": s["std"], "coef": s["coef"], "intercept": s["intercept"],
            "fidelity": s["fidelity_to_production_model"],
        }

    # ---- Load research feature data if available ----
    def load_json_optional(name):
        p = Path(MODEL_DIR) / name
        if p.exists():
            with open(p) as fh:
                return json.load(fh)
        return None

    itshap_data = load_json_optional("itshap_attributions.json")
    fidelity_data = load_json_optional("surrogate_fidelity.json")
    ood_data = load_json_optional("ood_detector.json")
    cf_data = load_json_optional("counterfactuals.json")
    drift_data = load_json_optional("drift_report.json")

    for p in patients_out:
        pid = p["patientId"]
        if itshap_data and pid in itshap_data:
            p["itshap"] = itshap_data[pid]
        if fidelity_data and pid in fidelity_data.get("patient_fidelity", {}):
            p["surrogateFidelity"] = fidelity_data["patient_fidelity"][pid]
        if ood_data and pid in ood_data.get("patient_ood", {}):
            p["oodStatus"] = ood_data["patient_ood"][pid]
        if cf_data and pid in cf_data.get("patient_counterfactuals", {}):
            p["counterfactuals"] = cf_data["patient_counterfactuals"][pid]

    export = {
        "modelMetrics": slim_metrics_for_dashboard(PROD_METRICS),
        "modelComparison": MODEL_COMPARISON,
        "featureCols": tabular_feature_cols,
        "surrogates": surrogates_for_js,
        "featureLabels": FEATURE_LABELS,
        "conditionLabels": CONDITION_LABELS,
        "displayVitals": DISPLAY_VITALS,
        "patients": patients_out,
        "cohortIndex": cohort_index,
        "productionModelType": "xgboost",
        "generatedFrom": "PhysioNet Sepsis Challenge training_setA (held-out test split)",
    }
    if fidelity_data and "condition_fidelity" in fidelity_data:
        export["surrogateFidelity"] = fidelity_data["condition_fidelity"]
    if drift_data:
        export["driftReport"] = {
            "predictionDrift": drift_data.get("prediction_drift", {}),
            "temporalWindows": drift_data.get("temporal_windows", []),
            "summary": drift_data.get("summary", {}),
        }
    if ood_data:
        export["oodSummary"] = {
            "trainingStats": ood_data.get("training_stats", {}),
            "testDistribution": ood_data.get("test_distribution", {}),
        }

    with open(OUT_PATH, "w") as f:
        json.dump(export, f)
    print(f"[PRECOMPUTE] Wrote dashboard dataset -> {OUT_PATH} "
          f"({len(patients_out)} full-detail patients, {len(cohort_index['patientId'])} cohort-index, "
          f"{sum(len(p['hourly']['hour']) for p in patients_out)} patient-hours)")
    print(f"[PRECOMPUTE] File size: {round(len(json.dumps(export))/1024, 1)} KB")


if __name__ == "__main__":
    main()
