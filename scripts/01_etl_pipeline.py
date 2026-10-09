"""
ICU-DT :: Data Acquisition & ETL Service (FR-1) + Preprocessing Module (FR-2)
--------------------------------------------------------------------------
Ingests per-patient .psv time series (PhysioNet Sepsis Challenge format),
validates schema at the FILE level and reliability at the ROW level, engineers
derived clinical features + deterioration proxy labels, and writes one
consolidated, versioned patient-hour table.

Data-quality philosophy (see README "Data Quality Policy"):
  - CORE_COLS must exist as columns in the file AND be non-null on a given
    row for that row to be usable (identifiers, the ground-truth label, and
    the primary vitals every ICU monitor records). A row missing a CORE
    value is dropped — a patient's file is quarantined only if a CORE
    column is absent from its schema entirely (which does not happen in
    this dataset; every file shares the same 41-column PhysioNet schema).
  - OPTIONAL_COLS (labs drawn intermittently, e.g. Lactate, WBC, PTT) are
    NEVER used to drop a row. They are forward-filled within a patient's
    stay, then any remaining gap is filled with the population median.
    Losing a whole patient-hour because one infrequent lab wasn't drawn
    that hour would throw away far more good signal than it protects.

Run:  python3 01_etl_pipeline.py
"""
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = str(BASE_DIR / "data" / "training_setA")
OUT_PATH = str(BASE_DIR / "data" / "processed_patient_hours.parquet")
AUDIT_LOG = str(BASE_DIR / "data" / "ingestion_audit_log.csv")
MEDIANS_PATH = str(BASE_DIR / "data" / "feature_medians.csv")

# Identifiers + ground-truth label (always present, 0% missing in this dataset)
# plus the vitals that are CONTINUOUSLY monitored (HR/SpO2/MAP/Resp — bedside
# monitor derived, low missingness). A row is only usable for training if
# these are all present.
CORE_COLS = ["ICULOS", "SepsisLabel", "Age", "Gender", "HospAdmTime", "HR", "O2Sat", "MAP", "Resp"]

# Temp/SBP/DBP are charted intermittently (manual measurement, not continuous
# monitoring) with real, clinically-normal gaps between readings — treated
# like labs: forward-filled/imputed, never used to drop a row. Requiring them
# per-row would discard good data for no accuracy benefit (see README).
IMPUTED_VITALS = ["Temp", "SBP", "DBP"]

# Everything else: intermittently-drawn labs. Never drops a row — imputed.
ALL_RAW_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess",
    "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
    "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel",
]
OPTIONAL_COLS = [c for c in ALL_RAW_COLS if c not in CORE_COLS]


def load_all_patients(raw_dir: str):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.psv")))
    frames = []
    audit_rows = []
    total_rows_seen = 0
    total_rows_dropped_core = 0

    for f in files:
        pid = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_csv(f, sep="|")
        except Exception as e:
            audit_rows.append({"patient_id": pid, "status": "ERROR", "detail": str(e),
                                "n_records": 0, "n_core_dropped": 0})
            continue

        missing_schema = [c for c in CORE_COLS if c not in df.columns]
        if missing_schema:
            audit_rows.append({"patient_id": pid, "status": "QUARANTINED_SCHEMA",
                                "detail": f"file missing CORE columns entirely: {missing_schema}",
                                "n_records": len(df), "n_core_dropped": 0})
            continue

        n_before = len(df)
        total_rows_seen += n_before
        # row-level filter: only on CORE columns. Optional labs stay NaN for now
        # (imputed in resample_and_impute) and never cause a row to be dropped.
        df_clean = df.dropna(subset=CORE_COLS)
        n_dropped = n_before - len(df_clean)
        total_rows_dropped_core += n_dropped

        if len(df_clean) == 0:
            audit_rows.append({"patient_id": pid, "status": "QUARANTINED_EMPTY_AFTER_FILTER",
                                "detail": "no rows had all CORE fields present",
                                "n_records": n_before, "n_core_dropped": n_dropped})
            continue

        df_clean = df_clean.copy()
        df_clean["patient_id"] = pid
        df_clean["hour"] = df_clean["ICULOS"].astype(int)
        frames.append(df_clean)
        audit_rows.append({"patient_id": pid, "status": "OK", "detail": "",
                            "n_records": len(df_clean), "n_core_dropped": n_dropped})

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(AUDIT_LOG, index=False)

    n_ok = audit_df["status"].eq("OK").sum()
    n_quarantined = audit_df["status"].str.startswith("QUARANTINED").sum()
    n_error = audit_df["status"].eq("ERROR").sum()
    print(f"[ETL] Files: {n_ok} OK / {len(audit_df)} total "
          f"({n_quarantined} quarantined [schema or fully-unusable], {n_error} errored)")
    print(f"[ETL] Row-level filtering: {total_rows_dropped_core}/{total_rows_seen} "
          f"({100*total_rows_dropped_core/max(total_rows_seen,1):.2f}%) rows dropped for missing "
          f"CORE fields; optional-lab gaps were imputed, not dropped.")
    print(f"[ETL] Full audit trail -> {AUDIT_LOG}")

    return pd.concat(frames, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """FR-2.5: derived clinical features + deterioration proxy labels."""
    df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)

    df["ShockIndex"] = df["HR"] / df["SBP"].replace(0, np.nan)
    df["PulsePressure"] = df["SBP"] - df["DBP"]
    if "SaO2" in df.columns and "FiO2" in df.columns:
        df["PF_ratio"] = df["SaO2"] / df["FiO2"].replace(0, np.nan)

    for col in ["HR", "MAP", "Resp", "Temp"]:
        df[f"Delta_{col}"] = df.groupby("patient_id")[col].diff()
    if "Lactate" in df.columns:
        df["Delta_Lactate"] = df.groupby("patient_id")["Lactate"].diff()
    if "Creatinine" in df.columns:
        df["Delta_Creatinine"] = df.groupby("patient_id")["Creatinine"].diff()

    df["qSOFA_proxy"] = (
        (df["Resp"] >= 22).astype(int) + (df["SBP"] <= 100).astype(int) + (df["MAP"] < 65).astype(int)
    )

    # --- Deterioration proxy labels (current-hour clinical state flags) ---
    # RULE-BASED PROXIES from established clinical thresholds — not adjudicated
    # ground truth (unlike SepsisLabel). See README "Label Provenance".

    # ARF: hypoxemic respiratory failure (Berlin ARDS PF<300, or severe hypoxemia+tachypnea)
    pf = df.get("PF_ratio")
    df["ARF_proxy"] = (
        ((pf < 300) if pf is not None else False) | ((df["O2Sat"] < 90) & (df["Resp"] >= 24))
    ).astype(int)

    # Cardiac/hemodynamic collapse: sustained hypotension + compensatory shock index
    # or hyperlactatemia (tissue hypoperfusion marker)
    lact = df.get("Lactate")
    df["Cardiac_proxy"] = (
        (df["MAP"] < 65) & ((df["ShockIndex"] > 1.0) | ((lact >= 4) if lact is not None else False))
    ).astype(int)

    # AKI (Acute Kidney Injury): KDIGO-inspired — creatinine rise of >=0.3 mg/dL
    # from the patient's rolling 48h minimum ("baseline"), or >=1.5x that baseline.
    if "Creatinine" in df.columns:
        def aki_flags(g):
            baseline = g["Creatinine"].rolling(48, min_periods=1).min()
            rise_abs = g["Creatinine"] - baseline
            rise_rel = g["Creatinine"] / baseline.replace(0, np.nan)
            return ((rise_abs >= 0.3) | (rise_rel >= 1.5)).astype(int)
        df["AKI_proxy"] = df.groupby("patient_id", group_keys=False).apply(aki_flags)
    else:
        df["AKI_proxy"] = 0

    # Coagulopathy / DIC-risk: thrombocytopenia (<100 K/uL) with prolonged PTT
    # (>40s) when PTT is available; platelet-only flag as a fallback.
    if "Platelets" in df.columns:
        plt_low = df["Platelets"] < 100
        if "PTT" in df.columns:
            df["Coagulopathy_proxy"] = (plt_low & (df["PTT"] > 40)).astype(int)
        else:
            df["Coagulopathy_proxy"] = plt_low.astype(int)
    else:
        df["Coagulopathy_proxy"] = 0

    return df


def resample_and_impute(df: pd.DataFrame, feature_cols: list):
    """FR-2.1/2.2: forward-fill within a patient's stay, then fill any
    remaining leading gap with the population median (documented, reproducible)."""
    df = df.sort_values(["patient_id", "hour"])
    df[feature_cols] = df.groupby("patient_id")[feature_cols].ffill()
    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)
    return df, medians


# Base vitals/labs windowed for the tabular baselines below (raw signals only —
# not the already-derived ShockIndex/qSOFA/Delta_* features, which are
# themselves single-hour summaries and would just be redundant here).
WINDOW_BASE_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2", "BaseExcess",
    "HCO3", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
    "Chloride", "Creatinine", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "Hct", "Hgb", "WBC", "Platelets",
]
WINDOW_HOURS = 24  # matches the LSTM's LOOKBACK exactly — this is the whole point


def build_window_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gives the tabular baselines (LR/RF/XGBoost) the SAME 24h of context the
    LSTM sees, just pre-aggregated into engineered summary statistics instead
    of a raw sequence the model has to learn temporal structure from itself.
    Without this, a comparison between the LSTM (24h window) and single-hour
    tabular models is confounded by information access, not just model
    capacity — this makes it apples-to-apples (see project discussion log).

    Adds, per base column: rolling mean/std/min/max over the trailing
    WINDOW_HOURS, plus 6h and 24h lagged differences (short- and long-term
    trend). New columns are prefixed 'W_' and are additive — 02_train_lstm.py
    ignores them (it uses its own FEATURE_COLS list), so this does not change
    LSTM training at all.
    """
    df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    g = df.groupby("patient_id")
    new_cols = {}
    for col in WINDOW_BASE_COLS:
        if col not in df.columns:
            continue
        roll = g[col].rolling(WINDOW_HOURS, min_periods=1)
        new_cols[f"W_mean_{col}"] = roll.mean().reset_index(level=0, drop=True)
        new_cols[f"W_std_{col}"] = roll.std().reset_index(level=0, drop=True).fillna(0)
        new_cols[f"W_min_{col}"] = roll.min().reset_index(level=0, drop=True)
        new_cols[f"W_max_{col}"] = roll.max().reset_index(level=0, drop=True)
        shifted6 = g[col].shift(6)
        shifted24 = g[col].shift(23)
        first_val = g[col].transform("first")
        new_cols[f"W_diff6h_{col}"] = (df[col] - shifted6.fillna(first_val))
        new_cols[f"W_diff24h_{col}"] = (df[col] - shifted24.fillna(first_val))
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def main():
    os.makedirs(BASE_DIR / "data", exist_ok=True)
    print("[ETL] Loading raw patient files ...")
    raw = load_all_patients(RAW_DIR)
    print(f"[ETL] Raw table shape after row-level CORE filtering: {raw.shape}")

    print("[ETL] Engineering derived clinical features + proxy labels ...")
    feat = engineer_features(raw)

    label_cols = {"SepsisLabel", "ARF_proxy", "Cardiac_proxy", "AKI_proxy", "Coagulopathy_proxy"}
    exclude = {"patient_id", "hour", "Unit1", "Unit2"} | label_cols
    feature_cols = [c for c in feat.columns if c not in exclude]

    print("[ETL] Resampling / imputing optional-lab gaps ...")
    feat, medians = resample_and_impute(feat, feature_cols)
    medians.to_csv(MEDIANS_PATH)
    print(f"[ETL] Feature medians (imputation reference) -> {MEDIANS_PATH}")

    print(f"[ETL] Building {WINDOW_HOURS}h window-summary features for tabular baselines "
          f"(mean/std/min/max/diff6h/diff24h per base vital/lab — matches LSTM's lookback exactly)...")
    feat = build_window_summary_features(feat)
    window_cols = [c for c in feat.columns if c.startswith("W_")]
    print(f"[ETL] Added {len(window_cols)} window-summary columns.")

    feat.to_parquet(OUT_PATH, index=False)
    print(f"[ETL] Saved processed patient-hour table -> {OUT_PATH}  shape={feat.shape}")

    print(f"[ETL] Patients: {feat['patient_id'].nunique()}")
    for lbl in ["SepsisLabel", "ARF_proxy", "Cardiac_proxy", "AKI_proxy", "Coagulopathy_proxy"]:
        hrs = feat[lbl].sum()
        pts = feat.groupby("patient_id")[lbl].max().sum()
        print(f"  {lbl:20s} hours={hrs:6.0f} ({100*hrs/len(feat):.2f}%)  "
              f"patients-ever-positive={pts} ({100*pts/feat['patient_id'].nunique():.1f}%)")


if __name__ == "__main__":
    main()
