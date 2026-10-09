"""
ICU-DT :: Shared constants and label-building logic (no torch dependency).
"""
import numpy as np
import pandas as pd

LOOKBACK = 24
HORIZONS = [6, 12, 24]
CONDITIONS = ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]
SOURCE_COL = {"Sepsis": "SepsisLabel", "ARF": "ARF_proxy", "Cardiac": "Cardiac_proxy",
              "AKI": "AKI_proxy", "Coagulopathy": "Coagulopathy_proxy"}

TASK_NAMES = [f"{c}_{h}h" for c in CONDITIONS for h in HORIZONS]

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


def forward_max_label(s: pd.Series, horizon: int) -> np.ndarray:
    rev = s[::-1].reset_index(drop=True)
    rolled = rev.shift(1).rolling(horizon, min_periods=1).max()
    return rolled[::-1].reset_index(drop=True).values


def build_multi_task_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    for cond in CONDITIONS:
        src = SOURCE_COL[cond]
        for h in HORIZONS:
            df[f"{cond}_{h}h"] = df.groupby("patient_id")[src].transform(lambda s: forward_max_label(s, h))
    return df
