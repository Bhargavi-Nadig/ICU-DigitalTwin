"""
ICU-DT :: IT-SHAP — Integrated Temporal SHAP for the existing LSTM
---------------------------------------------------------------------
Computes time-step x feature attribution matrices for the BiLSTM model.
This is an ADDITIONAL explainability path — the existing XGBoost +
TreeExplainer pipeline is completely unchanged.

Method: For each time step t in the 24h input window, we mask (zero out)
that time step and measure the change in the LSTM's predicted probability.
This gives a T x F attribution matrix showing which features at which
time points contributed most to the prediction.  We refine this with a
leave-one-feature-out pass at the most important time steps to get
per-feature granularity.

Output: model/itshap_attributions.json — per-patient, per-condition
temporal attribution data for dashboard consumption.
"""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
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
TASK_NAMES = train_mod.TASK_NAMES
LOOKBACK = train_mod.LOOKBACK

CONDITION_TASK_IDX = {c: i * len(HORIZONS) for i, c in enumerate(CONDITIONS)}


def load_lstm():
    with open(f"{MODEL_DIR}/model_config.json") as f:
        cfg = json.load(f)
    stats = np.load(f"{MODEL_DIR}/norm_stats.npz")
    mean, std = stats["mean"], stats["std"]

    model = train_mod.MultiTaskSepsisLSTM(
        n_features=len(FEATURE_COLS), n_tasks=len(TASK_NAMES),
        hidden=cfg["hidden_dim"], layers=cfg["layers"],
    )
    model.load_state_dict(torch.load(f"{MODEL_DIR}/sepsis_lstm.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, mean, std


def build_window(patient_df, t_idx, mean, std, lookback=LOOKBACK):
    X = ((patient_df[FEATURE_COLS].values - mean) / std).astype(np.float32)
    start = max(0, t_idx - lookback + 1)
    window = X[start:t_idx + 1]
    pad_len = lookback - window.shape[0]
    if pad_len > 0:
        window = np.vstack([np.zeros((pad_len, window.shape[1]), dtype=np.float32), window])
    return window


def temporal_attribution(model, window, condition):
    """Compute time-step level attribution by masking each time step."""
    task_idx = CONDITION_TASK_IDX[condition]
    T, F = window.shape

    base_input = torch.from_numpy(window).unsqueeze(0)
    with torch.no_grad():
        base_logit = model(base_input)[0, task_idx].item()
    base_prob = 1 / (1 + np.exp(-base_logit))

    timestep_importance = np.zeros(T)
    for t in range(T):
        masked = window.copy()
        masked[t, :] = 0.0
        with torch.no_grad():
            logit = model(torch.from_numpy(masked).unsqueeze(0))[0, task_idx].item()
        prob = 1 / (1 + np.exp(-logit))
        timestep_importance[t] = base_prob - prob

    return timestep_importance, base_prob


def feature_attribution_at_timestep(model, window, condition, t_idx):
    """For a specific time step, compute per-feature attribution."""
    task_idx = CONDITION_TASK_IDX[condition]
    F = window.shape[1]

    base_input = torch.from_numpy(window).unsqueeze(0)
    with torch.no_grad():
        base_logit = model(base_input)[0, task_idx].item()
    base_prob = 1 / (1 + np.exp(-base_logit))

    feature_importance = np.zeros(F)
    for f in range(F):
        masked = window.copy()
        masked[t_idx, f] = 0.0
        with torch.no_grad():
            logit = model(torch.from_numpy(masked).unsqueeze(0))[0, task_idx].item()
        prob = 1 / (1 + np.exp(-logit))
        feature_importance[f] = base_prob - prob

    return feature_importance


def compute_itshap_for_patient(model, patient_df, mean, std, top_timesteps=5, top_features=6):
    """Compute IT-SHAP attributions for a single patient at their last hour."""
    t_idx = len(patient_df) - 1
    window = build_window(patient_df, t_idx, mean, std)

    result = {}
    for cond in CONDITIONS:
        ts_importance, base_prob = temporal_attribution(model, window, cond)

        actual_hours = patient_df["hour"].values
        window_start = max(0, t_idx - LOOKBACK + 1)
        window_hours = actual_hours[window_start:t_idx + 1]
        pad_len = LOOKBACK - len(window_hours)
        if pad_len > 0:
            window_hours = np.concatenate([np.full(pad_len, -1), window_hours])

        top_t_indices = np.argsort(-np.abs(ts_importance))[:top_timesteps]

        temporal_detail = []
        for t in top_t_indices:
            feat_imp = feature_attribution_at_timestep(model, window, cond, int(t))
            top_f_indices = np.argsort(-np.abs(feat_imp))[:top_features]

            temporal_detail.append({
                "timestep": int(t),
                "hour": int(window_hours[t]) if window_hours[t] >= 0 else None,
                "timestep_importance": round(float(ts_importance[t]), 4),
                "features": [
                    {"feature": FEATURE_COLS[fi], "attribution": round(float(feat_imp[fi]), 4)}
                    for fi in top_f_indices
                ],
            })

        ts_for_heatmap = []
        for t in range(len(ts_importance)):
            if abs(ts_importance[t]) > 0.001:
                ts_for_heatmap.append({
                    "timestep": int(t),
                    "hour": int(window_hours[t]) if window_hours[t] >= 0 else None,
                    "importance": round(float(ts_importance[t]), 4),
                })

        result[cond] = {
            "lstm_risk": round(float(base_prob), 4),
            "temporal_detail": temporal_detail,
            "timestep_heatmap": ts_for_heatmap,
        }

    return result


def main():
    print("[IT-SHAP] Loading LSTM model...")
    model, mean, std = load_lstm()

    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)

    with open(f"{MODEL_DIR}/splits.json") as f:
        splits = json.load(f)

    with open(str(BASE_DIR / "artifacts" / "dashboard_data.json")) as f:
        dashboard = json.load(f)
    dashboard_pids = [p["patientId"] for p in dashboard["patients"]]

    test_df = df[df.patient_id.isin(splits["test"])]
    target_pids = [pid for pid in dashboard_pids if pid in test_df.patient_id.values]

    print(f"[IT-SHAP] Computing temporal attributions for {len(target_pids)} patients...")
    attributions = {}
    for i, pid in enumerate(target_pids):
        patient_df = test_df[test_df.patient_id == pid].sort_values("hour").reset_index(drop=True)
        if len(patient_df) < 3:
            continue
        attributions[pid] = compute_itshap_for_patient(model, patient_df, mean, std)
        if (i + 1) % 5 == 0:
            print(f"[IT-SHAP]   {i+1}/{len(target_pids)} patients done")

    out_path = f"{MODEL_DIR}/itshap_attributions.json"
    with open(out_path, "w") as f:
        json.dump(attributions, f)
    print(f"[IT-SHAP] Saved {len(attributions)} patient attributions -> {out_path}")

    for pid in list(attributions.keys())[:2]:
        for cond in CONDITIONS[:2]:
            data = attributions[pid][cond]
            print(f"  {pid}/{cond}: LSTM risk={data['lstm_risk']}, "
                  f"top timestep importance={data['temporal_detail'][0]['timestep_importance']:.4f}")


if __name__ == "__main__":
    main()
