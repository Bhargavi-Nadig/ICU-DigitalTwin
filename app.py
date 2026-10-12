"""
ICU-DT :: Serving backend (path to live deployment / cloud + MIMIC hot-swap)
------------------------------------------------------------------------------
This is a minimal, REAL, runnable FastAPI service — not just a design doc.
It loads whichever model version model_registry.json points at, and exposes
a /predict endpoint. Swapping to a MIMIC-trained model later means:
  1. Add the new version's folder + registry entry (see model_registry.json)
  2. Flip "active_version" in model_registry.json
  3. POST /admin/reload  <-- reloads models in-process, no restart needed
     (or restart the process if you'd rather not run an admin endpoint in
     prod — either way, no image rebuild / redeploy required, AS LONG AS
     model/ is a mounted volume, not baked into the container image)

Run locally:
    pip install fastapi uvicorn --break-system-packages
    uvicorn app:app --reload --port 8000
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"condition": "Sepsis", "horizon": 6, "features": {...}}'
"""
import importlib.util
import pandas as pd
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

BASE_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = BASE_DIR / "model_registry.json"

ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "dev-token-change-in-production")

CONDITIONS_VALID = {"Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"}
HORIZONS_VALID = {6, 12, 24}

app = FastAPI(title="ICU-DT Serving API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

_state = {
    "registry": None, "production_models": None, "shap_explainers": None,
    "active_version": None, "version_info": None,
    "itshap": None, "surrogate_fidelity": None,
    "ood_detector": None, "counterfactuals": None, "drift_report": None,
    "ensemble_models": None, "ensemble_metrics": None,
    "publication_evaluation": None,
}


def _load_json_optional(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _load_pickle_safe(path):
    """Load pickle with existence check — avoids loading arbitrary paths."""
    path = Path(path)
    if not path.exists():
        return None
    if not str(path.resolve()).startswith(str(BASE_DIR.resolve())):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def verify_admin(authorization: str = Header(None)):
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid admin token")


def load_active_model():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    active = registry["active_version"]
    version_info = registry["versions"][active]
    model_dir = BASE_DIR / registry.get("current_model_dir", f"model/{active}")

    production_models = _load_pickle_safe(model_dir / "production_models.pkl")
    if production_models is None:
        raise RuntimeError(f"production_models.pkl not found in {model_dir}")
    shap_explainers = _load_pickle_safe(model_dir / "shap_explainers.pkl")
    ensemble_models = _load_pickle_safe(model_dir / "ensemble_models.pkl")

    _state.update({
        "registry": registry, "production_models": production_models,
        "shap_explainers": shap_explainers, "active_version": active,
        "version_info": version_info,
        "itshap": _load_json_optional(model_dir / "itshap_attributions.json"),
        "surrogate_fidelity": _load_json_optional(model_dir / "surrogate_fidelity.json"),
        "ood_detector": _load_json_optional(model_dir / "ood_detector.json"),
        "counterfactuals": _load_json_optional(model_dir / "counterfactuals.json"),
        "drift_report": _load_json_optional(model_dir / "drift_report.json"),
        "ensemble_models": ensemble_models,
        "ensemble_metrics": _load_json_optional(model_dir / "ensemble_metrics.json"),
        "publication_evaluation": _load_json_optional(model_dir / "publication_evaluation.json"),
    })
    return active, len(production_models)


@app.on_event("startup")
def startup():
    version, n_tasks = load_active_model()
    extras = [k for k in ("itshap", "surrogate_fidelity", "ood_detector", "counterfactuals", "drift_report")
              if _state[k] is not None]
    print(f"[SERVE] Loaded model version '{version}' ({n_tasks} tasks, extras: {extras})")


class PredictRequest(BaseModel):
    condition: str
    horizon: int
    features: Dict[str, float]
    explain: Optional[bool] = False

    @validator("condition")
    def validate_condition(cls, v):
        if v not in CONDITIONS_VALID:
            raise ValueError(f"condition must be one of {CONDITIONS_VALID}")
        return v

    @validator("horizon")
    def validate_horizon(cls, v):
        if v not in HORIZONS_VALID:
            raise ValueError(f"horizon must be one of {HORIZONS_VALID}")
        return v

    @validator("features")
    def validate_features(cls, v):
        for key, val in v.items():
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                raise ValueError(f"Feature '{key}' must be a finite number")
        return v

class MimicScenarioRequest(BaseModel):
    hour: int
    changes: Dict[str, float]

    
    @validator("changes")
    def validate_changes(cls, v):
        allowed_ranges = {
            "HR": (20, 250),
            "SBP": (40, 300),
            "Resp": (1, 80),
            "TempC": (25, 45),
            "FiO2": (0.21, 1.0),
        }

        if not v:
            raise ValueError("Provide at least one observation to change")

        invalid = set(v) - set(allowed_ranges)
        if invalid:
            raise ValueError(f"Unsupported fields: {sorted(invalid)}")

        for key, value in v.items():
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite")

            low, high = allowed_ranges[key]
            if not low <= value <= high:
                raise ValueError(
                    f"{key} must be between {low} and {high}"
                )

        return v


@app.get("/health")
def health():
    return {"status": "ok", "active_version": _state["active_version"]}


@app.get("/admin/model-info")
def model_info(_=Depends(verify_admin)):
    return {
        "active_version": _state["active_version"],
        "version_info": _state["version_info"],
        "available_versions": list(_state["registry"]["versions"].keys()),
        "n_tasks_loaded": len(_state["production_models"]),
        "extras_loaded": [k for k in ("itshap", "surrogate_fidelity", "ood_detector",
                                       "counterfactuals", "drift_report",
                                       "ensemble_models", "publication_evaluation")
                          if _state[k] is not None],
        "ensemble_available": _state["ensemble_models"] is not None,
    }


@app.post("/admin/reload")
def reload_model(_=Depends(verify_admin)):
    try:
        version, n_tasks = load_active_model()
        return {"status": "reloaded", "active_version": version, "n_tasks": n_tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.post("/predict")
def predict(req: PredictRequest):
    task = f"{req.condition}_{req.horizon}h"
    models = _state["production_models"]
    if task not in models:
        raise HTTPException(status_code=404, detail=f"No model for task '{task}'. "
                             f"Available: {list(models.keys())}")
    entry = models[task]
    clf, mean, std, feature_cols = entry["model"], entry["mean"], entry["std"], entry["feature_cols"]

    missing = [c for c in feature_cols if c not in req.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing {len(missing)} required features, "
                             f"e.g. {missing[:5]}")

    x = np.array([[req.features[c] for c in feature_cols]])
    xz = (x - mean) / std
    risk = float(clf.predict_proba(xz)[:, 1][0])

    response = {"task": task, "risk": round(risk, 4), "model_version": _state["active_version"]}

    if req.explain and _state["shap_explainers"] is not None:
        explainer = _state["shap_explainers"].get(req.condition)
        if explainer is not None:
            shap_vals = explainer.shap_values(xz)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[-1]
            shap_vals = np.array(shap_vals).ravel()
            order = np.argsort(-np.abs(shap_vals))[:6]
            response["top_features"] = [
                {"feature": feature_cols[i], "shap": round(float(shap_vals[i]), 4)} for i in order
            ]

    return response


# ---- New endpoints for the 5 research features ----

@app.get("/itshap/{patient_id}")
def get_itshap(patient_id: str):
    if _state["itshap"] is None:
        raise HTTPException(status_code=404, detail="IT-SHAP data not available")
    data = _state["itshap"].get(patient_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No IT-SHAP data for patient {patient_id}")
    return data


@app.get("/surrogate-fidelity")
def get_surrogate_fidelity():
    if _state["surrogate_fidelity"] is None:
        raise HTTPException(status_code=404, detail="Surrogate fidelity data not available")
    return _state["surrogate_fidelity"]["condition_fidelity"]


@app.get("/surrogate-fidelity/{patient_id}")
def get_patient_fidelity(patient_id: str):
    if _state["surrogate_fidelity"] is None:
        raise HTTPException(status_code=404, detail="Surrogate fidelity data not available")
    data = _state["surrogate_fidelity"]["patient_fidelity"].get(patient_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No fidelity data for patient {patient_id}")
    return data


@app.get("/ood/{patient_id}")
def get_ood(patient_id: str):
    if _state["ood_detector"] is None:
        raise HTTPException(status_code=404, detail="OOD detection data not available")
    data = _state["ood_detector"]["patient_ood"].get(patient_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No OOD data for patient {patient_id}")
    return data


@app.get("/ood-summary")
def get_ood_summary():
    if _state["ood_detector"] is None:
        raise HTTPException(status_code=404, detail="OOD detection data not available")
    return {
        "training_stats": _state["ood_detector"]["training_stats"],
        "test_distribution": _state["ood_detector"]["test_distribution"],
    }


@app.get("/counterfactuals/{patient_id}")
def get_counterfactuals(patient_id: str):
    if _state["counterfactuals"] is None:
        raise HTTPException(status_code=404, detail="Counterfactual data not available")
    data = _state["counterfactuals"]["patient_counterfactuals"].get(patient_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No counterfactual data for patient {patient_id}")
    return data


@app.get("/drift-report")
def get_drift_report():
    if _state["drift_report"] is None:
        raise HTTPException(status_code=404, detail="Drift report not available")
    return {
        "prediction_drift": _state["drift_report"]["prediction_drift"],
        "temporal_windows": _state["drift_report"]["temporal_windows"],
        "summary": _state["drift_report"]["summary"],
    }


@app.get("/ensemble-metrics")
def get_ensemble_metrics():
    if _state["ensemble_metrics"] is None:
        raise HTTPException(status_code=404, detail="Ensemble metrics not available")
    return _state["ensemble_metrics"]


@app.get("/publication-evaluation/{condition}/{horizon}")
def get_publication_eval(condition: str, horizon: int):
    if condition not in CONDITIONS_VALID:
        raise HTTPException(status_code=422, detail=f"condition must be one of {CONDITIONS_VALID}")
    if horizon not in HORIZONS_VALID:
        raise HTTPException(status_code=422, detail=f"horizon must be one of {HORIZONS_VALID}")
    if _state["publication_evaluation"] is None:
        raise HTTPException(status_code=404, detail="Publication evaluation not available")
    task = f"{condition}_{horizon}h"
    data = _state["publication_evaluation"].get(task)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No evaluation data for {task}")
    return data


@app.get("/model-version")
def get_model_version():
    return {
        "active_version": _state["active_version"],
        "version_info": _state["version_info"],
        "ensemble_available": _state["ensemble_models"] is not None,
        "publication_evaluation_available": _state["publication_evaluation"] is not None,
        "proxy_label_disclaimer": (
            "ARF, Cardiac, AKI, and Coagulopathy labels are rule-based clinical "
            "proxies derived from established clinical thresholds, NOT adjudicated "
            "ground truth. Only SepsisLabel is expert-annotated."
        ),
    }
# ---- MIMIC-IV exploratory replay endpoint ----

@app.post("/mimic/replay")
def mimic_replay():
    import pandas as pd

    csv_path = (
    Path(os.getenv("MIMIC_DATA_DIR", str(Path(__file__).resolve().parent / "mimic_data")))
    / "mimic_demo_stay_32359580_features.csv"
)

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"MIMIC feature CSV not found: {csv_path}"
        )

    try:
        df = pd.read_csv(csv_path)
        results = []

        models = _state["production_models"]

        for _, row in df.iterrows():
            record = {"hour": int(row["hour"])}

            for task, entry in models.items():
                feature_cols = entry["feature_cols"]
                missing = [
                    col for col in feature_cols
                    if col not in df.columns or pd.isna(row[col])
                ]

                if missing:
                    raise HTTPException(
                        status_code=422,
                        detail=f"{task}: missing or invalid features: {missing[:5]}"
                    )

                x = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)

                if not np.isfinite(x).all():
                    raise HTTPException(
                        status_code=422,
                        detail=f"{task}: non-finite feature values"
                    )

                mean = entry["mean"]
                std = entry["std"]

                if np.any(std == 0):
                    raise HTTPException(
                        status_code=500,
                        detail=f"{task}: zero standard deviation in model"
                    )

                xz = (x - mean) / std
                score = float(entry["model"].predict_proba(xz)[0, 1])
                record[f"{task}_score"] = round(score, 4)

            record["interpretation"] = (
                "Exploratory only; not clinically validated"
            )
            results.append(record)

        return {
            "status": "success",
            "model_version": _state["active_version"],
            "rows_processed": len(results),
            "tasks": len(models),
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MIMIC replay failed: {e}"
        )

# ---- MIMIC-IV single-hour prediction endpoint ----

@app.get("/mimic/predict-hour/{hour}")
def mimic_predict_hour(hour: int):
    import pandas as pd

    csv_path = (
    Path(os.getenv("MIMIC_DATA_DIR", str(Path(__file__).resolve().parent / "mimic_data")))
    / "mimic_demo_stay_32359580_features.csv"
)

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"MIMIC feature CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    selected = df[df["hour"] == hour]

    if selected.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No MIMIC data available for hour {hour}"
        )

    row = selected.iloc[0]
    models = _state["production_models"]
    results = {}

    for task, entry in models.items():
        feature_cols = entry["feature_cols"]
        x = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)

        if not np.isfinite(x).all():
            raise HTTPException(
                status_code=422,
                detail=f"Invalid feature values for {task}"
            )

        mean = entry["mean"]
        std = entry["std"]

        if np.any(std == 0):
            raise HTTPException(
                status_code=500,
                detail=f"Zero standard deviation for {task}"
            )

        xz = (x - mean) / std
        results[f"{task}_score"] = round(
            float(entry["model"].predict_proba(xz)[0, 1]), 4
        )

    return {
        "status": "success",
        "hour": hour,
        "model_version": _state["active_version"],
        "tasks": results,
        "interpretation": "Exploratory only; not clinically validated"
    }

@app.get("/mimic/explain-hour/{hour}/{task}")
def mimic_explain_hour(hour: int, task: str):
    import pandas as pd
    import shap

    csv_path = (
    Path(os.getenv("MIMIC_DATA_DIR", str(Path(__file__).resolve().parent / "mimic_data")))
    / "mimic_demo_stay_32359580_features.csv"
)

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"MIMIC feature CSV not found: {csv_path}",
        )

    df = pd.read_csv(csv_path)
    selected = df[df["hour"] == hour]

    if selected.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No MIMIC data available for hour {hour}",
        )

    models = _state["production_models"]

    if task not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown prediction task: {task}",
        )

    entry = models[task]
    feature_cols = entry["feature_cols"]
    row = selected.iloc[0]
    x = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)

    if not np.isfinite(x).all():
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values for {task}",
        )

    mean = entry["mean"]
    std = entry["std"]

    if np.any(std == 0):
        raise HTTPException(
            status_code=500,
            detail=f"Zero standard deviation for {task}",
        )

    xz = (x - mean) / std
    model = entry["model"]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(xz)

        # Support common binary-classification SHAP output formats.
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1])[0]
        else:
            values = np.asarray(shap_values)

            if values.ndim == 3:
                values = values[0, :, -1]
            elif values.ndim == 2:
                values = values[0]

        contributions = [
            {
                "feature": str(name),
                "value": float(row[name]),
                "contribution": float(contribution),
            }
            for name, contribution in zip(feature_cols, values)
        ]

        contributions.sort(
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )

        return {
            "status": "success",
            "hour": hour,
            "task": task,
            "prediction": float(model.predict_proba(xz)[0, 1]),
            "features": contributions[:10],
            "interpretation": (
                "Exploratory model explanation"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "TreeExplainer could not explain this model. "
                f"Check model compatibility: {exc}"
            ),
        )

@app.post("/mimic/scenario")
def mimic_scenario(request: MimicScenarioRequest):
    csv_path = (
        Path(os.getenv("MIMIC_DATA_DIR", str(Path(__file__).resolve().parent / "mimic_data")))
        / "mimic_demo_stay_32359580_hourly.csv"
    )
    features_path = (
        Path(os.getenv("MIMIC_DATA_DIR", str(Path(__file__).resolve().parent / "mimic_data")))
        / "mimic_demo_stay_32359580_features.csv"
    )
    etl_path = Path(__file__).resolve().parent / "scripts" / "01_etl_pipeline.py"
    medians_path = Path(__file__).resolve().parent / "data" / "feature_medians.csv"

    for path in [csv_path, features_path, etl_path, medians_path]:
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Required file not found: {path}")

    raw = pd.read_csv(csv_path)
    baseline_features = pd.read_csv(features_path)

    original_row = raw.loc[raw["hour"] == request.hour]
    if original_row.empty:
        raise HTTPException(status_code=404, detail=f"No observations available for hour {request.hour}")

    baseline_row = baseline_features.loc[baseline_features["hour"] == request.hour]
    if baseline_row.empty:
        raise HTTPException(status_code=404, detail=f"No baseline features available for hour {request.hour}")

    # Keep baseline scores from the existing precomputed feature row.
    models = _state["production_models"]
    if not models:
        raise HTTPException(status_code=503, detail="Production models are not loaded")

    def score_row(row):
        scores = {}
        for task, entry in models.items():
            cols = entry["feature_cols"]
            x = row[cols].to_numpy(dtype=float).reshape(1, -1)
            if not np.isfinite(x).all():
                raise HTTPException(status_code=422, detail=f"Non-finite features for {task}")
            mean, std = entry["mean"], entry["std"]
            if np.any(std == 0):
                raise HTTPException(status_code=500, detail=f"Invalid scaling values for {task}")
            xz = (x - mean) / std
            scores[f"{task}_score"] = round(float(entry["model"].predict_proba(xz)[0, 1]), 4)
        return scores

    baseline_scores = score_row(baseline_row.iloc[0])

    # Apply hypothetical changes only to a copy of the raw observations.
    scenario_raw = raw.copy()
    for field, value in request.changes.items():
        scenario_raw.loc[scenario_raw["hour"] == request.hour, field] = value

    # Rebuild features using the project's existing ETL functions.
    spec = importlib.util.spec_from_file_location("scenario_etl", etl_path)
    etl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(etl)

    scenario_df = scenario_raw.rename(columns={"TempC": "Temp"}).copy()
    scenario_df["patient_id"] = "mimic_demo_32359580"
    scenario_df["hour"] = scenario_df["hour"].astype(int)
    scenario_df["ICULOS"] = scenario_df["hour"]

    for col in etl.ALL_RAW_COLS:
        if col not in scenario_df.columns:
            scenario_df[col] = np.nan

    scenario_df["SepsisLabel"] = np.nan
    scenario_df = etl.engineer_features(scenario_df)

    median_df = pd.read_csv(medians_path, index_col=0)
    training_medians = median_df.iloc[:, 0]

    scenario_df["SaO2"] = scenario_df["SaO2"].fillna(training_medians.get("SaO2"))
    scenario_df["PF_ratio"] = scenario_df["SaO2"] / scenario_df["FiO2"].replace(0, np.nan)
    scenario_df["PF_ratio"] = scenario_df["PF_ratio"].fillna(training_medians.get("PF_ratio"))

    excluded = {
        "patient_id", "hour", "Unit1", "Unit2", "SepsisLabel",
        "ARF_proxy", "Cardiac_proxy", "AKI_proxy", "Coagulopathy_proxy"
    }
    feature_cols = [c for c in scenario_df.columns if c not in excluded]
    scenario_df = scenario_df.sort_values(["patient_id", "hour"]).copy()
    scenario_df[feature_cols] = scenario_df.groupby("patient_id")[feature_cols].ffill()

    for col in feature_cols:
        median = training_medians.get(col)
        if median is not None and pd.notna(median):
            scenario_df[col] = scenario_df[col].fillna(median)

    scenario_df = etl.build_window_summary_features(scenario_df)

    scenario_row = scenario_df.loc[scenario_df["hour"] == request.hour]
    if scenario_row.empty:
        raise HTTPException(status_code=404, detail=f"Scenario features unavailable for hour {request.hour}")

    scenario_scores = score_row(scenario_row.iloc[0])

    return {
        "status": "success",
        "hour": request.hour,
        "changes": request.changes,
        "baseline_scores": baseline_scores,
        "scenario_scores": scenario_scores,
        "interpretation": "Exploratory what-if simulation only; not clinically validated."
    }


@app.get("/mimic/observations/{hour}")
def mimic_observations(hour: int):
    import pandas as pd

    csv_path = (
    Path(os.getenv("MIMIC_DATA_DIR", str(Path(__file__).resolve().parent / "mimic_data")))
    / "mimic_demo_stay_32359580_hourly.csv"
)

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"MIMIC observation CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    row = df.loc[df["hour"] == hour]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No observations available for hour {hour}"
        )

    fields = [
        "HR", "SBP", "DBP", "MAP", "Resp", "TempC",
        "O2Sat", "FiO2", "pH", "PaCO2", "Lactate",
        "Creatinine", "BUN", "WBC", "Platelets", "Glucose"
    ]

    record = row.iloc[0]
    observations = {}

    for field in fields:
        value = record.get(field)
        observations[field] = (
            None if pd.isna(value) else float(value)
        )

    return {
        "status": "success",
        "hour": hour,
        "observations": observations
    }

@app.get("/mimic/hours")
def mimic_available_hours():
    import pandas as pd

    csv_path = (
    Path(os.getenv("MIMIC_DATA_DIR", str(Path(__file__).resolve().parent / "mimic_data")))
    / "mimic_demo_stay_32359580_features.csv"
)

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"MIMIC feature CSV not found: {csv_path}",
        )

    df = pd.read_csv(csv_path)

    if "hour" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="The MIMIC feature CSV has no hour column",
        )

    hours = sorted(
        df["hour"].dropna().astype(int).unique().tolist()
    )

    return {
        "status": "success",
        "count": len(hours),
        "hours": hours,
    }
