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
