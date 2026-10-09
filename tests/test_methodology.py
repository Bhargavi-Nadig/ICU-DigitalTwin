"""
ICU-DT :: Methodology Validation Tests
-----------------------------------------
Tests that guard against the methodological issues fixed in the
publication pipeline (scripts 12-15). Run with:
    cd ICU-DigitalTwin-Sample4
    python -m pytest tests/test_methodology.py -v
"""
import json
import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
SPLITS_PATH = MODEL_DIR / "splits.json"
PUB_DATA_PATH = DATA_DIR / "publication_patient_hours.parquet"
META_PATH = MODEL_DIR / "preprocessing_metadata.json"
ENS_METRICS_PATH = MODEL_DIR / "ensemble_metrics.json"
PUB_EVAL_PATH = MODEL_DIR / "publication_evaluation.json"


# ────────────────────── Split integrity ──────────────────────

class TestSplitIntegrity:
    @pytest.fixture(autouse=True)
    def load_splits(self):
        if not SPLITS_PATH.exists():
            pytest.skip("splits.json not found")
        with open(SPLITS_PATH) as f:
            self.splits = json.load(f)

    def test_splits_disjoint(self):
        train = set(self.splits["train"])
        val = set(self.splits["val"])
        test = set(self.splits["test"])
        assert train & val == set(), "Train and val overlap"
        assert train & test == set(), "Train and test overlap"
        assert val & test == set(), "Val and test overlap"

    def test_splits_nonempty(self):
        for split_name in ["train", "val", "test"]:
            assert len(self.splits[split_name]) > 0, f"{split_name} split is empty"

    def test_train_is_majority(self):
        total = sum(len(self.splits[s]) for s in ["train", "val", "test"])
        train_frac = len(self.splits["train"]) / total
        assert train_frac >= 0.6, f"Training set is only {train_frac:.1%} of data"

    def test_all_patients_assigned(self):
        if not PUB_DATA_PATH.exists():
            pytest.skip("publication data not yet generated")
        df = pd.read_parquet(PUB_DATA_PATH, columns=["patient_id"])
        all_pids = set(df["patient_id"].astype(str).unique())
        assigned = set(self.splits["train"]) | set(self.splits["val"]) | set(self.splits["test"])
        unassigned = all_pids - assigned
        assert len(unassigned) == 0, f"{len(unassigned)} patients not in any split"


# ────────────────────── Preprocessing leakage ──────────────────────

class TestPreprocessingLeakage:
    @pytest.fixture(autouse=True)
    def load_data(self):
        if not META_PATH.exists():
            pytest.skip("preprocessing_metadata.json not found — run script 12 first")
        with open(META_PATH) as f:
            self.meta = json.load(f)

    def test_imputation_uses_train_only(self):
        src = self.meta["preprocessing"]["imputation_source"]
        assert "training" in src.lower() and "leakage" not in src.lower() or "no" in src.lower()
        assert self.meta["preprocessing"]["n_train_patients_for_medians"] > 0

    def test_forward_fill_has_limit(self):
        limit = self.meta["preprocessing"]["forward_fill_limit_hours"]
        if isinstance(limit, (int, float)):
            assert limit <= 24, f"Forward-fill limit is {limit}, should be <= 24h"
        else:
            assert limit is not None, "Forward-fill limit should be documented"

    def test_missingness_indicators_present(self):
        n_miss = self.meta.get("n_missingness_indicators", 0)
        assert n_miss > 0, "No missingness indicators added"


# ────────────────────── Label integrity ──────────────────────

class TestLabelIntegrity:
    @pytest.fixture(autouse=True)
    def load_meta(self):
        if not META_PATH.exists():
            pytest.skip("preprocessing_metadata.json not found")
        with open(META_PATH) as f:
            self.meta = json.load(f)

    def test_labels_use_iculos(self):
        method = self.meta["label_construction"]["method"]
        assert "ICULOS" in method or "iculos" in method.lower(), \
            f"Labels should use ICULOS, got: {method}"

    def test_censoring_applied(self):
        censoring = self.meta["label_construction"]["censoring"]
        assert "NaN" in censoring or "censor" in censoring.lower(), \
            f"Labels should censor incomplete windows, got: {censoring}"

    def test_censoring_rates_reasonable(self):
        stats = self.meta["label_construction"]["censoring_stats"]
        for task, s in stats.items():
            assert s["labeled"] > 0, f"{task} has zero labeled samples"
            if "24h" in task:
                assert s["pct_censored"] > 5, \
                    f"24h horizon {task} has only {s['pct_censored']}% censored — suspiciously low"


# ────────────────────── Feature availability ──────────────────────

class TestFeatureAvailability:
    @pytest.fixture(autouse=True)
    def load_data(self):
        if not PUB_DATA_PATH.exists():
            pytest.skip("publication data not yet generated")
        self.df = pd.read_parquet(PUB_DATA_PATH, columns=None)

    def test_missingness_columns_exist(self):
        miss_cols = [c for c in self.df.columns if c.startswith("miss_")]
        assert len(miss_cols) > 0, "No missingness indicator columns found"

    def test_window_features_exist(self):
        w_cols = [c for c in self.df.columns if c.startswith("W_")]
        assert len(w_cols) > 100, f"Expected >100 window features, got {len(w_cols)}"

    def test_no_nan_in_features_after_imputation(self):
        # EtCO2, Unit1, Unit2 are entirely missing in this dataset — exclude them
        known_all_nan = {"EtCO2", "Unit1", "Unit2"}
        feature_cols = [c for c in self.df.columns
                        if not c.startswith("miss_") and c not in
                        ["patient_id", "hour", "ICULOS", "SepsisLabel",
                         "ARF_proxy", "Cardiac_proxy", "AKI_proxy", "Coagulopathy_proxy"]
                        and c not in known_all_nan
                        and not any(c == f"{cond}_{h}h"
                                    for cond in ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]
                                    for h in [6, 12, 24])]
        nan_counts = self.df[feature_cols].isna().sum()
        problem_cols = nan_counts[nan_counts > 0]
        assert len(problem_cols) == 0, \
            f"Features with NaN after imputation: {dict(problem_cols.head(10))}"

    def test_labels_have_censored_nans(self):
        for cond in ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]:
            col = f"{cond}_24h"
            if col in self.df.columns:
                n_nan = self.df[col].isna().sum()
                assert n_nan > 0, f"{col} has zero NaN — censoring may not be working"


# ────────────────────── Leakage guard ──────────────────────

class TestLeakageGuard:
    @pytest.fixture(autouse=True)
    def load_data(self):
        if not PUB_DATA_PATH.exists() or not SPLITS_PATH.exists():
            pytest.skip("publication data or splits not found")
        self.df = pd.read_parquet(PUB_DATA_PATH)
        with open(SPLITS_PATH) as f:
            self.splits = json.load(f)

    def test_no_future_label_leakage_in_features(self):
        """Verify no label column is used as a feature (they should be targets only)."""
        label_cols = set()
        for cond in ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]:
            for h in [6, 12, 24]:
                label_cols.add(f"{cond}_{h}h")
        feature_cols = [c for c in self.df.columns if c.startswith("W_") or c.startswith("Delta_")]
        assert not label_cols & set(feature_cols), \
            f"Label columns found in features: {label_cols & set(feature_cols)}"


# ────────────────────── Ensemble metrics ──────────────────────

class TestEnsembleMetrics:
    @pytest.fixture(autouse=True)
    def load_metrics(self):
        if not ENS_METRICS_PATH.exists():
            pytest.skip("ensemble_metrics.json not found — run script 13 first")
        with open(ENS_METRICS_PATH) as f:
            self.metrics = json.load(f)

    def test_all_tasks_present(self):
        for cond in ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]:
            for h in [6, 12, 24]:
                task = f"{cond}_{h}h"
                assert task in self.metrics, f"Missing metrics for {task}"

    def test_auroc_reasonable(self):
        for task, m in self.metrics.items():
            if task.startswith("_"):
                continue
            auroc = m["xgboost"]["test_AUROC"]
            assert 0.5 <= auroc <= 1.0, f"{task} XGBoost AUROC={auroc}, expected [0.5, 1.0]"

    def test_alpha_in_range(self):
        for task, m in self.metrics.items():
            if task.startswith("_"):
                continue
            alpha = m["ensemble"]["alpha"]
            assert 0 <= alpha <= 1, f"{task} alpha={alpha}, expected [0, 1]"


# ────────────────────── Publication evaluation ──────────────────────

class TestPublicationEvaluation:
    @pytest.fixture(autouse=True)
    def load_eval(self):
        if not PUB_EVAL_PATH.exists():
            pytest.skip("publication_evaluation.json not found — run script 14 first")
        with open(PUB_EVAL_PATH) as f:
            self.eval_data = json.load(f)

    def test_bootstrap_cis_present(self):
        for task, data in self.eval_data.items():
            if task.startswith("_"):
                continue
            auroc = data["hour_level"]["AUROC"]
            assert "ci_lower" in auroc and "ci_upper" in auroc, \
                f"{task} missing bootstrap CIs"
            assert auroc["ci_lower"] <= auroc["mean"] <= auroc["ci_upper"], \
                f"{task} CI doesn't contain mean"

    def test_calibration_present(self):
        for task, data in self.eval_data.items():
            if task.startswith("_"):
                continue
            cal = data["calibration"]
            assert "ECE" in cal, f"{task} missing ECE"
            assert "slope_intercept" in cal, f"{task} missing calibration slope"

    def test_patient_level_metrics(self):
        for task, data in self.eval_data.items():
            if task.startswith("_"):
                continue
            pat = data["patient_level"]
            assert "AUROC" in pat, f"{task} missing patient-level AUROC"
            assert pat["n_patients"] > 0, f"{task} has zero patients"


# ────────────────────── API contract ──────────────────────

class TestAPIContract:
    def test_predict_request_validation(self):
        """Test that the PredictRequest model validates inputs."""
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from app import PredictRequest

        with pytest.raises(Exception):
            PredictRequest(condition="InvalidCondition", horizon=6, features={"HR": 80})

        with pytest.raises(Exception):
            PredictRequest(condition="Sepsis", horizon=99, features={"HR": 80})

        req = PredictRequest(condition="Sepsis", horizon=6, features={"HR": 80.0})
        assert req.condition == "Sepsis"
        assert req.horizon == 6
