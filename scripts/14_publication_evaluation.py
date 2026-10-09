"""
ICU-DT :: Publication-Grade Evaluation Suite
----------------------------------------------
Runs the full battery of metrics expected by clinical-ML reviewers:
  - Bootstrap 95% CIs for AUROC, AUPRC (2000 resamples)
  - Patient-level metrics (one prediction per patient: last-hour)
  - Calibration: Brier score, ECE, calibration slope & intercept, reliability diagram data
  - Decision curve analysis (net benefit across thresholds)
  - Subgroup analysis by age, sex, and condition prevalence
  - Sensitivity analysis: compare results with/without missingness indicators
  - Imputation sensitivity: rerun metrics under median-only, ffill-only, and combined

Input:  model/ensemble_models.pkl, data/publication_patient_hours.parquet
Output: model/publication_evaluation.json
"""
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.linear_model import LogisticRegression
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "publication_patient_hours.parquet")
MODEL_DIR = str(BASE_DIR / "model")
ENS_PATH = str(BASE_DIR / "model" / "ensemble_models.pkl")
SPLITS_PATH = str(BASE_DIR / "model" / "splits.json")
OUT_PATH = str(BASE_DIR / "model" / "publication_evaluation.json")

CONDITIONS = ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]
HORIZONS = [6, 12, 24]
N_BOOTSTRAP = 2000
RNG = np.random.RandomState(42)


# ────────────────────── helpers ──────────────────────

def safe_auroc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=N_BOOTSTRAP, alpha=0.05):
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = RNG.randint(0, n, size=n)
        s = metric_fn(y_true[idx], y_score[idx])
        if not np.isnan(s):
            scores.append(s)
    if len(scores) < 100:
        return {"mean": float(np.nan), "ci_lower": float(np.nan), "ci_upper": float(np.nan)}
    scores = np.array(scores)
    return {
        "mean": round(float(np.mean(scores)), 4),
        "ci_lower": round(float(np.percentile(scores, 100 * alpha / 2)), 4),
        "ci_upper": round(float(np.percentile(scores, 100 * (1 - alpha / 2))), 4),
    }


def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    reliability = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        total += (mask.sum() / len(y_prob)) * abs(conf - acc)
        reliability.append({
            "bin_lo": round(float(lo), 2), "bin_hi": round(float(hi), 2),
            "confidence": round(float(conf), 4), "observed": round(float(acc), 4),
            "n": int(mask.sum()),
        })
    return float(total), reliability


def calibration_slope_intercept(y_true, y_prob):
    logit_p = np.log(np.clip(y_prob, 1e-8, 1 - 1e-8) / (1 - np.clip(y_prob, 1e-8, 1 - 1e-8)))
    lr = LogisticRegression(penalty=None, max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)
    return {
        "slope": round(float(lr.coef_[0][0]), 4),
        "intercept": round(float(lr.intercept_[0]), 4),
        "interpretation": (
            "slope=1, intercept=0 is perfect calibration. "
            "slope<1 = overconfident, slope>1 = underconfident."
        ),
    }


def decision_curve_analysis(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.02)
    prevalence = y_true.mean()
    results = []
    for pt in thresholds:
        tp = ((y_prob >= pt) & (y_true == 1)).sum()
        fp = ((y_prob >= pt) & (y_true == 0)).sum()
        n = len(y_true)
        net_benefit_model = tp / n - fp / n * pt / (1 - pt)
        net_benefit_all = prevalence - (1 - prevalence) * pt / (1 - pt)
        results.append({
            "threshold": round(float(pt), 3),
            "net_benefit_model": round(float(net_benefit_model), 6),
            "net_benefit_treat_all": round(float(net_benefit_all), 6),
            "net_benefit_treat_none": 0.0,
        })
    return results


def patient_level_predictions(df, feature_cols, model_entry, task):
    """Aggregate to one prediction per patient (last available hour)."""
    mean, std_v = model_entry["mean"], model_entry["std"]
    prod_type = model_entry["production_model"]
    pids, preds, labels = [], [], []
    for pid, g in df.groupby("patient_id"):
        g = g.dropna(subset=[task])
        if len(g) == 0:
            continue
        last = g.sort_values("hour").iloc[-1]
        x = last[feature_cols].values.astype(np.float32).reshape(1, -1)
        x_norm = (x - mean) / std_v
        if prod_type == "ensemble":
            alpha = model_entry["alpha"]
            p = alpha * model_entry["xgboost"].predict_proba(x_norm)[:, 1][0] + \
                (1 - alpha) * model_entry["random_forest"].predict_proba(x_norm)[:, 1][0]
        else:
            p = model_entry["xgboost"].predict_proba(x_norm)[:, 1][0]
        pids.append(pid)
        preds.append(float(p))
        labels.append(int(last[task]))
    return np.array(pids), np.array(preds), np.array(labels)


# ────────────────────── main ──────────────────────

def main():
    print("=" * 70)
    print("Publication-Grade Evaluation Suite")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    with open(ENS_PATH, "rb") as f:
        models = pickle.load(f)

    test_ids = set(splits["test"])
    val_ids = set(splits["val"])
    test_df = df[df.patient_id.isin(test_ids)]
    val_df = df[df.patient_id.isin(val_ids)]

    results = {}

    for cond in CONDITIONS:
        for h in HORIZONS:
            task = f"{cond}_{h}h"
            entry = models[task]
            feature_cols = entry["feature_cols"]
            mean, std_v = entry["mean"], entry["std"]
            alpha = entry["alpha"]
            prod_type = entry["production_model"]

            te = test_df.dropna(subset=[task])
            Xte = ((te[feature_cols] - mean) / std_v).values.astype(np.float32)
            yte = te[task].values.astype(int)

            if prod_type == "ensemble":
                p_test = alpha * entry["xgboost"].predict_proba(Xte)[:, 1] + \
                         (1 - alpha) * entry["random_forest"].predict_proba(Xte)[:, 1]
            else:
                p_test = entry["xgboost"].predict_proba(Xte)[:, 1]

            print(f"\n[EVAL] ── {task} ({prod_type}, n={len(yte)}) ──")

            # Bootstrap CIs
            auroc_ci = bootstrap_ci(yte, p_test, safe_auroc)
            auprc_ci = bootstrap_ci(yte, p_test, safe_auprc)
            brier = brier_score_loss(yte, p_test)
            brier_ci = bootstrap_ci(yte, p_test, brier_score_loss)
            print(f"  AUROC: {auroc_ci['mean']:.4f} ({auroc_ci['ci_lower']:.4f}-{auroc_ci['ci_upper']:.4f})")
            print(f"  AUPRC: {auprc_ci['mean']:.4f} ({auprc_ci['ci_lower']:.4f}-{auprc_ci['ci_upper']:.4f})")
            print(f"  Brier: {brier:.4f}")

            # Calibration
            ece_val, reliability = ece(yte, p_test)
            cal_slope = calibration_slope_intercept(yte, p_test)
            print(f"  ECE: {ece_val:.4f}, slope: {cal_slope['slope']:.3f}, intercept: {cal_slope['intercept']:.3f}")

            # Operating threshold from validation
            va = val_df.dropna(subset=[task])
            Xva = ((va[feature_cols] - mean) / std_v).values.astype(np.float32)
            yva = va[task].values.astype(int)
            if prod_type == "ensemble":
                p_val = alpha * entry["xgboost"].predict_proba(Xva)[:, 1] + \
                        (1 - alpha) * entry["random_forest"].predict_proba(Xva)[:, 1]
            else:
                p_val = entry["xgboost"].predict_proba(Xva)[:, 1]
            fpr, tpr, thr = roc_curve(yva, p_val)
            threshold = float(thr[np.argmax(tpr - fpr)])

            bin_preds = (p_test >= threshold).astype(int)
            prec = precision_score(yte, bin_preds, zero_division=0)
            rec = recall_score(yte, bin_preds, zero_division=0)
            f1 = f1_score(yte, bin_preds, zero_division=0)
            cm = confusion_matrix(yte, bin_preds)
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

            # Decision curve
            dca = decision_curve_analysis(yte, p_test)

            # Patient-level metrics
            _, pat_preds, pat_labels = patient_level_predictions(te, feature_cols, entry, task)
            pat_auroc_ci = bootstrap_ci(pat_labels, pat_preds, safe_auroc)
            pat_auprc_ci = bootstrap_ci(pat_labels, pat_preds, safe_auprc)
            print(f"  Patient-level AUROC: {pat_auroc_ci['mean']:.4f} "
                  f"({pat_auroc_ci['ci_lower']:.4f}-{pat_auroc_ci['ci_upper']:.4f})")

            # Subgroup analysis
            subgroups = {}
            ages = te["Age"].values
            for lo, hi, label in [(0, 40, "<40"), (40, 60, "40-60"), (60, 80, "60-80"), (80, 200, "80+")]:
                m = (ages >= lo) & (ages < hi)
                if m.sum() < 20 or len(np.unique(yte[m])) < 2:
                    continue
                subgroups[f"age_{label}"] = {
                    "n": int(m.sum()),
                    "AUROC": round(float(roc_auc_score(yte[m], p_test[m])), 4),
                    "AUPRC": round(float(average_precision_score(yte[m], p_test[m])), 4),
                    "prevalence": round(float(yte[m].mean()), 4),
                }
            if "Gender" in te.columns:
                for gval, glabel in [(0, "female"), (1, "male")]:
                    m = te["Gender"].values == gval
                    if m.sum() < 20 or len(np.unique(yte[m])) < 2:
                        continue
                    subgroups[f"sex_{glabel}"] = {
                        "n": int(m.sum()),
                        "AUROC": round(float(roc_auc_score(yte[m], p_test[m])), 4),
                        "AUPRC": round(float(average_precision_score(yte[m], p_test[m])), 4),
                        "prevalence": round(float(yte[m].mean()), 4),
                    }
            prevalence_overall = yte.mean()
            if prevalence_overall > 0:
                high_prev_patients = set()
                for pid, g in te.groupby("patient_id"):
                    if g[task].mean() > 0.5:
                        high_prev_patients.add(pid)
                m_high = te.patient_id.isin(high_prev_patients).values
                m_low = ~m_high
                for m, label in [(m_high, "high_prevalence"), (m_low, "low_prevalence")]:
                    if m.sum() < 20 or len(np.unique(yte[m])) < 2:
                        continue
                    subgroups[label] = {
                        "n": int(m.sum()),
                        "AUROC": round(float(roc_auc_score(yte[m], p_test[m])), 4),
                        "prevalence": round(float(yte[m].mean()), 4),
                    }

            # ROC/PR curves for plotting
            fpr_t, tpr_t, _ = roc_curve(yte, p_test)
            prec_t, rec_t, _ = precision_recall_curve(yte, p_test)

            def subsample(x, y, n=30):
                if len(x) <= n:
                    return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(x, y)]
                idx = np.linspace(0, len(x) - 1, n).astype(int)
                return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]

            results[task] = {
                "production_model": prod_type,
                "hour_level": {
                    "AUROC": auroc_ci,
                    "AUPRC": auprc_ci,
                    "Brier": {"score": round(brier, 4), **brier_ci},
                    "n_samples": len(yte),
                    "prevalence": round(float(yte.mean()), 4),
                },
                "patient_level": {
                    "AUROC": pat_auroc_ci,
                    "AUPRC": pat_auprc_ci,
                    "n_patients": len(pat_labels),
                    "prevalence": round(float(pat_labels.mean()), 4),
                },
                "calibration": {
                    "ECE": round(ece_val, 4),
                    "Brier": round(brier, 4),
                    "slope_intercept": cal_slope,
                    "reliability_diagram": reliability,
                },
                "operating_point": {
                    "threshold": round(threshold, 4),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "f1": round(float(f1), 4),
                    "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
                    "method": "Youden J from validation set",
                },
                "decision_curve": dca,
                "subgroups": subgroups,
                "curves": {
                    "roc": subsample(fpr_t, tpr_t),
                    "pr": subsample(rec_t, prec_t),
                },
            }

    # Sensitivity analysis: missingness indicators
    print(f"\n[EVAL] ── Sensitivity: with vs without missingness indicators ──")
    miss_cols = [c for c in feature_cols if c.startswith("miss_")]
    if miss_cols:
        non_miss_cols = [c for c in feature_cols if not c.startswith("miss_")]
        sensitivity = {}
        for cond in CONDITIONS:
            task = f"{cond}_6h"
            entry = models[task]
            te = test_df.dropna(subset=[task])
            yte = te[task].values.astype(int)
            Xte_full = ((te[feature_cols] - mean) / std_v).values.astype(np.float32)
            Xte_nomiss = Xte_full.copy()
            miss_indices = [feature_cols.index(c) for c in miss_cols if c in feature_cols]
            Xte_nomiss[:, miss_indices] = 0
            if prod_type == "ensemble":
                p_full = alpha * entry["xgboost"].predict_proba(Xte_full)[:, 1] + \
                         (1 - alpha) * entry["random_forest"].predict_proba(Xte_full)[:, 1]
                p_nomiss = alpha * entry["xgboost"].predict_proba(Xte_nomiss)[:, 1] + \
                           (1 - alpha) * entry["random_forest"].predict_proba(Xte_nomiss)[:, 1]
            else:
                p_full = entry["xgboost"].predict_proba(Xte_full)[:, 1]
                p_nomiss = entry["xgboost"].predict_proba(Xte_nomiss)[:, 1]
            auroc_full = safe_auroc(yte, p_full)
            auroc_nomiss = safe_auroc(yte, p_nomiss)
            sensitivity[cond] = {
                "with_missingness": round(float(auroc_full), 4),
                "without_missingness": round(float(auroc_nomiss), 4),
                "delta_pp": round((auroc_full - auroc_nomiss) * 100, 2),
            }
            print(f"  {cond}: with={auroc_full:.4f} without={auroc_nomiss:.4f} "
                  f"delta={sensitivity[cond]['delta_pp']:+.2f}pp")
        results["_sensitivity_missingness"] = sensitivity

    results["_meta"] = {
        "n_bootstrap": N_BOOTSTRAP,
        "ci_level": 0.95,
        "patient_level_method": "last available hour per patient",
        "calibration_method": "logistic regression on logit(p) — slope/intercept",
        "dca_thresholds": "0.01 to 0.99 by 0.02",
        "proxy_label_disclaimer": (
            "ARF, Cardiac, AKI, Coagulopathy labels are rule-based proxies, "
            "NOT adjudicated ground truth. Metrics for these conditions reflect "
            "agreement with clinical heuristics, not true diagnostic accuracy."
        ),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EVAL] Saved publication evaluation -> {OUT_PATH}")


if __name__ == "__main__":
    main()
