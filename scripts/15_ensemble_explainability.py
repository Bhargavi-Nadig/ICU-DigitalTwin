"""
ICU-DT :: Ensemble Explainability — Weighted SHAP
----------------------------------------------------
Computes TreeSHAP for both XGBoost and Random Forest, then combines
them using the ensemble weight alpha:
    SHAP_ensemble = alpha * SHAP_xgb + (1 - alpha) * SHAP_rf

Limitation: this additive combination is an approximation. True Shapley
values for the blended model would require recomputing from scratch
on the ensemble function, which is computationally intractable for
tree ensembles. The approximation is standard practice but may slightly
misattribute in regions where the two models disagree strongly.

Output: model/ensemble_shap.json, model/feature_importance_comparison.json
"""
import json, pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "publication_patient_hours.parquet")
MODEL_DIR = str(BASE_DIR / "model")
ENS_PATH = str(BASE_DIR / "model" / "ensemble_models.pkl")
SPLITS_PATH = str(BASE_DIR / "model" / "splits.json")
OUT_SHAP = str(BASE_DIR / "model" / "ensemble_shap.json")
OUT_IMPORTANCE = str(BASE_DIR / "model" / "feature_importance_comparison.json")

CONDITIONS = ["Sepsis", "ARF", "Cardiac", "AKI", "Coagulopathy"]
N_BACKGROUND = 200
N_EXPLAIN = 50


def main():
    print("=" * 70)
    print("Ensemble Explainability: Weighted SHAP")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    with open(ENS_PATH, "rb") as f:
        models = pickle.load(f)

    train_ids = set(splits["train"])
    test_ids = set(splits["test"])
    train_df = df[df.patient_id.isin(train_ids)]
    test_df = df[df.patient_id.isin(test_ids)]

    shap_results = {}
    importance_comparison = {}

    for cond in CONDITIONS:
        task = f"{cond}_6h"
        entry = models[task]
        feature_cols = entry["feature_cols"]
        mean, std_v = entry["mean"], entry["std"]
        alpha = entry["alpha"]

        print(f"\n[SHAP] === {cond} (alpha={alpha}, prod={entry['production_model']}) ===")

        tr = train_df.dropna(subset=[task])
        te = test_df.dropna(subset=[task])

        Xtr = ((tr[feature_cols] - mean) / std_v).values.astype(np.float32)
        Xte = ((te[feature_cols] - mean) / std_v).values.astype(np.float32)
        yte = te[task].values.astype(int)

        bg_idx = np.random.RandomState(42).choice(len(Xtr), min(N_BACKGROUND, len(Xtr)), replace=False)
        X_bg = Xtr[bg_idx]

        # Pick a mix of positive and negative test cases
        pos_idx = np.where(yte == 1)[0]
        neg_idx = np.where(yte == 0)[0]
        n_pos = min(N_EXPLAIN // 2, len(pos_idx))
        n_neg = min(N_EXPLAIN - n_pos, len(neg_idx))
        explain_idx = np.concatenate([
            np.random.RandomState(42).choice(pos_idx, n_pos, replace=False),
            np.random.RandomState(42).choice(neg_idx, n_neg, replace=False),
        ])
        X_explain = Xte[explain_idx]

        # XGBoost SHAP via native predict(pred_contribs=True) — avoids
        # SHAP library's categorical-split check on newer XGBoost versions
        print(f"  Computing XGBoost TreeSHAP ({len(X_explain)} samples)...")
        import xgboost as xgb
        dmat = xgb.DMatrix(X_explain, feature_names=feature_cols)
        contribs = entry["xgboost"].get_booster().predict(dmat, pred_contribs=True)
        shap_xgb = contribs[:, :-1]  # drop bias column

        # Random Forest SHAP via SHAP library
        print(f"  Computing RF TreeSHAP ({len(X_explain)} samples)...")
        explainer_rf = shap.TreeExplainer(entry["random_forest"])
        shap_rf = explainer_rf.shap_values(X_explain)
        if isinstance(shap_rf, list):
            shap_rf = shap_rf[1]
        elif shap_rf.ndim == 3:
            shap_rf = shap_rf[:, :, 1]

        # Weighted combination
        shap_ensemble = alpha * shap_xgb + (1 - alpha) * shap_rf

        # Global feature importance (mean |SHAP|)
        imp_xgb = np.abs(shap_xgb).mean(axis=0)
        imp_rf = np.abs(shap_rf).mean(axis=0)
        imp_ens = np.abs(shap_ensemble).mean(axis=0)

        top_k = 20
        top_idx_ens = np.argsort(-imp_ens)[:top_k]

        global_importance = []
        for rank, fi in enumerate(top_idx_ens):
            global_importance.append({
                "rank": rank + 1,
                "feature": feature_cols[fi],
                "mean_abs_shap_xgb": round(float(imp_xgb[fi]), 6),
                "mean_abs_shap_rf": round(float(imp_rf[fi]), 6),
                "mean_abs_shap_ensemble": round(float(imp_ens[fi]), 6),
            })

        # Rank correlation between XGBoost and RF importance
        rank_xgb = np.argsort(-imp_xgb)
        rank_rf = np.argsort(-imp_rf)
        rank_map_xgb = {fi: r for r, fi in enumerate(rank_xgb)}
        rank_map_rf = {fi: r for r, fi in enumerate(rank_rf)}
        from scipy.stats import spearmanr
        ranks_x = [rank_map_xgb[i] for i in range(len(feature_cols))]
        ranks_r = [rank_map_rf[i] for i in range(len(feature_cols))]
        spearman_corr, spearman_p = spearmanr(ranks_x, ranks_r)

        # Per-sample explanations for a few patients
        sample_explanations = []
        for s_idx in range(min(5, len(X_explain))):
            patient_idx = explain_idx[s_idx]
            pid = te.iloc[patient_idx]["patient_id"]
            hour = int(te.iloc[patient_idx]["hour"])
            label = int(yte[patient_idx])

            top_features = np.argsort(-np.abs(shap_ensemble[s_idx]))[:8]
            features_detail = []
            for fi in top_features:
                features_detail.append({
                    "feature": feature_cols[fi],
                    "value": round(float(X_explain[s_idx, fi]), 3),
                    "shap_xgb": round(float(shap_xgb[s_idx, fi]), 6),
                    "shap_rf": round(float(shap_rf[s_idx, fi]), 6),
                    "shap_ensemble": round(float(shap_ensemble[s_idx, fi]), 6),
                })
            sample_explanations.append({
                "patient_id": str(pid),
                "hour": hour,
                "label": label,
                "top_features": features_detail,
            })

        shap_results[cond] = {
            "alpha": alpha,
            "production_model": entry["production_model"],
            "n_explained": len(X_explain),
            "global_importance": global_importance,
            "sample_explanations": sample_explanations,
        }

        importance_comparison[cond] = {
            "spearman_rank_correlation": round(float(spearman_corr), 4),
            "spearman_p_value": round(float(spearman_p), 6),
            "agreement": (
                "strong" if abs(spearman_corr) > 0.8
                else "moderate" if abs(spearman_corr) > 0.6
                else "weak"
            ),
            "top_10_xgb": [feature_cols[i] for i in rank_xgb[:10]],
            "top_10_rf": [feature_cols[i] for i in rank_rf[:10]],
            "top_10_ensemble": [feature_cols[top_idx_ens[i]] for i in range(min(10, len(top_idx_ens)))],
        }

        print(f"  Rank correlation (XGB vs RF): rho={spearman_corr:.3f}")
        print(f"  Top 5 ensemble features: {[feature_cols[i] for i in top_idx_ens[:5]]}")

    output = {
        "method": "weighted TreeSHAP: SHAP_ens = alpha * SHAP_xgb + (1-alpha) * SHAP_rf",
        "limitation": (
            "Additive SHAP combination is an approximation. True Shapley values for the "
            "blended function would require recomputing on the ensemble output, which is "
            "computationally expensive. In regions where XGBoost and RF disagree strongly, "
            "the weighted sum may slightly misattribute feature contributions."
        ),
        "n_background_samples": N_BACKGROUND,
        "conditions": shap_results,
    }

    with open(OUT_SHAP, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SHAP] Saved ensemble SHAP -> {OUT_SHAP}")

    with open(OUT_IMPORTANCE, "w") as f:
        json.dump(importance_comparison, f, indent=2)
    print(f"[SHAP] Saved importance comparison -> {OUT_IMPORTANCE}")


if __name__ == "__main__":
    main()
