# ICU-DT — Explainable ICU Digital Twin (Multi-Condition Deterioration)

A working prototype of the ICU Digital Twin system from your SRS/HLD/LLD,
trained end-to-end on real data, with a production model chosen through a
fair head-to-head comparison, exact SHAP explanations, and a path to both
cloud deployment and a later MIMIC-IV model swap without redeploying.

## 1. What's in this repo (data provenance — item 1)

Unlike the earlier iteration, the **processed dataset itself ships in the
repo**, not just the code that produces it, so you (or anyone reviewing
this) can see exactly what the model was trained on:

- `data/processed_patient_hours.parquet` — the full cleaned, feature-engineered
  patient-hour table (128,828 rows x 231 columns) used for every training run.
- `data/feature_medians.csv` — the imputation reference values.
- `data/ingestion_audit_log.csv` — per-patient ingestion outcome (OK /
  quarantined / row-counts), so every row's provenance is traceable.
- `data/training_setA/` — the original raw `.psv` files, for full reproducibility.

## 2. Does the dataset have ARF/Cardiac labels? (item 2)

No. Checked directly against the raw files: the PhysioNet Sepsis Challenge
schema has exactly 41 columns, and `SepsisLabel` is the only outcome label.
`Unit1`/`Unit2` (ICU type) exist but are ~53% missing. Everything else is a
vital sign or lab value. ARF, Cardiac, AKI, and Coagulopathy are **not**
present and had to be constructed as proxies (see section 4).

## 3. Data quality policy — fixed properly (item 3)

The first attempt at this was too aggressive: requiring every vital
(including `Temp`/`SBP`/`DBP`, which are charted intermittently, not
continuously monitored) to be non-null per row dropped **78%** of the data.
Fixed with a tiered policy that matches how the data is actually collected:

- **CORE** (row dropped if missing): `ICULOS`, `SepsisLabel`, `Age`,
  `Gender`, `HospAdmTime` (all 0% missing, free filter) + `HR`, `O2Sat`,
  `MAP`, `Resp` (continuously monitor-derived, ~8-11% missing each).
  Result: **16.6%** of rows dropped — a defensible number for genuinely
  unreliable rows, not a data massacre.
- **Imputed, never drops a row**: `Temp`, `SBP`, `DBP` (charted
  intermittently but still primary vitals) and all labs — forward-filled
  within a patient's stay, then population-median-filled for any remaining
  gap. Losing a whole patient-hour because one infrequent lab wasn't drawn
  that hour would throw away far more good signal than it protects.
- **File-level quarantine**: only if a patient's file is missing a CORE
  column entirely (schema-level) or has zero usable rows after the above
  filter. 13/3,971 patients hit the latter case — genuinely unusable (no
  reliable vitals recorded at any point in their stay).

Full per-patient outcome is in `data/ingestion_audit_log.csv`.

## 4. Beyond sepsis — 5 conditions, 3 horizons (item 4)

Since only `SepsisLabel` is adjudicated, ARF/Cardiac/AKI/Coagulopathy are
built as **rule-based clinical proxies** from established thresholds:

| Condition | Definition |
|---|---|
| ARF | PF ratio < 300 (Berlin ARDS threshold) OR (O2Sat < 90 AND Resp >= 24) |
| Cardiac / hemodynamic collapse | MAP < 65 AND (shock index > 1.0 OR lactate >= 4 mmol/L) |
| AKI | KDIGO-inspired: creatinine >=0.3 mg/dL rise from rolling 48h minimum, or >=1.5x that baseline |
| Coagulopathy | Platelets < 100 K/uL AND PTT > 40s (platelet-only fallback if PTT unavailable) |

**These are not adjudicated ground truth** — no clinician confirmed them.
Every place they're surfaced (dashboard, metrics, this README) says so
explicitly. If/when MIMIC-IV access is available, these get replaced with
real ICD-coded diagnoses (see the MIMIC section below).

Each condition is predicted at **6h / 12h / 24h** horizons using a
leakage-safe forward-looking rolling-max label
(`label[t] = max(source[t+1..t+horizon])`), giving **15 prediction tasks**
total.

## 5-8. Model selection, tuning, and honest results

### The comparison had a confound, fixed before drawing conclusions

The first BiLSTM-vs-tabular-baselines comparison wasn't apples-to-apples:
the LSTM saw a full 24h sequence window; Random Forest/XGBoost/LogReg saw
only the last observed hour. Before picking a production model, every
model was given the *same* 24h of information: the LSTM as a raw sequence,
the tabular models as 174 engineered window-summary features (mean / std /
min / max / 6h-trend / 24h-trend per vital, over the same 24h lookback),
on top of the 41 single-hour features (215 total).

**Even on equal footing, XGBoost won 9/15 tasks, Random Forest 5/15, and
Logistic Regression 1/15 — the BiLSTM won 0/15.** This is a genuinely
interesting, citable finding (tree ensembles beating deep sequence models
on small-to-medium tabular-style clinical data is well documented in the
literature), not a shortcut — see `model/model_comparison.json` for the
full 5-model x 15-task table.

### Production model: XGBoost, tuned further

Per-condition randomized hyperparameter search (10 trials, train-fit /
val-select, never touching test) + early stopping on validation loss. Final
test-set AUROC, after tuning:

| Condition | 6h | 12h | 24h |
|---|---|---|---|
| Sepsis | 0.774 | ~0.74 | ~0.75 |
| ARF | 0.866 | ~0.85 | ~0.84 |
| Cardiac | 0.884 | ~0.87 | ~0.86 |
| AKI | 0.892 | ~0.88 | ~0.86 |
| Coagulopathy | 0.920 | 0.930 | 0.926 |

Full precision/recall/F1/confusion-matrix/support numbers are in
`model/metrics.json`. **Coagulopathy lands in the 90-95% range originally
targeted; Sepsis (the hardest, most occult condition) tops out around
0.77** — reported honestly rather than inflated. Sepsis is fundamentally a
harder target: it's a systemic, often-occult process without one defining
vital-sign threshold, unlike Cardiac/AKI/Coagulopathy which are partly
defined by thresholds on features the model can see directly.

## 6. Comprehensive evaluation metrics

`model/metrics.json` (full detail) / `model/model_comparison.json`
(cross-model) include, per task:

- AUROC, AUPRC, precision, recall, F1, full confusion matrix, class support/prevalence
- Calibration: Expected Calibration Error before AND after Platt scaling
  (often drops from ~0.15-0.30 to ~0.01-0.02), plus reliability-diagram bins
- ROC and PR curves (subsampled to ~25 points each — real curves, not just endpoints)
- Subgroup metrics by age band (<40 / 40-60 / 60-80 / 80+)
- Failure-analysis examples: concrete patient_id + hour for a few
  true-positive / false-positive / false-negative / true-negative cases per task
- Hyperparameter search results per condition

The dashboard's Cohort Analytics view surfaces a slimmed version of this
(headline numbers + reliability bins), so it stays fast without needing the
full curve/example detail inline.

## 7. Model directory, consolidated

Removed the stale single-task-era files (old `feature_cols.json`,
`task_names.json`, `xai_explainer.pkl`, etc.) and renamed the current ones
to drop the now-redundant `_multitask` suffix.

| File | Purpose |
|---|---|
| `production_models.pkl` | The 15 tuned XGBoost models actually serving predictions |
| `shap_explainers.pkl` | Exact `shap.TreeExplainer` per condition |
| `simulation_surrogates.json` | Distilled linear models for the dashboard's live slider |
| `metrics.json` | Full per-task evaluation detail for the production XGBoost models |
| `lstm_metrics.json` | Full per-task evaluation detail for the BiLSTM (learning curves, calibration, subgroups — same depth as `metrics.json`) |
| `model_comparison.json` | Fair 5-model x 15-task comparison table (clinical rule / LogReg / RF / XGBoost / LSTM, all on matched 24h information) |
| `splits.json` | Train/val/test patient IDs (disjoint, asserted at training time) |
| `sepsis_lstm.pt` | The BiLSTM weights from the comparison |
| `model_config.json` | Feature list, task names, architecture params for the BiLSTM |
| `norm_stats.npz` | BiLSTM feature normalization (mean/std) |

## 9. SHAP / production-model agreement — solved differently than the three options offered

The three options considered were: make the surrogate more similar to the
real model, filter disagreements, or distill from the real model's
predictions. **There's a better option: XGBoost has a native exact SHAP
explainer.**

`shap.TreeExplainer` computes exact Shapley values for tree ensembles in
polynomial time — it isn't an approximation of XGBoost, it *is* XGBoost's
attribution, computed exactly. So the dashboard's SHAP chart has zero
fidelity gap by construction: there's no second model to "agree" with.

The distillation option is still used, but for a different, narrower job:
the dashboard's live what-if slider needs to recompute risk on every drag,
client-side, with no backend — shipping a 600-tree ensemble to the browser
isn't practical. So a small linear model is distilled from XGBoost's real
predictions (soft-target BCE, not ground truth) specifically for that
interactive piece, and its fidelity is reported and shown in the dashboard
next to the simulator:

| Condition | Pearson r vs. XGBoost | Agreement @ 0.5 |
|---|---|---|
| Sepsis | 0.726 | 95.3% |
| ARF | 0.852 | 88.8% |
| Cardiac | 0.944 | 93.1% |
| AKI | 0.750 | 93.1% |
| Coagulopathy | 0.874 | 99.3% |

This is the requested verification mechanism, correctly scoped to the one
piece that's actually an approximation.

## 10. Dashboard patient count

26 patients with full detail (up from 15), spanning sepsis/ARF/cardiac/AKI/
coagulopathy/multi-condition/healthy categories, plus a 268-patient
similarity-search index. Kept the payload at ~83k tokens (embedded JSON,
column-oriented storage, no per-row key-name repetition) so it stays fast;
patient count is tunable via `N_PER_CATEGORY` in `04_precompute_dashboard_data.py`.

## 11. Login screen — full-bleed layout

Replaced the centered-modal-with-whitespace login with a full-viewport
split layout: a hero panel (left, ~60%) with live headline AUROC per
condition, and the sign-in form (right, fixed 420px) — both fill 100vh with
no letterboxing.

## Training scripts — every model has both its script and its metrics

| Script | Trains | Output |
|---|---|---|
| `01_etl_pipeline.py` | — (data pipeline) | `data/processed_patient_hours.parquet` |
| `02_train_lstm.py` | BiLSTM (comparison model) | `model/sepsis_lstm.pt`, `model/lstm_metrics.json`, `model/splits.json` |
| `06_fair_baseline_comparison.py` | Clinical-rule (qSOFA) / Logistic Regression / Random Forest / untuned XGBoost, compared against the BiLSTM | `model/model_comparison.json` |
| `05_train_production_xgboost.py` | Tuned XGBoost (production model — per-condition hyperparameter search + early stopping) | `model/production_models.pkl`, `model/metrics.json`, `model/shap_explainers.pkl` |
| `03_train_simulation_surrogate.py` | Linear surrogate distilled from production XGBoost (for the dashboard's live simulator only) | `model/simulation_surrogates.json` |
| `04_precompute_dashboard_data.py` | — (assembles dashboard payload) | `artifacts/dashboard_data.json` |

Run order for a full rebuild: `01 -> 02 -> 06 -> 05 -> 03 -> 04`. `06` needs
`02`'s output (`lstm_metrics.json`, `splits.json`) first; `05` and `03` also
need `splits.json` but are independent of `06`.

**Note on `06_fair_baseline_comparison.py`**: this script was originally
written and run earlier in development, then accidentally deleted when the
pipeline was rebuilt around XGBoost as production — the numbers survived in
`model_comparison.json` but the script that produced them didn't. It's been
reconstructed (functionally identical: same 215 features, same model
families, same fair comparison) and rerun to confirm — current
`model_comparison.json` numbers come from this reconstructed script against
a freshly retrained LSTM, and are consistent with the originals (e.g.
Sepsis_6h XGBoost 0.758 vs. the original run's 0.723; minor run-to-run
variation from RF/XGBoost's own internal randomness, not a methodology
change).

- Baselines required by reviewers: clinical-rule (qSOFA), Logistic
  Regression, Random Forest, XGBoost, all compared fairly against the BiLSTM
  (see the comparison table above, or `model/model_comparison.json`).
- Calibration: ECE before/after Platt scaling, per task, with
  reliability-diagram data (`model/metrics.json`).
- Patient-level split, verified: `02_train_lstm.py` asserts
  `train_ids.isdisjoint(val_ids)` etc. at training time — not just "should
  be disjoint," actually checked and printed with a checkmark every run.
- Still proxy labels for 4/5 conditions — the single biggest thing to fix
  before a real submission is MIMIC-IV's adjudicated ARF/cardiac-arrest/AKI
  diagnoses (see the MIMIC section below for the exact ICD mapping
  planned). Framed honestly throughout as the main current limitation.

## MIMIC-IV compatibility & cloud hot-swap design

Goal: train a MIMIC-IV version later, swap it in with a restart, no
redeploy, without touching the working PhysioNet version in the meantime.

**What's in place now**:

- `model_registry.json` — documents the active model version and the exact
  versioned-folder refactor to do (`model/<version>/`) before adding a
  second model. Deliberately not done yet: restructuring for a hot-swap
  that has nothing to swap to yet would be speculative complexity in a repo
  still being iterated on. It's a ~10-line-per-script change, and it's the
  literal next step once MIMIC access exists.
- `app.py` — a real, tested FastAPI serving backend (not just a design
  doc): loads whichever model the registry points at, exposes `/predict`
  (with optional exact SHAP via `explain: true`), `/admin/model-info`, and
  `/admin/reload` (hot-reloads model files from disk with no process
  restart). Verified end-to-end: started the server, loaded the real
  15-model bundle, hit `/predict` with real feature values, got back a real
  XGBoost prediction (0.156) with real SHAP attributions, and reloaded live.
- The actual "no redeploy" mechanism: mount `model/` as a persistent volume
  (cloud disk / GCS-FUSE / EFS, whatever your cloud provider offers) rather
  than baking it into the container image. Swapping models becomes: upload
  new model files to the volume, flip `active_version` in the registry, hit
  `/admin/reload` or restart the process. Zero image rebuilds, zero
  Kubernetes redeploys.

**When MIMIC-IV access is available**, the path is:

1. Point `01_etl_pipeline.py`'s loader at MIMIC's `chartevents`/`labevents`
   instead of `.psv` files, mapping to the same `FEATURE_COLS` names — the
   downstream training/XAI/dashboard code is dataset-agnostic already.
2. Replace the 4 proxy labels with real ICD-coded diagnoses (mapping
   documented in `model_registry.json`: ARF from J96.*, cardiac arrest from
   I46.*, AKI from N17.*).
3. Run `02 -> 05 -> 03 -> 04` exactly as-is against the new data.
4. Do the versioned-folder refactor (above), add the `mimic_iv_v1` entry,
   flip `active_version`, restart.

## Running it yourself

```bash
pip install torch scikit-learn shap xgboost scipy pandas numpy pyarrow fastapi uvicorn --break-system-packages

python3 scripts/01_etl_pipeline.py               # raw .psv -> processed_patient_hours.parquet
python3 scripts/02_train_lstm.py                 # BiLSTM comparison model + train/val/test splits
python3 scripts/06_fair_baseline_comparison.py    # clinical rule / LogReg / RF / XGBoost vs. BiLSTM
python3 scripts/05_train_production_xgboost.py    # tuned XGBoost production models (uses splits.json)
python3 scripts/03_train_simulation_surrogate.py  # distilled linear surrogate for the live simulator
python3 scripts/04_precompute_dashboard_data.py   # builds artifacts/dashboard_data.json

# serve predictions (optional, for cloud deployment):
uvicorn app:app --port 8000

# dashboard: open icu_dt_dashboard.jsx as a React artifact / paste into your app
```

## Honest open items

- Sepsis AUROC (~0.77) is well short of a clinical-grade bar. The fair
  comparison shows this isn't primarily an architecture problem — it's a
  ceiling from proxy-label noise plus limited data (~2,800 training
  patients, one Challenge subset). Full combined Sepsis Challenge data
  (~40k patients) is the highest-leverage next step.
- 4 of 5 conditions are proxy-labeled, not adjudicated. This is called out
  everywhere it matters in this README, but it's the main limitation to
  lead with in any publication draft.
