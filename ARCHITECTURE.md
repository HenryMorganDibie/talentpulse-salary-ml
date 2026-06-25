# Architecture

This document describes the system design of the TalentPulse Salary ML Pipeline:
its module boundaries, data flow, key design decisions, and known limitations.

---

## 1. High-Level Data Flow

```
data/raw/
  └── jobs_dataset.xlsx
          │
          ▼
  src/data/loader.py
  ┌─────────────────────────────────────────┐
  │  load_raw()          schema validation  │
  │  get_labelled()      filter has_salary  │
  │  parse_skills_column()  skills → list   │
  │  train_test_split_stratified()          │
  └───────────┬─────────────────────────────┘
              │  df_labelled (2,295 rows)
              ▼
  src/features/engineering.py
  ┌─────────────────────────────────────────┐
  │  add_log_target()    log1p(salary_avg)  │
  │  add_seniority_features()  is_senior    │
  │  add_experience_tier()     ordinal 0–3  │
  │  add_skill_flags()   python/sql/spark/  │
  │                      aws binary flags   │
  │  add_remote_flag()   bool → int         │
  │  one_hot_encode()    job_level, country │
  └───────────┬─────────────────────────────┘
              │  df_ohe  (feature matrix)
              ├──────────────────┐
              ▼                  ▼
         X_train / y_train   X_test / y_test
              │
              ▼
  src/models/train.py
  ┌─────────────────────────────────────────┐
  │  tune_ridge_alpha()   30-pt log CV      │
  │  train_all()          4-model registry  │
  │    LinearRegression                     │
  │    Ridge (best_alpha)                   │
  │    RandomForest  ← GridSearchCV 5-fold  │
  │    GradientBoosting ← GridSearchCV      │
  │  rf_depth_curve()     overfit analysis  │
  │  save_model()         pickle            │
  └───────────┬─────────────────────────────┘
              │  fitted_models dict
              ▼
  src/evaluation/metrics.py
  ┌─────────────────────────────────────────┐
  │  score_all()          MAE/RMSE/R²       │
  │  build_residuals_frame()                │
  │  compute_vif()        multicollinearity │
  │  plot_*()             10 PNG figures    │
  └───────────┬─────────────────────────────┘
              │
              ▼
  reports/
    figures/   (10 PNGs)
    metrics.json
    benchmarking_memo.md

  models/
    best_model.pkl
```

---

## 2. Module Responsibilities

### `src/data/loader.py` — Ingestion & Validation

**Single responsibility:** get raw data off disk in a validated, clean state.

- Schema validation is strict (raises on missing columns) to surface data drift early.
- Country and job_level mismatches are warnings not errors — the pipeline degrades
  gracefully if a new market is added to the feed.
- `train_test_split_stratified` **preserves the original DataFrame index** by design.
  This is the contract the rest of the system depends on: `df_ohe.loc[test_df.index]`
  in `build_residuals_frame` is only correct because indices are never reset here.
  Do not add `reset_index()` — it will silently slice the wrong rows.

### `src/features/engineering.py` — Feature Engineering

**Single responsibility:** transform clean data into a model-ready feature matrix.

All functions are **stateless** — no fit-on-train transforms are applied here.
If a feature ever requires train-set statistics (e.g., target encoding, imputation
from train mean), it must be wrapped in a `sklearn.Pipeline` with a `fit`/`transform`
split before being added here.

Key invariant: **`salary_min` and `salary_max` are never features.**
Both are direct components of `salary_avg = (min + max) / 2`. Including them would
inflate R² artificially and make the model non-deployable on unseen job postings
(which have no salary information by definition).

The skill flag implementation uses the `_make_skill_checker(skill)` factory pattern
instead of a lambda inside a for loop. This is intentional — it closes over the
`skill` variable at call time, preventing Python's late-binding closure trap where
all flags would reflect the last iteration value in a lazy evaluation context.

### `src/models/train.py` — Training & Tuning

**Single responsibility:** fit and serialise models.

The `build_model_registry` function returns a dict of untrained estimators. This
separates configuration from execution and makes it straightforward to add a new
model (XGBoost, LightGBM) without touching the training loop.

GridSearchCV wrappers are returned as-is from `train_all`. The caller is responsible
for unwrapping `.best_estimator_` when needed. This keeps the fitted dict honest —
callers can inspect `.cv_results_` or `.best_params_` without the module needing
to know what the caller will do with them.

`rf_depth_curve` is a **diagnostic utility**, not a production training path.
It uses a fixed `_DEPTH_CURVE_N_ESTIMATORS` (100 trees) which is intentionally
smaller than the GridSearch range — speed matters more than accuracy for a
visualisation pass.

### `src/evaluation/metrics.py` — Scoring & Visualisation

**Single responsibility:** measure and visualise model behaviour.

`score_model` returns only serialisable values (floats). Predictions are not stored
in the results dict. Callers that need predictions call `model.predict(X_test)`
directly — this avoids the numpy-array-in-JSON problem and keeps the results dict
safe to `json.dump` without a `default=str` escape hatch.

`build_residuals_frame` lives here (not in `run_pipeline.py`) because assembling a
labelled residuals frame is evaluation logic, not orchestration. The function takes
`test_index: pd.Index` explicitly rather than deriving it from `y_test.index` —
this makes the index contract visible at the call site.

All plot functions accept `out_dir: Path` as an explicit argument. There is no
module-level `FIGURE_DIR` constant. This makes functions safe to call from any
working directory and easy to unit test without touching the filesystem.

### `run_pipeline.py` — Orchestration Only

This module wires the four src modules together. It should contain **no domain
logic** — if you find yourself writing a feature transformation or a metric
calculation here, it belongs in a src module instead.

Business constants (`LEGACY_MAE`, `INDUSTRY_TARGET_MAE`) live here because they
are pipeline-level configuration, not feature logic.

---

## 3. Configuration

All tuneable parameters live in `configs/pipeline.yaml`. The pipeline reads this
file at startup via `src/config.py`. Hardcoded numbers in src modules are a
maintenance smell — if a parameter might change between experiments, it belongs
in the config.

```yaml
# configs/pipeline.yaml (excerpt)
split:
  test_size: 0.2
  random_state: 42

ridge:
  alpha_min: 0.01
  alpha_max: 10000.0
  n_alphas: 30
  cv_folds: 5

random_forest:
  param_grid:
    n_estimators: [100, 200]
    max_depth: [8, 12, null]
  cv_folds: 5
```

---

## 4. Design Decisions & Trade-offs

### Why log1p(salary)?
Salary distributions are right-skewed. Log-transforming the target improves
linear model fit, reduces the influence of extreme outliers, and makes residuals
closer to normally distributed (confirmed by Q-Q plots). The inverse transform
(`np.expm1`) is applied before computing MAE/RMSE so metrics are interpretable
in USD.

### Why not StandardScaler?
Tree-based models (Random Forest, Gradient Boosting) are scale-invariant.
Linear models benefit from scaling, but Ridge already handles ill-conditioning
via L2 regularisation, and the VIF analysis confirms no severe multicollinearity
(all VIF < 3). Adding a scaler would have no material effect and would require
fitting on train only — adding complexity for no gain.

### Why GridSearchCV over RandomizedSearchCV?
The search space is small (12 RF combinations, 8 GB combinations). Exhaustive
grid search is reproducible and complete for this scale. At larger search spaces
(> ~100 combinations), switch to `RandomizedSearchCV` or `Optuna`.

### Why stratify on job_level?
Seniority is the strongest predictor of salary. Without stratification, a bad
random split could place all Lead roles in train or test, making evaluation
non-representative. Stratification guarantees proportional seniority distribution
in both partitions.

### Why pickle over ONNX / joblib?
Pickle is the simplest choice for a single-repo, same-Python-version workflow.
For cross-language serving or version-safe persistence, prefer `joblib` (faster
for large numpy arrays) or ONNX (language-portable inference). This is noted as
a v2 upgrade in the roadmap.

### Why not a `sklearn.Pipeline` for the full feature→model chain?
The current feature engineering is DataFrame-native and stateless — there is
nothing to fit on train and apply to test. A sklearn Pipeline adds value when
transforms have fit state (scalers, encoders fitted on train). If target encoding,
imputation from train statistics, or embeddings are added in v2, the feature
engineering module should be refactored into a `BaseEstimator`/`TransformerMixin`
and wrapped in a `Pipeline`.

---

## 5. Known Limitations

### India salary outliers
The India segment shows a mean residual of –$522,682, indicating extreme salary
values that are likely USD/INR currency mixing in the source data. Until this is
corrected in the ingestion pipeline, India results should not be used for pricing
decisions. See `reports/benchmarking_memo.md` §4 for details.

### 50.7% unlabelled data
Rows where `has_salary=False` (2,358 / 4,653) are excluded from training.
They are retained in `data/raw/` for future semi-supervised approaches.
Training on only 2,295 rows limits generalisation — currency normalisation
and semi-supervised label propagation are the highest-leverage v2 investments.

### No online / streaming path
The pipeline is batch-only. For real-time salary inference (e.g., pricing a
new job posting at submission time), the feature engineering step needs to be
wrapped in a stateless inference function and served behind an API endpoint.

### Model is global, not market-specific
A single model is trained across all five markets. Given the magnitude of
cross-country variance (and the India outlier), a market-stratified ensemble
or separate per-country models would likely outperform the global model.

---

## 6. Extension Points

| What to add | Where to add it |
|-------------|-----------------|
| New skill flags | `TOP_SKILLS` list in `src/features/engineering.py` |
| New model (XGBoost) | `build_model_registry()` in `src/models/train.py` |
| New country / job level | `EXPECTED_COUNTRIES` / `EXPECTED_JOB_LEVELS` in `src/data/loader.py` |
| API serving endpoint | New `src/serving/` module; call `load_model()` + `build_features()` |
| Drift monitoring | New `src/monitoring/` module; compare PSI of incoming features vs train distribution |
| Experiment tracking | Wrap `train_all()` with MLflow `mlflow.start_run()` context |

---

## 7. Repository Layout

```
talentpulse-salary-ml/
├── run_pipeline.py          # Orchestration entrypoint
├── Makefile                 # Developer UX: make test / run / lint
├── pyproject.toml           # Package metadata & dev dependencies
├── requirements.txt         # Pinned runtime dependencies
├── ARCHITECTURE.md          # This file
├── CHANGELOG.md             # Version history
├── CONTRIBUTING.md          # Dev setup & PR guidelines
│
├── configs/
│   └── pipeline.yaml        # All tuneable parameters
│
├── src/
│   ├── config.py            # Typed config loader (reads pipeline.yaml)
│   ├── data/
│   │   └── loader.py        # Ingestion, validation, splitting
│   ├── features/
│   │   └── engineering.py   # Stateless feature transforms
│   ├── models/
│   │   └── train.py         # Model registry, tuning, persistence
│   └── evaluation/
│       └── metrics.py       # Scoring, VIF, visualisation, residuals
│
├── data/
│   ├── raw/
│   │   ├── jobs_dataset.xlsx
│   │   └── README.md        # Data card: schema, provenance, known issues
│   └── processed/           # Reserved for cleaned / feature-engineered outputs
│
├── reports/
│   ├── figures/             # 10 PNG visualisations (auto-generated)
│   ├── metrics.json         # Model scorecard + bias diagnostics
│   └── benchmarking_memo.md # Analytical findings & recommendations
│
├── models/                  # Serialised best model (git-ignored)
│
├── tests/
│   └── test_pipeline.py     # 34 pytest unit tests
│
└── .github/
    └── workflows/
        └── ci.yml           # GitHub Actions: lint + test on push/PR
```
