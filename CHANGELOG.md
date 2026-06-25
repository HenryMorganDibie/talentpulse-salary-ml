# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-06-24

### Added
- End-to-end supervised regression pipeline across 5 modules (`loader`, `engineering`,
  `train`, `metrics`, `config`)
- Four model comparison: Linear Regression, Ridge, Random Forest, Gradient Boosting
- 5-fold GridSearchCV hyperparameter tuning for RF (depth, n_estimators) and GB
  (depth, learning rate, n_estimators)
- Ridge alpha selection via cross-validated MAE across 30 log-spaced candidates
- 10 publication-quality visualisations saved to `reports/figures/`
- Bias diagnostics: mean residuals segmented by country and job level
- Segment analysis: Premium (Senior/Lead) vs Standard (Junior/Mid) MAE comparison
- VIF analysis to diagnose multicollinearity in base feature set
- RF overfitting diagnostic: train vs test R² across depths 3–20
- Stratified 80/20 train/test split on `job_level`
- 34 pytest unit tests covering correctness, leakage guards, edge cases, and
  the late-binding closure regression test for skill flag generation
- `configs/pipeline.yaml` — all tuneable parameters externalised from code
- `src/config.py` — typed `PipelineConfig` dataclass loaded from YAML
- `ARCHITECTURE.md` — module responsibilities, data flow, design decisions
- `CONTRIBUTING.md` — dev setup, branching, PR guidelines
- `pyproject.toml` — package metadata, ruff config, coverage thresholds
- `Makefile` — `make test`, `make run`, `make lint`, `make clean`
- GitHub Actions CI (`ci.yml`) — lint + tests on push/PR across Python 3.10–3.12
- `data/raw/README.md` — data card with schema, provenance, and known issues

### Fixed (audit pass)
- **Index alignment bug**: `train_test_split_stratified` was calling `reset_index`,
  causing `build_residuals_frame` to silently slice the wrong rows from `df_ohe`
- **Late-binding closure**: `add_skill_flags` used a `lambda` in a `for` loop;
  replaced with `_make_skill_checker(skill)` factory to prevent all flags
  reflecting the last loop value in lazy evaluation contexts
- **Dead variables**: Removed unused `X`, `y` (full-dataset copies computed then
  discarded) and duplicate `y_test_aligned` binding in `run_pipeline.py`
- **Type annotation**: `tune_ridge_alpha` was annotated `-> float` but returned
  a `Tuple[float, List]`
- **Dead imports**: Removed `Pipeline`, `json`, `LabelEncoder`, `numpy` (loader),
  `List`, `Tuple` (train) — all imported but never referenced
- **Module-level CWD dependency**: Removed hardcoded `FIGURE_DIR = Path(...)`;
  all plot functions now require explicit `out_dir` parameter
- **Mixed concerns**: Moved `build_residuals_frame` from `run_pipeline.py` into
  `src/evaluation/metrics.py` where it belongs
- **Magic numbers**: Promoted `18500`, `8000`, `100` (depth curve estimators),
  and `["Senior", "Lead"]` to named constants
- **Pandas anti-pattern**: `df["has_salary"] == True` → `df["has_salary"]`
- **JSON safety**: Removed numpy `preds` array from `score_model` return dict;
  callers now call `model.predict()` directly

---

## [Unreleased]

### Planned (v2)
- Currency normalisation for India segment (USD/INR mixing correction)
- XGBoost / LightGBM model candidates
- NLP features from `description` column (TF-IDF or sentence embeddings)
- Country-stratified ensemble models
- Semi-supervised training on unlabelled salary rows
- `sklearn.Pipeline` wrapper for fit-on-train transforms (prerequisite for API serving)
- MLflow experiment tracking integration
- FastAPI inference endpoint with `/predict` route
