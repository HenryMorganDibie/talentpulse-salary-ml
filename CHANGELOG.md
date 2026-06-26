# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned (v2)
- Currency normalisation for India segment (USD/INR mixing correction)
- DummyRegressor baseline benchmark
- XGBoost / LightGBM model candidates
- NLP features from `description` column (TF-IDF or sentence embeddings)
- Country-stratified ensemble models
- Semi-supervised training on unlabelled salary rows
- `sklearn.Pipeline` wrapper for fit-on-train transforms (prerequisite for API serving)
- MLflow experiment tracking integration
- FastAPI inference endpoint with `/predict` route

---

## [1.1.0] — 2026-06-26

### Added
- 56 new tests across 4 new test files (total: 90 tests, 99% coverage)
  - `tests/test_loader.py` — load_raw, _validate_schema, get_labelled, split
  - `tests/test_train.py` — model registry, alpha tuning, train_all, depth curve, persistence
  - `tests/test_metrics.py` — score_model, score_all, VIF, build_residuals_frame
  - `tests/test_config.py` — load_config, all YAML sections, null→None normalisation
- `configs/pipeline.yaml` — all tuneable parameters externalised from code
- `src/config.py` — typed `PipelineConfig` dataclass loaded from YAML
- `ARCHITECTURE.md` — module responsibilities, data flow, design decisions, known limitations
- `CONTRIBUTING.md` — dev setup, branching, PR guidelines, key invariants
- `pyproject.toml` — package metadata, ruff config, coverage exclusions
- `Makefile` — `make test`, `make run`, `make lint`, `make clean`
- `data/raw/README.md` — data card with schema, provenance, and known issues
- GitHub Actions CI (`ci.yml`) — ruff lint + pytest on push/PR across Python 3.10–3.12
- Coverage `exclude_lines` for plot functions (require display, not unit-testable)

### Fixed (audit pass)
- **Index alignment bug:** `train_test_split_stratified` was calling `reset_index`,
  causing `build_residuals_frame` to silently slice the wrong rows from `df_ohe`
- **Late-binding closure:** `add_skill_flags` used a lambda in a for loop; replaced
  with `_make_skill_checker(skill)` factory
- **Dead variables:** Removed unused `X`, `y` and duplicate `y_test_aligned` in
  `run_pipeline.py`
- **Type annotation:** `tune_ridge_alpha` was annotated `-> float` but returned a tuple
- **Dead imports:** Removed across all src modules (numpy in loader, LabelEncoder in
  engineering, Pipeline/json/List/Tuple in train)
- **Module-level CWD dependency:** Removed hardcoded `FIGURE_DIR`; all plot functions
  now require explicit `out_dir: Path`
- **Mixed concerns:** Moved `build_residuals_frame` from `run_pipeline.py` into
  `src/evaluation/metrics.py`
- **Magic numbers:** Promoted `18500`, `8000`, `100` (depth curve), and
  `["Senior", "Lead"]` to named constants
- **Pandas anti-pattern:** `df["has_salary"] == True` → `df["has_salary"]`
- **JSON safety:** Removed numpy `preds` array from `score_model` return dict

### Changed
- `plot_model_comparison` now receives `legacy_mae` and `target_mae` as explicit
  parameters instead of using hardcoded values
- `_parse_skill_string` promoted to module-level in `loader.py` (was re-created on
  every call as an inline closure; now directly unit-testable)
- All `typing.List`, `typing.Dict`, `typing.Tuple`, `typing.Optional` replaced with
  built-in `list`, `dict`, `tuple`, and `X | None` union syntax (Python 3.10+)

---

## [1.0.0] — 2026-06-24

### Added
- End-to-end supervised regression pipeline across 5 modules
  (`loader`, `engineering`, `train`, `metrics`, `config`)
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
- 34 pytest unit tests covering correctness, leakage guards, edge cases, and the
  late-binding closure regression test for skill flag generation
- `reports/benchmarking_memo.md` — analytical findings & recommendations
