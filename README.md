# TalentPulse Salary ML Pipeline

> **End-to-end supervised regression pipeline for tech job market salary prediction.**  
> Built as a Senior Data Associate assessment submission for 10Alytics — HR Analytics & Labour Markets track.  
> Trained on 4,653 global job postings across 5 countries and 4 seniority tiers.

[![CI](https://github.com/HenryMorganDibie/talentpulse-salary-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/HenryMorganDibie/talentpulse-salary-ml/actions/workflows/ci.yml)

---

## Business Context

TalentPulse Analytics needed a data-driven salary benchmarking model to replace manual lookup tables. The core problem: existing tooling ignores seniority effects, geographic market variance, and in-demand skill premiums — leading to systematically inaccurate salary recommendations.

This pipeline addresses those gaps through a structured ML regression approach:

- Seniority encoding (Senior/Lead roles command a documented 45–80% salary premium)
- Country-level market segmentation across 5 English-speaking tech markets
- Skill scarcity flags for Python, SQL, Spark, and AWS
- Leakage-free feature engineering (`salary_min`/`salary_max` explicitly excluded)

---

## Project Structure

```
talentpulse-salary-ml/
├── run_pipeline.py              # Main entrypoint — runs Steps 1–5 end-to-end
├── Makefile                     # Developer UX: make test / run / lint / clean
├── pyproject.toml               # Package metadata, ruff config, coverage thresholds
├── requirements.txt             # Runtime dependencies
│
├── configs/
│   └── pipeline.yaml            # All tuneable parameters (grid, splits, thresholds)
│
├── src/
│   ├── config.py                # Typed config loader (reads pipeline.yaml)
│   ├── data/
│   │   └── loader.py            # Schema validation, labelled split, skills parsing
│   ├── features/
│   │   └── engineering.py       # Leak-free feature engineering pipeline
│   ├── models/
│   │   └── train.py             # Model registry, GridSearchCV tuning, persistence
│   └── evaluation/
│       └── metrics.py           # Scoring, VIF, bias diagnostics, 10 visualisations
│
├── data/
│   └── raw/
│       ├── jobs_dataset.xlsx    # TalentPulse Jobs Dataset (4,653 rows)
│       └── README.md            # Data card: schema, provenance, known issues
│
├── reports/
│   ├── benchmarking_memo.md     # Analytical findings & business recommendations
│   ├── metrics.json             # Model scorecard + bias diagnostics
│   └── figures/                 # 10 PNG visualisations (auto-generated)
│
├── presentation/
│   ├── TalentPulse_SalaryML_Henry_Dibie.pptx  # 12-slide assessment presentation deck
│   └── README.md                              # Slide index and presenter notes guide
│
├── models/                      # Serialised best model — git-ignored, created on run
│
└── tests/
    ├── test_pipeline.py         # Feature engineering tests (34 tests)
    ├── test_loader.py           # Data loading & validation tests (17 tests)
    ├── test_train.py            # Model training & persistence tests (18 tests)
    ├── test_metrics.py          # Scoring & evaluation tests (17 tests)
    └── test_config.py           # Config loader tests (10 tests)
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/HenryMorganDibie/talentpulse-salary-ml.git
cd talentpulse-salary-ml

# 2. Install (runtime + dev)
pip install -e ".[dev]"

# 3. Run full pipeline
make run
# or directly:
python run_pipeline.py --data data/raw/jobs_dataset.xlsx --out reports

# 4. Run tests with coverage
make test

# 5. Lint
make lint
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `src/data/loader.py` | Load dataset, schema validation, EDA distributions |
| 2 | `src/data/loader.py` | Filter labelled rows (has_salary=True), drop duplicates, parse skills |
| 3 | `src/features/engineering.py` | `is_senior`, `experience_tier`, skill flags, OHE, log1p target transform |
| 4 | `src/models/train.py` | Train LR, Ridge, RF, GB with 5-fold GridSearchCV; Ridge alpha CV search |
| 5 | `src/evaluation/metrics.py` | MAE/RMSE/R², feature importances, VIF, residual bias plots, segment analysis |

---

## Model Results

All metrics computed on a stratified 20% hold-out test set (459 rows, stratified by `job_level`). Target is `log1p(salary_avg)`; metrics are reported on the inverse-transformed (USD) scale.

| Model             |     MAE |       RMSE |     R²  |
|-------------------|--------:|-----------:|--------:|
| Linear Regression | $42,832 |  $140,653  | 0.6217  |
| Ridge Regression  | $42,834 |  $140,672  | 0.6217  |
| **Random Forest** | **$42,064** | **$142,764** | **0.6104** |
| Gradient Boosting | $43,804 |  $149,747  | 0.6029  |

**Random Forest** achieves the lowest MAE — best average prediction accuracy.  
**Linear/Ridge** achieve the highest R² — strongest overall variance explanation and more robustness to extreme outliers.

> ⚠️ **Note on RMSE:** All models show high RMSE relative to MAE. This is driven by extreme salary outliers in the India segment (likely USD/INR currency mixing in the source data — median India salary is ~10× other markets). See [`reports/benchmarking_memo.md`](reports/benchmarking_memo.md) §4 for full diagnostics and the data card for the investigation recommendation.

---

## Feature Engineering

All features are stateless transforms — no train-set statistics are applied, making the pipeline safe for direct inference on new postings.

| Feature | Type | Rationale |
|---------|------|-----------|
| `is_senior` | Binary | Senior/Lead flag; captures documented 45–80% seniority premium |
| `experience_tier` | Ordinal (0–3) | Buckets 0–2yr / 3–5yr / 6–9yr / 10+yr; robust to 90% missingness |
| `skill_python/sql/spark/aws` | Binary | Scarcity-premium flags for high-demand technical skills |
| `num_skills` | Integer | Role complexity proxy; correlates with seniority |
| `is_remote_int` | Binary | Boolean cast to int for model compatibility |
| OHE: `job_level` | 4 columns | Junior / Mid / Senior / Lead one-hot encoded |
| OHE: `country` | 5 columns | USA / UK / Canada / Australia / India one-hot encoded |

**Leakage prevention:** `salary_min` and `salary_max` are never used as features — both are direct inputs to `salary_avg = (min + max) / 2`. Including them would inflate R² artificially and make the model non-deployable on real postings that lack salary disclosure.

---

## Visualisations

| Figure | Description |
|--------|-------------|
| `fig1_salary_distribution.png` | Raw vs log1p-transformed salary histogram |
| `fig2_salary_by_group.png` | Salary boxplots by job level and country |
| `fig3_correlation_heatmap.png` | Pearson correlation matrix (base features vs salary) |
| `fig4_model_comparison.png` | MAE / RMSE / R² bar charts across all models |
| `fig5_feature_importances.png` | Top-15 feature importances (Random Forest) |
| `fig6_residuals.png` | Residual boxplots by country and job level |
| `fig7_overfitting.png` | RF train vs test R² across depth 3–20 |
| `fig8_qq_plots.png` | Q-Q plots: raw vs log-transformed salary |
| `fig9_ridge_alpha.png` | Ridge CV error vs regularisation strength (alpha) |
| `fig10_segment_comparison.png` | Premium (Senior/Lead) vs Standard role accuracy |

---

## Presentation

The `presentation/` folder contains the 12-slide deck prepared for the 10Alytics assessment panel.

| Slide | Section |
|-------|---------|
| 1 | Title |
| 2 | Agenda |
| 3 | Business Problem |
| 4 | Data & EDA |
| 5 | Feature Engineering & Design Decisions |
| 6 | Model Training & Tuning |
| 7 | Results & Model Comparison |
| 8 | Bias Diagnostics |
| 9 | Feature Importances & R² Explained |
| 10 | Engineering Quality |
| 11 | Limitations & Next Steps |
| 12 | Summary |

Each slide includes presenter notes with anticipated panel questions and prepared answers.

---

## Tests

```
90 tests across 5 files — pytest tests/ -v
Coverage: 99% (src/features: 100%, src/data: 100%, src/evaluation: 100%, src/models: 98%, src/config: 97%)
```

| File | Tests | Scope |
|------|-------|-------|
| `test_pipeline.py` | 34 | Feature engineering, leakage guards, late-binding closure regression |
| `test_loader.py` | 17 | load_raw, schema validation, get_labelled, parse_skills, split |
| `test_train.py` | 18 | Model registry, alpha tuning, train_all, depth curve, save/load |
| `test_metrics.py` | 17 | score_model, score_all, VIF, build_residuals_frame |
| `test_config.py` | 10 | Config loader, all YAML sections, null→None normalisation |

---

## Tech Stack

Python 3.10–3.12 · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · SciPy · statsmodels · PyYAML · ruff · pytest

---

## Key Design Decisions

- **Log1p target transform** — salary distributions are right-skewed; log transform reduces outlier influence and improves linear model fit (confirmed by Q-Q plots)
- **No StandardScaler** — tree models are scale-invariant; Ridge handles ill-conditioning via L2; VIF confirms no multicollinearity (all VIF < 3)
- **Stratified split on job_level** — prevents a bad random draw from placing all Lead/Senior roles in one partition
- **Index preserved through split** — `train_test_split_stratified` does not call `reset_index`, ensuring `df_ohe.loc[test_df.index]` alignment is correct in residuals analysis

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design rationale and [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup.
