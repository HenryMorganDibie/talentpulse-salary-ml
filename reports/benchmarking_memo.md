# TalentPulse Analytics — Salary Benchmarking Memo

**To:** TalentPulse Leadership & HR Analytics Team  
**From:** Henry Dibie, ML Systems Engineering  
**Date:** June 2026  
**Re:** Tech Job Market Salary Prediction — ML Regression Assessment Results

---

## Executive Summary

I trained and evaluated four supervised regression models on the TalentPulse Jobs Dataset
(4,653 job postings, 5 countries) with the goal of building a data-driven salary prediction
system for the HR Analytics and Labour Market domain.

The **Random Forest** model achieved the lowest MAE of **$42,064** on the held-out test set,
making it the most accurate model for average salary prediction. **Ridge Regression** achieved
the highest R² (0.6217) and lowest RMSE ($140,672), indicating stronger robustness to extreme
outliers and better overall variance explanation. The choice between the two depends on whether
the business prioritises average prediction accuracy (MAE → Random Forest) or stability across
the full salary distribution (RMSE/R² → Ridge).

| Model             |     MAE |       RMSE |     R²  |
|-------------------|--------:|-----------:|--------:|
| Linear Regression | $42,832 |  $140,653  | 0.6217  |
| Ridge Regression  | $42,834 |  $140,672  | 0.6217  |
| **Random Forest** | **$42,064** | **$142,764** | **0.6104** |
| Gradient Boosting | $43,804 |  $149,747  | 0.6029  |

> **Note on RMSE:** High RMSE values across all models are driven by extreme salary
> outliers in the India segment (likely USD/INR currency mixing — see §4). Excluding
> India from evaluation reduces RMSE substantially across all models.

---

## 1. Data & EDA Findings

**Dataset:** 4,653 job postings across 5 countries (USA, UK, Canada, Australia, India)
covering Data Engineer, Data Scientist, ML Engineer, Business Analyst, and Data Analyst roles
at Junior, Mid, Senior, and Lead seniority levels.

**Missing salary data:** 2,358 rows (50.7%) have no salary information (`has_salary=False`).
These were **excluded from supervised training** — imputing salary for ~51% of records would
introduce unacceptable noise. They are retained for potential semi-supervised use in future work.

**Experience data quality:** `experience_required` is missing for approximately 90% of rows.
This feature was retained using a conservative default (tier=1, mid-level) but its contribution
to model performance is limited by this missingness. It should not be considered a reliable
predictor in the current dataset without significant data quality improvement upstream.

**Salary distribution:** Right-skewed with a median of approximately $98,000 (USD).
A log1p transformation was applied to the target before modelling. The Q-Q plots show that
log1p(salary_avg) substantially reduces skewness and improves distributional symmetry —
this is beneficial for both linear model assumptions and tree model stability.

**Correlation analysis (Pearson):**
- `is_senior` shows the strongest positive correlation with salary (r ≈ 0.39)
- `experience_tier` moderately correlates (r ≈ 0.31), with the caveat of 90% missingness
- Skill flags (Python, Spark, AWS) show modest positive correlations (r ≈ 0.10–0.18)
- `is_remote_int` shows near-zero linear correlation — consistent with the known non-linear
  interaction between remote work and experience level

**VIF analysis:** All engineered features have VIF < 3, confirming no problematic
multicollinearity in the baseline feature set.

---

## 2. Feature Engineering Decisions

| Feature | Rationale |
|---------|-----------|
| `is_senior` | Binary flag for Senior/Lead roles; captures the documented 45–80% salary premium |
| `experience_tier` | Ordinal bucket (0–3) makes the feature robust to non-linearity and missing values |
| `skill_python/sql/spark/aws` | Binary presence flags for high-demand, scarcity-premium skills |
| `is_remote_int` | Boolean cast to int for model compatibility |
| `num_skills` | Proxy for role complexity; correlates with seniority level |
| OHE: `job_level` | One-hot encodes all four seniority levels |
| OHE: `country` | One-hot encodes five country markets |

**Leakage prevention:** `salary_min` and `salary_max` are intentionally excluded.
Both columns are direct inputs to `salary_avg = (min + max) / 2` — including them would
inflate R² artificially and make the model non-deployable on real job postings where
salary is undisclosed.

---

## 3. Hyperparameter Tuning

All hyperparameter selection was performed using **5-fold cross-validation on the training
partition only**. The test set was held out and never used during tuning.

**Ridge alpha:** Cross-validated over 30 log-spaced values (0.01–10,000).
Optimal alpha = 0.01, indicating that the feature set is well-conditioned and
minimal L2 regularisation is sufficient.

**Random Forest:**
- Best `max_depth = 12`, `n_estimators = 200` (5-fold CV GridSearchCV)
- Overfitting diagnostic (depth 3–20): the train/test R² gap remains within acceptable
  bounds at depth 12, validating the selected depth

**Gradient Boosting:**
- Best `learning_rate = 0.05`, `max_depth = 4`, `n_estimators = 200`
- Conservative depth (4) is appropriate for boosting — prevents accumulation of variance
  across the ensemble

---

## 4. Evaluation & Bias Diagnostics

### Systematic Under-prediction by Country

All models systematically under-predict salary (negative mean residuals). The India segment
shows the most severe bias (mean residual ≈ –$522,682). This is almost certainly caused by
USD/INR currency mixing in the source data — India salary values appear to be stored in
Indian Rupees without conversion (e.g., ₹10,00,000 stored as 10000000 rather than ~$12,000).
This hypothesis is supported by the median India salary being approximately 10× other markets.
This is a data quality issue that warrants urgent investigation before production deployment.

| Country   | Mean Residual (Predicted − Actual) |
|-----------|------------------------------------|
| UK        | –$5,441                            |
| USA       | –$10,969                           |
| Canada    | –$19,746                           |
| Australia | –$46,608                           |
| India     | –$522,682 ⚠️ (likely currency issue) |

### Systematic Under-prediction by Job Level

All seniority levels show negative residuals. This is consistent with the India outlier
effect pulling down average predictions across the board.

| Level   | Mean Residual |
|---------|---------------|
| Junior  | –$3,769       |
| Lead    | –$11,142      |
| Senior  | –$12,305      |
| Mid     | –$22,977      |

### Segment Analysis: Premium vs Standard Roles

| Segment                 | MAE       |
|-------------------------|-----------|
| Premium (Senior/Lead)   | $27,576   |
| Standard (Junior/Mid)   | $44,804   |

The model performs better on Premium roles (lower MAE). This is explained by the India
outlier effect disproportionately inflating Standard role errors — India postings span all
levels but the currency mixing issue creates the most noise in the mid-salary range.

### Why is R² ~0.62?

The dataset contains substantial missingness (90% missing experience, 51% missing salary),
heterogeneous salary scales across countries, and potential currency inconsistencies in the
India segment. Despite those constraints, the model explains approximately 62% of salary
variance using only structured job-posting attributes — a reasonable baseline that would
improve materially with currency normalisation and NLP features from the description field.

---

## 5. Top Feature Importances (Random Forest)

Country-related features ranked among the most influential predictors, followed by
seniority and experience signals:

1. **Country encoding** — Geography is the single strongest predictor of compensation
2. **is_senior** — Confirms the documented 45–80% seniority premium
3. **experience_tier** — Ordinal experience bucket provides meaningful signal despite missingness
4. **num_skills** — Role complexity proxy
5. **Skill flags** (Python, AWS, Spark) — In-demand technical skills command measurable premiums

---

## 6. Recommendations

### Immediate (Before Production Deployment)

1. **Resolve India currency contamination** — Implement USD/INR normalisation in the
   data ingestion pipeline. The mean residual of –$522k for India makes the model
   unsuitable for India market pricing until this is corrected.

2. **Add a baseline DummyRegressor** — Compare all models against
   `DummyRegressor(strategy="median")` to formally demonstrate that ML adds value over
   a naive baseline. This is standard practice for regression assessment submissions.

3. **Add remote × experience interaction term** — The non-linear interaction between
   `is_remote` and `experience_required` is not captured by the current feature set.
   Adding `remote_x_experience = is_remote_int * experience_tier` is a low-cost improvement.

### Medium-Term

4. **XGBoost / LightGBM evaluation** — Both typically outperform scikit-learn's
   GradientBoostingRegressor and should be benchmarked as v2 candidates.

5. **NLP features from job descriptions** — TF-IDF or sentence embeddings from the
   `description` column would surface salary signals not captured by structured features
   (e.g. "Series B startup", "FAANG", "equity compensation"). This is the highest-leverage
   single feature improvement available.

6. **Country-stratified models** — Given the magnitude of cross-country salary variance,
   training separate models per market (or a country-conditioned ensemble) would likely
   outperform the current global model.

7. **Semi-supervised training on unlabelled rows** — The 2,358 rows with `has_salary=False`
   can be incorporated via label propagation after currency normalisation, potentially
   doubling the effective training set size.

---

## 7. Reproducibility

All code, figures, and metrics are version-controlled in the project repository.

```bash
git clone https://github.com/HenryMorganDibie/talentpulse-salary-ml
cd talentpulse-salary-ml
pip install -r requirements.txt
python run_pipeline.py
```

Outputs are written to `reports/figures/` (10 PNG visualisations) and
`reports/metrics.json` (full scorecard with bias diagnostics).

---

*All model metrics are computed on an unseen 20% hold-out test set (459 rows,
stratified by job_level). Hyperparameters were selected using 5-fold cross-validation
on the training partition only — no test-set data was used during model selection.*
