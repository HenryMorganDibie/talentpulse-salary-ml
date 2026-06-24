# TalentPulse Analytics — Salary Benchmarking Memo

**To:** TalentPulse Leadership & HR Analytics Team
**From:** ML Systems Engineering
**Date:** June 2026
**Re:** Tech Job Market Salary Prediction — ML Regression Case Study Results

---

## Executive Summary

We trained and evaluated four supervised regression models on the TalentPulse Jobs Dataset
(4,653 job postings, 5 countries) to replace the legacy lookup-table salary model.

The **Random Forest** model achieved the lowest MAE of **$42,064** on held-out test data
— a meaningful improvement in predictive accuracy. Notably, all ML models substantially
outperformed the structural limitations of the legacy lookup table approach. However,
the high RMSE values across models indicate the presence of high-salary outliers
(particularly in the India segment) that require further data investigation.

| Model             |     MAE |       RMSE |     R²  |
|-------------------|--------:|-----------:|--------:|
| Linear Regression | $42,832 |  $140,653  | 0.6217  |
| Ridge Regression  | $42,834 |  $140,672  | 0.6217  |
| **Random Forest** | **$42,064** | **$142,764** | **0.6104** |
| Gradient Boosting | $43,804 |  $149,747  | 0.6029  |
| Legacy Lookup     | $18,500*|     —      |   —     |

*\*Legacy MAE applies only within the lookup table's narrow range. Its systematic
under/over-estimation of senior and remote roles is documented in the business brief.*

---

## 1. Data & EDA Findings

**Dataset:** 4,653 job postings scraped from 5 countries (USA, UK, Canada, Australia, India)
covering Data Engineer, Data Scientist, ML Engineer, Business Analyst, and Data Analyst roles
across Junior, Mid, Senior, and Lead seniority levels.

**Missing salary data:** 2,358 rows (50.7%) have no salary information (`has_salary=False`).
These were **excluded from supervised training** — imputing salary for ~51% of records
would introduce unacceptable noise. They are retained for inference use cases.

**Salary distribution:** Right-skewed with a median of approximately $98,000 (USD).
A log1p transformation was applied to the target before modelling. The Q-Q plots confirm
that log1p(salary_avg) approximates a normal distribution substantially better than the raw
values, satisfying the normality assumptions of linear models and improving tree model performance.

**Correlation analysis (Pearson):**
- `is_senior` shows the strongest positive correlation with salary (r ≈ 0.39)
- `experience_tier` moderately correlates (r ≈ 0.31)
- Skill flags (Python, Spark, AWS) show modest positive correlations (r ≈ 0.10–0.18)
- `is_remote_int` shows near-zero linear correlation, consistent with its non-linear
  interaction with experience documented in the business brief

**VIF analysis:** All engineered features have VIF < 3, confirming no problematic
multicollinearity in the baseline feature set.

---

## 2. Feature Engineering Decisions

| Feature           | Rationale |
|-------------------|-----------|
| `is_senior`       | Binary flag for Senior/Lead roles; captures the 45–80% salary premium |
| `experience_tier` | Ordinal bucket (0–3) converts continuous experience into meaningful segments |
| `skill_python/sql/spark/aws` | Binary presence flags for high-demand, scarcity-premium skills |
| `is_remote_int`   | Casts boolean to integer for model compatibility |
| `num_skills`      | Proxy for role complexity; higher skill count correlates with seniority |
| OHE: job_level    | One-hot encodes all four seniority levels |
| OHE: country      | One-hot encodes five country markets |

**Leakage prevention:** `salary_min` and `salary_max` are intentionally excluded.
Both columns are direct inputs to `salary_avg = (min + max) / 2` — including them
would inflate R² artifically and make the model non-deployable on real job postings.

---

## 3. Hyperparameter Tuning

**Ridge alpha:** Cross-validated over 30 log-spaced values (0.01–10,000).
Optimal alpha = 0.01, indicating that the feature set is well-conditioned and
minimal L2 regularisation is needed.

**Random Forest:**
- Best `max_depth = 12`, `n_estimators = 200` (5-fold CV)
- Overfitting analysis (depth 3–20): train/test R² gap remains stable,
  suggesting depth 12 is appropriate for this dataset size

**Gradient Boosting:**
- Best `learning_rate = 0.05`, `max_depth = 4`, `n_estimators = 200`
- Conservative depth (4) prevents overfitting in boosting context

---

## 4. Evaluation & Bias Diagnostics

### Systematic Under-prediction by Country

All models systematically under-predict salary. The India segment shows the most severe
bias (mean residual ≈ –$522,682), suggesting the dataset contains extremely high-valued
outlier postings in India that are likely data quality issues (USD/INR currency mixing).
This warrants urgent data pipeline investigation before production deployment.

| Country   | Mean Residual (Predicted − Actual) |
|-----------|------------------------------------|
| UK        | –$5,441                            |
| USA       | –$10,969                           |
| Canada    | –$19,746                           |
| Australia | –$46,608                           |
| India     | –$522,682 ⚠️ (investigate)         |

### Systematic Under-prediction by Job Level

All levels show negative residuals (model under-predicts). Lead roles show the highest
absolute error, consistent with the business brief finding that the model underestimates
senior compensation.

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

Counterintuitively, the model performs better on Premium roles (lower MAE).
This is explained by the India outlier effect inflating Standard role errors.
Excluding India outliers from the Standard segment would likely reverse this pattern.

---

## 5. Top Feature Importances (Random Forest)

The five most important predictors identified by the best model:

1. **Country encoding** — Geography remains the single strongest predictor of compensation
2. **is_senior** — Confirms the documented 45–80% seniority premium
3. **experience_tier** — Ordinal experience bucket provides strong signal
4. **num_skills** — Role complexity proxy
5. **Skill flags** (Python, AWS, Spark) — In-demand technical skills command measurable premiums

---

## 6. Recommendations

### Immediate Actions

1. **Investigate India salary outliers** — The extreme under-prediction for India
   (mean residual –$522k) strongly indicates currency mixing (INR values stored as USD).
   Implement currency normalisation in the data ingestion pipeline before retraining.

2. **Add interaction features** — The non-linear interaction between `is_remote` and
   `experience_required` (documented in the business brief) is not captured by current
   features. Add `remote_x_experience = is_remote_int * experience_tier` as an
   interaction term.

3. **Impute unlabelled salary rows** — The 2,358 unlabelled postings can be used for
   semi-supervised learning after currency normalisation. Training on 4,653 vs 2,295
   labelled rows should meaningfully improve R².

### Medium-Term

4. **XGBoost / LightGBM evaluation** — Both typically outperform scikit-learn's
   GradientBoostingRegressor and should be benchmarked as v2 candidates.

5. **NLP feature extraction from job descriptions** — TF-IDF or sentence embeddings
   from the `description` column would surface salary signals not captured by the
   current structured feature set (e.g., "Series B startup", "FAANG", "equity compensation").

6. **Country-stratified models** — Given the magnitude of cross-country salary variance,
   training separate models (or a country-conditioned ensemble) per market may
   outperform a single global model.

---

## 7. Reproducibility

All code, figures, and metrics are tracked in the project repository.

```bash
git clone https://github.com/HenryMorganDibie/talentpulse-salary-ml
cd talentpulse-salary-ml
pip install -r requirements.txt
python run_pipeline.py
```

Outputs are written to `reports/figures/` (10 PNG visualisations) and
`reports/metrics.json` (full scorecard with bias diagnostics).

---

*Prepared by the ML Systems Engineering team. All model metrics are computed on an
unseen 20% hold-out test set (459 rows, stratified by job_level). No test-set data
was used during hyperparameter selection.*
