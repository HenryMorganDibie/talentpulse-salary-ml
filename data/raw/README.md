# Data Card — TalentPulse Jobs Dataset

## Overview

| Attribute | Value |
|-----------|-------|
| File | `jobs_dataset.xlsx` |
| Rows | 4,653 |
| Columns | 17 |
| Labelled rows (`has_salary=True`) | 2,295 (49.3%) |
| Unlabelled rows (`has_salary=False`) | 2,358 (50.7%) |
| Countries | USA, UK, Canada, Australia, India |
| Job levels | Junior, Mid, Senior, Lead |
| Date collected | June 2026 |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | int | Unique job posting identifier |
| `job_title` | str | Raw job title from the posting |
| `company` | str | Hiring company name |
| `location` | str | City / region of the role |
| `salary_min` | float | Minimum salary from posting (USD) |
| `salary_max` | float | Maximum salary from posting (USD) |
| `salary_avg` | float | `(salary_min + salary_max) / 2` — **target variable** |
| `has_salary` | bool | True if salary was disclosed in the posting |
| `description` | str | Full job description text |
| `country` | str | One of: USA, UK, Canada, Australia, India |
| `job_level` | str | One of: Junior, Mid, Senior, Lead |
| `search_keyword` | str | Job category keyword used in scraping |
| `experience_required` | float | Years of experience required (nullable) |
| `degree_required` | str | Degree requirement (e.g., Bachelor, Master) |
| `skills` | str | Stringified Python list of required skills |
| `num_skills` | int | Count of skills in the `skills` column |
| `is_remote` | bool | Whether the role is fully remote |

## Target Variable

`salary_avg` is derived from `salary_min` and `salary_max`. For this reason,
**`salary_min` and `salary_max` are excluded from the model feature set** to
prevent data leakage — they are direct components of the target.

The target is log1p-transformed before modelling to reduce right-skew.

## Known Issues

### 1. India salary outliers ⚠️
Several India postings contain salary values in the range of $1M–$10M USD.
These are almost certainly the result of INR (Indian Rupee) values being stored
without currency conversion (e.g., ₹10,00,000 stored as 10000000 instead of
~$12,000). This causes severe model under-prediction for India (mean residual
≈ –$522,000). **Do not use India salary predictions for pricing decisions
until this is corrected in the ingestion pipeline.**

### 2. 50.7% missing salary
Job postings that do not disclose salary are marked `has_salary=False` and
excluded from supervised training. This leaves only 2,295 labelled rows.
The unlabelled rows are retained in `data/raw/` for potential semi-supervised
use in future versions.

### 3. No timestamp
The dataset has no posting date column. Salary trend analysis over time is
not possible with this version of the data.

### 4. `skills` as stringified list
The `skills` column stores lists as Python string representations
(e.g., `"['python', 'sql']"`). This is parsed by `src/data/loader.py`
using `ast.literal_eval`. Malformed entries default to an empty list.

## Provenance

Data sourced from a TalentPulse Analytics scraping pipeline covering
tech job postings (Data Scientist, ML Engineer, Data Engineer,
Data Analyst, Business Analyst) across five English-speaking markets.
