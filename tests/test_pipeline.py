"""
Unit tests for TalentPulse ML pipeline.

Run with:  pytest tests/ -v
"""

import ast
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import (
    add_log_target,
    add_seniority_features,
    add_experience_tier,
    add_skill_flags,
    add_remote_flag,
    get_feature_columns,
    build_features,
)
from src.data.loader import parse_skills_column


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "job_title": ["Data Scientist", "ML Engineer", "Data Analyst", "Lead Engineer"],
        "company": ["A", "B", "C", "D"],
        "location": ["New York", "London", "Toronto", "Sydney"],
        "salary_min": [80000, 90000, 60000, 120000],
        "salary_max": [100000, 110000, 80000, 160000],
        "description": ["desc"] * 4,
        "country": ["USA", "UK", "Canada", "Australia"],
        "search_keyword": ["data"] * 4,
        "experience_required": [2, 5, 1, 10],
        "degree_required": ["Bachelor"] * 4,
        "skills": [
            "['python', 'sql']",
            "['spark', 'aws', 'python']",
            "['sql', 'excel']",
            "['python', 'aws', 'spark', 'sql']",
        ],
        "num_skills": [2, 3, 2, 4],
        "job_level": ["Junior", "Mid", "Junior", "Senior"],
        "is_remote": [True, False, True, False],
        "salary_avg": [90000, 100000, 70000, 140000],
        "has_salary": [True, True, True, True],
    })


# ── Data loader tests ────────────────────────────────────────────────────────

def test_parse_skills_column(sample_df):
    df = parse_skills_column(sample_df)
    assert "skills_list" in df.columns
    assert isinstance(df["skills_list"].iloc[0], list)
    assert "python" in df["skills_list"].iloc[0]


def test_parse_skills_handles_malformed():
    df = pd.DataFrame({"skills": ["not a list", None, "['python']"]})
    result = parse_skills_column(df)
    assert result["skills_list"].iloc[0] == []
    assert result["skills_list"].iloc[1] == []
    assert result["skills_list"].iloc[2] == ["python"]


# ── Feature engineering tests ────────────────────────────────────────────────

def test_add_log_target(sample_df):
    df = add_log_target(sample_df)
    assert "log_salary" in df.columns
    assert np.allclose(df["log_salary"], np.log1p(df["salary_avg"]))


def test_log_target_positive(sample_df):
    df = add_log_target(sample_df)
    assert (df["log_salary"] > 0).all()


def test_add_seniority_features(sample_df):
    df = add_seniority_features(sample_df)
    assert "is_senior" in df.columns
    assert df.loc[df["job_level"] == "Senior", "is_senior"].all()
    assert not df.loc[df["job_level"] == "Junior", "is_senior"].any()


def test_lead_is_senior():
    df = pd.DataFrame({"job_level": ["Lead", "Junior", "Mid"]})
    result = add_seniority_features(df)
    assert result.loc[0, "is_senior"] == 1
    assert result.loc[1, "is_senior"] == 0


def test_experience_tier_buckets(sample_df):
    df = add_experience_tier(sample_df)
    assert df.loc[df["experience_required"] == 2, "experience_tier"].iloc[0] == 0
    assert df.loc[df["experience_required"] == 5, "experience_tier"].iloc[0] == 1
    assert df.loc[df["experience_required"] == 10, "experience_tier"].iloc[0] == 3


def test_experience_tier_nan_defaults_to_1():
    df = pd.DataFrame({"experience_required": [np.nan, None]})
    result = add_experience_tier(df)
    assert (result["experience_tier"] == 1).all()


def test_skill_flags(sample_df):
    df = parse_skills_column(sample_df)
    df = add_skill_flags(df)
    assert "skill_python" in df.columns
    assert df.loc[0, "skill_python"] == 1   # row 0 has python
    assert df.loc[2, "skill_spark"] == 0    # row 2 has no spark


def test_skill_flags_require_skills_list(sample_df):
    with pytest.raises(ValueError, match="parse_skills_column"):
        add_skill_flags(sample_df)


def test_remote_flag(sample_df):
    df = add_remote_flag(sample_df)
    assert "is_remote_int" in df.columns
    assert df.loc[0, "is_remote_int"] == 1
    assert df.loc[1, "is_remote_int"] == 0


def test_no_salary_leakage_in_feature_cols(sample_df):
    df = parse_skills_column(sample_df)
    df_feat = build_features(df)
    cols = get_feature_columns(df_feat)
    leaky = [c for c in cols if "salary_min" in c or "salary_max" in c]
    assert leaky == [], f"Leaky features detected: {leaky}"


def test_feature_cols_all_present(sample_df):
    df = parse_skills_column(sample_df)
    df_feat = build_features(df)
    cols = get_feature_columns(df_feat)
    missing = [c for c in cols if c not in df_feat.columns]
    assert missing == [], f"Feature columns missing from DataFrame: {missing}"


def test_build_features_returns_log_salary(sample_df):
    df = parse_skills_column(sample_df)
    df_feat = build_features(df)
    assert "log_salary" in df_feat.columns
