"""
Unit tests for TalentPulse ML pipeline.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    _parse_skill_string,
    get_labelled,
    parse_skills_column,
    train_test_split_stratified,
)
from src.features.engineering import (
    LEVEL_ORDER,
    SENIOR_LEVELS,
    TOP_SKILLS,
    add_experience_tier,
    add_log_target,
    add_remote_flag,
    add_seniority_features,
    add_skill_flags,
    build_features,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "job_id":               [1, 2, 3, 4],
        "job_title":            ["Data Scientist", "ML Engineer", "Data Analyst", "Lead Engineer"],
        "company":              ["A", "B", "C", "D"],
        "location":             ["New York", "London", "Toronto", "Sydney"],
        "salary_min":           [80_000, 90_000, 60_000, 120_000],
        "salary_max":           [100_000, 110_000, 80_000, 160_000],
        "description":          ["desc"] * 4,
        "country":              ["USA", "UK", "Canada", "Australia"],
        "search_keyword":       ["data"] * 4,
        "experience_required":  [2, 5, 1, 10],
        "degree_required":      ["Bachelor"] * 4,
        "skills": [
            "['python', 'sql']",
            "['spark', 'aws', 'python']",
            "['sql', 'excel']",
            "['python', 'aws', 'spark', 'sql']",
        ],
        "num_skills":   [2, 3, 2, 4],
        "job_level":    ["Junior", "Mid", "Junior", "Senior"],
        "is_remote":    [True, False, True, False],
        "salary_avg":   [90_000, 100_000, 70_000, 140_000],
        "has_salary":   [True, True, True, True],
    })


# ---------------------------------------------------------------------------
# Data loader — _parse_skill_string
# ---------------------------------------------------------------------------

def test_parse_skill_string_valid():
    assert _parse_skill_string("['python', 'sql']") == ["python", "sql"]


def test_parse_skill_string_malformed():
    assert _parse_skill_string("not a list") == []


def test_parse_skill_string_none():
    assert _parse_skill_string(None) == []


def test_parse_skill_string_non_list_literal():
    # A valid literal that isn't a list should return []
    assert _parse_skill_string("{'python'}") == []


# ---------------------------------------------------------------------------
# Data loader — parse_skills_column
# ---------------------------------------------------------------------------

def test_parse_skills_column_creates_list_column(sample_df):
    df = parse_skills_column(sample_df)
    assert "skills_list" in df.columns
    assert isinstance(df["skills_list"].iloc[0], list)
    assert "python" in df["skills_list"].iloc[0]


def test_parse_skills_column_does_not_mutate_input(sample_df):
    original_cols = sample_df.columns.tolist()
    parse_skills_column(sample_df)
    assert sample_df.columns.tolist() == original_cols


# ---------------------------------------------------------------------------
# Data loader — get_labelled
# ---------------------------------------------------------------------------

def test_get_labelled_filters_correctly(sample_df):
    mixed = sample_df.copy()
    mixed.loc[0, "has_salary"] = False
    result = get_labelled(mixed)
    assert len(result) == 3
    assert result["has_salary"].all()


def test_get_labelled_drops_duplicates(sample_df):
    duped = pd.concat([sample_df, sample_df]).reset_index(drop=True)
    result = get_labelled(duped)
    assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# Data loader — train_test_split_stratified
# ---------------------------------------------------------------------------

def test_split_preserves_original_index(sample_df):
    """Index must NOT be reset — downstream alignment depends on it."""
    df = parse_skills_column(sample_df)
    df_feat = build_features(df)
    train_df, test_df = train_test_split_stratified(df_feat, test_size=0.25, random_state=0)
    # All indices must be valid labels in the original df_feat
    assert train_df.index.isin(df_feat.index).all()
    assert test_df.index.isin(df_feat.index).all()


def test_split_sizes(sample_df):
    df = parse_skills_column(sample_df)
    df_feat = build_features(df)
    train_df, test_df = train_test_split_stratified(df_feat, test_size=0.25, random_state=42)
    assert len(train_df) + len(test_df) == len(df_feat)


# ---------------------------------------------------------------------------
# Feature engineering — log target
# ---------------------------------------------------------------------------

def test_add_log_target_correctness(sample_df):
    df = add_log_target(sample_df)
    assert "log_salary" in df.columns
    assert np.allclose(df["log_salary"], np.log1p(df["salary_avg"]))


def test_add_log_target_positive(sample_df):
    df = add_log_target(sample_df)
    assert (df["log_salary"] > 0).all()


def test_add_log_target_does_not_mutate(sample_df):
    add_log_target(sample_df)
    assert "log_salary" not in sample_df.columns


# ---------------------------------------------------------------------------
# Feature engineering — seniority
# ---------------------------------------------------------------------------

def test_seniority_senior_is_flagged(sample_df):
    df = add_seniority_features(sample_df)
    assert df.loc[df["job_level"] == "Senior", "is_senior"].all()


def test_seniority_junior_not_flagged(sample_df):
    df = add_seniority_features(sample_df)
    assert not df.loc[df["job_level"] == "Junior", "is_senior"].any()


def test_seniority_lead_is_flagged():
    """Lead must map to is_senior=1 — it's in SENIOR_LEVELS."""
    df = pd.DataFrame({"job_level": ["Lead", "Junior", "Mid"]})
    result = add_seniority_features(df)
    assert result.loc[0, "is_senior"] == 1
    assert result.loc[1, "is_senior"] == 0


def test_senior_levels_constant_matches_implementation():
    """SENIOR_LEVELS constant must stay in sync with add_seniority_features logic."""
    for level in SENIOR_LEVELS:
        df = pd.DataFrame({"job_level": [level]})
        assert add_seniority_features(df).loc[0, "is_senior"] == 1


# ---------------------------------------------------------------------------
# Feature engineering — experience tier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("years,expected_tier", [
    (0,   0),
    (2,   0),
    (3,   1),
    (5,   1),
    (6,   2),
    (9,   2),
    (10,  3),
    (20,  3),
])
def test_experience_tier_boundaries(years, expected_tier):
    df = pd.DataFrame({"experience_required": [years]})
    result = add_experience_tier(df)
    assert result.loc[0, "experience_tier"] == expected_tier


def test_experience_tier_nan_defaults_to_1():
    df = pd.DataFrame({"experience_required": [np.nan, None]})
    result = add_experience_tier(df)
    assert (result["experience_tier"] == 1).all()


# ---------------------------------------------------------------------------
# Feature engineering — skill flags
# ---------------------------------------------------------------------------

def test_skill_flags_present(sample_df):
    df = parse_skills_column(sample_df)
    df = add_skill_flags(df)
    for sk in TOP_SKILLS:
        assert f"skill_{sk}" in df.columns


def test_skill_flags_correct_values(sample_df):
    df = parse_skills_column(sample_df)
    df = add_skill_flags(df)
    assert df.loc[0, "skill_python"] == 1   # row 0 has python
    assert df.loc[2, "skill_spark"] == 0    # row 2 has no spark


def test_skill_flags_no_late_binding_bug():
    """
    Each skill flag must reflect its own skill — not the last one in the loop.
    This catches the classic Python late-binding closure trap.
    """
    df = pd.DataFrame({
        "skills_list": [
            ["python"],
            ["sql"],
            ["spark"],
            ["aws"],
        ]
    })
    result = add_skill_flags(df, skills=["python", "sql", "spark", "aws"])
    assert result.loc[0, "skill_python"] == 1 and result.loc[0, "skill_sql"] == 0
    assert result.loc[1, "skill_sql"] == 1    and result.loc[1, "skill_python"] == 0
    assert result.loc[2, "skill_spark"] == 1  and result.loc[2, "skill_aws"] == 0
    assert result.loc[3, "skill_aws"] == 1    and result.loc[3, "skill_spark"] == 0


def test_skill_flags_require_skills_list(sample_df):
    with pytest.raises(ValueError, match="parse_skills_column"):
        add_skill_flags(sample_df)


# ---------------------------------------------------------------------------
# Feature engineering — remote flag
# ---------------------------------------------------------------------------

def test_remote_flag_values(sample_df):
    df = add_remote_flag(sample_df)
    assert "is_remote_int" in df.columns
    assert df.loc[0, "is_remote_int"] == 1
    assert df.loc[1, "is_remote_int"] == 0


# ---------------------------------------------------------------------------
# Feature engineering — leakage guard
# ---------------------------------------------------------------------------

def test_no_salary_leakage_in_feature_cols(sample_df):
    """salary_min and salary_max must never appear as model features."""
    df = parse_skills_column(sample_df)
    df_feat = build_features(df)
    cols = get_feature_columns(df_feat)
    leaky = [c for c in cols if "salary_min" in c or "salary_max" in c]
    assert leaky == [], f"Leaky features detected: {leaky}"


def test_all_feature_cols_present_in_dataframe(sample_df):
    df = parse_skills_column(sample_df)
    df_feat = build_features(df)
    cols = get_feature_columns(df_feat)
    missing = [c for c in cols if c not in df_feat.columns]
    assert missing == [], f"Feature columns missing from DataFrame: {missing}"


def test_build_features_produces_log_salary(sample_df):
    df = parse_skills_column(sample_df)
    df_feat = build_features(df)
    assert "log_salary" in df_feat.columns
