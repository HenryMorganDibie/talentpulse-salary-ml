"""
Tests for src/data/loader.py — covering load_raw, _validate_schema,
get_labelled, parse_skills_column, train_test_split_stratified.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    REQUIRED_COLUMNS,
    _validate_schema,
    get_labelled,
    load_raw,
    parse_skills_column,
    train_test_split_stratified,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid DataFrame matching REQUIRED_COLUMNS."""
    base = {
        "job_id": [1, 2, 3, 4],
        "job_title": ["Data Scientist", "ML Engineer", "Analyst", "Lead Eng"],
        "company": ["A", "B", "C", "D"],
        "location": ["NY", "London", "Toronto", "Sydney"],
        "salary_min": [80_000, 90_000, 60_000, 120_000],
        "salary_max": [100_000, 110_000, 80_000, 160_000],
        "description": ["d"] * 4,
        "country": ["USA", "UK", "Canada", "Australia"],
        "search_keyword": ["data"] * 4,
        "experience_required": [2.0, 5.0, 1.0, 10.0],
        "degree_required": ["Bachelor"] * 4,
        "skills": ["['python']", "['sql']", "['excel']", "['aws']"],
        "num_skills": [1, 1, 1, 1],
        "job_level": ["Junior", "Mid", "Junior", "Senior"],
        "is_remote": [True, False, True, False],
        "salary_avg": [90_000.0, 100_000.0, 70_000.0, 140_000.0],
        "has_salary": [True, True, True, True],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------

def test_load_raw_csv(tmp_path):
    df = _minimal_df()
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    result = load_raw(p)
    assert len(result) == 4


def test_load_raw_xlsx(tmp_path):
    df = _minimal_df()
    p = tmp_path / "data.xlsx"
    df.to_excel(p, index=False)
    result = load_raw(p)
    assert len(result) == 4


def test_load_raw_missing_file():
    with pytest.raises(FileNotFoundError):
        load_raw("/nonexistent/path/data.csv")


def test_load_raw_unsupported_format(tmp_path):
    p = tmp_path / "data.parquet"
    p.write_text("fake")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_raw(p)


# ---------------------------------------------------------------------------
# _validate_schema
# ---------------------------------------------------------------------------

def test_validate_schema_passes_on_valid_df():
    df = _minimal_df()
    _validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_column():
    df = _minimal_df().drop(columns=["salary_avg"])
    with pytest.raises(ValueError, match="Missing required columns"):
        _validate_schema(df)


def test_validate_schema_warns_on_unexpected_country(caplog):
    import logging
    df = _minimal_df(country=["USA", "UK", "Canada", "NewZealand"])
    with caplog.at_level(logging.WARNING, logger="src.data.loader"):
        _validate_schema(df)
    assert "NewZealand" in caplog.text


def test_validate_schema_warns_on_unexpected_level(caplog):
    import logging
    df = _minimal_df(job_level=["Junior", "Mid", "Staff", "Senior"])
    with caplog.at_level(logging.WARNING, logger="src.data.loader"):
        _validate_schema(df)
    assert "Staff" in caplog.text


# ---------------------------------------------------------------------------
# get_labelled
# ---------------------------------------------------------------------------

def test_get_labelled_keeps_only_has_salary():
    df = _minimal_df(has_salary=[True, False, True, False])
    result = get_labelled(df)
    assert len(result) == 2
    assert result["has_salary"].all()


def test_get_labelled_drops_duplicates():
    df = _minimal_df()
    duped = pd.concat([df, df]).reset_index(drop=True)
    result = get_labelled(duped)
    assert len(result) == len(df)


def test_get_labelled_resets_index():
    df = _minimal_df(has_salary=[True, False, True, False])
    result = get_labelled(df)
    assert list(result.index) == [0, 1]


# ---------------------------------------------------------------------------
# parse_skills_column
# ---------------------------------------------------------------------------

def test_parse_skills_column_all_rows():
    df = _minimal_df()
    result = parse_skills_column(df)
    assert "skills_list" in result.columns
    assert all(isinstance(v, list) for v in result["skills_list"])


def test_parse_skills_column_handles_bad_values():
    df = pd.DataFrame({"skills": ["not_a_list", None, "['python']"]})
    result = parse_skills_column(df)
    assert result["skills_list"].iloc[0] == []
    assert result["skills_list"].iloc[1] == []
    assert result["skills_list"].iloc[2] == ["python"]


# ---------------------------------------------------------------------------
# train_test_split_stratified
# ---------------------------------------------------------------------------

def test_split_total_size():
    from src.features.engineering import build_features
    df = parse_skills_column(_minimal_df())
    df_feat = build_features(df)
    train, test = train_test_split_stratified(df_feat, test_size=0.25, random_state=0)
    assert len(train) + len(test) == len(df_feat)


def test_split_index_is_subset_of_original():
    from src.features.engineering import build_features
    df = parse_skills_column(_minimal_df())
    df_feat = build_features(df)
    train, test = train_test_split_stratified(df_feat, test_size=0.25, random_state=0)
    assert set(train.index).issubset(set(df_feat.index))
    assert set(test.index).issubset(set(df_feat.index))


def test_split_train_test_no_overlap():
    from src.features.engineering import build_features
    df = parse_skills_column(_minimal_df())
    df_feat = build_features(df)
    train, test = train_test_split_stratified(df_feat, test_size=0.25, random_state=0)
    assert len(set(train.index) & set(test.index)) == 0
