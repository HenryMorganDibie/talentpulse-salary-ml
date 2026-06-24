"""
Data loading and validation for the TalentPulse Jobs Dataset.

Handles raw ingestion, schema validation, and train/test splitting.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "job_id", "job_title", "company", "location", "salary_min", "salary_max",
    "description", "country", "search_keyword", "experience_required",
    "degree_required", "skills", "num_skills", "job_level", "is_remote",
    "salary_avg", "has_salary",
]

EXPECTED_COUNTRIES = {"USA", "UK", "Canada", "Australia", "India"}
EXPECTED_JOB_LEVELS = {"Junior", "Mid", "Senior", "Lead"}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw jobs dataset and run schema checks."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = path.suffix.lower()
    if ext == ".xlsx":
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    logger.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    unexpected_countries = set(df["country"].dropna().unique()) - EXPECTED_COUNTRIES
    if unexpected_countries:
        logger.warning("Unexpected country values: %s", unexpected_countries)

    unexpected_levels = set(df["job_level"].dropna().unique()) - EXPECTED_JOB_LEVELS
    if unexpected_levels:
        logger.warning("Unexpected job_level values: %s", unexpected_levels)

    logger.info(
        "Schema OK | salary missing: %d / %d (%.1f%%)",
        df["salary_avg"].isna().sum(),
        len(df),
        df["salary_avg"].isna().mean() * 100,
    )


def get_labelled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows with salary information.

    Rows where has_salary=False cannot be used for supervised regression
    training; they are preserved separately for inference if needed.
    """
    labelled = df[df["has_salary"] == True].copy().drop_duplicates()
    labelled.reset_index(drop=True, inplace=True)
    logger.info("Labelled rows (has_salary=True): %d", len(labelled))
    return labelled


def parse_skills_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse stringified list in the skills column into actual Python lists."""
    def _parse(s):
        try:
            return ast.literal_eval(s)
        except Exception:
            return []

    df = df.copy()
    df["skills_list"] = df["skills"].apply(_parse)
    return df


def train_test_split_stratified(
    df: pd.DataFrame,
    target_col: str = "log_salary",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    80/20 split. Returns train and test DataFrames with all columns intact.
    Stratification is applied on job_level buckets to ensure balanced
    representation of seniority tiers in both partitions.
    """
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df.get("job_level"),
    )
    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)
    logger.info("Split → train: %d, test: %d", len(train_df), len(test_df))
    return train_df, test_df
