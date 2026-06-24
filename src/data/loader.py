"""
Data loading and validation for the TalentPulse Jobs Dataset.

Handles raw ingestion, schema validation, skills parsing, and train/test splitting.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = [
    "job_id", "job_title", "company", "location", "salary_min", "salary_max",
    "description", "country", "search_keyword", "experience_required",
    "degree_required", "skills", "num_skills", "job_level", "is_remote",
    "salary_avg", "has_salary",
]

EXPECTED_COUNTRIES: frozenset = frozenset({"USA", "UK", "Canada", "Australia", "India"})
EXPECTED_JOB_LEVELS: frozenset = frozenset({"Junior", "Mid", "Senior", "Lead"})


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
        raise ValueError(f"Unsupported file type: {ext!r}. Expected .xlsx or .csv")

    logger.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise on missing required columns; warn on unexpected categorical values."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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
    Return only rows that have salary information.

    Rows where has_salary=False cannot be used for supervised regression
    training; they are preserved separately for inference if needed.
    """
    labelled = df[df["has_salary"]].copy().drop_duplicates()
    labelled.reset_index(drop=True, inplace=True)
    logger.info("Labelled rows (has_salary=True): %d", len(labelled))
    return labelled


def _parse_skill_string(s: object) -> List[str]:
    """Safely parse a stringified list of skills into a Python list."""
    try:
        result = ast.literal_eval(str(s))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_skills_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse stringified list in the skills column into actual Python lists."""
    df = df.copy()
    df["skills_list"] = df["skills"].apply(_parse_skill_string)
    return df


def train_test_split_stratified(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 split on job_level to ensure balanced seniority
    representation in both partitions.

    Returns train and test DataFrames with contiguous integer indices
    (reset_index is intentionally NOT applied so callers can align on
    the original df index if needed).
    """
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df.get("job_level"),
    )
    train_df = df.loc[train_idx]
    test_df  = df.loc[test_idx]
    logger.info("Split → train: %d, test: %d", len(train_df), len(test_df))
    return train_df, test_df
