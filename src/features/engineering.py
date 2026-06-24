"""
Feature engineering pipeline for TalentPulse salary prediction.

Design principles
-----------------
- No leakage: salary_min / salary_max are never used as features
  (they are direct components of the target salary_avg).
- All transformations are stateless; no fit-on-train state is stored here.
- Features are grouped into semantic buckets for interpretability.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Skills to create binary flag features for
TOP_SKILLS: List[str] = ["python", "sql", "spark", "aws"]

# Ordered seniority levels (subset that maps to is_senior=1)
SENIOR_LEVELS: List[str] = ["Senior", "Lead"]

# Canonical level ordering — kept here so callers don't redefine it
LEVEL_ORDER: List[str] = ["Junior", "Mid", "Senior", "Lead"]


# ---------------------------------------------------------------------------
# Target engineering
# ---------------------------------------------------------------------------

def add_log_target(df: pd.DataFrame, col: str = "salary_avg") -> pd.DataFrame:
    """Log1p-transform the salary target to reduce right-skew."""
    df = df.copy()
    df["log_salary"] = np.log1p(df[col])
    return df


# ---------------------------------------------------------------------------
# Seniority features
# ---------------------------------------------------------------------------

def add_seniority_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_senior binary flag (Senior or Lead = 1, Junior/Mid = 0).
    Captures the 45–80% salary premium documented in the TalentPulse brief.
    """
    df = df.copy()
    senior_mask = df.get("job_level", pd.Series(dtype=str)).isin(SENIOR_LEVELS)
    df["is_senior"] = senior_mask.astype(int)
    return df


# ---------------------------------------------------------------------------
# Experience tier
# ---------------------------------------------------------------------------

def add_experience_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket years-of-experience into an ordinal tier:
        0 → 0–2 yrs  (entry level)
        1 → 3–5 yrs  (mid level)   ← default when value is missing
        2 → 6–9 yrs  (senior)
        3 → 10+ yrs  (lead / principal)
    """
    def _tier(x: object) -> int:
        if pd.isna(x):
            return 1
        years = float(x)
        if years <= 2:
            return 0
        if years <= 5:
            return 1
        if years <= 9:
            return 2
        return 3

    df = df.copy()
    df["experience_tier"] = df["experience_required"].apply(_tier)
    return df


# ---------------------------------------------------------------------------
# Skill flags
# ---------------------------------------------------------------------------

def _make_skill_checker(skill: str):
    """
    Return a vectorised skill-presence checker for a single skill keyword.

    Defined as a factory (not a lambda inside a loop) to avoid Python's
    late-binding closure trap — each returned function permanently captures
    its own `skill` reference.
    """
    skill_lower = skill.lower()

    def _check(skill_list: list) -> int:
        return int(any(skill_lower in str(item).lower() for item in skill_list))

    return _check


def add_skill_flags(df: pd.DataFrame, skills: List[str] = TOP_SKILLS) -> pd.DataFrame:
    """
    Binary flag for each high-value skill (Python, SQL, Spark, AWS).
    Requires the 'skills_list' column produced by parse_skills_column().
    """
    if "skills_list" not in df.columns:
        raise ValueError(
            "'skills_list' column not found. "
            "Call parse_skills_column() before add_skill_flags()."
        )
    df = df.copy()
    for sk in skills:
        df[f"skill_{sk}"] = df["skills_list"].apply(_make_skill_checker(sk))
    return df


# ---------------------------------------------------------------------------
# Remote flag
# ---------------------------------------------------------------------------

def add_remote_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Cast boolean is_remote to integer for model compatibility."""
    df = df.copy()
    df["is_remote_int"] = df["is_remote"].astype(int)
    return df


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def one_hot_encode(
    df: pd.DataFrame,
    columns: List[str] = ("job_level", "country"),
) -> pd.DataFrame:
    """One-hot encode categorical columns. All categories are preserved (drop_first=False)."""
    return pd.get_dummies(df, columns=list(columns), drop_first=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Call this on the labelled dataset after parse_skills_column().
    Returns a DataFrame ready for model training.

    Leaky features (salary_min, salary_max) are intentionally excluded —
    both are direct inputs to salary_avg = (min + max) / 2.
    """
    df = add_log_target(df)
    df = add_seniority_features(df)
    df = add_experience_tier(df)
    df = add_skill_flags(df)
    df = add_remote_flag(df)
    df = one_hot_encode(df, columns=["job_level", "country"])
    logger.info("Feature matrix shape after engineering: %s", df.shape)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the ordered list of model input features from an engineered DataFrame.

    OHE columns are discovered dynamically so this stays correct if the dataset
    gains new countries or job levels in future.
    """
    base = [
        "num_skills",
        "experience_tier",
        "is_senior",
        "is_remote_int",
        "skill_python",
        "skill_sql",
        "skill_spark",
        "skill_aws",
    ]
    ohe = sorted(
        c for c in df.columns
        if c.startswith("job_level_") or c.startswith("country_")
    )
    return [c for c in base + ohe if c in df.columns]
