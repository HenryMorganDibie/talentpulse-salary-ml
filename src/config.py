"""
Typed configuration loader for TalentPulse ML Pipeline.

Reads configs/pipeline.yaml and exposes a single PipelineConfig dataclass.
All src modules should import constants from here rather than hardcoding values.

Usage
-----
    from src.config import load_config

    cfg = load_config()                          # reads configs/pipeline.yaml
    cfg = load_config("configs/pipeline.yaml")   # explicit path

    print(cfg.split.test_size)         # 0.2
    print(cfg.evaluation.legacy_mae)   # 18500.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default config path relative to repo root
_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "pipeline.yaml"


# ---------------------------------------------------------------------------
# Dataclass hierarchy
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    raw_path: str
    expected_countries: list[str]
    expected_job_levels: list[str]
    senior_levels: list[str]


@dataclass
class FeaturesConfig:
    top_skills: list[str]
    ohe_columns: list[str]
    experience_tier_breaks: list[int]


@dataclass
class SplitConfig:
    test_size: float
    random_state: int
    stratify_col: str


@dataclass
class RidgeConfig:
    alpha_log_min: float
    alpha_log_max: float
    n_alphas: int
    cv_folds: int


@dataclass
class RandomForestConfig:
    param_grid: dict
    cv_folds: int
    depth_curve_n_estimators: int


@dataclass
class GradientBoostingConfig:
    param_grid: dict
    cv_folds: int


@dataclass
class EvaluationConfig:
    legacy_mae: float
    industry_target_mae: float
    vif_threshold: float
    overfit_gap_threshold: float


@dataclass
class OutputConfig:
    reports_dir: str
    figures_dir: str
    models_dir: str
    metrics_file: str


@dataclass
class PipelineConfig:
    data: DataConfig
    features: FeaturesConfig
    split: SplitConfig
    ridge: RidgeConfig
    random_forest: RandomForestConfig
    gradient_boosting: GradientBoostingConfig
    evaluation: EvaluationConfig
    output: OutputConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path = _DEFAULT_CONFIG) -> PipelineConfig:
    """
    Load and parse pipeline.yaml into a typed PipelineConfig.

    Parameters
    ----------
    path : path to the YAML config file (defaults to configs/pipeline.yaml)

    Raises
    ------
    FileNotFoundError  if the config file does not exist
    KeyError           if a required section is missing
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load the pipeline config. "
            "Install it with: pip install pyyaml"
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    logger.info("Loaded config from %s", path)

    # Normalise null max_depth values from YAML (null → None in Python)
    rf_grid = raw["random_forest"]["param_grid"].copy()
    rf_grid["max_depth"] = [
        None if v is None else v for v in rf_grid.get("max_depth", [])
    ]

    return PipelineConfig(
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        split=SplitConfig(**raw["split"]),
        ridge=RidgeConfig(**raw["ridge"]),
        random_forest=RandomForestConfig(
            param_grid=rf_grid,
            cv_folds=raw["random_forest"]["cv_folds"],
            depth_curve_n_estimators=raw["random_forest"]["depth_curve_n_estimators"],
        ),
        gradient_boosting=GradientBoostingConfig(
            param_grid=raw["gradient_boosting"]["param_grid"],
            cv_folds=raw["gradient_boosting"]["cv_folds"],
        ),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        output=OutputConfig(**raw["output"]),
    )
