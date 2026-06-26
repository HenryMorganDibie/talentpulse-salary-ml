"""
Tests for src/config.py — covering load_config, PipelineConfig structure,
and YAML null → None normalisation for max_depth.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = textwrap.dedent("""
    data:
      raw_path: "data/raw/jobs_dataset.xlsx"
      expected_countries: [USA, UK]
      expected_job_levels: [Junior, Mid, Senior, Lead]
      senior_levels: [Senior, Lead]

    features:
      top_skills: [python, sql]
      ohe_columns: [job_level, country]
      experience_tier_breaks: [2, 5, 9]

    split:
      test_size: 0.2
      random_state: 42
      stratify_col: job_level

    ridge:
      alpha_log_min: -2
      alpha_log_max: 4
      n_alphas: 30
      cv_folds: 5

    random_forest:
      param_grid:
        n_estimators: [100, 200]
        max_depth: [8, null]
      cv_folds: 5
      depth_curve_n_estimators: 100

    gradient_boosting:
      param_grid:
        n_estimators: [100]
        learning_rate: [0.1]
        max_depth: [4]
      cv_folds: 5

    evaluation:
      legacy_mae: 18500
      industry_target_mae: 8000
      vif_threshold: 10
      overfit_gap_threshold: 0.05

    output:
      reports_dir: "reports"
      figures_dir: "reports/figures"
      models_dir: "models"
      metrics_file: "reports/metrics.json"
""")


@pytest.fixture
def config_file(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text(MINIMAL_YAML)
    return p


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_returns_pipeline_config(config_file):
    from src.config import PipelineConfig
    cfg = load_config(config_file)
    assert isinstance(cfg, PipelineConfig)


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/pipeline.yaml")


def test_load_config_data_section(config_file):
    cfg = load_config(config_file)
    assert cfg.data.raw_path == "data/raw/jobs_dataset.xlsx"
    assert "USA" in cfg.data.expected_countries
    assert "Senior" in cfg.data.senior_levels


def test_load_config_split_section(config_file):
    cfg = load_config(config_file)
    assert cfg.split.test_size == 0.2
    assert cfg.split.random_state == 42


def test_load_config_evaluation_section(config_file):
    cfg = load_config(config_file)
    assert cfg.evaluation.legacy_mae == 18500
    assert cfg.evaluation.industry_target_mae == 8000
    assert cfg.evaluation.vif_threshold == 10


def test_load_config_ridge_section(config_file):
    cfg = load_config(config_file)
    assert cfg.ridge.n_alphas == 30
    assert cfg.ridge.cv_folds == 5


def test_load_config_null_max_depth_becomes_none(config_file):
    """YAML null in max_depth list must be converted to Python None."""
    cfg = load_config(config_file)
    depths = cfg.random_forest.param_grid["max_depth"]
    assert None in depths, f"Expected None in max_depth list, got: {depths}"


def test_load_config_features_section(config_file):
    cfg = load_config(config_file)
    assert "python" in cfg.features.top_skills
    assert cfg.features.experience_tier_breaks == [2, 5, 9]


def test_load_config_output_section(config_file):
    cfg = load_config(config_file)
    assert cfg.output.reports_dir == "reports"
    assert cfg.output.metrics_file == "reports/metrics.json"
