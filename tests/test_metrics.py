"""
Tests for src/evaluation/metrics.py — covering score_model, score_all,
compute_vif, and build_residuals_frame.

Plot functions are excluded from coverage (they require a display and
produce side-effect PNG files — tested via integration run, not unit tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    build_residuals_frame,
    compute_vif,
    score_all,
    score_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_model():
    """A trivially fitted model that always predicts log1p(50_000)."""
    import numpy as np
    from sklearn.dummy import DummyRegressor
    m = DummyRegressor(strategy="constant", constant=np.log1p(50_000))
    m.fit([[0]], [np.log1p(50_000)])
    return m


@pytest.fixture
def xy_test():
    """Small log-salary test arrays."""
    rng = np.random.default_rng(0)
    n = 20
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = pd.Series(np.log1p(rng.uniform(40_000, 120_000, n)))
    return X, y


@pytest.fixture
def residuals_df():
    """Minimal df_ohe + test_index + preds for build_residuals_frame."""
    n = 10
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "salary_avg":      rng.uniform(50_000, 150_000, n),
        "country_USA":     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "country_UK":      [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "job_level_Junior":[1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "job_level_Senior":[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "is_senior":       [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    y_test = pd.Series(np.log1p(df["salary_avg"].values), index=df.index)
    preds  = y_test.values + rng.standard_normal(n) * 0.05
    return df, df.index, preds, y_test


# ---------------------------------------------------------------------------
# score_model
# ---------------------------------------------------------------------------

def test_score_model_returns_required_keys(dummy_model, xy_test):
    X, y = xy_test
    result = score_model(dummy_model, X, y, name="Dummy")
    assert {"name", "MAE", "RMSE", "R²"} == set(result.keys())


def test_score_model_name_preserved(dummy_model, xy_test):
    X, y = xy_test
    result = score_model(dummy_model, X, y, name="MyModel")
    assert result["name"] == "MyModel"


def test_score_model_metrics_are_floats(dummy_model, xy_test):
    X, y = xy_test
    result = score_model(dummy_model, X, y, name="Dummy")
    assert isinstance(result["MAE"], float)
    assert isinstance(result["RMSE"], float)
    assert isinstance(result["R²"], float)


def test_score_model_mae_nonnegative(dummy_model, xy_test):
    X, y = xy_test
    result = score_model(dummy_model, X, y, name="Dummy")
    assert result["MAE"] >= 0
    assert result["RMSE"] >= 0


def test_score_model_no_preds_in_dict(dummy_model, xy_test):
    """Predictions must NOT be stored in the result dict (JSON safety)."""
    X, y = xy_test
    result = score_model(dummy_model, X, y, name="Dummy")
    assert "preds" not in result
    import json
    json.dumps(result)  # must not raise


def test_score_model_perfect_predictions():
    """A model that predicts perfectly should have MAE=0 and R²=1."""
    from sklearn.dummy import DummyRegressor

    y_vals = np.log1p(np.array([50_000.0, 80_000.0, 120_000.0]))
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series(y_vals)

    class PerfectModel:
        def predict(self, X):
            return y_vals

    result = score_model(PerfectModel(), X, y, name="Perfect")
    assert result["MAE"] < 1e-6
    assert abs(result["R²"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# score_all
# ---------------------------------------------------------------------------

def test_score_all_returns_all_models(dummy_model, xy_test):
    X, y = xy_test
    fitted = {"ModelA": dummy_model, "ModelB": dummy_model}
    results = score_all(fitted, X, y)
    assert set(results.keys()) == {"ModelA", "ModelB"}


def test_score_all_each_result_has_metrics(dummy_model, xy_test):
    X, y = xy_test
    results = score_all({"M": dummy_model}, X, y)
    assert {"name", "MAE", "RMSE", "R²"} == set(results["M"].keys())


# ---------------------------------------------------------------------------
# compute_vif
# ---------------------------------------------------------------------------

def test_compute_vif_returns_dataframe():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6], "c": [5, 3, 1, 4, 2]})
    result = compute_vif(X, cols=["a", "b", "c"])
    assert isinstance(result, pd.DataFrame)
    assert "Feature" in result.columns
    assert "VIF" in result.columns


def test_compute_vif_length_matches_cols():
    X = pd.DataFrame({"x": range(20), "y": range(20), "z": range(20)})
    result = compute_vif(X, cols=["x", "y"])
    assert len(result) == 2


def test_compute_vif_sorted_descending():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
    result = compute_vif(X, cols=["a", "b", "c"])
    assert result["VIF"].iloc[0] >= result["VIF"].iloc[-1]


# ---------------------------------------------------------------------------
# build_residuals_frame
# ---------------------------------------------------------------------------

def test_build_residuals_frame_columns(residuals_df):
    df_ohe, test_idx, preds, y_test = residuals_df
    result = build_residuals_frame(df_ohe, test_idx, preds, y_test)
    for col in ["pred_log", "pred_sal", "actual_sal", "residual",
                "country_name", "level_name", "segment"]:
        assert col in result.columns, f"Missing column: {col}"


def test_build_residuals_frame_length(residuals_df):
    df_ohe, test_idx, preds, y_test = residuals_df
    result = build_residuals_frame(df_ohe, test_idx, preds, y_test)
    assert len(result) == len(test_idx)


def test_build_residuals_frame_residual_formula(residuals_df):
    df_ohe, test_idx, preds, y_test = residuals_df
    result = build_residuals_frame(df_ohe, test_idx, preds, y_test)
    expected_residual = result["pred_sal"] - result["actual_sal"]
    np.testing.assert_allclose(result["residual"].values, expected_residual.values)


def test_build_residuals_frame_segment_values(residuals_df):
    df_ohe, test_idx, preds, y_test = residuals_df
    result = build_residuals_frame(df_ohe, test_idx, preds, y_test)
    valid_segments = {"Premium (Senior/Lead)", "Standard (Junior/Mid)"}
    assert set(result["segment"].unique()).issubset(valid_segments)


def test_build_residuals_frame_country_names(residuals_df):
    df_ohe, test_idx, preds, y_test = residuals_df
    result = build_residuals_frame(df_ohe, test_idx, preds, y_test)
    assert set(result["country_name"].unique()).issubset({"USA", "UK"})
