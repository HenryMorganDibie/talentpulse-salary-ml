"""
Tests for src/models/train.py — covering model registry, alpha tuning,
train_all, rf_depth_curve, and model persistence.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import (
    build_model_registry,
    load_model,
    rf_depth_curve,
    save_model,
    train_all,
    tune_ridge_alpha,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_xy():
    """Small deterministic regression dataset — fast to train on."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((80, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(X["f0"] * 2 + X["f1"] + rng.standard_normal(80) * 0.1)
    return X, y


@pytest.fixture
def split_xy(small_xy):
    X, y = small_xy
    return X[:60], X[60:], y[:60], y[60:]


# ---------------------------------------------------------------------------
# tune_ridge_alpha
# ---------------------------------------------------------------------------

def test_tune_ridge_alpha_returns_float(split_xy):
    X_train, _, y_train, _ = split_xy
    best_alpha, pairs = tune_ridge_alpha(X_train, y_train)
    assert isinstance(best_alpha, float)
    assert best_alpha > 0


def test_tune_ridge_alpha_returns_pairs(split_xy):
    X_train, _, y_train, _ = split_xy
    alphas = np.logspace(-1, 2, 5)
    _, pairs = tune_ridge_alpha(X_train, y_train, alphas=alphas)
    assert len(pairs) == 5
    assert all(isinstance(a, float) and isinstance(s, float) for a, s in pairs)


def test_tune_ridge_alpha_custom_alphas(split_xy):
    X_train, _, y_train, _ = split_xy
    alphas = np.array([0.1, 1.0, 10.0])
    best_alpha, _ = tune_ridge_alpha(X_train, y_train, alphas=alphas)
    assert best_alpha in alphas.tolist()


# ---------------------------------------------------------------------------
# build_model_registry
# ---------------------------------------------------------------------------

def test_registry_contains_four_models():
    registry = build_model_registry(best_alpha=1.0)
    assert len(registry) == 4


def test_registry_model_names():
    registry = build_model_registry(best_alpha=1.0)
    expected = {"Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"}
    assert set(registry.keys()) == expected


def test_registry_ridge_uses_best_alpha():
    registry = build_model_registry(best_alpha=7.5)
    assert registry["Ridge Regression"].alpha == 7.5


def test_registry_tree_models_are_gridsearch():
    registry = build_model_registry(best_alpha=1.0)
    assert isinstance(registry["Random Forest"], GridSearchCV)
    assert isinstance(registry["Gradient Boosting"], GridSearchCV)


def test_registry_linear_models_are_not_gridsearch():
    registry = build_model_registry(best_alpha=1.0)
    assert isinstance(registry["Linear Regression"], LinearRegression)
    assert isinstance(registry["Ridge Regression"], Ridge)


# ---------------------------------------------------------------------------
# train_all (uses small grid to keep test fast)
# ---------------------------------------------------------------------------

def test_train_all_returns_fitted_models(split_xy):
    X_train, X_test, y_train, _ = split_xy
    fitted = train_all(X_train, y_train, best_alpha=1.0)
    assert set(fitted.keys()) == {"Linear Regression", "Ridge Regression",
                                   "Random Forest", "Gradient Boosting"}
    # Each model must be able to predict
    for name, model in fitted.items():
        preds = model.predict(X_test)
        assert len(preds) == len(X_test), f"{name} returned wrong prediction length"


def test_train_all_gridsearch_has_best_params(split_xy):
    X_train, _, y_train, _ = split_xy
    fitted = train_all(X_train, y_train, best_alpha=1.0)
    assert hasattr(fitted["Random Forest"], "best_params_")
    assert hasattr(fitted["Gradient Boosting"], "best_params_")


# ---------------------------------------------------------------------------
# rf_depth_curve
# ---------------------------------------------------------------------------

def test_rf_depth_curve_returns_dataframe(split_xy):
    X_train, X_test, y_train, y_test = split_xy
    curve = rf_depth_curve(X_train, y_train, X_test, y_test, depths=[3, 5, 7])
    assert isinstance(curve, pd.DataFrame)
    assert list(curve.columns) == ["max_depth", "train_r2", "test_r2"]
    assert len(curve) == 3


def test_rf_depth_curve_r2_in_valid_range(split_xy):
    X_train, X_test, y_train, y_test = split_xy
    curve = rf_depth_curve(X_train, y_train, X_test, y_test, depths=[3, 5])
    assert (curve["train_r2"] >= 0).all()
    assert (curve["train_r2"] <= 1).all()


def test_rf_depth_curve_train_gte_test(split_xy):
    """Train R² should be >= test R² for all depths (overfitting expectation)."""
    X_train, X_test, y_train, y_test = split_xy
    curve = rf_depth_curve(X_train, y_train, X_test, y_test, depths=[5, 10, 15])
    assert (curve["train_r2"] >= curve["test_r2"] - 0.05).all()


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_and_load_model(tmp_path, split_xy):
    X_train, X_test, y_train, _ = split_xy
    model = LinearRegression().fit(X_train, y_train)
    path = tmp_path / "model.pkl"

    save_model(model, path)
    assert path.exists()

    loaded = load_model(path)
    original_preds = model.predict(X_test)
    loaded_preds = loaded.predict(X_test)
    np.testing.assert_array_equal(original_preds, loaded_preds)


def test_save_model_creates_parent_dirs(tmp_path, split_xy):
    X_train, _, y_train, _ = split_xy
    model = LinearRegression().fit(X_train, y_train)
    nested_path = tmp_path / "nested" / "deep" / "model.pkl"

    save_model(model, nested_path)
    assert nested_path.exists()
