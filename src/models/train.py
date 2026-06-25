"""
Model training, hyperparameter tuning, and serialisation for TalentPulse.

Four algorithm families are compared:
  1. Linear Regression  — baseline, fully interpretable
  2. Ridge Regression   — L2-regularised baseline; alpha tuned via CV
  3. Random Forest      — non-linear ensemble; depth + n_estimators tuned
  4. Gradient Boosting  — boosted ensemble; learning rate + depth tuned

All tree models are tuned with 5-fold GridSearchCV to prevent overfitting.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

logger = logging.getLogger(__name__)

# Number of trees used for the depth-curve diagnostic (separate from GridSearch)
_DEPTH_CURVE_N_ESTIMATORS = 100


# ---------------------------------------------------------------------------
# Ridge alpha selection
# ---------------------------------------------------------------------------

def tune_ridge_alpha(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: Optional[np.ndarray] = None,
    cv: int = 5,
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Select optimal L2 regularisation strength for Ridge via cross-validated MAE.
    Searches log-space from 1e-2 to 1e4 by default.

    Returns
    -------
    best_alpha : float
    alpha_cv_pairs : list of (alpha, cv_mae) tuples for plotting
    """
    if alphas is None:
        alphas = np.logspace(-2, 4, 30)

    cv_scores = [
        -cross_val_score(
            Ridge(alpha=a), X_train, y_train,
            cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1,
        ).mean()
        for a in alphas
    ]
    best_alpha = float(alphas[np.argmin(cv_scores)])
    logger.info("Ridge CV — best alpha: %.4f  (CV MAE: %.5f)", best_alpha, min(cv_scores))
    return best_alpha, list(zip(alphas.tolist(), cv_scores))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_model_registry(best_alpha: float) -> Dict[str, Any]:
    """Return a dict of untrained estimators keyed by model name."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=best_alpha),
        "Random Forest": GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [8, 12, None],
            },
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        ),
        "Gradient Boosting": GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [4, 6],
            },
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        ),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_alpha: float,
) -> Dict[str, Any]:
    """
    Fit all models in the registry on training data.

    Returns
    -------
    dict mapping model name → fitted estimator (GridSearchCV or base estimator)
    """
    registry = build_model_registry(best_alpha)
    fitted: Dict[str, Any] = {}

    for name, estimator in registry.items():
        logger.info("Training: %s", name)
        estimator.fit(X_train, y_train)
        fitted[name] = estimator
        if hasattr(estimator, "best_params_"):
            logger.info("  Best params: %s", estimator.best_params_)

    return fitted


# ---------------------------------------------------------------------------
# Overfitting analysis
# ---------------------------------------------------------------------------

def rf_depth_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    depths: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Train a Random Forest at each depth in `depths` and record train/test R².

    Used to diagnose the overfitting cliff and visually justify the depth
    selected by GridSearchCV. Uses a fixed n_estimators for speed; this is
    a diagnostic pass, not a production model.

    Returns a DataFrame with columns: max_depth, train_r2, test_r2.
    """
    if depths is None:
        depths = list(range(3, 21))

    records = []
    for d in depths:
        m = RandomForestRegressor(
            n_estimators=_DEPTH_CURVE_N_ESTIMATORS,
            max_depth=d,
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X_train, y_train)
        records.append({
            "max_depth": d,
            "train_r2": r2_score(y_train, m.predict(X_train)),
            "test_r2":  r2_score(y_test,  m.predict(X_test)),
        })
        logger.debug("  depth=%2d  train_r2=%.4f  test_r2=%.4f",
                     d, records[-1]["train_r2"], records[-1]["test_r2"])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path) -> None:
    """Serialise a fitted model to disk via pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", path)


def load_model(path: str | Path) -> Any:
    """Deserialise a model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded ← %s", path)
    return model
