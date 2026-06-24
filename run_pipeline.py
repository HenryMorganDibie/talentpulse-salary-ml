#!/usr/bin/env python3
"""
TalentPulse Salary ML Pipeline — Main Entrypoint
=================================================

Runs the full supervised regression pipeline end-to-end:

    Step 1 · Data loading & EDA
    Step 2 · Data cleaning
    Step 3 · Feature engineering
    Step 4 · Model training & hyperparameter tuning
    Step 5 · Evaluation, visualisation & bias diagnostics

Outputs
-------
    reports/figures/    — 10 PNG visualisations
    reports/metrics.json — model scorecard + bias diagnostics
    models/             — serialised best model (pickle)

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --data data/raw/jobs_dataset.xlsx --out reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# ── Ensure src is on the path when run from repo root ──────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import (
    load_raw,
    get_labelled,
    parse_skills_column,
    train_test_split_stratified,
)
from src.features.engineering import build_features, get_feature_columns
from src.models.train import tune_ridge_alpha, train_all, rf_depth_curve, save_model
from src.evaluation.metrics import (
    plot_salary_distribution,
    plot_salary_by_group,
    plot_correlation_heatmap,
    plot_qq,
    compute_vif,
    plot_ridge_alpha_curve,
    plot_rf_depth_curve,
    score_all,
    plot_model_comparison,
    plot_feature_importances,
    plot_residuals,
    plot_segment_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TalentPulse Salary ML Pipeline")
    p.add_argument("--data", default="data/raw/jobs_dataset.xlsx",
                   help="Path to the raw jobs dataset (.xlsx or .csv)")
    p.add_argument("--out", default="reports",
                   help="Output directory for figures and metrics")
    p.add_argument("--models-dir", default="models",
                   help="Directory to save serialised models")
    return p.parse_args()


def build_residuals_frame(
    df_ohe: pd.DataFrame,
    y_test: pd.Series,
    best_preds: np.ndarray,
) -> pd.DataFrame:
    """Assemble test-set residuals with country / level / segment labels."""
    df_t = df_ohe.loc[y_test.index].copy()
    df_t["pred_log"]   = best_preds
    df_t["pred_sal"]   = np.expm1(df_t["pred_log"])
    df_t["actual_sal"] = np.expm1(y_test)
    df_t["residual"]   = df_t["pred_sal"] - df_t["actual_sal"]

    cc = [c for c in df_t.columns if c.startswith("country_")]
    lc = [c for c in df_t.columns if c.startswith("job_level_")]
    df_t["country_name"] = (
        df_t[cc].idxmax(axis=1).str.replace("country_", "", regex=False) if cc else "Unknown"
    )
    df_t["level_name"] = (
        df_t[lc].idxmax(axis=1).str.replace("job_level_", "", regex=False) if lc else "Unknown"
    )
    df_t["segment"] = df_t["is_senior"].map(
        {1: "Premium (Senior/Lead)", 0: "Standard (Junior/Mid)"}
    )
    return df_t


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.out) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Load & EDA ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Data Loading & EDA")
    logger.info("=" * 60)

    df_raw = load_raw(args.data)
    salary_all = df_raw["salary_avg"].dropna()

    plot_salary_distribution(salary_all, out_dir=figures_dir)
    plot_salary_by_group(df_raw[df_raw["has_salary"] == True], out_dir=figures_dir)

    # ── STEP 2: Clean ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Data Cleaning")
    logger.info("=" * 60)

    df_labelled = get_labelled(df_raw)
    df_labelled = parse_skills_column(df_labelled)
    logger.info(
        "Unlabelled rows (held out for inference): %d",
        len(df_raw) - len(df_labelled),
    )

    # ── STEP 3: Feature Engineering ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Feature Engineering")
    logger.info("=" * 60)

    df_ohe = build_features(df_labelled)
    feature_cols = get_feature_columns(df_ohe)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    plot_correlation_heatmap(df_ohe, feature_cols, out_dir=figures_dir)
    plot_qq(salary_all, out_dir=figures_dir)

    X = df_ohe[feature_cols].fillna(0)
    y = df_ohe["log_salary"]

    # stratified split — preserves job_level distribution
    train_df, test_df = train_test_split_stratified(df_ohe)
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["log_salary"]
    X_test  = test_df[feature_cols].fillna(0)
    y_test  = test_df["log_salary"]

    # VIF check (base numeric features only, no OHE dummies)
    base_feats = [c for c in feature_cols if not c.startswith(("job_level_", "country_"))]
    vif_df = compute_vif(X, cols=base_feats)

    # ── STEP 4: Model Training ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Model Training & Hyperparameter Tuning")
    logger.info("=" * 60)

    alphas = np.logspace(-2, 4, 30)
    best_alpha, alpha_cv_pairs = tune_ridge_alpha(X_train, y_train, alphas=alphas)
    alpha_scores = [v for _, v in alpha_cv_pairs]
    plot_ridge_alpha_curve(alphas, alpha_scores, best_alpha, out_dir=figures_dir)

    fitted_models = train_all(X_train, y_train, best_alpha)

    # RF overfitting curve
    curve_df = rf_depth_curve(X_train, y_train, X_test, y_test)
    overfit_depth = plot_rf_depth_curve(curve_df, out_dir=figures_dir)

    # ── STEP 5: Evaluation ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 — Evaluation & Bias Diagnostics")
    logger.info("=" * 60)

    results = score_all(fitted_models, X_test, y_test)
    models_plot = list(fitted_models.keys())
    plot_model_comparison(results, models_plot, out_dir=figures_dir)

    best_name = min(results, key=lambda n: results[n]["MAE"])
    best_fitted = fitted_models[best_name]
    # Unwrap GridSearchCV to get the actual estimator
    best_estimator = (
        best_fitted.best_estimator_
        if hasattr(best_fitted, "best_estimator_")
        else best_fitted
    )

    logger.info("Best model: %s  MAE $%,.0f", best_name, results[best_name]["MAE"])
    logger.info("Legacy MAE: $18,500 → improvement $%,.0f (%.1f%%)",
                18500 - results[best_name]["MAE"],
                (18500 - results[best_name]["MAE"]) / 18500 * 100)

    if hasattr(best_estimator, "feature_importances_"):
        plot_feature_importances(best_estimator, feature_cols, best_name, out_dir=figures_dir)

    # Residuals
    best_preds = results[best_name]["preds"]
    # Re-align: score_all uses test_df indices
    y_test_aligned = test_df["log_salary"]
    df_test_res = build_residuals_frame(df_ohe, y_test_aligned, best_preds)
    plot_residuals(df_test_res, best_name, out_dir=figures_dir)
    plot_segment_comparison(df_test_res, out_dir=figures_dir)

    # Bias diagnostics
    bias_country = df_test_res.groupby("country_name")["residual"].mean().to_dict()
    bias_level   = df_test_res.groupby("level_name")["residual"].mean().to_dict()
    seg_mae = (
        df_test_res.groupby("segment")
        .apply(lambda g: float(mean_absolute_error(g["actual_sal"], g["pred_sal"])))
        .to_dict()
    )

    logger.info("Mean residual by country: %s",
                {k: f"${v:,.0f}" for k, v in bias_country.items()})
    logger.info("Mean residual by job level: %s",
                {k: f"${v:,.0f}" for k, v in bias_level.items()})
    logger.info("Segment MAE: %s", {k: f"${v:,.0f}" for k, v in seg_mae.items()})

    # ── Save model + metrics ──────────────────────────────────────────────
    save_model(best_estimator, f"{args.models_dir}/best_model.pkl")

    metrics_out = {
        "best_model": best_name,
        "results": {
            n: {"MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R²"]}
            for n, r in results.items()
        },
        "legacy_mae": 18500,
        "industry_target_mae": 8000,
        "bias_country": bias_country,
        "bias_level": bias_level,
        "segment_mae": seg_mae,
        "rf_best_params": getattr(
            fitted_models.get("Random Forest"), "best_params_", {}
        ),
        "gb_best_params": getattr(
            fitted_models.get("Gradient Boosting"), "best_params_", {}
        ),
        "best_ridge_alpha": float(best_alpha),
        "overfit_depth": overfit_depth,
        "vif_table": vif_df.to_dict(orient="records"),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_cols": feature_cols,
    }

    metrics_path = Path(args.out) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    logger.info("Metrics saved → %s", metrics_path)

    logger.info("=" * 60)
    logger.info("Pipeline complete. Figures: %s", figures_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
