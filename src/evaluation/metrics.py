"""
Model evaluation, visualisation, and bias diagnostics for TalentPulse.

All plots are written to reports/figures/ and named by step for traceability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 13, "axes.labelsize": 11})

FIGURE_DIR = Path("reports/figures")
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def _savefig(fig: plt.Figure, name: str, out_dir: Path = FIGURE_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_salary_distribution(salary: pd.Series, out_dir: Path = FIGURE_DIR) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EDA — Salary Distribution & Log-Transform", fontsize=14, fontweight="bold")

    axes[0].hist(salary, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(salary.median(), color="red", linestyle="--", lw=1.8,
                    label=f"Median ${salary.median():,.0f}")
    axes[0].set_title("salary_avg — Raw")
    axes[0].set_xlabel("Salary (USD)"); axes[0].set_ylabel("Count"); axes[0].legend()

    axes[1].hist(np.log1p(salary), bins=60, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_title("log1p(salary_avg) — After Transform")
    axes[1].set_xlabel("log(Salary + 1)"); axes[1].set_ylabel("Count")

    plt.tight_layout()
    _savefig(fig, "fig1_salary_distribution.png", out_dir)


def plot_salary_by_group(df: pd.DataFrame, out_dir: Path = FIGURE_DIR) -> None:
    level_order = [l for l in ["Junior", "Mid", "Senior", "Lead"] if l in df["job_level"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("EDA — Salary by Job Level & Country", fontsize=14, fontweight="bold")

    sns.boxplot(data=df, x="job_level", y="salary_avg", order=level_order,
                ax=axes[0], palette="Blues_d")
    axes[0].set_title("By Job Level"); axes[0].set_xlabel(""); axes[0].set_ylabel("Salary (USD)")

    sns.boxplot(data=df, x="country", y="salary_avg", ax=axes[1], palette="Oranges_d")
    axes[1].set_title("By Country"); axes[1].set_xlabel(""); axes[1].set_ylabel("Salary (USD)")

    plt.tight_layout()
    _savefig(fig, "fig2_salary_by_group.png", out_dir)


def plot_correlation_heatmap(df_ohe: pd.DataFrame, feature_cols: List[str],
                              out_dir: Path = FIGURE_DIR) -> None:
    num_cols = [c for c in feature_cols if c in df_ohe.columns] + ["salary_avg"]
    num_cols = [c for c in num_cols if c in df_ohe.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_ohe[num_cols].corr(), annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax, square=True, linewidths=0.4)
    ax.set_title("EDA — Pearson Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "fig3_correlation_heatmap.png", out_dir)


def plot_qq(salary: pd.Series, out_dir: Path = FIGURE_DIR) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Feature Engineering — Q-Q Plots: Raw vs Log-Transformed Salary",
                 fontsize=13, fontweight="bold")
    sp_stats.probplot(salary.values, dist="norm", plot=axes[0])
    axes[0].set_title("Raw salary_avg")
    sp_stats.probplot(np.log1p(salary.values), dist="norm", plot=axes[1])
    axes[1].set_title("log1p(salary_avg)")
    plt.tight_layout()
    _savefig(fig, "fig8_qq_plots.png", out_dir)


# ---------------------------------------------------------------------------
# Hyperparameter tuning plots
# ---------------------------------------------------------------------------

def plot_ridge_alpha_curve(alphas: np.ndarray, cv_scores: List[float],
                            best_alpha: float, out_dir: Path = FIGURE_DIR) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(alphas, cv_scores, "o-", color="#55A868", lw=2)
    ax.axvline(best_alpha, color="red", linestyle="--", lw=1.5,
               label=f"Optimal α={best_alpha:.4f}")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("5-Fold CV MAE (log-salary units)")
    ax.set_title("Hyperparameter Tuning — Ridge Regression: CV Error vs Alpha",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "fig9_ridge_alpha.png", out_dir)


def plot_rf_depth_curve(curve_df: pd.DataFrame, out_dir: Path = FIGURE_DIR) -> None:
    overfit_depth = None
    for _, row in curve_df.iterrows():
        if row["train_r2"] - row["test_r2"] > 0.05:
            overfit_depth = int(row["max_depth"])
            break

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve_df["max_depth"], curve_df["train_r2"], "o-",
            color="#4C72B0", lw=2, label="Train R²")
    ax.plot(curve_df["max_depth"], curve_df["test_r2"], "s-",
            color="#C44E52", lw=2, label="Test R²")
    if overfit_depth:
        ax.axvline(overfit_depth, color="grey", linestyle=":", lw=1.5,
                   label=f"Overfitting ≈ depth {overfit_depth}")
    ax.fill_between(curve_df["max_depth"], curve_df["train_r2"],
                    curve_df["test_r2"], alpha=0.08, color="red")
    ax.set_xlabel("max_depth"); ax.set_ylabel("R²")
    ax.set_title("Hyperparameter Tuning — RF Train vs Test R² by Depth",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "fig7_overfitting.png", out_dir)
    return overfit_depth


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def score_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> Dict[str, float]:
    """Compute MAE, RMSE, R² on original salary scale (inverse log1p)."""
    p_log = model.predict(X_test)
    a_log = y_test.values
    p_sal = np.expm1(p_log)
    a_sal = np.expm1(a_log)
    return {
        "name":  name,
        "MAE":   float(mean_absolute_error(a_sal, p_sal)),
        "RMSE":  float(np.sqrt(mean_squared_error(a_sal, p_sal))),
        "R²":    float(r2_score(a_log, p_log)),
        "preds": p_log,
    }


def score_all(
    fitted_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict]:
    results = {name: score_model(m, X_test, y_test, name)
               for name, m in fitted_models.items()}
    logger.info("\n--- Model Scorecard ---")
    for n, r in results.items():
        logger.info("  %-22s MAE $%,.0f  RMSE $%,.0f  R² %.4f",
                    n, r["MAE"], r["RMSE"], r["R²"])
    return results


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def plot_model_comparison(results: Dict, models_plot: List[str],
                           out_dir: Path = FIGURE_DIR) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Evaluation — MAE, RMSE, R² Across All Models",
                 fontsize=14, fontweight="bold")

    for ax, metric, title in zip(
        axes,
        ["MAE", "RMSE", "R²"],
        ["MAE (USD) — lower better", "RMSE (USD) — lower better", "R² — higher better"],
    ):
        vals = [results[m][metric] for m in models_plot]
        bars = ax.bar(models_plot, vals, color=COLORS, width=0.55, edgecolor="white")
        ax.set_title(title); ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            lbl = f"${v:,.0f}" if metric != "R²" else f"{v:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    lbl, ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticklabels(models_plot, rotation=18, ha="right", fontsize=8)
        if metric == "MAE":
            ax.axhline(8000,  color="red",  linestyle="--", lw=1.5, label="$8k target")
            ax.axhline(18500, color="grey", linestyle=":",  lw=1.2, label="Legacy $18.5k")
            ax.legend(fontsize=8)

    plt.tight_layout()
    _savefig(fig, "fig4_model_comparison.png", out_dir)


def plot_feature_importances(
    model: Any, feature_cols: List[str], model_name: str,
    out_dir: Path = FIGURE_DIR,
) -> None:
    fi = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(15)
    )
    pal = ["#C44E52" if ("level" in c or "senior" in c) else "#4C72B0"
           for c in fi.index[::-1]]
    fig, ax = plt.subplots(figsize=(10, 6))
    fi.sort_values().plot.barh(ax=ax, color=pal, edgecolor="white")
    ax.set_title(f"Model Evaluation — Top-15 Feature Importances ({model_name})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.legend(handles=[
        mpatches.Patch(color="#C44E52", label="Seniority"),
        mpatches.Patch(color="#4C72B0", label="Other"),
    ], fontsize=9)
    plt.tight_layout()
    _savefig(fig, "fig5_feature_importances.png", out_dir)


def plot_residuals(df_test: pd.DataFrame, best_name: str,
                   out_dir: Path = FIGURE_DIR) -> None:
    lo = [l for l in ["Junior", "Mid", "Senior", "Lead"]
          if l in df_test["level_name"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Model Evaluation — Residuals by Country & Level ({best_name})",
        fontsize=13, fontweight="bold",
    )
    sns.boxplot(data=df_test, x="country_name", y="residual",
                ax=axes[0], palette="Blues_d")
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_title("By Country"); axes[0].set_xlabel("")
    axes[0].set_ylabel("Predicted − Actual Salary (USD)")

    sns.boxplot(data=df_test, x="level_name", y="residual",
                order=lo, ax=axes[1], palette="Oranges_d")
    axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title("By Job Level"); axes[1].set_xlabel("")
    axes[1].set_ylabel("Predicted − Actual Salary (USD)")

    plt.tight_layout()
    _savefig(fig, "fig6_residuals.png", out_dir)


def plot_segment_comparison(df_test: pd.DataFrame,
                             out_dir: Path = FIGURE_DIR) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Segment Analysis — Premium vs Standard Role Accuracy",
                 fontsize=13, fontweight="bold")

    for i, (seg, grp) in enumerate(df_test.groupby("segment")):
        mae_s = mean_absolute_error(grp["actual_sal"], grp["pred_sal"])
        axes[0].scatter(grp["actual_sal"], grp["pred_sal"], alpha=0.25, s=14,
                        label=f"{seg}\nMAE ${mae_s:,.0f}",
                        color=["#4C72B0", "#C44E52"][i])

    mn, mx = df_test["actual_sal"].min(), df_test["actual_sal"].max()
    axes[0].plot([mn, mx], [mn, mx], "k--", lw=1.2, label="Perfect prediction")
    axes[0].set_xlabel("Actual (USD)"); axes[0].set_ylabel("Predicted (USD)")
    axes[0].set_title("Actual vs Predicted"); axes[0].legend(fontsize=8)

    sns.boxplot(data=df_test, x="segment", y="residual",
                palette=["#4C72B0", "#C44E52"], ax=axes[1])
    axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title("Residuals by Segment")
    axes[1].set_xlabel(""); axes[1].set_ylabel("Residual (USD)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=10, fontsize=9)

    plt.tight_layout()
    _savefig(fig, "fig10_segment_comparison.png", out_dir)


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor for multicollinearity diagnosis."""
    X_sub = X[cols].fillna(0)
    vif_data = pd.DataFrame({
        "Feature": cols,
        "VIF": [variance_inflation_factor(X_sub.values, i) for i in range(len(cols))],
    }).sort_values("VIF", ascending=False)
    high = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
    if high:
        logger.warning("Features with VIF > 10 (multicollinearity risk): %s", high)
    else:
        logger.info("VIF check passed — no features exceed VIF=10")
    return vif_data
