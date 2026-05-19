"""
Visualize BMI regression results from OOF predictions.

Generates three plots per model:
  1. Scatter: true vs predicted BMI (with identity line + Pearson r)
  2. Residual: residual vs true BMI (detect systematic bias)
  3. Error distribution: histogram of absolute errors

Usage:
    conda run -n Alz_face_main_analysis python scripts/bmi/plot_bmi_results.py
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BMI_ANALYSIS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GROUP_COLORS = {"P": "#E74C3C", "NAD": "#3498DB", "ACS": "#2ECC71"}


def _parse_group(sid: str) -> str:
    if sid.startswith("P"):
        return "P"
    if sid.startswith("NAD"):
        return "NAD"
    if sid.startswith("ACS"):
        return "ACS"
    return "OTHER"


def plot_scatter(df, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    groups = df["group"]
    for grp in ["P", "NAD", "ACS"]:
        mask = groups == grp
        if mask.sum() == 0:
            continue
        ax.scatter(df.loc[mask, "y_true"], df.loc[mask, "y_pred"],
                   c=GROUP_COLORS[grp], alpha=0.4, s=18, label=f"{grp} (n={mask.sum()})")

    lo = min(df["y_true"].min(), df["y_pred"].min()) - 1
    hi = max(df["y_true"].max(), df["y_pred"].max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)

    r, p = stats.pearsonr(df["y_true"], df["y_pred"])
    mae = np.mean(np.abs(df["y_true"] - df["y_pred"]))
    ax.set_title(f"{model_name.upper()}  |  r={r:.3f}  MAE={mae:.2f}", fontsize=14)
    ax.set_xlabel("True BMI", fontsize=12)
    ax.set_ylabel("Predicted BMI (OOF)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()

    out = out_dir / f"scatter_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_residual(df, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    residuals = df["y_true"] - df["y_pred"]

    for grp in ["P", "NAD", "ACS"]:
        mask = df["group"] == grp
        if mask.sum() == 0:
            continue
        ax.scatter(df.loc[mask, "y_true"], residuals[mask],
                   c=GROUP_COLORS[grp], alpha=0.35, s=14, label=grp)

    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("True BMI", fontsize=12)
    ax.set_ylabel("Residual (True - Pred)", fontsize=12)
    ax.set_title(f"{model_name.upper()} residuals", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()

    out = out_dir / f"residual_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_error_dist(df, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    abs_err = np.abs(df["y_true"] - df["y_pred"])

    ax.hist(abs_err, bins=50, color="#5DADE2", edgecolor="white", alpha=0.8)
    mean_ae = abs_err.mean()
    median_ae = np.median(abs_err)
    ax.axvline(mean_ae, color="red", ls="--", lw=1.5, label=f"Mean AE = {mean_ae:.2f}")
    ax.axvline(median_ae, color="orange", ls="--", lw=1.5, label=f"Median AE = {median_ae:.2f}")

    ax.set_xlabel("Absolute Error", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{model_name.upper()} error distribution", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()

    out = out_dir / f"error_dist_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["ridge", "xgb"])
    args = parser.parse_args()

    n_plots = 0
    for model_name in args.models:
        oof_csv = BMI_ANALYSIS_DIR / f"oof_{model_name}.csv"
        if not oof_csv.exists():
            logger.warning(f"No OOF file for {model_name}: {oof_csv}")
            continue

        df = pd.read_csv(oof_csv)
        df["group"] = df["ID"].apply(_parse_group)
        logger.info(f"{model_name}: {len(df)} OOF predictions loaded")

        for plot_fn in (plot_scatter, plot_residual, plot_error_dist):
            out = plot_fn(df, model_name, BMI_ANALYSIS_DIR)
            logger.info(f"  {out.name}")
            n_plots += 1

    logger.info(f"Done: {n_plots} plots written to {BMI_ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
