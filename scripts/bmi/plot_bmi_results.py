"""
Visualize BMI regression results: train (resubstitution) vs test (OOF).

Layout: 2 rows × 3 columns
    Row 1: Train — P / NAD / ACS  (how well the model fits training data)
    Row 2: Test  — P / NAD / ACS  (actual generalization, OOF predictions)

Requires oof_{model}.csv and resub_{model}.csv in the analysis directory.

Usage:
    conda run -n Alz_face_main_analysis python scripts/bmi/plot_bmi_results.py
    conda run -n Alz_face_main_analysis python scripts/bmi/plot_bmi_results.py --models svr
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
GROUP_LABELS = {"P": "Dementia (P)", "NAD": "Non-Dementia (NAD)", "ACS": "ACS"}


def _parse_group(sid: str) -> str:
    if sid.startswith("P"):
        return "P"
    if sid.startswith("NAD"):
        return "NAD"
    if sid.startswith("ACS"):
        return "ACS"
    return "OTHER"


def _metrics_text(y_true, y_pred):
    r, _ = stats.pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return f"r={r:.3f}  MAE={mae:.2f}  R²={r2:.3f}"


def plot_scatter_train_test(df_resub, df_test, model_name, out_dir):
    grp_order = [g for g in ["P", "NAD", "ACS"]
                 if (df_test["group"] == g).sum() > 0]
    n_cols = len(grp_order)

    fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 14))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    all_y = np.concatenate([df_resub["y_true"], df_resub["y_pred"],
                            df_test["y_true"], df_test["y_pred"]])
    lo, hi = all_y.min() - 1, all_y.max() + 1

    for col_i, grp in enumerate(grp_order):
        # Row 0: Train (resubstitution)
        ax = axes[0, col_i]
        sub = df_resub[df_resub["group"] == grp]
        ax.scatter(sub["y_true"], sub["y_pred"],
                   c=GROUP_COLORS[grp], alpha=0.3, s=16)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_title(f"Train — {GROUP_LABELS[grp]}  (n={len(sub)})\n"
                     f"{_metrics_text(sub['y_true'].values, sub['y_pred'].values)}",
                     fontsize=13)
        ax.set_xlabel("True BMI", fontsize=11)
        ax.set_ylabel("Predicted BMI", fontsize=11)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        # Row 1: Test (OOF)
        ax = axes[1, col_i]
        sub = df_test[df_test["group"] == grp]
        ax.scatter(sub["y_true"], sub["y_pred"],
                   c=GROUP_COLORS[grp], alpha=0.4, s=20)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_title(f"Test — {GROUP_LABELS[grp]}  (n={len(sub)})\n"
                     f"{_metrics_text(sub['y_true'].values, sub['y_pred'].values)}",
                     fontsize=13)
        ax.set_xlabel("True BMI", fontsize=11)
        ax.set_ylabel("Predicted BMI (OOF)", fontsize=11)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    fig.suptitle(f"{model_name.upper()}", fontsize=18, y=1.005)
    fig.tight_layout()

    out = out_dir / f"scatter_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["svr", "ridge", "xgb"])
    args = parser.parse_args()

    n_plots = 0
    for model_name in args.models:
        oof_csv = BMI_ANALYSIS_DIR / f"oof_{model_name}.csv"
        resub_csv = BMI_ANALYSIS_DIR / f"resub_{model_name}.csv"
        if not oof_csv.exists():
            logger.warning(f"No OOF file: {oof_csv}")
            continue
        if not resub_csv.exists():
            logger.warning(f"No resub file: {resub_csv} (re-run train_bmi.py)")
            continue

        df_test = pd.read_csv(oof_csv)
        df_test["group"] = df_test["ID"].apply(_parse_group)

        df_resub = pd.read_csv(resub_csv)
        df_resub["group"] = df_resub["ID"].apply(_parse_group)

        out = plot_scatter_train_test(df_resub, df_test, model_name, BMI_ANALYSIS_DIR)
        logger.info(f"  {out.name}")
        n_plots += 1

    logger.info(f"Done: {n_plots} plots written to {BMI_ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
