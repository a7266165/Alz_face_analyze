"""
scripts/age/error/lines.py
Prediction-residual (real − predicted) line plots by true age, for the
internal ACS / NAD / P groups.

Cohort is built with the canonical ``src.common.cohort.cohort_list`` — the same
gold-standard filtering ``histogram.py`` uses — so the CDR/MMSE/visit filters
actually applied match the output directory's cohort name. Residuals are computed
directly from the raw MiVOLO predictions (no age-calibration involved).

Outputs (under <AGE_ANALYSIS_DIR>/<visit_dir>/<cdr_mmse_dir>/lines/):
  no_sliding_window/lines_internal.png       — residual by integer age (mean ± std)
  sliding_window_10/lines_internal_sw10.png  — residual by 10-y sliding window

Usage:
  conda run -n Alz_face_main_analysis python scripts/age/error/lines.py
  conda run -n Alz_face_main_analysis python scripts/age/error/lines.py \
      --cohort-mode p_all_cdrall_hc_all_cdrall_or_mmseall
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    AGE_ANALYSIS_DIR,
    DEFAULT_COHORT_MODE,
    VALID_COHORT_CHOICES,
    cohort_path,
)
from src.age.error_table import load_age_error_table

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLORS = {
    "ACS": "#4CAF50",
    "NAD": "#2196F3",
    "P":   "#F44336",
}
GROUPS = ["ACS", "NAD", "P"]

# ── data loading ─────────────────────────────────────────────────────────────

def build_internal(cohort_mode: str) -> pd.DataFrame:
    """ACS/NAD/P residual = real − predicted, on the canonical cohort.

    The cohort × predictions join is delegated to the shared
    ``src.age.error_table.load_age_error_table``; the residual is kept under the
    legacy column name ``error_before`` for the plot helpers below.
    """
    df = load_age_error_table(cohort_mode)
    df["error_before"] = df["age_error"]
    return df[["group", "real_age", "predicted_age", "error_before", "age_int"]]

# ── plot functions ───────────────────────────────────────────────────────────

def plot_combined(df_all, output_path, groups, title, ylabel, y_col):
    fig, ax = plt.subplots(figsize=(14, 5))
    for grp in groups:
        color = COLORS.get(grp)
        if color is None:
            continue
        sub = df_all[df_all["group"] == grp]
        if sub.empty:
            continue
        st = sub.groupby("age_int")[y_col].agg(["mean", "std", "count"])
        st = st[st["count"] >= 3].sort_index()
        if st.empty:
            continue
        ages, means, stds = st.index.values, st["mean"].values, st["std"].fillna(0).values
        ax.plot(ages, means, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{grp} (n={len(sub)})")
        ax.fill_between(ages, means - stds, means + stds, color=color, alpha=0.15)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("True Age (y)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(50, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {output_path}")


def plot_sliding_window(df_all, output_path, groups, title, ylabel, y_col,
                        window=10, step=1, min_count=5, xlim=(50, 100)):
    fig, ax = plt.subplots(figsize=(14, 5))
    lo_x, hi_x = xlim
    starts = np.arange(lo_x, hi_x - window + 1 + step, step)

    for grp in groups:
        color = COLORS.get(grp)
        if color is None:
            continue
        sub = df_all[df_all["group"] == grp]
        if sub.empty:
            continue
        real, vals = sub["real_age"].values, sub[y_col].values
        xs, ms, ss = [], [], []
        for s in starts:
            mask = (real >= s) & (real < s + window)
            if mask.sum() < min_count:
                continue
            xs.append(s + window / 2.0)
            ms.append(vals[mask].mean())
            ss.append(vals[mask].std(ddof=0))
        if not xs:
            continue
        xs, ms, ss = np.array(xs), np.array(ms), np.array(ss)
        ax.plot(xs, ms, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{grp} (n={len(sub)})")
        ax.fill_between(xs, ms - ss, ms + ss, color=color, alpha=0.15)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel(f"True Age (y) — {window}-y sliding window center", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(lo_x, hi_x)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {output_path}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort-mode", default=DEFAULT_COHORT_MODE,
                    choices=VALID_COHORT_CHOICES,
                    help=f"canonical cohort name (預設: {DEFAULT_COHORT_MODE})")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="覆寫輸出目錄；留空依 cohort-mode 自動決定")
    args = ap.parse_args()

    output_dir = args.output_dir or (
        AGE_ANALYSIS_DIR / cohort_path(args.cohort_mode) / "lines")
    dir_py = output_dir / "no_sliding_window"
    dir_sw = output_dir / "sliding_window_10"

    logger.info(f"cohort-mode = {args.cohort_mode}")
    logger.info(f"output-dir  = {output_dir}")

    df_int = build_internal(args.cohort_mode)
    logger.info(f"internal rows: {len(df_int)} "
                f"({df_int['group'].value_counts().to_dict()})")

    ylabel = "Prediction Residual (real − predicted)"
    title_state = "Residual = real − predicted"

    plot_combined(
        df_int, dir_py / "lines_internal.png", groups=GROUPS,
        title=f"{title_state} by True Age — ACS / NAD / P (mean ± std)",
        ylabel=ylabel, y_col="error_before")
    plot_sliding_window(
        df_int, dir_sw / "lines_internal_sw10.png", groups=GROUPS,
        title=f"{title_state} by True Age — ACS / NAD / P (10-y sliding window)",
        ylabel=ylabel, y_col="error_before")


if __name__ == "__main__":
    main()
