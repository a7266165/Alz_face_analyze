"""
scripts/age/lines.py
Error-by-age line plots with internal + external dataset combinations.

Outputs:
  lines/correction/{no_sliding_window,sliding_window_10}/  — post-bootstrap corrected
  lines/error/{no_sliding_window,sliding_window_10}/       — raw residual (before correction)

Each directory contains 8 source combinations × {lines,merged} views.

Usage:
  conda run -n Alz_face_age python scripts/age/lines.py
  conda run -n Alz_face_age python scripts/age/lines.py --output-dir <cohort-dir>
"""

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    AGE_LINES_DIR,
    BOOTSTRAP_DIR,
    DEMOGRAPHICS_DIR,
    PREDICTED_AGES_FILE,
)
from src.age.calibration import load_predicted_ages

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLORS = {
    "ACS":       "#4CAF50",
    "NAD":       "#2196F3",
    "P":         "#F44336",
    "UTKFace":   "#9C27B0",
    "AgeDB":     "#17becf",
    "APPA-REAL": "#bcbd22",
}
EXTERNAL_SOURCES = ["AgeDB", "APPA-REAL", "UTKFace"]

# ── data loading ─────────────────────────────────────────────────────────────

def load_bootstrap_mean_coefs(coefs_csv: Path):
    df = pd.read_csv(coefs_csv, encoding="utf-8-sig")
    row = df[df["iter"].astype(str) == "mean"].iloc[0]
    a, b = float(row["a"]), float(row["b"])
    logger.info(f"bootstrap mean coefs: a={a:.4f}, b={b:.4f}")
    return a, b


def load_corrected_internal(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    logger.info(f"internal corrected: {len(df)} rows")
    return df


def build_external_corrected(source, a, b, preds):
    demo = pd.read_csv(DEMOGRAPHICS_DIR / "EACS.csv", encoding="utf-8-sig")
    demo = demo[demo["Source"] == source].copy()
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["predicted_age"] = demo["ID"].map(preds)
    demo = demo.dropna(subset=["Age", "predicted_age"]).reset_index(drop=True)
    real = demo["Age"].astype(float)
    pred = demo["predicted_age"].astype(float)
    corrected = pred + (a * real + b)
    return pd.DataFrame({
        "ID": demo["ID"], "group": source,
        "real_age": real, "predicted_age": pred,
        "corrected_age": corrected,
        "error_before": real - pred,
        "error_after": real - corrected,
        "age_int": real.astype(int),
    })

# ── plot functions ───────────────────────────────────────────────────────────

def plot_combined(df_all, output_path, groups, title, ylabel, y_col,
                  group_labels=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = group_labels or {}
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
        display = labels.get(grp, grp)
        ax.plot(ages, means, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{display} (n={len(sub)})")
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
                        group_labels=None, window=10, step=1, min_count=5,
                        xlim=(50, 100)):
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = group_labels or {}
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
        display = labels.get(grp, grp)
        ax.plot(xs, ms, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{display} (n={len(sub)})")
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

# ── helpers ──────────────────────────────────────────────────────────────────

def combo_suffix(sources):
    if not sources:
        return "internal"
    return "_".join(s.lower().replace("-", "") for s in sorted(sources, key=str.lower))


def combo_label(sources):
    return " + ".join(sorted(sources, key=str.lower))


def all_combos():
    return [()] + [tuple(c) for k in (1, 2, 3)
                   for c in combinations(EXTERNAL_SOURCES, k)]

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap-dir", type=Path, default=BOOTSTRAP_DIR)
    ap.add_argument("--output-dir", type=Path, default=AGE_LINES_DIR)
    args = ap.parse_args()

    DIR_CORR = args.output_dir / "correction" / "no_sliding_window"
    DIR_CORR_SW = args.output_dir / "correction" / "sliding_window_10"
    DIR_ERR = args.output_dir / "error" / "no_sliding_window"
    DIR_ERR_SW = args.output_dir / "error" / "sliding_window_10"

    logger.info(f"bootstrap-dir = {args.bootstrap_dir}")
    logger.info(f"output-dir    = {args.output_dir}")

    a, b = load_bootstrap_mean_coefs(args.bootstrap_dir / "data" / "bootstrap_coefficients.csv")
    df_internal = load_corrected_internal(args.bootstrap_dir / "data" / "corrected_ages.csv")

    preds = load_predicted_ages(PREDICTED_AGES_FILE)

    keep = ["group", "real_age", "predicted_age", "corrected_age",
            "error_before", "error_after", "age_int"]
    df_int = df_internal[keep]
    df_ext = {src: build_external_corrected(src, a, b, preds)[keep]
              for src in EXTERNAL_SOURCES}

    settings = [
        (DIR_CORR, DIR_CORR_SW,
         "Error (after bootstrap correction)", "error_after", "Corrected Error"),
        (DIR_ERR, DIR_ERR_SW,
         "Residual = real - predicted", "error_before", "Prediction Residual (Before Correction)"),
    ]

    for combo in all_combos():
        parts = [df_int] + [df_ext[s] for s in combo]
        df_combo = pd.concat(parts, ignore_index=True)
        groups_lines = ["ACS", "NAD", "P"] + sorted(combo, key=str.lower)
        suffix = combo_suffix(combo)
        label = combo_label(combo)

        df_merged = df_combo.copy()
        if combo:
            df_merged.loc[df_merged["group"].isin(combo), "group"] = "ACS"
        merged_label = f"ACS + {label}" if combo else "ACS"

        for dir_py, dir_sw, ylabel, y_col, title_state in settings:
            title_l = f"{title_state} by True Age — {' / '.join(groups_lines)} (mean ± std)"
            plot_combined(df_combo, dir_py / f"lines_{suffix}.png",
                          groups=groups_lines, title=title_l,
                          ylabel=ylabel, y_col=y_col)
            plot_sliding_window(df_combo, dir_sw / f"lines_{suffix}_sw10.png",
                                groups=groups_lines,
                                title=f"{title_state} by True Age — {' / '.join(groups_lines)} (10-y sliding window)",
                                ylabel=ylabel, y_col=y_col)
            if combo:
                title_m = f"{title_state} by True Age — {merged_label} / NAD / P (mean ± std)"
                plot_combined(df_merged, dir_py / f"merged_{suffix}.png",
                              groups=["ACS", "NAD", "P"], title=title_m,
                              ylabel=ylabel, y_col=y_col,
                              group_labels={"ACS": merged_label})
                plot_sliding_window(df_merged, dir_sw / f"merged_{suffix}_sw10.png",
                                    groups=["ACS", "NAD", "P"],
                                    title=f"{title_state} by True Age — {merged_label} / NAD / P (10-y sliding window)",
                                    ylabel=ylabel, y_col=y_col,
                                    group_labels={"ACS": merged_label})


if __name__ == "__main__":
    main()
