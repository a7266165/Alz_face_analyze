"""
scripts/age/lines.py
Prediction-residual (real − predicted) line plots by true age, with internal +
external dataset combinations. No age-calibration involved — residuals are
computed directly from the raw MiVOLO predictions.

Outputs (under AGE_LINES_DIR):
  no_sliding_window/   — residual by integer age (mean ± std)
  sliding_window_10/   — residual by 10-year sliding window

Each directory contains 8 source combinations × {lines, merged} views.

Usage:
  conda run -n Alz_face_age python scripts/age/lines.py
  conda run -n Alz_face_age python scripts/age/lines.py --output-dir <cohort-dir>
"""

import argparse
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
    DEMOGRAPHICS_DIR,
    PREDICTED_AGES_FILE,
)
from src.age.utils import load_predicted_ages

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

_COLS = ["group", "real_age", "predicted_age", "error_before", "age_int"]

# ── data loading ─────────────────────────────────────────────────────────────

def build_internal(preds: dict) -> pd.DataFrame:
    """ACS/NAD/P residual = real − predicted, from raw predictions."""
    dfs = []
    for csv_name in ["ACS.csv", "NAD.csv", "P.csv"]:
        demo = pd.read_csv(DEMOGRAPHICS_DIR / csv_name, encoding="utf-8-sig")
        grp = csv_name.replace(".csv", "")
        demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
        demo["predicted_age"] = demo["ID"].map(preds)
        demo = demo.dropna(subset=["Age", "predicted_age"])
        real = demo["Age"].astype(float)
        pred = demo["predicted_age"].astype(float)
        dfs.append(pd.DataFrame({
            "group": grp,
            "real_age": real,
            "predicted_age": pred,
            "error_before": real - pred,
            "age_int": real.astype(int),
        }))
    return pd.concat(dfs, ignore_index=True)


def build_external(source: str, preds: dict) -> pd.DataFrame:
    """One EACS source's residual = real − predicted, from raw predictions."""
    path = DEMOGRAPHICS_DIR / "EACS.csv"
    if not path.exists():
        return pd.DataFrame(columns=_COLS)
    demo = pd.read_csv(path, encoding="utf-8-sig")
    if "Source" not in demo.columns:
        return pd.DataFrame(columns=_COLS)
    demo = demo[demo["Source"] == source].copy()
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["predicted_age"] = demo["ID"].map(preds)
    demo = demo.dropna(subset=["Age", "predicted_age"]).reset_index(drop=True)
    real = demo["Age"].astype(float)
    pred = demo["predicted_age"].astype(float)
    return pd.DataFrame({
        "group": source,
        "real_age": real,
        "predicted_age": pred,
        "error_before": real - pred,
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
    ap.add_argument("--output-dir", type=Path, default=AGE_LINES_DIR)
    args = ap.parse_args()

    dir_py = args.output_dir / "no_sliding_window"
    dir_sw = args.output_dir / "sliding_window_10"

    logger.info(f"output-dir = {args.output_dir}")

    preds = load_predicted_ages(PREDICTED_AGES_FILE)
    df_int = build_internal(preds)
    df_ext = {src: build_external(src, preds) for src in EXTERNAL_SOURCES}

    ylabel = "Prediction Residual (real − predicted)"
    title_state = "Residual = real − predicted"

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

        title_l = f"{title_state} by True Age — {' / '.join(groups_lines)} (mean ± std)"
        plot_combined(df_combo, dir_py / f"lines_{suffix}.png",
                      groups=groups_lines, title=title_l,
                      ylabel=ylabel, y_col="error_before")
        plot_sliding_window(df_combo, dir_sw / f"lines_{suffix}_sw10.png",
                            groups=groups_lines,
                            title=f"{title_state} by True Age — {' / '.join(groups_lines)} (10-y sliding window)",
                            ylabel=ylabel, y_col="error_before")
        if combo:
            title_m = f"{title_state} by True Age — {merged_label} / NAD / P (mean ± std)"
            plot_combined(df_merged, dir_py / f"merged_{suffix}.png",
                          groups=["ACS", "NAD", "P"], title=title_m,
                          ylabel=ylabel, y_col="error_before",
                          group_labels={"ACS": merged_label})
            plot_sliding_window(df_merged, dir_sw / f"merged_{suffix}_sw10.png",
                                groups=["ACS", "NAD", "P"],
                                title=f"{title_state} by True Age — {merged_label} / NAD / P (10-y sliding window)",
                                ylabel=ylabel, y_col="error_before",
                                group_labels={"ACS": merged_label})


if __name__ == "__main__":
    main()
