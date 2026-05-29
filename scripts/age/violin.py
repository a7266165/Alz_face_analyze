"""
scripts/age/violin.py
Age prediction error violin plots — 1:1 age-matched and unmatched comparisons.

Comparisons:
  AD vs HC/NAD/ACS         — age-matched 1:1 + unmatched
  MMSE/CASI high vs low    — within AD, age-matched 1:1 + unmatched

Output:
  violin/1by1match/{comparison}/matched_violin.png
  violin/all/{comparison}/unmatched_violin.png

Usage:
  conda run -n Alz_face_age python scripts/age/violin.py
  conda run -n Alz_face_age python scripts/age/violin.py --comparison ad_vs_hc
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    AGE_VIOLIN_DIR,
    DEMOGRAPHICS_DIR,
    PREDICTED_AGES_FILE,
)
from src.age.utils import load_predicted_ages

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HC_COMPARISONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
HILO_COMPARISONS = ["mmse_high_vs_low", "casi_high_vs_low"]

# ── data loading ─────────────────────────────────────────────────────────────

def load_all_with_error():
    preds = load_predicted_ages(PREDICTED_AGES_FILE)
    # 唯一讀取點：cohort.load_demographics() 已組好 ID(完整鍵) 並解析 Age/MMSE/CASI。
    from src.common.cohort import load_demographics
    df = load_demographics()
    df["group"] = df["Group"]
    df["predicted_age"] = df["ID"].map(preds)
    df = df.dropna(subset=["Age", "predicted_age"])
    df["age_error"] = df["Age"] - df["predicted_age"]
    df["subject"] = df["ID"].apply(lambda x: x.rsplit("-", 1)[0])
    return df.reset_index(drop=True)

# ── matching ─────────────────────────────────────────────────────────────────

def match_1to1_by_age(df_minor, df_major, caliper=5.0):
    """1:1 nearest-age matching without replacement."""
    minor = df_minor.sample(frac=1, random_state=42).reset_index(drop=True)
    major_pool = df_major.copy().reset_index(drop=True)
    used = set()
    pairs = []
    for _, row_m in minor.iterrows():
        dists = (major_pool["Age"] - row_m["Age"]).abs()
        dists.loc[list(used)] = np.inf
        idx = dists.idxmin()
        if dists[idx] <= caliper:
            used.add(idx)
            pairs.append((row_m["ID"], major_pool.loc[idx, "ID"]))
    matched_ids = {p[0] for p in pairs} | {p[1] for p in pairs}
    return df_minor[df_minor["ID"].isin(matched_ids)].copy(), \
           df_major[df_major["ID"].isin(matched_ids)].copy()

# ── violin drawing ───────────────────────────────────────────────────────────

def _draw_violin(ax, left_vals, right_vals, left_label, right_label,
                 title, left_color="#4C72B0", right_color="#C44E52"):
    data = [left_vals, right_vals]
    parts = ax.violinplot(data, showmedians=True)
    for pc, color in zip(parts["bodies"], [left_color, right_color]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{left_label}\n(n={len(left_vals)})",
                        f"{right_label}\n(n={len(right_vals)})"])
    ax.set_ylabel("Age error (real - predicted)")
    ax.set_title(title)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)


def save_violin(left_vals, right_vals, left_label, right_label,
                title, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_violin(ax, left_vals, right_vals, left_label, right_label, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out_path}")

# ── comparison runners ───────────────────────────────────────────────────────

def run_hc_comparison(df_all, comparison, output_dir, caliper=5.0):
    """AD vs {HC, NAD, ACS} violin."""
    cmp_map = {
        "ad_vs_hc": (["NAD", "ACS"], "HC"),
        "ad_vs_nad": (["NAD"], "NAD"),
        "ad_vs_acs": (["ACS"], "ACS"),
    }
    groups, label = cmp_map[comparison]
    df_hc = df_all[df_all["group"].isin(groups)]
    df_ad = df_all[df_all["group"] == "P"]

    # unmatched
    hc_err = df_hc["age_error"].dropna().values
    ad_err = df_ad["age_error"].dropna().values
    if len(hc_err) > 0 and len(ad_err) > 0:
        save_violin(hc_err, ad_err, label, "AD",
                    f"Age prediction error: {label} vs AD (all)",
                    output_dir / "all" / comparison / "unmatched_violin.png")

    # 1:1 matched
    hc_m, ad_m = match_1to1_by_age(df_hc, df_ad, caliper=caliper)
    if not hc_m.empty and not ad_m.empty:
        save_violin(hc_m["age_error"].values, ad_m["age_error"].values,
                    label, "AD",
                    f"Age prediction error: {label} vs AD\n(age-matched 1:1, caliper={caliper}y)",
                    output_dir / "1by1match" / comparison / "matched_violin.png")


def run_hilo_comparison(df_all, metric, output_dir, caliper=5.0):
    """MMSE/CASI high vs low within AD (and NAD)."""
    comparison = f"{metric.lower()}_high_vs_low"

    for grp, grp_label in [("P", "ad"), ("NAD", "nad")]:
        df_grp = df_all[df_all["group"] == grp].copy()
        df_grp[metric] = pd.to_numeric(df_grp[metric], errors="coerce")
        df_valid = df_grp.dropna(subset=[metric, "age_error"])
        if df_valid.empty:
            continue
        median_val = df_valid[metric].median()
        df_high = df_valid[df_valid[metric] >= median_val]
        df_low = df_valid[df_valid[metric] < median_val]

        # unmatched
        if not df_high.empty and not df_low.empty:
            save_violin(
                df_high["age_error"].values, df_low["age_error"].values,
                f"High {metric}", f"Low {metric}",
                f"Age error by {metric} ({grp_label.upper()}, all)\nmedian={median_val:.0f}",
                output_dir / "all" / comparison / f"{grp_label}_unmatched_violin.png")

        # 1:1 matched
        hi_m, lo_m = match_1to1_by_age(df_high, df_low, caliper=caliper)
        if not hi_m.empty and not lo_m.empty:
            save_violin(
                hi_m["age_error"].values, lo_m["age_error"].values,
                f"High {metric}", f"Low {metric}",
                f"Age error by {metric} ({grp_label.upper()})\n(age-matched 1:1, caliper={caliper}y)",
                output_dir / "1by1match" / comparison / f"{grp_label}_matched_violin.png")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=AGE_VIOLIN_DIR)
    ap.add_argument("--comparison", default=None,
                    choices=HC_COMPARISONS + HILO_COMPARISONS,
                    help="Run only one comparison; default runs all")
    ap.add_argument("--caliper", type=float, default=5.0)
    args = ap.parse_args()

    df_all = load_all_with_error()
    logger.info(f"loaded {len(df_all)} rows (ACS={int((df_all['group']=='ACS').sum())}, "
                f"NAD={int((df_all['group']=='NAD').sum())}, P={int((df_all['group']=='P').sum())})")

    targets = [args.comparison] if args.comparison else HC_COMPARISONS + HILO_COMPARISONS

    for cmp in targets:
        logger.info(f"--- {cmp} ---")
        if cmp in HC_COMPARISONS:
            run_hc_comparison(df_all, cmp, args.output_dir, caliper=args.caliper)
        elif cmp == "mmse_high_vs_low":
            run_hilo_comparison(df_all, "MMSE", args.output_dir, caliper=args.caliper)
        elif cmp == "casi_high_vs_low":
            run_hilo_comparison(df_all, "CASI", args.output_dir, caliper=args.caliper)


if __name__ == "__main__":
    main()
