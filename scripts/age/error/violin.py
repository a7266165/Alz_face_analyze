"""
scripts/age/error/violin.py
Age prediction error violin plots — full cohort and 1:1 age-matched comparisons.

Cohort is built with the canonical ``src.common.cohort.cohort_list`` (same
gold-standard filtering as histogram / stat / lines). Comparisons:
  AD vs HC / NAD / ACS      — group matching      (canonical match_cohort)
  MMSE / CASI high vs low   — score-median match  (canonical match_by_score)

Each comparison outputs both the full cohort and the 1:1 age-matched subset:
  violin/full/{comparison}/...
  violin/1by1matched/{comparison}/...

Usage:
  conda run -n Alz_face_main_analysis python scripts/age/error/violin.py
  conda run -n Alz_face_main_analysis python scripts/age/error/violin.py --comparison ad_vs_hc
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    AGE_ANALYSIS_DIR,
    DEFAULT_COHORT_MODE,
    VALID_COHORT_CHOICES,
    cohort_path,
)
from src.age.error_table import (
    load_age_error_table,
    matched_ad_vs_hc,
    matched_by_score,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HC_COMPARISONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
HILO_COMPARISONS = ["mmse_high_vs_low", "casi_high_vs_low"]

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


def save_violin(left_vals, right_vals, left_label, right_label, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_violin(ax, left_vals, right_vals, left_label, right_label, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out_path}")

# ── comparison runners ───────────────────────────────────────────────────────

def run_hc_comparison(df_all, comparison, output_dir, caliper=1.0):
    """AD vs {HC, NAD, ACS} violin — full cohort + 1:1 age-matched (match_cohort).

    ``controls`` 餵給 canonical match_cohort：None → 全部非 P（NAD+ACS）；指定
    組則只取該組當對照。
    """
    cmp_map = {
        "ad_vs_hc": (None, ["NAD", "ACS"], "HC"),
        "ad_vs_nad": (["NAD"], ["NAD"], "NAD"),
        "ad_vs_acs": (["ACS"], ["ACS"], "ACS"),
    }
    controls, hc_groups, label = cmp_map[comparison]

    # full (unmatched)
    df_hc = df_all[df_all["group"].isin(hc_groups)]
    df_ad = df_all[df_all["group"] == "P"]
    hc_err = df_hc["age_error"].dropna().values
    ad_err = df_ad["age_error"].dropna().values
    if len(hc_err) > 0 and len(ad_err) > 0:
        save_violin(hc_err, ad_err, label, "AD",
                    f"Age prediction error: {label} vs AD (full)",
                    output_dir / "full" / comparison / "violin.png")

    # 1:1 age-matched (canonical)
    msub = matched_ad_vs_hc(df_all, controls=controls, caliper=caliper)
    m_hc = msub[msub["group"] != "P"]
    m_ad = msub[msub["group"] == "P"]
    if not m_hc.empty and not m_ad.empty:
        save_violin(m_hc["age_error"].values, m_ad["age_error"].values, label, "AD",
                    f"Age prediction error: {label} vs AD\n"
                    f"(age-matched 1:1, caliper={caliper}y)",
                    output_dir / "1by1matched" / comparison / "violin.png")


def run_hilo_comparison(df_all, metric, output_dir, caliper=1.0):
    """MMSE/CASI high vs low within AD (and NAD) — full + 1:1 age-matched.

    1:1 matched 子集委派給 canonical match_by_score（在該組內以中位數切 high/low
    再做年齡配對）；繪圖時以同一中位數把被選中的列分回兩臂。
    """
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

        # full (unmatched)
        if not df_high.empty and not df_low.empty:
            save_violin(
                df_high["age_error"].values, df_low["age_error"].values,
                f"High {metric}", f"Low {metric}",
                f"Age error by {metric} ({grp_label.upper()}, full)\nmedian={median_val:.0f}",
                output_dir / "full" / comparison / f"{grp_label}_violin.png")

        # 1:1 age-matched (canonical match_by_score; same median split)
        msub = matched_by_score(df_valid, metric, cut="median", caliper=caliper)
        if not msub.empty:
            sm = pd.to_numeric(msub[metric], errors="coerce")
            hi = msub[sm >= median_val]["age_error"].values
            lo = msub[sm < median_val]["age_error"].values
            if len(hi) > 0 and len(lo) > 0:
                save_violin(
                    hi, lo, f"High {metric}", f"Low {metric}",
                    f"Age error by {metric} ({grp_label.upper()})\n"
                    f"(age-matched 1:1, caliper={caliper}y, median={median_val:.0f})",
                    output_dir / "1by1matched" / comparison / f"{grp_label}_violin.png")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort-mode", default=DEFAULT_COHORT_MODE,
                    choices=VALID_COHORT_CHOICES,
                    help=f"canonical cohort name (預設: {DEFAULT_COHORT_MODE})")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="覆寫輸出目錄；留空依 cohort-mode 自動決定")
    ap.add_argument("--comparison", default=None,
                    choices=HC_COMPARISONS + HILO_COMPARISONS,
                    help="Run only one comparison; default runs all")
    ap.add_argument("--caliper", type=float, default=1.0)
    args = ap.parse_args()

    output_dir = args.output_dir or (
        AGE_ANALYSIS_DIR / cohort_path(args.cohort_mode) / "violin")
    logger.info(f"cohort-mode = {args.cohort_mode}")
    logger.info(f"output-dir  = {output_dir}")

    df_all = load_age_error_table(args.cohort_mode)
    logger.info(f"loaded {len(df_all)} rows "
                f"({df_all['group'].value_counts().to_dict()})")

    targets = [args.comparison] if args.comparison else HC_COMPARISONS + HILO_COMPARISONS
    for cmp in targets:
        logger.info(f"--- {cmp} ---")
        if cmp in HC_COMPARISONS:
            run_hc_comparison(df_all, cmp, output_dir, caliper=args.caliper)
        elif cmp == "mmse_high_vs_low":
            run_hilo_comparison(df_all, "MMSE", output_dir, caliper=args.caliper)
        elif cmp == "casi_high_vs_low":
            run_hilo_comparison(df_all, "CASI", output_dir, caliper=args.caliper)


if __name__ == "__main__":
    main()
