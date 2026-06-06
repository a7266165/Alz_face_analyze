"""Age-prediction error violin plots for ACS / NAD / P
— full cohort + AD-vs-HC 1:1 age-matched subset.

Outputs under <AGE_ANALYSIS_DIR>/<cohort>/violin/{full,1by1matched}/<comparison>/:
  AD vs HC / NAD / ACS     — slices of ONE ACS-first AD-vs-HC match (see run_hc_comparison)
  MMSE / CASI high vs low  — score-median match (match_by_score)
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
    cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.common.cohort import cohort_list
from src.age.utils import build_cohort_with_age_error
from src.common.matching import (match_by_age, match_by_score,
                                 split_by_group, split_by_score)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HC_COMPARISONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
HILO_COMPARISONS = ["mmse_high_vs_low", "casi_high_vs_low"]

# ── 小提琴繪圖 ───────────────────────────────────────────────────────────

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
    ax.set_ylim(-35, 35)  # 固定軸範圍，供跨 cohort/版本比較


def save_violin(left_vals, right_vals, left_label, right_label, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_violin(ax, left_vals, right_vals, left_label, right_label, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out_path}")

# ── 比較執行器 ───────────────────────────────────────────────────────

def run_hc_comparison(df_all, tokens, comparison, output_dir, caliper=1.0):
    """AD vs {HC, NAD, ACS} violin — full cohort + 1:1 age-matched.

    The matched arm is a *slice* of ONE ACS-first global AD-vs-HC match
    (``priority=["ACS"]``): ACS controls match first, the rest against NAD. The
    NAD / ACS comparisons are then the NAD-paired / ACS-paired slices of that
    single match — so they stay consistent with ad_vs_hc and scatter/lines/stat
    (HC = NAD ∪ ACS) instead of being an independent re-match.
    """
    hc_subgroups, label = {
        "ad_vs_hc":  ({"NAD", "ACS"}, "HC"),
        "ad_vs_nad": ({"NAD"},        "NAD"),
        "ad_vs_acs": ({"ACS"},        "ACS"),
    }[comparison]

    # full（未配對）：把 AD(P) 與此比較的對照組切兩臂
    df_split = split_by_group(df_all, "P", control=list(hc_subgroups))
    ad_err = df_split[df_split["arm"] == "high"]["age_error"].dropna().values
    hc_err = df_split[df_split["arm"] == "low"]["age_error"].dropna().values
    if len(hc_err) > 0 and len(ad_err) > 0:
        save_violin(hc_err, ad_err, label, "AD",
                    f"Age prediction error: {label} vs AD (full)",
                    output_dir / "full" / comparison / "violin.png")

    # 1:1 age-matched：依 HC 子群把單一 ACS 優先全域配對切片
    group_of = cohort_list(*tokens).set_index("ID")["Group"]
    p_ids, hc_ids = match_by_age(*tokens, priority=["ACS"], caliper=caliper)
    keep_hc = [h for h in hc_ids if group_of.get(h) in hc_subgroups]
    keep_p = [p for p, h in zip(p_ids, hc_ids) if group_of.get(h) in hc_subgroups]
    msub = df_all[df_all["ID"].isin(set(keep_p) | set(keep_hc))]
    m_hc = msub[msub["group"].isin(hc_subgroups)]["age_error"].dropna().values
    m_ad = msub[msub["group"] == "P"]["age_error"].dropna().values
    if len(m_hc) > 0 and len(m_ad) > 0:
        save_violin(m_hc, m_ad, label, "AD",
                    f"Age prediction error: {label} vs AD\n"
                    f"(age-matched 1:1, caliper={caliper}y)",
                    output_dir / "1by1matched" / comparison / "violin.png")


def run_hilo_comparison(df_all, tokens, metric, output_dir, caliper=1.0):
    """MMSE/CASI high vs low within AD (and NAD) — full + 1:1 age-matched.

    1:1 matched 子集委派給 canonical match_by_score（在該組內以中位數切 high/low
    再做年齡配對）；high/low 兩臂直接由回傳的兩個 ID list 決定。
    """
    comparison = f"{metric.lower()}_high_vs_low"

    for grp, grp_label in [("P", "ad"), ("NAD", "nad")]:
        df_grp = df_all[df_all["group"] == grp].dropna(subset=["age_error"])
        df_split = split_by_score(df_grp, metric)  # 第二切：組內按分數中位數標 high/low
        if df_split.empty:
            continue
        median_val = pd.to_numeric(df_split[metric], errors="coerce").median()
        df_high = df_split[df_split["arm"] == "high"]
        df_low = df_split[df_split["arm"] == "low"]

        # full（未配對）
        if not df_high.empty and not df_low.empty:
            save_violin(
                df_high["age_error"].values, df_low["age_error"].values,
                f"High {metric}", f"Low {metric}",
                f"Age error by {metric} ({grp_label.upper()}, full)\nmedian={median_val:.0f}",
                output_dir / "full" / comparison / f"{grp_label}_violin.png")

        # 1:1 age-matched（canonical match_by_score；high/low 由回傳兩 list 直接決定）
        high_ids, low_ids = match_by_score(
            *tokens, grp, metric, "median", caliper=caliper)
        hi = df_all[df_all["ID"].isin(high_ids)]["age_error"].dropna().values
        lo = df_all[df_all["ID"].isin(low_ids)]["age_error"].dropna().values
        if len(hi) > 0 and len(lo) > 0:
            save_violin(
                hi, lo, f"High {metric}", f"Low {metric}",
                f"Age error by {metric} ({grp_label.upper()})\n"
                f"(age-matched 1:1, caliper={caliper}y)",
                output_dir / "1by1matched" / comparison / f"{grp_label}_violin.png")

# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="覆寫輸出目錄；留空依 cohort 自動決定")
    ap.add_argument("--comparison", default=None,
                    choices=HC_COMPARISONS + HILO_COMPARISONS,
                    help="只跑單一比較；預設全跑")
    ap.add_argument("--caliper", type=float, default=1.0)
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    output_dir = args.output_dir or (
        AGE_ANALYSIS_DIR / cohort_path(*cohort) / "violin")
    logger.info(f"cohort = {cohort}")
    logger.info(f"output-dir  = {output_dir}")

    df_all = build_cohort_with_age_error(*cohort)
    logger.info(f"loaded {len(df_all)} rows "
                f"({df_all['group'].value_counts().to_dict()})")

    targets = [args.comparison] if args.comparison else HC_COMPARISONS + HILO_COMPARISONS
    for cmp in targets:
        logger.info(f"--- {cmp} ---")
        if cmp in HC_COMPARISONS:
            run_hc_comparison(df_all, cohort, cmp, output_dir, caliper=args.caliper)
        elif cmp == "mmse_high_vs_low":
            run_hilo_comparison(df_all, cohort, "MMSE", output_dir, caliper=args.caliper)
        elif cmp == "casi_high_vs_low":
            run_hilo_comparison(df_all, cohort, "CASI", output_dir, caliper=args.caliper)


if __name__ == "__main__":
    main()
