"""
Age distribution before vs after 1:1 age-matching.

2×2 panels: (before/after) × (visits/subjects).
Stacked histograms for NAD, ACS, P with mean±std legend.

Usage:
    conda run -n Alz_face_main_analysis python scripts/stat/histogram.py
    conda run -n Alz_face_main_analysis python scripts/stat/histogram.py \
        --cohort-mode p_first_cdrall_hc_all_cdrall_or_mmseall
"""

import argparse
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
    AGE_ANALYSIS_DIR, cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
)
from src.common.cohort import base_id_of, cohort_list
from src.common.matching import match_by_age

COLORS = {"NAD": "#9ecae1", "ACS": "#6baed6", "P": "#fc9272"}
BINS = np.arange(35, 110, 2)


def _legend_label(group, ages):
    return f"{group} {ages.mean():.1f}±{ages.std():.1f}"


def _plot_panel(ax, cohort, unit, title):
    if unit == "subjects":
        df = cohort.drop_duplicates("base_id")
    else:
        df = cohort

    for grp in ["NAD", "ACS", "P"]:
        sub = df[df["group"] == grp]
        if sub.empty:
            continue
        ax.hist(sub["Age"], bins=BINS, alpha=0.55, color=COLORS[grp],
                label=_legend_label(grp, sub["Age"]), edgecolor="white",
                linewidth=0.3)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("subjects" if unit == "subjects" else "visits")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
    ax.grid(True, alpha=0.2)


def run(cohort, priority_groups=None):
    full = cohort_list(*cohort)
    full["group"] = full["Group"]
    full["base_id"] = full["ID"].map(base_id_of)
    p_ids, hc_ids = match_by_age(*cohort, priority=priority_groups, level="subject")
    matched = full[full["ID"].isin(set(p_ids) | set(hc_ids))].copy()
    pv_ids, hv_ids = match_by_age(*cohort, priority=priority_groups, level="visit")
    matched_visit = full[full["ID"].isin(set(pv_ids) | set(hv_ids))].copy()

    n_pairs = len(p_ids)
    priority_tag = ""
    if priority_groups:
        priority_tag = f"_{'_'.join(g.lower() for g in priority_groups)}_first"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    _plot_panel(axes[0, 0], full, "visits",
                "before matching  (visits-level)")
    _plot_panel(axes[0, 1], matched_visit, "visits",
                "after matching  (visits-level)")
    _plot_panel(axes[1, 0], full, "subjects",
                "before matching  (subjects-level)")
    _plot_panel(axes[1, 1], matched, "subjects",
                "after matching  (subjects-level)")

    axes[1, 0].set_xlabel("Age (years)")
    axes[1, 1].set_xlabel("Age (years)")

    fig.suptitle(
        f"Age distribution before vs after 1:1 age-matching\n"
        f"cohort = {cohort}  ·  {n_pairs} matched pairs",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = AGE_ANALYSIS_DIR / cohort_path(*cohort) / "histogram"
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"age_hist_before_after_matching{priority_tag}.png"
    fig.savefig(str(out_dir / fname), dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_rows = []
    for label, df in [("before_visit", full),
                       ("before_subject", full.drop_duplicates("base_id")),
                       ("after_visit", matched_visit),
                       ("after_subject", matched)]:
        for grp in ["NAD", "ACS", "P"]:
            sub = df[df["group"] == grp]
            if sub.empty:
                continue
            csv_rows.append({
                "stage": label, "group": grp, "n": len(sub),
                "age_mean": round(sub["Age"].mean(), 1),
                "age_std": round(sub["Age"].std(), 1),
            })
    csv_path = out_dir / f"age_hist_before_after_matching{priority_tag}_data.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"Saved: {out_dir / fname}")
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    # 預設掃 p_visit ∈ {p_all, p_first}(其餘 token 固定),對應舊的兩個 cohort_mode 預設。
    parser.add_argument("--p-visit", nargs="*", choices=list(P_VISIT_TOKENS),
                        default=["p_all", "p_first"])
    parser.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default="p_cdrall")
    parser.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default="hc_all")
    parser.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                        default="hc_cdrall_or_mmseall")
    parser.add_argument("--match-priority", nargs="*", default=None)
    args = parser.parse_args()

    for pv in args.p_visit:
        cohort = (pv, args.p_score, args.hc_visit, args.hc_score)
        print(f"\n{'='*60}")
        print(f"Cohort: {cohort}")
        print(f"{'='*60}")
        run(cohort, priority_groups=args.match_priority)


if __name__ == "__main__":
    main()
