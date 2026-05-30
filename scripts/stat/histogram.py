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

from src.config import cohort_name, cohort_spec_from_name, AGE_ANALYSIS_DIR
from src.common.cohort import cohort_list
from src.common.legacy.matching import match_cohort_ad_vs_hc

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


def run(cohort_mode, priority_groups=None):
    spec = cohort_spec_from_name(cohort_name(cohort_mode))

    full = cohort_list(
        f"p_{spec.p_visit}", f"p_{spec.p_cdr}", f"hc_{spec.hc_visit}",
        "hc_cdr0_or_mmse26" if spec.hc_strict else "hc_cdrall_or_mmseall")
    full["group"] = full["Group"]
    full["base_id"] = full["Group"] + full["Number"].astype(str)  # ID 已是完整鍵
    matched, pairs = match_cohort_ad_vs_hc(
        full, priority_groups=priority_groups, match_level="subject")
    matched_visit, _ = match_cohort_ad_vs_hc(
        full, priority_groups=priority_groups, match_level="visit")

    n_pairs = len(pairs) if pairs is not None else 0
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
        f"cohort = {cohort_mode}  ·  {n_pairs} matched pairs",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = (AGE_ANALYSIS_DIR
               / spec.visit_dir / spec.cdr_mmse_dir / "histogram")
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
    parser.add_argument("--cohort-mode", nargs="*",
                        default=["p_all_cdrall_hc_all_cdrall_or_mmseall",
                                 "p_first_cdrall_hc_all_cdrall_or_mmseall"])
    parser.add_argument("--match-priority", nargs="*", default=None)
    args = parser.parse_args()

    for cm in args.cohort_mode:
        print(f"\n{'='*60}")
        print(f"Cohort: {cm}")
        print(f"{'='*60}")
        run(cm, priority_groups=args.match_priority)


if __name__ == "__main__":
    main()
