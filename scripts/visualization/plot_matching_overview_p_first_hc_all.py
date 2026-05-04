"""
Cohort matching overview for the p_first_hc_all cohort.

Two-row layout:
  Row 1 — PRE-match: AD pool vs full HC pool (HC, NAD, ACS).
  Row 2 — POST-match: subject-first 1:1 age matching (caliper = 2 yr).

Writes:
  workspace/arms_analysis/p_first_hc_all/cohort_matching_overview.png
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARMS_P_FIRST_HC_ALL_DIR

OUT_PNG = ARMS_P_FIRST_HC_ALL_DIR / "cohort_matching_overview.png"
EMBEDDING_FEAT_DIR = (PROJECT_ROOT / "workspace" / "embedding" / "features"
                      / "arcface" / "original")

COLOR_AD = "#C44E52"
COLOR_HC = "#4C72B0"
CALIPER = 2.0


def _has_npy(visit_id):
    return (EMBEDDING_FEAT_DIR / f"{visit_id}.npy").exists()


def filter_to_npy_available(coh):
    """Drop rows whose visit-level ID has no .npy embedding file. For AD
    (label=1) we keep at least one visit per subject if any visit has .npy
    — first-visit + .npy fallback already handled upstream, so most rows
    survive. For HC (label=0, p_first_hc_all keeps every visit) we drop
    each visit independently."""
    keep = coh["ID"].apply(_has_npy)
    return coh[keep].copy().reset_index(drop=True)


def _load_grid():
    spec = importlib.util.spec_from_file_location(
        "run_4arm_deep_dive",
        PROJECT_ROOT / "scripts" / "experiments" / "run_4arm_deep_dive.py",
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.COHORT_MODE = "p_first_hc_all"
    return m


def _build_arm_b_with_mode(grid, hc_source, match_mode):
    """Replicate build_cohort_ad_vs_HCgroup arm B but force a specific
    match_mode (so we can compare visit vs subject_first side-by-side)."""
    import os
    # Build the AD + HC pools the same way the grid does
    frames = []
    groups_to_load = ["P", "NAD"]
    if grid.HC_SOURCE_MODE != "EACS":
        groups_to_load.append("ACS")
    for grp in groups_to_load:
        df = pd.read_csv(grid.DEMOGRAPHICS_DIR / f"{grp}.csv")
        if "ID" not in df.columns:
            for col in df.columns:
                if col in ("ACS", "NAD"):
                    df = df.rename(columns={col: "ID"})
                    break
        df["group"] = grp
        df["Source"] = "internal"
        frames.append(df)
    demo = pd.concat(frames, ignore_index=True)
    if "Source" not in demo.columns:
        demo["Source"] = "internal"
    demo["Source"] = demo["Source"].fillna("internal")
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["Global_CDR"] = pd.to_numeric(demo.get("Global_CDR"), errors="coerce")
    demo["MMSE"] = pd.to_numeric(demo.get("MMSE"), errors="coerce")
    demo["base_id"] = demo["ID"].str.extract(r"^(.+)-\d+$")
    demo["visit"] = demo["ID"].str.extract(r"-(\d+)$").astype(float)

    ad_all = demo[(demo["group"] == "P") & (demo["Global_CDR"] >= 0.5) &
                    demo["Age"].notna()].copy()
    ad_all = ad_all.sort_values(["base_id", "visit"])
    ad = grid._pick_first_visit_with_features(ad_all)
    ad["label"] = 1

    hc_all = grid._strict_hc_filter_all_visits(demo, hc_source)
    hc_all = hc_all[hc_all["Age"].notna()].copy()
    hc = hc_all.reset_index(drop=True).copy()  # p_first_hc_all keeps all visits

    prep = pd.concat([ad, hc], ignore_index=True)
    prep["mmse_group"] = np.where(prep["label"] == 1, "high", "low")
    prep["MMSE"] = prep["MMSE"].fillna(999)
    matched, pairs, _ = grid.match_1to1(prep, caliper=CALIPER, seed=42,
                                          match_mode=match_mode)
    cohort = matched.merge(prep[["ID", "base_id", "group", "Age", "MMSE",
                                   "Global_CDR", "label"]].drop_duplicates("ID"),
                            on="ID", how="left", suffixes=("", "_p"))
    cohort = cohort.drop(columns=[c for c in cohort.columns if c.endswith("_p")])
    return cohort, pairs


def overlay(ax, ad_ages, hc_ages, hc_label, bins, xlim, title, n_subj_ad=None,
            n_subj_hc=None):
    ad_lbl = f"AD (n={len(ad_ages)})"
    if n_subj_ad is not None and n_subj_ad != len(ad_ages):
        ad_lbl = f"AD (n={len(ad_ages)} visits / {n_subj_ad} subj)"
    hc_lbl = f"{hc_label} (n={len(hc_ages)})"
    if n_subj_hc is not None and n_subj_hc != len(hc_ages):
        hc_lbl = f"{hc_label} (n={len(hc_ages)} visits / {n_subj_hc} subj)"
    ax.hist(ad_ages, bins=bins, range=xlim, alpha=0.55, color=COLOR_AD,
            label=ad_lbl, edgecolor="white", linewidth=0.5)
    ax.hist(hc_ages, bins=bins, range=xlim, alpha=0.55, color=COLOR_HC,
            label=hc_lbl, edgecolor="white", linewidth=0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    grid = _load_grid()

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True)
    bins = 30
    all_ages = []

    pre_data = {}
    post_subj = {}
    for cmp in ("HC", "NAD", "ACS"):
        coh_a, _ = grid.build_cohort_ad_vs_HCgroup(cmp, arm="A")
        coh_a = filter_to_npy_available(coh_a)
        ad_pre = coh_a.loc[coh_a["label"] == 1, "Age"].dropna().values
        hc_pre = coh_a.loc[coh_a["label"] == 0, "Age"].dropna().values
        n_subj_ad_pre = coh_a.loc[coh_a["label"] == 1, "base_id"].nunique()
        n_subj_hc_pre = coh_a.loc[coh_a["label"] == 0, "base_id"].nunique()
        pre_data[cmp] = (ad_pre, hc_pre, n_subj_ad_pre, n_subj_hc_pre)
        all_ages.extend(ad_pre); all_ages.extend(hc_pre)

        coh_s, _ = _build_arm_b_with_mode(grid, cmp, "subject_first")
        coh_s = filter_to_npy_available(coh_s)
        ad_s = coh_s.loc[coh_s["label"] == 1, "Age"].dropna().values
        hc_s = coh_s.loc[coh_s["label"] == 0, "Age"].dropna().values
        n_ad_s = coh_s.loc[coh_s["label"] == 1, "base_id"].nunique()
        n_hc_s = coh_s.loc[coh_s["label"] == 0, "base_id"].nunique()
        post_subj[cmp] = (ad_s, hc_s, n_ad_s, n_hc_s)

    xlim = (np.floor(min(all_ages) / 5) * 5, np.ceil(max(all_ages) / 5) * 5)

    for i, cmp in enumerate(("HC", "NAD", "ACS")):
        ad_pre, hc_pre, nad_pre_s, nhc_pre_s = pre_data[cmp]
        overlay(axes[0, i], ad_pre, hc_pre, cmp, bins, xlim,
                f"PRE-match · AD vs {cmp}",
                n_subj_ad=nad_pre_s, n_subj_hc=nhc_pre_s)

        ad_s, hc_s, n_ad_s, n_hc_s = post_subj[cmp]
        overlay(axes[1, i], ad_s, hc_s, cmp, bins, xlim,
                f"POST-match · AD vs {cmp}",
                n_subj_ad=n_ad_s, n_subj_hc=n_hc_s)

    # Row labels on left
    for i, label in enumerate(("PRE", "POST")):
        axes[i, 0].text(-0.18, 0.5, label, transform=axes[i, 0].transAxes,
                         fontsize=12, fontweight="bold", rotation=90,
                         va="center", ha="center")

    fig.suptitle(
        "Cohort matching overview — p_first_hc_all\n"
        "(P first-visit + CDR≥0.5 + .npy fallback;  HC = ALL NAD/ACS visits, no strict HC)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'cmp':4s}  {'AD pairs':>8s}  {'HC visits':>9s}  {'HC unique subj':>14s}  {'visits/subj':>11s}")
    for cmp in ("HC", "NAD", "ACS"):
        ad_s, hc_s, _, n_hc_s = post_subj[cmp]
        print(f"{cmp:4s}  {len(ad_s):8d}  {len(hc_s):9d}  {n_hc_s:14d}  "
              f"{len(hc_s)/n_hc_s:11.2f}")


if __name__ == "__main__":
    main()
