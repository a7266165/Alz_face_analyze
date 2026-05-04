"""
Arm B AD vs {HC, NAD, ACS} 1:1 age-matched analysis.

Mirror of run_mmse_hilo_standalone.py but for the 3 non-hi-lo comparisons.
Reuses cohort builder from run_4arm_deep_dive (build_cohort_ad_vs_HCgroup) and
the violin plotter pattern from mmse_hilo_standalone.

Outputs per comparison (HC / NAD / ACS) into:
  workspace/arms_analysis/per_arm/arm_b/ad_vs_<comparison>/
    ├── matched_features.csv
    ├── matched_pairs.csv
    ├── matching_report.txt
    ├── summary_stats.csv
    └── age/fig_age_error_violin.png

Usage:
    conda run -n Alz_face_main_analysis python \
        scripts/experiments/run_arm_b_ad_vs_hcgroups.py
"""
import argparse
import importlib.util
import json
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


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_grid = _load_module("run_4arm_deep_dive",
                      PROJECT_ROOT / "scripts" / "experiments" /
                      "run_4arm_deep_dive.py")
build_cohort_ad_vs_HCgroup = _grid.build_cohort_ad_vs_HCgroup

DEFAULT_ARM_B_DIR = (PROJECT_ROOT / "workspace" / "arms_analysis" /
                     "p_first_hc_strict" / "per_arm" / "arm_b")
AGES_FILE = (PROJECT_ROOT / "workspace" / "age" / "age_prediction" /
             "predicted_ages.json")
CALIPER = 2.0  # max age diff for matching (years)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) +
                      (len(b) - 1) * b.var(ddof=1)) /
                     (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled


def attach_age_error(cohort, pred_ages):
    """Add age_error column = Age - predicted_age."""
    cohort = cohort.copy()
    cohort["predicted_age"] = cohort["ID"].map(pred_ages)
    cohort["age_error"] = cohort["Age"] - cohort["predicted_age"]
    return cohort


def plot_violin(cohort, comparison_name, out_path):
    """Violin: comparison group (left, blue) vs AD (right, red)."""
    hc = cohort[cohort["label"] == 0]["age_error"].dropna()
    ad = cohort[cohort["label"] == 1]["age_error"].dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot([hc, ad], showmedians=True)
    for pc, color in zip(parts["bodies"], ["#4C72B0", "#C44E52"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{comparison_name}\n(n={len(hc)})",
                        f"AD\n(n={len(ad)})"])
    ax.set_ylabel("Age error (real − predicted)")
    ax.set_title(f"Age prediction error: {comparison_name} vs AD\n"
                 f"(age-matched 1:1, caliper={CALIPER}y)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_group_stats(cohort, col):
    """Welch t-test stats: AD (1) vs comparison (0). Returns dict."""
    a = cohort[cohort["label"] == 1][col].dropna().astype(float)
    b = cohort[cohort["label"] == 0][col].dropna().astype(float)
    out = {
        "ad_n": len(a), "ad_mean": float(a.mean()) if len(a) else np.nan,
        "ad_std": float(a.std(ddof=1)) if len(a) > 1 else np.nan,
        "hc_n": len(b), "hc_mean": float(b.mean()) if len(b) else np.nan,
        "hc_std": float(b.std(ddof=1)) if len(b) > 1 else np.nan,
    }
    if len(a) > 1 and len(b) > 1:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        out["welch_t"] = float(t)
        out["welch_p"] = float(p)
    else:
        out["welch_t"] = np.nan
        out["welch_p"] = np.nan
    out["cohen_d"] = cohens_d(a, b)
    return out


def fmt_stats(label, s, fmt=".2f"):
    """Format one stat row for the matching report."""
    return (f"{label}: AD {s['ad_mean']:{fmt}} ± {s['ad_std']:{fmt}} vs "
            f"{s['hc_mean']:{fmt}} ± {s['hc_std']:{fmt}}  "
            f"(Welch t={s['welch_t']:.2f}, p={s['welch_p']:.3g}, "
            f"d={s['cohen_d']:.3f})")


def run_one(comparison, arm_b_dir):
    logger.info(f"=== AD vs {comparison} ===")
    cohort, pairs = build_cohort_ad_vs_HCgroup(comparison, arm="B",
                                                 caliper=CALIPER)
    n_pairs_raw = len(pairs) if pairs is not None else len(cohort) // 2

    with open(AGES_FILE) as f:
        pred_ages = json.load(f)

    # Drop pairs where either side lacks predicted_age — keeps n balanced
    # (mirror of mmse_hilo_standalone's pre-filter at L508-511, applied
    # post-hoc here since matching is done by the grid module).
    if pairs is not None:
        keep = pairs["minor_id"].isin(pred_ages) & pairs["major_id"].isin(pred_ages)
        pairs = pairs[keep].copy()
        kept_ids = set(pairs["minor_id"]).union(pairs["major_id"])
        cohort = cohort[cohort["ID"].isin(kept_ids)].copy()
    n_pairs = len(pairs) if pairs is not None else len(cohort) // 2
    logger.info(f"Matched pairs: {n_pairs}/{n_pairs_raw} after predicted_age filter "
                f"(caliper={CALIPER}y)")

    cohort = attach_age_error(cohort, pred_ages)

    out_dir = arm_b_dir / f"ad_vs_{comparison.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    age_dir = out_dir / "age"
    age_dir.mkdir(parents=True, exist_ok=True)

    cohort.to_csv(out_dir / "matched_features.csv", index=False)
    if pairs is not None:
        pairs.to_csv(out_dir / "matched_pairs.csv", index=False)

    age_s = per_group_stats(cohort, "Age")
    mmse_s = per_group_stats(cohort, "MMSE")
    err_s = per_group_stats(cohort, "age_error")

    summary = pd.DataFrame([
        {"variable": "Age", **age_s},
        {"variable": "MMSE", **mmse_s},
        {"variable": "age_error", **err_s},
    ])
    summary.to_csv(out_dir / "summary_stats.csv", index=False)

    with open(out_dir / "matching_report.txt", "w", encoding="utf-8") as f:
        f.write(f"AD vs {comparison} 1:1 age-matched (caliper={CALIPER}y)\n")
        f.write(f"Matched pairs: {n_pairs}\n\n")
        f.write(fmt_stats("Age   ", age_s) + "\n")
        f.write(fmt_stats("MMSE  ", mmse_s) + "\n")
        f.write(fmt_stats("AgeErr", err_s) + "\n")

    plot_violin(cohort, comparison, age_dir / "fig_age_error_violin.png")
    logger.info(f"Saved {out_dir}/")
    logger.info("  " + fmt_stats("Age   ", age_s))
    logger.info("  " + fmt_stats("MMSE  ", mmse_s))
    logger.info("  " + fmt_stats("AgeErr", err_s))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-mode", choices=["default", "p_first_hc_all"],
                         default="default",
                         help="default=原 strict HC + first-visit；"
                              "p_first_hc_all=first-visit P + ALL NAD/ACS")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARM_B_DIR,
                         help="arm_b 輸出根目錄 (default: workspace/"
                              "arms_analysis/per_arm/arm_b)")
    args = parser.parse_args()

    # 切 grid module 的 COHORT_MODE，build_cohort_ad_vs_HCgroup 會吃這個 global
    _grid.COHORT_MODE = args.cohort_mode
    logger.info(f"cohort_mode={args.cohort_mode}  arm_b_dir={args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cmp in ("HC", "NAD", "ACS"):
        run_one(cmp, args.output_dir)


if __name__ == "__main__":
    main()
