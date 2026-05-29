"""
L2-norm AUC analysis for embedding asymmetry features.

Computes asymmetry = sqrt(sum(f_i^2)) per subject, then uses that scalar
directly as a discriminative score (no classifier training). Plots ROC
curves and reports AUC with 95% CI for each (model, feature_type, partition)
under each (match_level, eval_unit) combination.

Output:
    <classification_root>/<cohort>/<bg_mode>/<emb>/<feat>/mean/no_drop/
      _summary/l2_auc/
        <match_level>/<eval_unit>/<match_strategy>/
          roc_curves.png
          l2_auc_metrics.csv

Usage:
    conda run -n Alz_face_main_analysis python \\
        scripts/embedding/plot_l2_auc.py \\
        --cohort-mode p_first_cdrall_hc_all_cdrall_or_mmseall \\
        --bg-mode no_background --match-priority ACS
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
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_CLASSIFICATION_DIR,
    EMBEDDING_FEATURES_DIR,
    VALID_COHORT_CHOICES,
    cohort_path,
    cohort_spec_from_name,
)
from src.common.cohort import cohort_list
from src.common.legacy.matching import match_cohort_ad_vs_hc

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["arcface", "dlib", "topofr"]
FEATURE_TYPES = [
    "difference", "absolute_difference",
    "relative_differences", "absolute_relative_differences",
]
FEATURE_TYPE_LABELS = {
    "difference": "diff",
    "absolute_difference": "abs_diff",
    "relative_differences": "rel_diff",
    "absolute_relative_differences": "abs_rel_diff",
}

PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
PARTITION_KEEP_GROUPS = {
    "ad_vs_hc": None,
    "ad_vs_nad": {"P", "NAD"},
    "ad_vs_acs": {"P", "ACS"},
}
PARTITION_COMPARE = {
    "ad_vs_hc": "HC", "ad_vs_nad": "NAD", "ad_vs_acs": "ACS",
}

MATCH_LEVELS = ["subject_match", "visit_match"]
EVAL_UNITS = ["eval_by_subject", "eval_by_visit"]
MATCH_LEVEL_ARG = {"subject_match": "subject", "visit_match": "visit"}

PARTITION_COLORS = {
    "ad_vs_hc": "#4C72B0",
    "ad_vs_nad": "#DD8452",
    "ad_vs_acs": "#55A868",
}
PARTITION_LABELS = {
    "ad_vs_hc": "HC (NAD+ACS)",
    "ad_vs_nad": "NAD only",
    "ad_vs_acs": "ACS only",
}


def _load_l2_norms(ids, model, feature_type, bg_mode):
    feat_dir = EMBEDDING_FEATURES_DIR / model / bg_mode / feature_type
    results = {}
    for sid in ids:
        npy = feat_dir / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy, allow_pickle=True)
        if a.dtype == object:
            a = list(a.item().values())[0]
        vec = a.mean(axis=0) if a.ndim == 2 else a
        results[sid] = float(np.linalg.norm(vec))
    return results


def _aggregate_to_subject(scores, cohort):
    """Mean-pool L2 norms per base_id."""
    df = pd.DataFrame([
        {"ID": sid, "l2": val} for sid, val in scores.items()
    ])
    if df.empty:
        return df
    df["base_id"] = df["ID"].astype(str).str.extract(r"^(.+)-\d+$")[0]
    df["base_id"] = df["base_id"].fillna(df["ID"])
    agg = df.groupby("base_id")["l2"].mean().reset_index()
    agg = agg.rename(columns={"base_id": "ID"})
    return agg


def _compute_auc_ci(y_true, y_score, n_bootstrap=2000, seed=42):
    """AUC with bootstrap 95% CI."""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aucs:
        return auc, np.nan, np.nan
    lo = float(np.percentile(aucs, 2.5))
    hi = float(np.percentile(aucs, 97.5))
    return auc, lo, hi


def run_one_cell(cohort, scores_dict, eval_unit, keep_groups=None):
    """Compute AUC + ROC for one (partition, match_level, eval_unit) cell."""
    if keep_groups is not None:
        cohort = cohort[cohort["group"].isin(keep_groups)].copy()

    if eval_unit == "eval_by_subject":
        score_df = _aggregate_to_subject(scores_dict, cohort)
        if "base_id" not in cohort.columns:
            cohort = cohort.copy()
            cohort["base_id"] = (cohort["ID"].astype(str)
                                 .str.extract(r"^(.+)-\d+$")[0])
        coh = cohort.groupby("base_id").agg(
            {"label": "first"}).reset_index().rename(
            columns={"base_id": "ID"})
    else:
        score_df = pd.DataFrame([
            {"ID": sid, "l2": val} for sid, val in scores_dict.items()
        ])
        coh = cohort[["ID", "label"]].copy()

    if score_df.empty:
        return None

    merged = coh.merge(score_df, on="ID", how="inner").dropna(subset=["l2"])
    if len(merged) < 10:
        return None

    y_true = merged["label"].to_numpy(dtype=int)
    y_score = merged["l2"].to_numpy(dtype=float)

    auc, ci_lo, ci_hi = _compute_auc_ci(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    return {
        "auc": auc, "auc_ci_lo": ci_lo, "auc_ci_hi": ci_hi,
        "n": len(merged), "n_pos": n_pos, "n_neg": n_neg,
        "fpr": fpr, "tpr": tpr,
    }


def plot_roc_multi_partition(results_by_part, title, out_path):
    """Plot ROC curves for multiple partitions on one figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)

    for part in PARTITIONS:
        if part not in results_by_part or results_by_part[part] is None:
            continue
        r = results_by_part[part]
        color = PARTITION_COLORS[part]
        label = (f"{PARTITION_LABELS[part]}  "
                 f"AUC={r['auc']:.3f} [{r['auc_ci_lo']:.3f}-{r['auc_ci_hi']:.3f}]"
                 f"  (n={r['n_pos']}+{r['n_neg']})")
        ax.plot(r["fpr"], r["tpr"], color=color, linewidth=2, label=label)

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _match_strategy_dir(priority_groups):
    if not priority_groups:
        return "no_priority"
    return f"priority_{'_'.join(g.lower() for g in priority_groups)}"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-mode",
                   choices=VALID_COHORT_CHOICES,
                   default="p_first_cdrall_hc_all_cdrall_or_mmseall")
    p.add_argument("--bg-mode",
                   choices=["no_background", "background"],
                   default="no_background")
    p.add_argument("--match-priority", nargs="*", default=None)
    args = p.parse_args()

    spec = cohort_spec_from_name(args.cohort_mode)
    cohort_dir = spec.visit_dir + "/" + spec.cdr_mmse_dir
    match_dir = _match_strategy_dir(args.match_priority)

    all_metrics = []

    # Pre-build full (unmatched) cohort for expanding visits
    cohort_naive = cohort_list(
        f"p_{spec.p_visit}", f"p_{spec.p_cdr}", f"hc_{spec.hc_visit}",
        "hc_cdr0_or_mmse26" if spec.hc_strict else "hc_cdrall_or_mmseall")
    cohort_naive["group"] = cohort_naive["Group"]
    cohort_naive["base_id"] = cohort_naive["Group"] + cohort_naive["ID"].astype(str)
    cohort_naive["ID"] = cohort_naive["base_id"] + "-" + cohort_naive["Photo_Session"].astype(str)
    cohort_naive["label"] = (cohort_naive["group"] == "P").astype(int)

    for model in MODELS:
        for feat in FEATURE_TYPES:
            ftl = FEATURE_TYPE_LABELS[feat]
            logger.info(f"=== {model} / {feat} ===")

            for ml in MATCH_LEVELS:
                ml_arg = MATCH_LEVEL_ARG[ml]

                cohort_matched, _ = match_cohort_ad_vs_hc(
                    cohort_naive,
                    match_level=ml_arg,
                    priority_groups=args.match_priority,
                )
                if "base_id" not in cohort_matched.columns:
                    cohort_matched["base_id"] = (
                        cohort_matched["ID"].astype(str)
                        .str.extract(r"^(.+)-\d+$")[0])

                # All visits of matched subjects (for eval_by_visit)
                matched_bids = set(cohort_matched["base_id"].unique())
                cohort_expanded = cohort_naive[
                    cohort_naive["base_id"].isin(matched_bids)
                ].copy()

                # Load L2 norms for all visit IDs
                all_ids = list(set(
                    cohort_matched["ID"].astype(str).tolist()
                    + cohort_expanded["ID"].astype(str).tolist()
                ))
                scores = _load_l2_norms(all_ids, model, feat, args.bg_mode)

                for eu in EVAL_UNITS:
                    if ml == "subject_match":
                        cell_cohort = cohort_expanded
                    else:
                        cell_cohort = cohort_matched
                    results_by_part = {}
                    for part in PARTITIONS:
                        keep = PARTITION_KEEP_GROUPS[part]
                        r = run_one_cell(cell_cohort, scores, eu,
                                         keep_groups=keep)
                        results_by_part[part] = r
                        if r is not None:
                            all_metrics.append({
                                "model": model,
                                "feature_type": feat,
                                "match_level": ml,
                                "eval_unit": eu,
                                "partition": part,
                                "auc": r["auc"],
                                "auc_ci_lo": r["auc_ci_lo"],
                                "auc_ci_hi": r["auc_ci_hi"],
                                "n": r["n"],
                                "n_pos": r["n_pos"],
                                "n_neg": r["n_neg"],
                            })

                    out_base = (EMBEDDING_CLASSIFICATION_DIR
                                / cohort_dir / args.bg_mode
                                / model / feat / "mean" / "no_drop"
                                / "_summary" / "l2_auc"
                                / ml / eu / match_dir)
                    out_base.mkdir(parents=True, exist_ok=True)
                    title = (f"L2-norm ROC — {model} / {ftl}\n"
                             f"{ml} / {eu}")
                    plot_roc_multi_partition(
                        results_by_part, title,
                        out_base / "roc_curves.png")
                    logger.info(f"  {ml}/{eu}: wrote {out_base}")

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        summary_root = (EMBEDDING_CLASSIFICATION_DIR
                        / cohort_dir / args.bg_mode
                        / "_summary" / "l2_auc")
        summary_root.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_root / "l2_auc_all.csv", index=False)
        logger.info(f"Summary: {summary_root / 'l2_auc_all.csv'} "
                    f"({len(df)} rows)")


if __name__ == "__main__":
    main()
