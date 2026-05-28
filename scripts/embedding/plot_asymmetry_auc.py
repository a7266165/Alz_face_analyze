"""
Asymmetry scoring AUC: three methods for converting embedding asymmetry
vectors into discriminative scores, without classifier training.

Methods:
    1. L2 norm:          √(Σᵢ fᵢ²)
    2. Centroid distance: cos_dist(x, μ_HC) − cos_dist(x, μ_AD)
    3. LDA projection:   Fisher's discriminant → 1D (10-fold CV)

Evaluates across ad_vs_hc / ad_vs_nad / ad_vs_acs under 4 combinations:
    subject_match × eval_by_subject / eval_by_visit
    visit_match   × eval_by_subject / eval_by_visit

Output:
    <classification_root>/<cohort>/<bg_mode>/<emb>/<feat>/mean/no_drop/
      _summary/asymmetry_auc/
        <match_level>/<eval_unit>/<match_strategy>/
          roc_l2_norm.png
          roc_centroid_dist.png
          roc_lda_projection.png
    <classification_root>/<cohort>/<bg_mode>/_summary/asymmetry_auc/
      asymmetry_auc_all.csv

Usage:
    conda run -n Alz_face_main_analysis python \\
        scripts/embedding/plot_asymmetry_auc.py \\
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
from scipy.spatial.distance import cosine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_CLASSIFICATION_DIR,
    EMBEDDING_FEATURES_DIR,
    VALID_COHORT_CHOICES,
    cohort_path,
    cohort_spec_from_name,
)
from src.cohort import build_cohort_ad_vs_HCgroup

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

MATCH_LEVELS = ["subject_match", "visit_match"]
EVAL_UNITS = ["eval_by_subject", "eval_by_visit"]
MATCH_LEVEL_ARG = {"subject_match": "subject", "visit_match": "visit"}

SCORING_METHODS = ["l2_norm", "centroid_dist", "lda_projection"]

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
METHOD_TITLES = {
    "l2_norm": "L2 Norm",
    "centroid_dist": "Centroid Distance",
    "lda_projection": "LDA Projection",
}


# ============================================================
# Feature loading
# ============================================================

def _load_npy_mean(feat_dir, sid):
    npy = feat_dir / f"{sid}.npy"
    if not npy.exists():
        return None
    a = np.load(npy, allow_pickle=True)
    if a.dtype == object:
        a = list(a.item().values())[0]
    return a.mean(axis=0) if a.ndim == 2 else a


def _load_vectors(ids, model, feature_type, bg_mode):
    """Load full vectors + L2 norms for all IDs. Returns dict[ID → vector]."""
    feat_dir = EMBEDDING_FEATURES_DIR / model / bg_mode / feature_type
    results = {}
    for sid in ids:
        vec = _load_npy_mean(feat_dir, sid)
        if vec is not None:
            results[sid] = vec
    return results


# ============================================================
# Scoring methods
# ============================================================

def _score_l2_norm(vectors, y_dict):
    """L2 norm of each vector."""
    return {sid: float(np.linalg.norm(vec))
            for sid, vec in vectors.items()}


def _score_centroid_dist(vectors, y_dict, groups_dict=None,
                         n_folds=10, seed=42):
    """cos_dist(x, centroid_HC) - cos_dist(x, centroid_AD) with CV.
    Centroids computed on training fold only to avoid leakage."""
    sids = [s for s in vectors if s in y_dict]
    if len(sids) < 20:
        return {}
    X = np.array([vectors[s] for s in sids])
    y = np.array([y_dict[s] for s in sids])
    if len(np.unique(y)) < 2:
        return {}

    if groups_dict is not None:
        groups = np.array([groups_dict.get(s, s) for s in sids])
        n_groups = len(np.unique(groups))
        k = min(n_folds, n_groups)
        if k < 2:
            return {}
        cv = GroupKFold(n_splits=k)
        splits = list(cv.split(X, y, groups))
    else:
        k = min(n_folds, min(np.bincount(y)))
        if k < 2:
            return {}
        cv = StratifiedKFold(n_splits=k, shuffle=True,
                             random_state=seed)
        splits = list(cv.split(X, y))

    scores = np.full(len(sids), np.nan)
    for train_idx, test_idx in splits:
        c_ad = X[train_idx][y[train_idx] == 1].mean(axis=0)
        c_hc = X[train_idx][y[train_idx] == 0].mean(axis=0)
        for i in test_idx:
            scores[i] = cosine(X[i], c_hc) - cosine(X[i], c_ad)

    return {sid: float(sc) for sid, sc in zip(sids, scores)
            if not np.isnan(sc)}


def _score_lda_cv(vectors, y_dict, groups_dict=None, n_folds=10, seed=42):
    """LDA projection with cross-validation. Returns dict[ID → score]."""
    sids = [s for s in vectors if s in y_dict]
    if len(sids) < 20:
        return {}
    X = np.array([vectors[s] for s in sids])
    y = np.array([y_dict[s] for s in sids])
    if len(np.unique(y)) < 2:
        return {}

    lda = LinearDiscriminantAnalysis(n_components=1)
    if groups_dict is not None:
        groups = np.array([groups_dict.get(s, s) for s in sids])
        n_groups = len(np.unique(groups))
        k = min(n_folds, n_groups)
        if k < 2:
            return {}
        cv = GroupKFold(n_splits=k)
        scores = cross_val_predict(lda, X, y, cv=cv, groups=groups,
                                   method="decision_function")
    else:
        k = min(n_folds, min(np.bincount(y)))
        if k < 2:
            return {}
        cv = StratifiedKFold(n_splits=k, shuffle=True,
                             random_state=seed)
        scores = cross_val_predict(lda, X, y, cv=cv,
                                   method="decision_function")
    return {sid: float(sc) for sid, sc in zip(sids, scores)}


SCORE_FUNCS = {
    "l2_norm": _score_l2_norm,
    "centroid_dist": _score_centroid_dist,
    "lda_projection": _score_lda_cv,
}


# ============================================================
# Aggregation helpers
# ============================================================

def _add_base_id(cohort):
    if "base_id" not in cohort.columns:
        cohort = cohort.copy()
        cohort["base_id"] = (cohort["ID"].astype(str)
                             .str.extract(r"^(.+)-\d+$")[0])
    return cohort


def _aggregate_scores_to_subject(scores_dict):
    """Mean-pool scores per base_id."""
    df = pd.DataFrame([
        {"ID": sid, "score": val} for sid, val in scores_dict.items()
    ])
    if df.empty:
        return {}
    df["base_id"] = df["ID"].astype(str).str.extract(r"^(.+)-\d+$")[0]
    df["base_id"] = df["base_id"].fillna(df["ID"])
    agg = df.groupby("base_id")["score"].mean()
    return agg.to_dict()


def _aggregate_cohort_to_subject(cohort):
    cohort = _add_base_id(cohort)
    agg = {"label": "first"}
    for col in ("group",):
        if col in cohort.columns:
            agg[col] = "first"
    out = cohort.groupby("base_id").agg(agg).reset_index()
    out = out.rename(columns={"base_id": "ID"})
    return out


# ============================================================
# AUC + ROC
# ============================================================

def _compute_auc_ci(y_true, y_score, n_bootstrap=2000, seed=42):
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
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _eval_one_partition(scores_dict, cohort, eval_unit, keep_groups=None):
    """Compute AUC + ROC for one partition."""
    if keep_groups is not None:
        cohort = cohort[cohort["group"].isin(keep_groups)].copy()

    if eval_unit == "eval_by_subject":
        agg_scores = _aggregate_scores_to_subject(scores_dict)
        coh = _aggregate_cohort_to_subject(cohort)
        score_col = {sid: agg_scores[sid] for sid in agg_scores
                     if sid in coh.set_index("ID").index}
    else:
        coh = cohort[["ID", "label"]].copy()
        score_col = scores_dict

    rows = []
    for _, row in coh.iterrows():
        sid = row["ID"]
        if sid in score_col:
            rows.append({"label": row["label"], "score": score_col[sid]})
    if len(rows) < 10:
        return None

    df = pd.DataFrame(rows).dropna(subset=["score"])
    y_true = df["label"].to_numpy(dtype=int)
    y_score = df["score"].to_numpy(dtype=float)

    auc, ci_lo, ci_hi = _compute_auc_ci(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return {
        "auc": auc, "auc_ci_lo": ci_lo, "auc_ci_hi": ci_hi,
        "n": len(df), "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "fpr": fpr, "tpr": tpr,
    }


# ============================================================
# Plotting
# ============================================================

def _plot_roc(results_by_part, title, out_path):
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


# ============================================================
# Main
# ============================================================

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

    cohort_naive, _ = build_cohort_ad_vs_HCgroup(
        "HC", design="cross_naive",
        cohort_mode=args.cohort_mode,
        hc_source_mode="ACS",
    )
    cohort_naive = _add_base_id(cohort_naive)

    for model in MODELS:
        for feat in FEATURE_TYPES:
            ftl = FEATURE_TYPE_LABELS[feat]
            logger.info(f"=== {model} / {feat} ===")

            for ml in MATCH_LEVELS:
                ml_arg = MATCH_LEVEL_ARG[ml]

                cohort_matched, _ = build_cohort_ad_vs_HCgroup(
                    "HC", design="cross_matched",
                    cohort_mode=args.cohort_mode,
                    hc_source_mode="ACS",
                    match_level=ml_arg,
                    priority_groups=args.match_priority,
                )
                cohort_matched = _add_base_id(cohort_matched)

                matched_bids = set(cohort_matched["base_id"].unique())
                cohort_expanded = cohort_naive[
                    cohort_naive["base_id"].isin(matched_bids)
                ].copy()

                all_ids = list(set(
                    cohort_matched["ID"].astype(str).tolist()
                    + cohort_expanded["ID"].astype(str).tolist()
                ))
                vectors = _load_vectors(all_ids, model, feat, args.bg_mode)

                for eu in EVAL_UNITS:
                    if ml == "subject_match":
                        cell_cohort = cohort_expanded
                    else:
                        cell_cohort = cohort_matched

                    # Build y_dict and groups_dict for scoring
                    y_dict = dict(zip(
                        cell_cohort["ID"].astype(str),
                        cell_cohort["label"].astype(int)))
                    groups_dict = dict(zip(
                        cell_cohort["ID"].astype(str),
                        cell_cohort["base_id"].astype(str)))

                    for method in SCORING_METHODS:
                        func = SCORE_FUNCS[method]
                        if method == "l2_norm":
                            scores = func(vectors, y_dict)
                        else:
                            scores = func(vectors, y_dict,
                                          groups_dict=groups_dict)

                        results_by_part = {}
                        for part in PARTITIONS:
                            keep = PARTITION_KEEP_GROUPS[part]
                            r = _eval_one_partition(
                                scores, cell_cohort, eu,
                                keep_groups=keep)
                            results_by_part[part] = r
                            if r is not None:
                                all_metrics.append({
                                    "method": method,
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
                                    / "_summary" / "asymmetry_auc"
                                    / ml / eu / match_dir)
                        out_base.mkdir(parents=True, exist_ok=True)
                        title = (f"{METHOD_TITLES[method]} — "
                                 f"{model} / {ftl}\n"
                                 f"{ml} / {eu}")
                        _plot_roc(results_by_part, title,
                                  out_base / f"roc_{method}.png")

                    logger.info(f"  {ml}/{eu}: wrote 3 ROC plots")

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        summary_root = (EMBEDDING_CLASSIFICATION_DIR
                        / cohort_dir / args.bg_mode
                        / "_summary" / "asymmetry_auc")
        summary_root.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_root / "asymmetry_auc_all.csv", index=False)
        logger.info(f"Summary: {summary_root / 'asymmetry_auc_all.csv'} "
                    f"({len(df)} rows)")


if __name__ == "__main__":
    main()
