"""
Forward / Reverse evaluation for asymmetry scoring methods.

Follows the same evaluation framework as run_fwd_rev.py but uses three
score-based methods instead of trained classifiers:

    l2_norm:        √(Σᵢ fᵢ²)
    centroid_dist:  cos_dist(x, μ_HC) − cos_dist(x, μ_AD)   (CV)
    lda_projection: Fisher discriminant → 1D                  (CV)

Forward: score full cohort via CV → evaluate on full + matched subset
Reverse: score matched cohort via CV → evaluate matched OOF + predict unmatched

Output mirrors run_fwd_rev.py directory structure:
    <classification>/<cohort>/<bg>/<model>/<feat>/mean/no_drop/
      <method>/fwd/1by1matched/<ml>/<eu>/<match_strategy>/<partition>/metrics.json
      <method>/rev/1by1matched/<ml>/<eu>/<match_strategy>/<partition>/metrics.json

Usage:
    conda run -n Alz_face_main_analysis python \\
        scripts/embedding/run_asymmetry_fwd_rev.py \\
        --cohort-mode p_first_cdrall_hc_all_cdrall_or_mmseall \\
        --bg-mode no_background --match-priority ACS \\
        --embedding arcface --feature-type difference
"""
import argparse
import json
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
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_CLASSIFICATION_DIR,
    EMBEDDING_FEATURES_DIR,
    VALID_COHORT_CHOICES,
    cohort_spec_from_name,
)
from src.common.cohort import cohort_list
from src.common.legacy.matching import match_cohort_ad_vs_hc
from scripts.utilities.stats_helpers import bootstrap_auc_ci

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

METHODS = ["l2_norm", "centroid_dist", "lda_projection"]
MODELS = ["arcface", "dlib", "topofr", "vggface"]
FEATURE_TYPES = [
    "difference", "absolute_difference",
    "relative_differences", "absolute_relative_differences",
]
PARTITIONS_HC = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
KEEP_GROUPS = {
    "ad_vs_hc": None,
    "ad_vs_nad": {"P", "NAD"},
    "ad_vs_acs": {"P", "ACS"},
}

DISTANCE_METRIC = {
    "arcface": "cosine", "dlib": "euclidean",
    "topofr": "euclidean", "vggface": "euclidean",
}


# ============================================================
# Feature loading
# ============================================================

def _load_vectors_for_cohort(cohort, model, feature_type, bg_mode):
    """Load mean-pooled vectors for all IDs in cohort.
    Returns (X, ids, base_ids, y) aligned arrays."""
    feat_dir = EMBEDDING_FEATURES_DIR / model / bg_mode / feature_type
    rows = []
    for _, r in cohort.iterrows():
        sid = str(r["ID"])
        npy = feat_dir / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy, allow_pickle=True)
        if a.dtype == object:
            a = list(a.item().values())[0]
        vec = a.mean(axis=0) if a.ndim == 2 else a
        bid = r["base_id"] if "base_id" in r.index else sid.rsplit("-", 1)[0]
        rows.append({"ID": sid, "base_id": bid,
                     "label": int(r["label"]), "vec": vec})
    if not rows:
        return None, None, None, None
    X = np.array([r["vec"] for r in rows])
    ids = np.array([r["ID"] for r in rows])
    base_ids = np.array([r["base_id"] for r in rows])
    y = np.array([r["label"] for r in rows])
    return X, ids, base_ids, y


# ============================================================
# Scoring functions (per-fold)
# ============================================================

def _score_l2(X_train, y_train, X_test):
    return np.linalg.norm(X_test, axis=1)


def _score_centroid(X_train, y_train, X_test):
    c_ad = X_train[y_train == 1].mean(axis=0)
    c_hc = X_train[y_train == 0].mean(axis=0)
    scores = np.array([cosine(x, c_hc) - cosine(x, c_ad) for x in X_test])
    return scores


def _score_lda(X_train, y_train, X_test):
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_train, y_train)
    return lda.decision_function(X_test)


SCORE_FUNC = {
    "l2_norm": _score_l2,
    "centroid_dist": _score_centroid,
    "lda_projection": _score_lda,
}


# ============================================================
# Metrics (same as run_fwd_rev.py)
# ============================================================

def compute_metrics(y_true, y_score, threshold=0.5, seed=42):
    y_pred = (y_score >= threshold).astype(int)
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    else:
        tn = fp = fn = tp = 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = float(roc_auc_score(y_true, y_score)) \
        if len(np.unique(y_true)) > 1 else float("nan")
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_score, n=1000, seed=seed) \
        if len(np.unique(y_true)) > 1 else (float("nan"), float("nan"))
    return {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "auc": auc,
        "auc_ci_low": float(ci_low) if not np.isnan(ci_low) else None,
        "auc_ci_high": float(ci_high) if not np.isnan(ci_high) else None,
        "balacc": float(balanced_accuracy_score(y_true, y_pred))
            if len(np.unique(y_true)) > 1 else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
            if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sens": float(sens) if not np.isnan(sens) else None,
        "spec": float(spec) if not np.isnan(spec) else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _aggregate_to_subject(ids, base_ids, y, scores):
    """Mean-pool scores per base_id."""
    df = pd.DataFrame({"base_id": base_ids, "y_true": y, "y_score": scores})
    agg = df.groupby("base_id").agg(
        y_true=("y_true", "first"), y_score=("y_score", "mean")).reset_index()
    return agg["y_true"].to_numpy(int), agg["y_score"].to_numpy(float)


def _filter_by_groups(ids, base_ids, y, scores, cohort, keep_groups):
    """Filter to subjects in keep_groups, using pair_id to keep matched pairs."""
    if keep_groups is None:
        return y, scores
    target_groups = set(keep_groups) - {"P"}
    if "pair_id" in cohort.columns:
        target_pairs = cohort[
            cohort["group"].isin(target_groups)]["pair_id"].unique()
        keep_bids = set(cohort[
            cohort["pair_id"].isin(target_pairs)]["base_id"].astype(str))
    else:
        keep_bids = set(cohort[
            cohort["group"].isin(keep_groups)]["base_id"].astype(str))
    mask = np.array([b in keep_bids for b in base_ids])
    return y[mask], scores[mask]


# ============================================================
# Forward strategy
# ============================================================

def run_forward(X, y, base_ids, ids, method, n_folds=10, seed=42):
    """CV on full cohort → OOF scores."""
    score_fn = SCORE_FUNC[method]
    n_groups = len(np.unique(base_ids))
    k = min(n_folds, n_groups)
    gkf = GroupKFold(n_splits=k)

    oof = np.full(len(y), np.nan)

    if method == "l2_norm":
        oof = score_fn(None, None, X)
    else:
        for tri, tei in gkf.split(X, y, groups=base_ids):
            oof[tei] = score_fn(X[tri], y[tri], X[tei])

    m_full = compute_metrics(y, oof, seed=seed)
    return oof, m_full


def forward_eval(oof, ids, base_ids, y, matched, keep_groups, seed=42):
    """Evaluate OOF scores on matched subset (subject + visit level)."""
    matched_bids = set(matched["base_id"].astype(str))
    mask = np.array([b in matched_bids for b in base_ids])

    if keep_groups is not None:
        target_groups = set(keep_groups) - {"P"}
        target_pairs = matched[
            matched["group"].isin(target_groups)]["pair_id"].unique()
        pair_bids = set(matched[
            matched["pair_id"].isin(target_pairs)]["base_id"].astype(str))
        mask = mask & np.array([b in pair_bids for b in base_ids])

    y_m, sc_m = y[mask], oof[mask]
    bids_m = base_ids[mask]

    # Visit-level
    m_visit = compute_metrics(y_m, sc_m, seed=seed) if len(y_m) >= 10 else None
    # Subject-level
    y_subj, sc_subj = _aggregate_to_subject(ids[mask], bids_m, y_m, sc_m)
    m_subj = compute_metrics(y_subj, sc_subj, seed=seed) \
        if len(y_subj) >= 10 else None

    return m_subj, m_visit, (y_subj, sc_subj), (y_m, sc_m)


# ============================================================
# ROC plotting
# ============================================================

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


def plot_roc(roc_data, title, out_path):
    """Plot ROC curves for multiple partitions.
    roc_data: dict[partition] → (y_true, y_score) or None"""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    for part in ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]:
        if part not in roc_data or roc_data[part] is None:
            continue
        yt, ys = roc_data[part]
        if len(yt) < 10 or len(np.unique(yt)) < 2:
            continue
        auc = roc_auc_score(yt, ys)
        fpr, tpr, _ = roc_curve(yt, ys)
        n_pos, n_neg = int((yt == 1).sum()), int((yt == 0).sum())
        label = (f"{PARTITION_LABELS[part]}  AUC={auc:.3f}"
                 f"  (n={n_pos}+{n_neg})")
        ax.plot(fpr, tpr, color=PARTITION_COLORS[part],
                linewidth=2, label=label)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


METHOD_TITLES = {
    "l2_norm": "L2 Norm",
    "centroid_dist": "Centroid Distance",
    "lda_projection": "LDA Projection",
}


def plot_roc_combined(all_roc, title, out_path):
    """3-panel ROC: one subplot per method, 3 partition lines each.
    all_roc: dict[method] → dict[partition] → (y_true, y_score)"""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, method in zip(axes, METHODS):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        roc_data = all_roc.get(method, {})
        for part in ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]:
            if part not in roc_data or roc_data[part] is None:
                continue
            yt, ys = roc_data[part]
            if len(yt) < 10 or len(np.unique(yt)) < 2:
                continue
            auc = roc_auc_score(yt, ys)
            fpr, tpr, _ = roc_curve(yt, ys)
            n_pos, n_neg = int((yt == 1).sum()), int((yt == 0).sum())
            label = (f"{PARTITION_LABELS[part]}  AUC={auc:.3f}"
                     f"  (n={n_pos}+{n_neg})")
            ax.plot(fpr, tpr, color=PARTITION_COLORS[part],
                    linewidth=2, label=label)
        ax.set_xlabel("FPR", fontsize=12)
        ax.set_ylabel("TPR", fontsize=12)
        ax.set_title(METHOD_TITLES[method], fontsize=15, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Reverse strategy
# ============================================================

def run_reverse(X_matched, y_matched, bids_matched,
                X_full, y_full, bids_full, ids_full,
                method, n_folds=10, seed=42):
    """Train on matched → OOF for matched + predict unmatched."""
    score_fn = SCORE_FUNC[method]
    n_groups = len(np.unique(bids_matched))
    k = min(n_folds, n_groups)
    gkf = GroupKFold(n_splits=k)

    oof_matched = np.full(len(y_matched), np.nan)

    if method == "l2_norm":
        oof_matched = score_fn(None, None, X_matched)
        scores_full = score_fn(None, None, X_full)
    else:
        scores_full_accum = np.zeros(len(y_full))
        n_folds_accum = 0
        for tri, tei in gkf.split(X_matched, y_matched, groups=bids_matched):
            oof_matched[tei] = score_fn(
                X_matched[tri], y_matched[tri], X_matched[tei])
            scores_full_accum += score_fn(
                X_matched[tri], y_matched[tri], X_full)
            n_folds_accum += 1
        scores_full = scores_full_accum / n_folds_accum

    # Split full scores into matched vs unmatched
    matched_bid_set = set(bids_matched)
    unmatched_mask = np.array([b not in matched_bid_set for b in bids_full])

    m_oof_subj_y, m_oof_subj_sc = _aggregate_to_subject(
        None, bids_matched, y_matched, oof_matched)
    m_matched_oof = compute_metrics(m_oof_subj_y, m_oof_subj_sc, seed=seed)
    m_matched_oof_visit = compute_metrics(y_matched, oof_matched, seed=seed)

    if unmatched_mask.sum() >= 10:
        y_un = y_full[unmatched_mask]
        sc_un = scores_full[unmatched_mask]
        m_unmatched = compute_metrics(y_un, sc_un, seed=seed)
    else:
        m_unmatched = None

    return {
        "metrics_matched_oof": m_matched_oof,
        "metrics_matched_oof_visit": m_matched_oof_visit,
        "metrics_unmatched": m_unmatched,
        "oof_matched": oof_matched,
        "scores_full": scores_full,
    }


# ============================================================
# Output helpers
# ============================================================

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _match_strategy_dir(priority_groups):
    if not priority_groups:
        return "no_priority"
    return f"priority_{priority_groups[0].lower()}"


def cell_dir(base, method, strategy, match_level="subject_match",
             eval_unit="eval_by_subject", match_strategy="no_priority",
             partition="ad_vs_hc"):
    return (base / method / strategy / "1by1matched"
            / match_level / eval_unit / match_strategy / partition)


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-mode", choices=VALID_COHORT_CHOICES,
                   default="p_first_cdrall_hc_all_cdrall_or_mmseall")
    p.add_argument("--bg-mode", choices=["no_background", "background"],
                   default="no_background")
    p.add_argument("--match-priority", nargs="*", default=None)
    p.add_argument("--embedding", default=None,
                   help="Limit to one model (default: all)")
    p.add_argument("--feature-type", default=None,
                   help="Limit to one feature type (default: all)")
    args = p.parse_args()

    spec = cohort_spec_from_name(args.cohort_mode)
    cohort_dir = spec.visit_dir + "/" + spec.cdr_mmse_dir
    match_dir = _match_strategy_dir(args.match_priority)

    models = [args.embedding] if args.embedding else MODELS
    feat_types = [args.feature_type] if args.feature_type else FEATURE_TYPES

    # Build cohorts once (shared across ad_vs_hc/nad/acs)
    full_cohort = cohort_list(
        f"p_{spec.p_visit}", f"p_{spec.p_cdr}", f"hc_{spec.hc_visit}",
        "hc_cdr0_or_mmse26" if spec.hc_strict else "hc_cdrall_or_mmseall")
    full_cohort["group"] = full_cohort["Group"]
    full_cohort["base_id"] = full_cohort["Group"] + full_cohort["ID"].astype(str)
    full_cohort["ID"] = full_cohort["base_id"] + "-" + full_cohort["Photo_Session"].astype(str)
    full_cohort["label"] = (full_cohort["group"] == "P").astype(int)

    matched_subj, _ = match_cohort_ad_vs_hc(
        full_cohort, match_level="subject",
        priority_groups=args.match_priority)
    if "base_id" not in matched_subj.columns:
        matched_subj["base_id"] = (matched_subj["ID"].astype(str)
                                   .str.extract(r"^(.+)-\d+$")[0])

    matched_visit, _ = match_cohort_ad_vs_hc(
        full_cohort, match_level="visit",
        priority_groups=args.match_priority)
    if "base_id" not in matched_visit.columns:
        matched_visit["base_id"] = (matched_visit["ID"].astype(str)
                                    .str.extract(r"^(.+)-\d+$")[0])

    # Expanded: all visits of subject-matched subjects
    subj_bids = set(matched_subj["base_id"].unique())
    expanded_subj = full_cohort[
        full_cohort["base_id"].isin(subj_bids)].copy()

    for model in models:
        for feat in feat_types:
            logger.info(f"=== {model} / {feat} ===")
            base = (EMBEDDING_CLASSIFICATION_DIR / cohort_dir
                    / args.bg_mode / model / feat / "mean" / "no_drop")

            # Load features for full cohort
            X_full, ids_full, bids_full, y_full = \
                _load_vectors_for_cohort(full_cohort, model, feat, args.bg_mode)
            if X_full is None:
                logger.warning(f"  no features for {model}/{feat}")
                continue

            ftl = feat.replace("absolute_relative_differences",
                "abs_rel_diff").replace("relative_differences",
                "rel_diff").replace("absolute_difference",
                "abs_diff").replace("difference", "diff")

            # Pre-compute OOF scores for all methods + save full scores
            l2_scores = _score_l2(None, None, X_full)
            method_oof = {"l2_norm": l2_scores}
            for method in ["centroid_dist", "lda_projection"]:
                oof, m_full = run_forward(
                    X_full, y_full, bids_full, ids_full, method)
                method_oof[method] = oof

            # Save full-cohort OOF scores per method
            for method, oof in method_oof.items():
                oof_df = pd.DataFrame({
                    "ID": ids_full, "base_id": bids_full,
                    "y_true": y_full, "y_score": oof,
                })
                if method == "l2_norm":
                    score_dir = base / method
                else:
                    score_dir = base / method / "fwd"
                score_dir.mkdir(parents=True, exist_ok=True)
                oof_df.to_csv(score_dir / "oof_scores_visit.csv",
                              index=False)
                subj_df = oof_df.groupby("base_id").agg(
                    y_true=("y_true", "first"),
                    y_score=("y_score", "mean")).reset_index()
                subj_df.to_csv(score_dir / "oof_scores_subject.csv",
                               index=False)

            for ml, matched in [("subject_match", matched_subj),
                                ("visit_match", matched_visit)]:
                # Collect ROC data across methods for combined plot
                combined_roc_subj = {}  # method → {part → (y, sc)}
                combined_roc_visit = {}

                for method in METHODS:
                    logger.info(f"  {method} / {ml}")
                    oof = method_oof[method]

                    roc_subj = {}
                    roc_visit = {}
                    for partition in PARTITIONS_HC:
                        keep = KEEP_GROUPS[partition]
                        m_subj, m_visit, raw_subj, raw_visit = \
                            forward_eval(oof, ids_full, bids_full,
                                         y_full, matched, keep, seed=42)
                        roc_subj[partition] = raw_subj
                        roc_visit[partition] = raw_visit

                        for eu, m, raw in [
                            ("eval_by_subject", m_subj, raw_subj),
                            ("eval_by_visit", m_visit, raw_visit),
                        ]:
                            if m is None:
                                continue
                            if method == "l2_norm":
                                out = (base / method / "1by1matched"
                                       / ml / eu / match_dir / partition)
                                write_json(out / "metrics.json", {
                                    "partition": partition,
                                    "embedding": model,
                                    "method": method,
                                    "metrics": m,
                                })
                            else:
                                out = cell_dir(base, method, "fwd",
                                               ml, eu, match_dir,
                                               partition)
                                write_json(out / "metrics.json", {
                                    "partition": partition,
                                    "embedding": model,
                                    "method": method,
                                    "strategy": "forward",
                                    "metrics_matched_subset": m,
                                })
                            yt, ys = raw
                            pd.DataFrame({
                                "y_true": yt, "y_score": ys,
                            }).to_csv(out / "scores.csv", index=False)

                    combined_roc_subj[method] = roc_subj
                    combined_roc_visit[method] = roc_visit

                    # Per-method ROC
                    for eu, roc in [("eval_by_subject", roc_subj),
                                    ("eval_by_visit", roc_visit)]:
                        if method == "l2_norm":
                            png = (base / method / "1by1matched"
                                   / ml / eu / match_dir / "roc_curve.png")
                        else:
                            png = cell_dir(base, method, "fwd",
                                           ml, eu, match_dir,
                                           "roc_curve.png")
                        plot_roc(roc,
                                 f"{METHOD_TITLES[method]} — "
                                 f"{model} / {ftl}\n{ml} / {eu}",
                                 png)

                # Combined 3-panel ROC → _summary/
                for eu, combined in [("eval_by_subject", combined_roc_subj),
                                     ("eval_by_visit", combined_roc_visit)]:
                    out_png = (base / "_summary" / "asymmetry_roc"
                               / ml / eu / match_dir
                               / "roc_combined.png")
                    plot_roc_combined(
                        combined,
                        f"{model} / {ftl} — {ml} / {eu}",
                        out_png)

            # === REVERSE (centroid / LDA only) ===
            for method in ["centroid_dist", "lda_projection"]:
                for ml, matched_coh in [("subject_match", matched_subj),
                                        ("visit_match", matched_visit)]:
                    X_m, ids_m, bids_m, y_m = \
                        _load_vectors_for_cohort(
                            matched_coh, model, feat, args.bg_mode)
                    if X_m is None:
                        continue
                    rev = run_reverse(
                        X_m, y_m, bids_m,
                        X_full, y_full, bids_full, ids_full,
                        method)
                    for partition in PARTITIONS_HC:
                        for eu, m_key in [
                            ("eval_by_subject", "metrics_matched_oof"),
                            ("eval_by_visit",
                             "metrics_matched_oof_visit"),
                        ]:
                            out = cell_dir(base, method, "rev",
                                           ml, eu, match_dir,
                                           partition)
                            write_json(out / "metrics.json", {
                                "partition": partition,
                                "embedding": model,
                                "method": method,
                                "strategy": "reverse",
                                "metrics_matched_oof": rev[m_key],
                                "metrics_unmatched":
                                    rev["metrics_unmatched"],
                            })

            logger.info(f"  done")

    logger.info("All done.")


if __name__ == "__main__":
    main()
