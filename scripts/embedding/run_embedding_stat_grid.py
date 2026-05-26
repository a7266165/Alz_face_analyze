"""
Embedding asymmetry stat grid: 4 feature types × 3 models × 2 tests = 24 rows.

Tests group-level differences (AD vs HC/NAD/ACS, MMSE/CASI hi-lo)
using cross_matched design with 4 evaluation combinations:
    subject_match × eval_by_subject
    subject_match × eval_by_visit
    visit_match   × eval_by_subject
    visit_match   × eval_by_visit

Feature types:
    difference, absolute_difference, relative_differences,
    absolute_relative_differences

Models:
    arcface (cosine), dlib (euclidean), topofr (euclidean)

Tests per model:
    L2 scalar   → Welch t-test + Mann-Whitney U + AUC (logistic)
    full vector → PERMANOVA + AUC (XGB)

Output:
    workspace/embedding/analysis/feature_stat/stat_grid/
      <visit>/<cdr_mmse>/<bg_mode>/<hc_source>/
        <match_level>/<eval_unit>/
          stat_grid_long.csv / wide / png / md
          difference/ absolute_difference/ ...

Usage:
    conda run -n Alz_face_main_analysis python \\
        scripts/embedding/run_embedding_stat_grid.py \\
        --cohort-mode p_first_cdr05_hc_first_cdrall_or_mmseall \\
        --hc-source ACS \\
        --bg-mode no_background
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_FEATURES_DIR,
    EMBEDDING_STAT_GRID_DIR,
    VALID_COHORT_CHOICES,
    cohort_path,
)
from scripts.utilities.cohort import (
    build_cohort_ad_hi_lo, build_cohort_ad_vs_HCgroup,
)
from scripts.utilities.stats_helpers import bh_fdr
from scripts.overview.run_stat_grid import (
    MIN_CELL_N,
    COMPARISONS,
    _compute_cell_header_stats,
    _hilo_matched_csv,
    run_scalar_modality,
    run_permanova_modality,
)
from scripts.overview.plot_deep_dive_grid import render_grid

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["arcface", "dlib", "topofr"]
FEATURE_TYPES = [
    "difference", "absolute_difference",
    "relative_differences", "absolute_relative_differences",
]
DISTANCE = {"arcface": "cosine", "dlib": "euclidean", "topofr": "euclidean"}

FEATURE_TYPE_LABELS = {
    "difference": "diff",
    "absolute_difference": "abs_diff",
    "relative_differences": "rel_diff",
    "absolute_relative_differences": "abs_rel_diff",
}

MATCH_LEVELS = ["subject_match", "visit_match"]
EVAL_UNITS = ["eval_by_subject", "eval_by_visit"]
MATCH_LEVEL_ARG = {"subject_match": "subject", "visit_match": "visit"}


# ============================================================
# Feature loaders (bg_mode-aware, multi feature-type)
# ============================================================

def _load_npy_mean(feat_dir, sid):
    npy = feat_dir / f"{sid}.npy"
    if not npy.exists():
        return None
    a = np.load(npy, allow_pickle=True)
    if a.dtype == object:
        a = list(a.item().values())[0]
    return a.mean(axis=0) if a.ndim == 2 else a


def load_l2_scalar(ids, model, feature_type, bg_mode):
    feat_dir = EMBEDDING_FEATURES_DIR / model / bg_mode / feature_type
    col = f"{FEATURE_TYPE_LABELS[feature_type]}_{model}_l2"
    rows = []
    for sid in ids:
        vec = _load_npy_mean(feat_dir, sid)
        rows.append({
            "subject_id": sid,
            col: float(np.linalg.norm(vec)) if vec is not None else np.nan,
        })
    return pd.DataFrame(rows)


def load_full_vector(ids, model, feature_type, bg_mode):
    feat_dir = EMBEDDING_FEATURES_DIR / model / bg_mode / feature_type
    rows = []
    for sid in ids:
        vec = _load_npy_mean(feat_dir, sid)
        if vec is None:
            continue
        rows.append({
            "subject_id": sid,
            **{f"dim_{i}": float(vec[i]) for i in range(vec.shape[0])},
        })
    return pd.DataFrame(rows) if rows else None


# ============================================================
# Subject-level aggregation
# ============================================================

def _aggregate_features_to_subject(X_df):
    """Mean-pool feature columns per base_id (subject)."""
    if X_df is None or X_df.empty:
        return X_df
    if "subject_id" not in X_df.columns:
        return X_df
    X_df = X_df.copy()
    X_df["_base"] = X_df["subject_id"].astype(str).str.extract(
        r"^(.+)-\d+$")[0]
    X_df["_base"] = X_df["_base"].fillna(X_df["subject_id"])
    feat_cols = [c for c in X_df.columns
                 if c not in ("subject_id", "_base")]
    agg = X_df.groupby("_base")[feat_cols].mean().reset_index()
    agg = agg.rename(columns={"_base": "subject_id"})
    return agg


def _aggregate_cohort_to_subject(cohort):
    """Collapse cohort to one row per base_id, keeping label + Age."""
    if "base_id" not in cohort.columns:
        cohort = cohort.copy()
        cohort["base_id"] = (cohort["ID"].astype(str)
                             .str.extract(r"^(.+)-\d+$")[0])
    agg = {"label": "first"}
    for col in ("Age", "MMSE", "Global_CDR", "CASI", "group"):
        if col in cohort.columns:
            agg[col] = "first"
    out = cohort.groupby("base_id").agg(agg).reset_index()
    out = out.rename(columns={"base_id": "ID"})
    return out


# ============================================================
# Modality spec generation
# ============================================================

def _build_modality_specs():
    specs = []
    for ft in FEATURE_TYPES:
        ftl = FEATURE_TYPE_LABELS[ft]
        for model in MODELS:
            parent = f"{ftl}_{model}"
            specs.append((parent, "L2 scalar", "scalar", model, ft))
            specs.append((parent, "full vector", "permanova", model, ft))
    return specs


MODALITY_SPECS = _build_modality_specs()
ROW_ORDER = [(s[0], s[1]) for s in MODALITY_SPECS]

DIRECTION_MAP = {}
for ft in FEATURE_TYPES:
    ftl = FEATURE_TYPE_LABELS[ft]
    for model in MODELS:
        DIRECTION_MAP[f"{ftl}_{model}"] = ft


# ============================================================
# Test dispatch
# ============================================================

def _load_features(test_kind, model, feature_type, bg_mode, ids):
    if test_kind == "scalar":
        return load_l2_scalar(ids, model, feature_type, bg_mode)
    return load_full_vector(ids, model, feature_type, bg_mode)


def _dispatch_test(test_kind, model, name, X_df, cohort):
    if X_df is None or (hasattr(X_df, "empty") and X_df.empty):
        return {"modality": name, "n": 0, "test": test_kind,
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "skip_reason": "feature unavailable"}
    if test_kind == "scalar":
        return run_scalar_modality(name, X_df, cohort)
    metric = DISTANCE[model]
    return run_permanova_modality(name, X_df, cohort, metric=metric)


# ============================================================
# Orchestrator
# ============================================================

def _get_ids(cohort):
    id_col = "ID" if "ID" in cohort.columns else "base_id"
    return cohort[id_col].astype(str).tolist()


def run_one_combo(output_dir, cohort_mode, hc_source_mode,
                  match_level, eval_unit, bg_mode, priority_groups=None):
    """Run stat grid for one (match_level, eval_unit) combination."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ml_arg = MATCH_LEVEL_ARG[match_level]
    logger.info(f"--- {match_level} / {eval_unit} ---")

    cohorts = {}
    feasibility_rows = []
    header_stat_rows = []

    for compare in COMPARISONS:
        try:
            if compare == "mmse-hi-lo":
                cohort, _ = build_cohort_ad_hi_lo(
                    design="cross_matched", cohort_mode=cohort_mode,
                    metric="MMSE",
                    matched_features_csv=_hilo_matched_csv(
                        cohort_mode, "mmse"),
                )
            elif compare == "casi-hi-lo":
                cohort, _ = build_cohort_ad_hi_lo(
                    design="cross_matched", cohort_mode=cohort_mode,
                    metric="CASI",
                    matched_features_csv=_hilo_matched_csv(
                        cohort_mode, "casi"),
                )
            else:
                cohort, _ = build_cohort_ad_vs_HCgroup(
                    compare, design="cross_matched",
                    cohort_mode=cohort_mode,
                    hc_source_mode=hc_source_mode,
                    match_level=ml_arg,
                    priority_groups=priority_groups,
                )
            n = len(cohort) if cohort is not None else 0
            n_pos = int((cohort["label"] == 1).sum()) if n > 0 else 0
            n_neg = int((cohort["label"] == 0).sum()) if n > 0 else 0
            status = "ok"
            note = ""
            if min(n_pos, n_neg) < MIN_CELL_N:
                status = (f"n/a (min-cell-n {min(n_pos, n_neg)} "
                          f"< {MIN_CELL_N})")
                note = "pair count below threshold"
        except Exception as e:
            cohort = None
            n, n_pos, n_neg = 0, 0, 0
            status = f"error: {e}"
            note = str(e)
        feasibility_rows.append({
            "match_level": match_level, "eval_unit": eval_unit,
            "comparison": compare,
            "n_total": n, "n_pos": n_pos, "n_neg": n_neg,
            "status": status, "note": note,
        })
        header_stat_rows.append(
            _compute_cell_header_stats("cross_matched", compare, cohort))
        cohorts[compare] = cohort

    feas_df = pd.DataFrame(feasibility_rows)
    feas_df.to_csv(output_dir / "feasibility_report.csv", index=False)
    hdr_df = pd.DataFrame(header_stat_rows)
    hdr_df.to_csv(output_dir / "cell_header_stats.csv", index=False)
    ok_count = (feas_df["status"] == "ok").sum()
    logger.info(f"Active cells: {ok_count}/{len(COMPARISONS)}")

    long_rows = []
    for compare in COMPARISONS:
        cohort = cohorts[compare]
        fstat = feas_df[feas_df.comparison == compare].iloc[0]
        if fstat["status"] != "ok":
            for (parent, sub, *_) in MODALITY_SPECS:
                long_rows.append({
                    "comparison": compare,
                    "modality_parent": parent, "modality_sub": sub,
                    "test": "n/a", "n": 0,
                    "p": np.nan, "q": np.nan,
                    "statistic": np.nan, "effect": np.nan,
                    "skip_reason": fstat["status"],
                })
            continue

        n_pos = fstat["n_pos"]
        n_neg = fstat["n_neg"]
        logger.info(f"=== {compare} : n_pos={n_pos} n_neg={n_neg} ===")
        ids = _get_ids(cohort)

        test_cohort = cohort
        if eval_unit == "eval_by_subject":
            test_cohort = _aggregate_cohort_to_subject(cohort)

        cell_results = []
        for (parent, sub, test_kind, model, ft) in MODALITY_SPECS:
            name = f"{parent}::{sub}"
            try:
                X_df = _load_features(test_kind, model, ft, bg_mode, ids)
                if eval_unit == "eval_by_subject":
                    X_df = _aggregate_features_to_subject(X_df)
                res = _dispatch_test(
                    test_kind, model, name, X_df, test_cohort)
            except Exception as e:
                logger.warning(f"{compare} {name}: {e}")
                res = {"modality": name, "n": 0, "p": np.nan,
                       "statistic": np.nan, "effect": np.nan,
                       "error": str(e)}
            res["comparison"] = compare
            res["modality_parent"] = parent
            res["modality_sub"] = sub
            cell_results.append(res)
            logger.info(
                f"  {name}: p={res.get('p')} eff={res.get('effect')}")

        pvals = np.array([r.get("p", np.nan) for r in cell_results],
                         dtype=float)
        valid = ~np.isnan(pvals)
        qvals = np.full(len(pvals), np.nan)
        if valid.sum() > 0:
            qvals[valid] = bh_fdr(pvals[valid])
        for r, q in zip(cell_results, qvals):
            r["q"] = float(q) if not np.isnan(q) else np.nan
            long_rows.append(r)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(output_dir / "stat_grid_long.csv", index=False)
    logger.info(f"Long-form saved: {len(long_df)} rows")

    # Wide pivot
    wide = long_df.copy()
    wide["label"] = (wide["modality_parent"]
                     + wide["modality_sub"].apply(
                         lambda s: "" if pd.isna(s) else f" [{s}]"))
    pivot = wide.pivot_table(
        index="label", columns="comparison",
        values=["statistic", "p", "q", "effect", "auc_auc",
                "auc_auc_ci_low", "auc_auc_ci_high", "n"],
        aggfunc="first"
    )
    pivot.to_csv(output_dir / "stat_grid_wide.csv")

    # Per-feature-type sub-grids
    long_df["direction"] = long_df["modality_parent"].map(DIRECTION_MAP)
    for direction, sub in long_df.groupby("direction"):
        d_dir = output_dir / direction
        d_dir.mkdir(parents=True, exist_ok=True)
        sub.drop(columns=["direction"]).to_csv(
            d_dir / "stat_grid_long.csv", index=False)
    logger.info(f"Sub-grids: "
                f"{sorted(long_df['direction'].dropna().unique())}")

    return long_df, feas_df, hdr_df


def run_all(root_dir, cohort_mode, hc_source_mode, bg_mode,
            priority_groups=None):
    """Run all 4 (match_level × eval_unit) combinations."""
    for ml in MATCH_LEVELS:
        for eu in EVAL_UNITS:
            out = root_dir / ml / eu
            run_one_combo(out, cohort_mode, hc_source_mode,
                          ml, eu, bg_mode,
                          priority_groups=priority_groups)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-mode",
                   choices=VALID_COHORT_CHOICES,
                   default="p_first_cdr05_hc_first_cdrall_or_mmseall")
    p.add_argument("--hc-source", choices=["ACS", "ACS_ext", "EACS"],
                   default="ACS")
    p.add_argument("--bg-mode",
                   choices=["no_background", "background"],
                   default="no_background")
    p.add_argument("--match-priority", nargs="*", default=None,
                   help="HC sub-group matching priority (e.g. ACS NAD)")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    if args.output_dir is not None:
        root_dir = args.output_dir
    else:
        root_dir = (EMBEDDING_STAT_GRID_DIR
                    / cohort_path(args.cohort_mode)
                    / args.bg_mode
                    / args.hc_source.lower())
    root_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"hc_source={args.hc_source}  cohort_mode={args.cohort_mode}  "
                f"bg_mode={args.bg_mode}  root={root_dir}")

    run_all(root_dir,
            cohort_mode=args.cohort_mode,
            hc_source_mode=args.hc_source,
            bg_mode=args.bg_mode,
            priority_groups=args.match_priority)


if __name__ == "__main__":
    main()
