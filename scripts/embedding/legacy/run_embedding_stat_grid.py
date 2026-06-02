"""
Embedding asymmetry stat grid: 4 feature types × 4 models × 2 tests = 32 rows.

L2 scalar (Welch t, Cohen's d) + full vector (PERMANOVA, R²) for each
(feature_type, model) combination across ad_vs_hc / ad_vs_nad / ad_vs_acs,
under 4 evaluation combinations:
    subject_match × eval_by_subject
    subject_match × eval_by_visit
    visit_match   × eval_by_subject
    visit_match   × eval_by_visit

Output:
    workspace/embedding/analysis/feature_stat/
      <visit>/<cdr_mmse>/<bg_mode>/
        <match_level>/<eval_unit>/<match_strategy>/<partition>/
          stat_grid.csv

Usage:
    conda run -n Alz_face_main_analysis python \\
        scripts/embedding/run_embedding_stat_grid.py \\
        --p-visit p_first --p-score p_cdrall \\
        --hc-visit hc_all --hc-score hc_cdrall_or_mmseall \\
        --bg-mode no_background --match-priority ACS
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_FEATURES_DIR,
    EMBEDDING_FEATURE_STAT_DIR,
    P_VISIT_TOKENS,
    P_SCORE_TOKENS,
    HC_VISIT_TOKENS,
    HC_SCORE_TOKENS,
    cohort_path,
)
from src.common.cohort import cohort_list
from src.common.matching import match_by_age
from scripts.utilities.stats_helpers import bh_fdr, permanova, welch_t_test

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MIN_CELL_N = 20

MODELS = ["arcface", "dlib", "topofr", "vggface"]
DISTANCE = {"arcface": "cosine", "dlib": "euclidean",
            "topofr": "euclidean", "vggface": "euclidean"}
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

MATCH_LEVELS = ["subject_match", "visit_match"]
EVAL_UNITS = ["eval_by_subject", "eval_by_visit"]
MATCH_LEVEL_ARG = {"subject_match": "subject", "visit_match": "visit"}

PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]

PARTITION_KEEP_GROUPS = {
    "ad_vs_hc": None,
    "ad_vs_nad": {"P", "NAD"},
    "ad_vs_acs": {"P", "ACS"},
}

_PARTITION_TO_COMPARE = {
    "ad_vs_hc": "HC", "ad_vs_nad": "NAD", "ad_vs_acs": "ACS",
}


# ============================================================
# Statistical helpers (internalised from overview)
# ============================================================

def _group_split(X_df, cohort, feat_cols, label_col="label"):
    """Align features with cohort labels; returns (X1, X2, merged, feat_cols)."""
    m = cohort[[label_col] + (["ID"] if "ID" in cohort.columns else ["base_id"])]
    id_col = "ID" if "ID" in cohort.columns else "base_id"
    feat_id_col = "subject_id" if "subject_id" in X_df.columns else "base_id"
    merged = m.merge(X_df, left_on=id_col, right_on=feat_id_col, how="inner")
    feat_cols_present = [c for c in feat_cols if c in merged.columns]
    merged = merged.dropna(subset=feat_cols_present)
    X = merged[feat_cols_present].to_numpy(dtype=float)
    y = merged[label_col].to_numpy(dtype=int)
    X1 = X[y == 1]; X2 = X[y == 0]
    return X1, X2, merged, feat_cols_present


def run_scalar_modality(name, X_df, cohort):
    feat_cols = [c for c in X_df.columns if c not in ("subject_id", "base_id", "ID")]
    if len(feat_cols) != 1:
        raise ValueError(f"scalar modality {name} must be 1 feature")
    X1, X2, merged, feat_cols = _group_split(X_df, cohort, feat_cols)
    if len(X1) < MIN_CELL_N or len(X2) < MIN_CELL_N:
        return {"modality": name, "n": len(merged), "test": "welch_t",
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "effect_type": "cohens_d"}
    res = welch_t_test(X1[:, 0], X2[:, 0])
    return {
        "modality": name, "n": res["n1"] + res["n2"],
        "n1": res["n1"], "n2": res["n2"],
        "test": "welch_t", "statistic": res["t"],
        "p": res["p_welch"], "p_secondary": res["p_mw"],
        "effect": res["d"], "effect_type": "cohens_d",
        "mean1": res["mean1"], "mean2": res["mean2"],
    }


def _compute_cell_header_stats(design, compare, cohort):
    """Per-cell header stats: group sizes and age balance."""
    row = {"design": design, "comparison": compare}
    if cohort is None or len(cohort) == 0:
        return row
    if "base_id" in cohort.columns:
        bid_col = cohort["base_id"].astype(str)
    else:
        bid_col = cohort["ID"].astype(str).str.extract(r"^(.+)-\d+$")[0]
    row["n_all_1"] = int((cohort["label"] == 1).sum())
    row["n_all_0"] = int((cohort["label"] == 0).sum())
    row["n_all"] = row["n_all_1"] + row["n_all_0"]
    row["n_unique_1"] = int(bid_col[cohort["label"] == 1].nunique())
    row["n_unique_0"] = int(bid_col[cohort["label"] == 0].nunique())
    row["n_unique"] = row["n_unique_1"] + row["n_unique_0"]

    age_col = "Age"
    if age_col in cohort.columns:
        g1 = cohort.loc[cohort["label"] == 1, age_col].dropna().astype(float)
        g2 = cohort.loc[cohort["label"] == 0, age_col].dropna().astype(float)
        row["age_mean_1"] = float(g1.mean()) if len(g1) else np.nan
        row["age_sd_1"] = float(g1.std(ddof=1)) if len(g1) > 1 else np.nan
        row["age_mean_2"] = float(g2.mean()) if len(g2) else np.nan
        row["age_sd_2"] = float(g2.std(ddof=1)) if len(g2) > 1 else np.nan
        if len(g1) >= 2 and len(g2) >= 2:
            _, p = stats.ttest_ind(g1, g2, equal_var=False)
            row["age_p"] = float(p)
        else:
            row["age_p"] = np.nan
    return row


# ============================================================
# Feature loaders (bg_mode-aware, multi feature-type)
# ============================================================

def _load_npy_mean(feat_dir, sid):
    npy = feat_dir / f"{sid}.npy"
    if not npy.exists():
        return None
    a = np.load(npy)
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


def run_permanova_modality(name, X_df, cohort, metric="euclidean",
                           n_perms=1000):
    feat_cols = [c for c in X_df.columns
                 if c not in ("subject_id", "base_id", "ID")]
    X1, X2, merged, feat_cols = _group_split(X_df, cohort, feat_cols)
    if len(X1) < MIN_CELL_N or len(X2) < MIN_CELL_N:
        return {"modality": name, "n": len(merged), "test": "permanova",
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "effect_type": "R2"}
    res = permanova(X1, X2, metric=metric, n_perms=n_perms)
    return {
        "modality": name, "n": res["n1"] + res["n2"],
        "n1": res["n1"], "n2": res["n2"], "dim": len(feat_cols),
        "test": f"permanova_{metric}", "statistic": res["pseudo_F"],
        "p": res["p_perm"],
        "effect": res["R2"], "effect_type": "R2",
        "omega2": res["omega2"],
    }


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


# ============================================================
# Orchestrator
# ============================================================

def _get_ids(cohort):
    id_col = "ID" if "ID" in cohort.columns else "base_id"
    return cohort[id_col].astype(str).tolist()


def _filter_matched_by_keep_groups(matched, keep_groups):
    if keep_groups is None:
        return matched
    target_groups = set(keep_groups) - {"P"}
    target_pair_ids = matched[
        matched["group"].isin(target_groups)]["pair_id"].unique()
    return matched[matched["pair_id"].isin(target_pair_ids)].copy()


def run_one_partition(partition_dir, partition, cohort, bg_mode, eval_unit):
    """Run stat grid for a single partition."""
    partition_dir.mkdir(parents=True, exist_ok=True)
    compare = _PARTITION_TO_COMPARE[partition]

    if cohort is None or len(cohort) == 0:
        return None

    n = len(cohort)
    n_pos = int((cohort["label"] == 1).sum())
    n_neg = int((cohort["label"] == 0).sum())

    if min(n_pos, n_neg) < MIN_CELL_N:
        logger.info(f"  {partition}: skipped (min-cell-n {min(n_pos, n_neg)} < {MIN_CELL_N})")
        return None

    logger.info(f"=== {partition} : n_pos={n_pos} n_neg={n_neg} ===")

    hdr = _compute_cell_header_stats("cross_matched", compare, cohort)

    ids = _get_ids(cohort)
    test_cohort = cohort
    if eval_unit == "eval_by_subject":
        test_cohort = _aggregate_cohort_to_subject(cohort)

    cell_results = []
    for (parent, sub, test_kind, model, ft) in MODALITY_SPECS:
        name = f"{parent}::{sub}"
        try:
            if test_kind == "scalar":
                X_df = load_l2_scalar(ids, model, ft, bg_mode)
            else:
                X_df = load_full_vector(ids, model, ft, bg_mode)
            if eval_unit == "eval_by_subject":
                X_df = _aggregate_features_to_subject(X_df)
            if X_df is None or X_df.empty:
                res = {"modality": name, "n": 0, "test": test_kind,
                       "statistic": np.nan, "p": np.nan, "effect": np.nan,
                       "skip_reason": "feature unavailable"}
            elif test_kind == "scalar":
                res = run_scalar_modality(name, X_df, test_cohort)
            else:
                metric = DISTANCE[model]
                res = run_permanova_modality(name, X_df, test_cohort,
                                            metric=metric)
        except Exception as e:
            logger.warning(f"{partition} {name}: {e}")
            res = {"modality": name, "n": 0, "p": np.nan,
                   "statistic": np.nan, "effect": np.nan,
                   "error": str(e)}
        res["partition"] = partition
        res["modality_parent"] = parent
        res["modality_sub"] = sub
        cell_results.append(res)
        logger.info(f"  {name}: p={res.get('p')} eff={res.get('effect')}")

    pvals = np.array([r.get("p", np.nan) for r in cell_results], dtype=float)
    valid = ~np.isnan(pvals)
    qvals = np.full(len(pvals), np.nan)
    if valid.sum() > 0:
        qvals[valid] = bh_fdr(pvals[valid])
    for r, q in zip(cell_results, qvals):
        r["q"] = float(q) if not np.isnan(q) else np.nan

    result_df = pd.DataFrame(cell_results)
    result_df["modality"] = (result_df["modality_parent"]
                             + result_df["modality_sub"].apply(
                                 lambda s: "" if pd.isna(s) else f" [{s}]"))
    out_cols = ["modality", "test", "n", "statistic", "p", "p_secondary",
                "q", "effect", "effect_type", "mean1", "mean2",
                "omega2", "dim"]
    out = result_df[[c for c in out_cols if c in result_df.columns]].copy()
    out.set_index("modality", inplace=True)

    meta = pd.DataFrame([hdr])
    meta_path = partition_dir / "stat_grid.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        f.write("# cohort summary\n")
        meta.to_csv(f, index=False)
        f.write("\n# stat grid (L2: Welch t / Cohen's d; vector: PERMANOVA / R²)\n")
        out.to_csv(f)

    return result_df


def run_one_combo(output_dir, cohort,
                  match_level, eval_unit, bg_mode, priority_groups=None):
    """Run stat grid for one (match_level, eval_unit) combination."""
    ml_arg = MATCH_LEVEL_ARG[match_level]
    match_strategy = (f"priority_{priority_groups[0].lower()}"
                      if priority_groups else "no_priority")
    logger.info(f"--- {match_level} / {eval_unit} / {match_strategy} ---")

    roster = cohort_list(*cohort)
    roster["group"] = roster["Group"]
    roster["base_id"] = roster["Group"] + roster["Number"].astype(str)
    p_ids, hc_ids = match_by_age(*cohort, level=ml_arg, priority=priority_groups)
    hc_cohort = roster[roster["ID"].isin(set(p_ids) | set(hc_ids))].copy()

    all_long = []
    for partition in PARTITIONS:
        part_dir = output_dir / match_level / eval_unit / match_strategy / partition
        cohort = _filter_matched_by_keep_groups(
            hc_cohort, PARTITION_KEEP_GROUPS[partition])
        long_df = run_one_partition(
            part_dir, partition, cohort, bg_mode, eval_unit)
        if long_df is not None:
            all_long.append(long_df)

    if all_long:
        combined = pd.concat(all_long, ignore_index=True)
        combo_dir = output_dir / match_level / eval_unit / match_strategy
        combined.to_csv(combo_dir / "stat_grid_all_partitions.csv", index=False)

    return all_long


def run_all(root_dir, cohort, bg_mode,
            priority_groups=None):
    """Run all 4 (match_level × eval_unit) combinations."""
    for ml in MATCH_LEVELS:
        for eu in EVAL_UNITS:
            run_one_combo(root_dir, cohort,
                          ml, eu, bg_mode,
                          priority_groups=priority_groups)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p-visit", choices=list(P_VISIT_TOKENS),
                   default="p_first")
    p.add_argument("--p-score", choices=list(P_SCORE_TOKENS),
                   default="p_cdrall")
    p.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS),
                   default="hc_all")
    p.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                   default="hc_cdrall_or_mmseall")
    p.add_argument("--bg-mode",
                   choices=["no_background", "background"],
                   default="no_background")
    p.add_argument("--match-priority", nargs="*", default=None,
                   help="HC sub-group matching priority (e.g. ACS NAD). "
                        "Output goes to priority_<group>/; "
                        "without this flag output goes to no_priority/.")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)

    if args.output_dir is not None:
        root_dir = args.output_dir
    else:
        root_dir = (EMBEDDING_FEATURE_STAT_DIR
                    / cohort_path(*cohort)
                    / args.bg_mode)
    root_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"cohort={cohort}  "
                f"bg_mode={args.bg_mode}  root={root_dir}")

    run_all(root_dir,
            cohort=cohort,
            bg_mode=args.bg_mode,
            priority_groups=args.match_priority)


if __name__ == "__main__":
    main()
