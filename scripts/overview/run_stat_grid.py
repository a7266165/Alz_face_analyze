"""
Cross-design × cross-modality statistical grid.

14 data rows (10 modality parents) × 20 comparison cells (4 design × 5
comparison):

  Designs:  cross_naive | cross_matched | longitudinal_naive | longitudinal_matched
  Comparisons: HC | NAD | ACS | mmse-hi-lo | casi-hi-lo

  Cell primary  = per-modality test (Welch t / Hotelling T² / PERMANOVA /
                  per-method-Hotelling-Fisher) + effect size + BH-FDR q
  Cell secondary = AUC + 95% CI

Usage:
    conda run -n Alz_face_test_2 python scripts/overview/run_stat_grid.py \\
        --cohort-mode default --hc-source ACS \\
        --designs cross_naive cross_matched longitudinal_naive longitudinal_matched
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    LONGITUDINAL_FEATURES_DIR, OVERVIEW_DIR, VALID_COHORT_CHOICES, cohort_name,
)
from scripts.utilities.cohort import (
    build_cohort_ad_hi_lo, build_cohort_ad_vs_HCgroup, VALID_DESIGNS,
)
from scripts.utilities.feature_loaders import (
    EMBEDDING_MODELS, EMOTION_METHODS, EMOTIONS, LANDMARK_REGIONS,
    load_embedding_asymmetry as load_embedding_asymmetry_l2,
    load_embedding_asymmetry_vec, load_embedding_mean, load_emotion_matrix,
    load_landmark_raw, load_landmark_region_l2,
)
from scripts.utilities.stats_helpers import (
    auc_with_ci, bh_fdr, fishers_method, hotelling_t2, permanova, welch_t_test,
)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
from src.config import PREDICTED_AGES_FILE as AGES_FILE
LANDMARK_FEATURES_CSV = (PROJECT_ROOT / "workspace" / "asymmetry" /
                         "features" / "pair_features.csv")
LONGITUDINAL_CSV = LONGITUDINAL_FEATURES_DIR / "patient_deltas.csv"
AD_DELTAS_CSV = LONGITUDINAL_FEATURES_DIR / "ad_patient_deltas.csv"
HC_LONGITUDINAL_CSV = LONGITUDINAL_FEATURES_DIR / "hc_patient_deltas.csv"
VECTOR_DELTAS_NPZ = LONGITUDINAL_FEATURES_DIR / "vector_deltas.npz"

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
N_PERMS = int(os.environ.get("STAT_GRID_N_PERMS", 1000))
SEED = 42
MIN_CELL_N = 20  # cells with min(n_pos,n_neg) < 20 → marked n/a

DESIGNS = ("cross_naive", "cross_matched",
           "longitudinal_naive", "longitudinal_matched")
COMPARISONS = ("HC", "NAD", "ACS", "mmse-hi-lo", "casi-hi-lo")

# Modality direction grouping for per-direction grid subfolders.
DIRECTION_MAP = {
    "age_only": "age",
    "age_error": "age",
    "embedding_arcface_mean": "embedding_mean",
    "embedding_dlib_mean": "embedding_mean",
    "embedding_topofr_mean": "embedding_mean",
    "embedding_arcface_asymmetry": "embedding_asymmetry",
    "embedding_dlib_asymmetry": "embedding_asymmetry",
    "embedding_topofr_asymmetry": "embedding_asymmetry",
    "landmark_asymmetry": "landmark_asymmetry",
    "emotion_8methods": "emotion",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Modality dispatch
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
    auc_row = auc_with_ci(merged[feat_cols].values, merged["label"].values,
                          model_cls="logistic")
    return {
        "modality": name, "n": res["n1"] + res["n2"],
        "n1": res["n1"], "n2": res["n2"],
        "test": "welch_t", "statistic": res["t"],
        "p": res["p_welch"], "p_secondary": res["p_mw"],
        "effect": res["d"], "effect_type": "cohens_d",
        "mean1": res["mean1"], "mean2": res["mean2"],
        **{f"auc_{k}": v for k, v in auc_row.items()},
    }


def run_hotelling_modality(name, X_df, cohort, n_perms=N_PERMS):
    feat_cols = [c for c in X_df.columns if c not in ("subject_id", "base_id", "ID")]
    X1, X2, merged, feat_cols = _group_split(X_df, cohort, feat_cols)
    if len(X1) < MIN_CELL_N or len(X2) < MIN_CELL_N:
        return {"modality": name, "n": len(merged), "test": "hotelling_t2",
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "effect_type": "mahalanobis_D2"}
    res = hotelling_t2(X1, X2, n_perms=n_perms)
    auc_row = auc_with_ci(merged[feat_cols].values, merged["label"].values,
                          model_cls="xgb")
    return {
        "modality": name, "n": res["n1"] + res["n2"],
        "n1": res["n1"], "n2": res["n2"], "dim": res["p"],
        "test": "hotelling_t2", "statistic": res["T2"],
        "p": res["p_perm"], "p_secondary": res["p_F"],
        "effect": res["D2"], "effect_type": "mahalanobis_D2",
        **{f"auc_{k}": v for k, v in auc_row.items()},
    }


def run_permanova_modality(name, X_df, cohort, metric="euclidean", n_perms=N_PERMS):
    feat_cols = [c for c in X_df.columns if c not in ("subject_id", "base_id", "ID")]
    X1, X2, merged, feat_cols = _group_split(X_df, cohort, feat_cols)
    if len(X1) < MIN_CELL_N or len(X2) < MIN_CELL_N:
        return {"modality": name, "n": len(merged), "test": "permanova",
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "effect_type": "R2"}
    res = permanova(X1, X2, metric=metric, n_perms=n_perms)
    auc_row = auc_with_ci(merged[feat_cols].values, merged["label"].values,
                          model_cls="xgb")
    return {
        "modality": name, "n": res["n1"] + res["n2"],
        "n1": res["n1"], "n2": res["n2"], "dim": len(feat_cols),
        "test": f"permanova_{metric}", "statistic": res["pseudo_F"],
        "p": res["p_perm"],
        "effect": res["R2"], "effect_type": "R2",
        "omega2": res["omega2"],
        **{f"auc_{k}": v for k, v in auc_row.items()},
    }


def run_emotion_modality(name, X_df, cohort, n_perms=N_PERMS):
    """Emotion: per-method Hotelling + Fisher combine over 8 methods."""
    all_feat_cols = [c for c in X_df.columns
                     if c not in ("subject_id", "base_id", "ID")]
    per_method_p, per_method_stat, per_method_n = [], [], []
    for method in EMOTION_METHODS:
        mcols = [c for c in all_feat_cols if c.startswith(f"{method}__")]
        if not mcols:
            continue
        sub_df = X_df[["subject_id"] + mcols]
        X1, X2, merged, mcols_p = _group_split(sub_df, cohort, mcols)
        if len(X1) < MIN_CELL_N or len(X2) < MIN_CELL_N or len(mcols_p) < 2:
            continue
        res = hotelling_t2(X1, X2, n_perms=n_perms)
        per_method_p.append(res["p_perm"])
        per_method_stat.append(res["T2"])
        per_method_n.append(res["n1"] + res["n2"])

    fish = fishers_method(per_method_p)
    X1_full, X2_full, merged_full, feat_cols = _group_split(
        X_df, cohort, all_feat_cols)
    auc_row = auc_with_ci(merged_full[feat_cols].values,
                          merged_full["label"].values, model_cls="xgb")
    return {
        "modality": name,
        "n": int(np.mean(per_method_n)) if per_method_n else len(merged_full),
        "dim": len(all_feat_cols),
        "test": "hotelling_per_method_fisher",
        "statistic": float(np.mean(per_method_stat)) if per_method_stat else np.nan,
        "p": fish["p"], "p_secondary": fish["chi2"],
        "effect": float(np.mean(per_method_stat)) if per_method_stat else np.nan,
        "effect_type": "mean_T2",
        "k_methods_used": fish["k"],
        **{f"auc_{k}": v for k, v in auc_row.items()},
    }


# ============================================================
# Modality specs (14 data rows)
# ============================================================

MODALITY_SPECS = [
    # (parent, sub_label, test_kind, extra)
    ("age_only", None, "scalar_age", {}),
    ("embedding_arcface_mean", None, "permanova_cosine", {"model": "arcface"}),
    ("embedding_dlib_mean", None, "permanova_euclidean", {"model": "dlib"}),
    ("embedding_topofr_mean", None, "permanova_euclidean", {"model": "topofr"}),
    ("embedding_arcface_asymmetry", "L2 scalar", "scalar_embasym_l2",
     {"model": "arcface"}),
    ("embedding_arcface_asymmetry", "full vector", "permanova_cosine_embasym",
     {"model": "arcface"}),
    ("embedding_dlib_asymmetry", "L2 scalar", "scalar_embasym_l2",
     {"model": "dlib"}),
    ("embedding_dlib_asymmetry", "full vector", "permanova_euclidean_embasym",
     {"model": "dlib"}),
    ("embedding_topofr_asymmetry", "L2 scalar", "scalar_embasym_l2",
     {"model": "topofr"}),
    ("embedding_topofr_asymmetry", "full vector", "permanova_euclidean_embasym",
     {"model": "topofr"}),
    ("landmark_asymmetry", "4-d per-region L2", "hotelling_landmark4", {}),
    ("landmark_asymmetry", "130-d raw xy", "permanova_euclidean_landmark130", {}),
    ("emotion_8methods", None, "emotion_fisher", {}),
    ("age_error", None, "scalar_age_error", {}),
]


# ============================================================
# Feature DF builder per modality / design
# ============================================================

def _get_subject_ids(cohort):
    if "ID" in cohort.columns:
        return cohort["ID"].dropna().astype(str).tolist()
    return cohort["base_id"].dropna().astype(str).tolist()


def _load_age_error_df(ids):
    """Compute age_error = Age - predicted_age per subject_id."""
    with open(AGES_FILE) as f:
        pred = json.load(f)
    demo_all = []
    for grp in ["P", "NAD", "ACS"]:
        df = pd.read_csv(DEMOGRAPHICS_DIR / f"{grp}.csv")
        if "ID" not in df.columns:
            for col in df.columns:
                if col in ("ACS", "NAD"):
                    df = df.rename(columns={col: "ID"}); break
        demo_all.append(df[["ID", "Age"]])
    demo_all = pd.concat(demo_all, ignore_index=True).drop_duplicates("ID")
    demo_all["Age"] = pd.to_numeric(demo_all["Age"], errors="coerce")
    rows = []
    for sid in ids:
        row = demo_all[demo_all["ID"] == sid]
        if len(row) == 0:
            continue
        a = float(row.iloc[0]["Age"])
        pa = pred.get(sid)
        if pa is None or np.isnan(a):
            continue
        rows.append({"subject_id": sid, "age_error": a - float(pa)})
    return pd.DataFrame(rows)


_VECTOR_DELTAS_CACHE = None


def _get_vector_deltas():
    global _VECTOR_DELTAS_CACHE
    if _VECTOR_DELTAS_CACHE is None:
        if not VECTOR_DELTAS_NPZ.exists():
            _VECTOR_DELTAS_CACHE = (None, {})
            return _VECTOR_DELTAS_CACHE
        z = np.load(VECTOR_DELTAS_NPZ, allow_pickle=False)
        base_ids = z["base_ids"]
        mats = {k: z[k] for k in z.files if k != "base_ids"}
        _VECTOR_DELTAS_CACHE = (base_ids, mats)
    return _VECTOR_DELTAS_CACHE


def _load_vector_delta(cohort, vec_key):
    """Return feature DF with subject_id + full-vector Δ cols from NPZ."""
    base_ids, mats = _get_vector_deltas()
    if base_ids is None or vec_key not in mats:
        return None
    mat = mats[vec_key]
    bid_to_i = {str(b): i for i, b in enumerate(base_ids)}
    idc = "base_id" if "base_id" in cohort.columns else "ID"
    cohort_bids = cohort[idc].astype(str).values
    D = mat.shape[1]
    out_rows = []
    for bid in cohort_bids:
        i = bid_to_i.get(bid)
        if i is None:
            continue
        vec = mat[i]
        if np.any(np.isnan(vec)):
            continue
        row = {"subject_id": bid}
        for d in range(D):
            row[f"{vec_key}_{d}"] = float(vec[d])
        out_rows.append(row)
    if not out_rows:
        return None
    return pd.DataFrame(out_rows)


def _longitudinal_feature_df(test_kind, extra, cohort):
    """Build feature DF for longitudinal designs using pre-computed ann_* Δ cols.

    Cohort comes from build_cohort_ad_vs_HCgroup with design='longitudinal_*'
    or build_cohort_ad_hi_lo with design='longitudinal_*'.
    """
    has_ann = any(c.startswith("ann_") for c in cohort.columns)
    if not has_ann:
        return None
    idc = "base_id"
    if test_kind == "scalar_age":
        return cohort[[idc, "first_age"]].rename(
            columns={idc: "subject_id", "first_age": "value_age"})
    if test_kind == "scalar_age_error":
        col = "ann_delta_age_error"
        if col not in cohort.columns:
            return None
        return cohort[[idc, col]].rename(columns={idc: "subject_id"})
    if test_kind == "scalar_embasym_l2":
        col = f"ann_delta_embasym_{extra['model']}"
        if col not in cohort.columns:
            return None
        return cohort[[idc, col]].rename(columns={idc: "subject_id"})
    if test_kind.endswith("embasym"):
        return _load_vector_delta(cohort, f"emb_asym_delta_vec_{extra['model']}")
    if test_kind == "hotelling_landmark4":
        cols = [f"ann_lmk_delta_{r}_l2" for r in LANDMARK_REGIONS]
        avail = [c for c in cols if c in cohort.columns]
        if len(avail) < 4:
            return None
        return cohort[[idc] + avail].rename(columns={idc: "subject_id"})
    if test_kind == "permanova_euclidean_landmark130":
        return _load_vector_delta(cohort, "lmk_raw_xy_delta")
    if test_kind in ("permanova_cosine", "permanova_euclidean"):
        return _load_vector_delta(cohort, f"emb_drift_vec_{extra['model']}")
    if test_kind == "emotion_fisher":
        cols = [f"ann_delta_{m}__{e}" for m in EMOTION_METHODS for e in EMOTIONS]
        avail = [c for c in cols if c in cohort.columns]
        if len(avail) < 7:
            return None
        sub = cohort[[idc] + avail].rename(columns={idc: "subject_id"})
        rename = {f"ann_delta_{m}__{e}": f"{m}__{e}_mean"
                  for m in EMOTION_METHODS for e in EMOTIONS
                  if f"ann_delta_{m}__{e}" in sub.columns}
        sub = sub.rename(columns=rename)
        return sub
    return None


def _feature_df_for_modality(test_kind, extra, cohort, design):
    """Load the feature DF for a given modality spec and cohort.

    For longitudinal designs, features come from the cohort's pre-computed
    ann_* Δ columns. For cross-sec designs, features come from raw npy/csv
    loaders keyed by visit-level ID.
    """
    if design.startswith("longitudinal_"):
        return _longitudinal_feature_df(test_kind, extra, cohort)

    ids = _get_subject_ids(cohort)
    if test_kind == "scalar_age":
        if "Age" in cohort.columns:
            idc = "ID" if "ID" in cohort.columns else "base_id"
            return cohort[[idc, "Age"]].rename(
                columns={idc: "subject_id", "Age": "value_age"})
        return None
    if test_kind == "scalar_age_error":
        return _load_age_error_df(ids)
    if test_kind == "scalar_embasym_l2":
        df = load_embedding_asymmetry_l2(ids)
        if df is None or df.empty:
            return None
        col = f"embasym__{extra['model']}_l2"
        if col not in df.columns:
            return None
        return df[["subject_id", col]].copy()
    if test_kind.endswith("embasym"):
        return load_embedding_asymmetry_vec(ids, extra["model"])
    if test_kind == "hotelling_landmark4":
        return load_landmark_region_l2(ids)
    if test_kind == "permanova_euclidean_landmark130":
        return load_landmark_raw(ids)
    if test_kind in ("permanova_cosine", "permanova_euclidean"):
        return load_embedding_mean(ids, extra["model"])
    if test_kind == "emotion_fisher":
        emo_df = load_emotion_matrix(ids)
        if emo_df is None:
            return None
        keep_cols = ["subject_id"] + [c for c in emo_df.columns
                                      if c != "subject_id" and c.endswith("_mean")]
        return emo_df[keep_cols].copy()
    return None


def _dispatch_test(test_kind, extra, name, X_df, cohort, design):
    if X_df is None or (hasattr(X_df, "empty") and X_df.empty):
        return {"modality": name, "n": 0, "test": test_kind,
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "skip_reason": "feature unavailable"}

    # Longitudinal: heuristic feature-count test selection (n_feat 1 → scalar,
    # ≤10 → Hotelling, else PERMANOVA with metric matching test_kind).
    if design.startswith("longitudinal_"):
        n_feat = len([c for c in X_df.columns if c not in ("subject_id",)])
        if n_feat == 1:
            return run_scalar_modality(name, X_df, cohort)
        if test_kind == "emotion_fisher":
            return run_emotion_modality(name, X_df, cohort)
        if n_feat <= 10:
            return run_hotelling_modality(name, X_df, cohort)
        if test_kind == "permanova_cosine" or test_kind.endswith("cosine_embasym"):
            return run_permanova_modality(name, X_df, cohort, metric="cosine")
        return run_permanova_modality(name, X_df, cohort, metric="euclidean")

    # Cross-sectional: declared test kind drives the call.
    if test_kind in ("scalar_age", "scalar_age_error", "scalar_embasym_l2"):
        return run_scalar_modality(name, X_df, cohort)
    if test_kind == "hotelling_landmark4":
        return run_hotelling_modality(name, X_df, cohort)
    if test_kind == "permanova_cosine" or test_kind.endswith("cosine_embasym"):
        return run_permanova_modality(name, X_df, cohort, metric="cosine")
    if test_kind.startswith("permanova_euclidean"):
        return run_permanova_modality(name, X_df, cohort, metric="euclidean")
    if test_kind == "emotion_fisher":
        return run_emotion_modality(name, X_df, cohort)
    return {"modality": name, "n": 0, "test": test_kind,
            "statistic": np.nan, "p": np.nan, "effect": np.nan}


# ============================================================
# Cell header stats
# ============================================================

def _compute_cell_header_stats(design, compare, cohort):
    """Per-cell header stats for the multi-row thead in plot_deep_dive_grid.py."""
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

    age_col = "first_age" if "first_age" in cohort.columns else "Age"
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

    if compare in ("mmse-hi-lo", "casi-hi-lo"):
        cog_triples = [
            ("mmse", "MMSE", "first_MMSE"),
            ("casi", "CASI", "first_CASI"),
            ("cdr",  "Global_CDR", "first_Global_CDR"),
        ]
        for name, ab_col, cd_col in cog_triples:
            col = ab_col if ab_col in cohort.columns else (
                cd_col if cd_col in cohort.columns else None)
            if col is None:
                for k in ("mean_1", "sd_1", "mean_2", "sd_2", "p"):
                    row[f"{name}_{k}"] = np.nan
                continue
            g1 = cohort.loc[cohort["label"] == 1, col].dropna().astype(float)
            g2 = cohort.loc[cohort["label"] == 0, col].dropna().astype(float)
            row[f"{name}_mean_1"] = float(g1.mean()) if len(g1) else np.nan
            row[f"{name}_sd_1"] = float(g1.std(ddof=1)) if len(g1) > 1 else np.nan
            row[f"{name}_mean_2"] = float(g2.mean()) if len(g2) else np.nan
            row[f"{name}_sd_2"] = float(g2.std(ddof=1)) if len(g2) > 1 else np.nan
            if len(g1) >= 2 and len(g2) >= 2:
                _, p = stats.ttest_ind(g1, g2, equal_var=False)
                row[f"{name}_p"] = float(p)
            else:
                row[f"{name}_p"] = np.nan
    return row


# ============================================================
# Orchestrator
# ============================================================

def run_all(output_dir, cohort_mode, hc_source_mode, designs_to_run,
            modality_keys=None):
    """Run the cross-design × cross-modality stat grid.

    output_dir: where stat_grid_long.csv etc. land
    cohort_mode: 'default' | 'p_first_hc_all' | 'p_all_hc_all'
    hc_source_mode: 'ACS' | 'ACS_ext' | 'EACS'
    designs_to_run: subset of DESIGNS to run (others marked n/a in grid)
    modality_keys: subset of MODALITY_SPECS' parent names (None → all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if modality_keys is None:
        active_specs = list(MODALITY_SPECS)
    else:
        active_specs = [s for s in MODALITY_SPECS if s[0] in modality_keys]
    logger.info(f"designs={list(designs_to_run)}  "
                f"modalities={[s[0] for s in active_specs]}")

    cells = [(d, c) for d in DESIGNS for c in COMPARISONS]
    cohorts = {}
    feasibility_rows = []
    header_stat_rows = []

    for design, compare in cells:
        if design not in designs_to_run:
            feasibility_rows.append({
                "design": design, "comparison": compare,
                "n_total": 0, "n_pos": 0, "n_neg": 0,
                "status": "n/a (design skipped)",
                "note": "design not in --designs",
            })
            header_stat_rows.append(_compute_cell_header_stats(design, compare, None))
            cohorts[(design, compare)] = (None, None)
            continue
        try:
            if compare == "mmse-hi-lo":
                cohort, pairs = build_cohort_ad_hi_lo(
                    design=design, cohort_mode=cohort_mode, metric="MMSE",
                    matched_features_csv=_hilo_matched_csv(cohort_mode, "mmse"),
                )
            elif compare == "casi-hi-lo":
                cohort, pairs = build_cohort_ad_hi_lo(
                    design=design, cohort_mode=cohort_mode, metric="CASI",
                    matched_features_csv=_hilo_matched_csv(cohort_mode, "casi"),
                )
            else:
                cohort, pairs = build_cohort_ad_vs_HCgroup(
                    compare, design=design, cohort_mode=cohort_mode,
                    hc_source_mode=hc_source_mode,
                )
            n = len(cohort) if cohort is not None else 0
            n_pos = int((cohort["label"] == 1).sum()) if n > 0 else 0
            n_neg = int((cohort["label"] == 0).sum()) if n > 0 else 0
            status = "ok"; note = ""
            if min(n_pos, n_neg) < MIN_CELL_N:
                status = f"n/a (min-cell-n {min(n_pos, n_neg)} < {MIN_CELL_N})"
                note = "pair count below threshold"
        except Exception as e:
            cohort, pairs = None, None
            n, n_pos, n_neg = 0, 0, 0
            status = f"error: {e}"; note = str(e)
        feasibility_rows.append({
            "design": design, "comparison": compare,
            "n_total": n, "n_pos": n_pos, "n_neg": n_neg,
            "status": status, "note": note,
        })
        header_stat_rows.append(_compute_cell_header_stats(design, compare, cohort))
        cohorts[(design, compare)] = (cohort, pairs)
    feas_df = pd.DataFrame(feasibility_rows)
    feas_df.to_csv(output_dir / "feasibility_report.csv", index=False)
    hdr_df = pd.DataFrame(header_stat_rows)
    hdr_df.to_csv(output_dir / "cell_header_stats.csv", index=False)
    logger.info(f"Feasibility + header stats saved. Active cells: "
                f"{(feas_df['status']=='ok').sum()}/{len(cells)}")

    # Run stats per modality × active cell
    long_rows = []
    active_keys = {s[0] for s in active_specs}
    for design, compare in cells:
        cohort, _ = cohorts[(design, compare)]
        fstat = feas_df[(feas_df.design == design) &
                        (feas_df.comparison == compare)].iloc[0]
        if fstat["status"] != "ok":
            for (parent, sub, test_kind, _) in MODALITY_SPECS:
                long_rows.append({
                    "design": design, "comparison": compare,
                    "modality_parent": parent, "modality_sub": sub,
                    "test": test_kind, "n": 0, "p": np.nan, "q": np.nan,
                    "statistic": np.nan, "effect": np.nan,
                    "skip_reason": fstat["status"],
                })
            continue
        logger.info(f"=== {design} × {compare} : "
                    f"n_pos={fstat['n_pos']} n_neg={fstat['n_neg']} ===")
        cell_results = []
        for (parent, sub, test_kind, extra) in MODALITY_SPECS:
            name = parent if sub is None else f"{parent}::{sub}"
            if parent not in active_keys:
                res = {"modality": name, "n": 0, "p": np.nan,
                       "statistic": np.nan, "effect": np.nan,
                       "skip_reason": "modality not in --modalities"}
                res["design"] = design; res["comparison"] = compare
                res["modality_parent"] = parent; res["modality_sub"] = sub
                cell_results.append(res)
                continue
            try:
                X_df = _feature_df_for_modality(test_kind, extra, cohort, design)
                res = _dispatch_test(test_kind, extra, name, X_df, cohort, design)
            except Exception as e:
                logger.warning(f"{design}×{compare} {name}: {e}")
                res = {"modality": name, "n": 0, "p": np.nan,
                       "statistic": np.nan, "effect": np.nan,
                       "error": str(e)}
            res["design"] = design; res["comparison"] = compare
            res["modality_parent"] = parent; res["modality_sub"] = sub
            cell_results.append(res)
            logger.info(f"  {name}: p={res.get('p')} eff={res.get('effect')}")

        # FDR within cell
        pvals = np.array([r.get("p", np.nan) for r in cell_results], dtype=float)
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

    # Pivot wide
    wide = long_df.copy()
    wide["cell"] = wide["design"] + "-" + wide["comparison"]
    wide["label"] = wide["modality_parent"] + wide["modality_sub"].apply(
        lambda s: "" if pd.isna(s) else f" [{s}]")
    pivot = wide.pivot_table(
        index="label", columns="cell",
        values=["statistic", "p", "q", "effect", "auc_auc",
                "auc_auc_ci_low", "auc_auc_ci_high", "n"],
        aggfunc="first"
    )
    pivot.to_csv(output_dir / "stat_grid_wide.csv")
    logger.info("Wide grid saved")

    # Per-direction subfolders
    long_df_dir = long_df.copy()
    long_df_dir["direction"] = long_df_dir["modality_parent"].map(DIRECTION_MAP)
    for direction, sub in long_df_dir.groupby("direction"):
        d_dir = output_dir / direction
        d_dir.mkdir(parents=True, exist_ok=True)
        sub.drop(columns=["direction"]).to_csv(
            d_dir / "stat_grid_long.csv", index=False)
        sub_wide = sub.copy()
        sub_wide["cell"] = sub_wide["design"] + "-" + sub_wide["comparison"]
        sub_wide["label"] = sub_wide["modality_parent"] + sub_wide["modality_sub"].apply(
            lambda s: "" if pd.isna(s) else f" [{s}]")
        sub_pivot = sub_wide.pivot_table(
            index="label", columns="cell",
            values=["statistic", "p", "q", "effect", "auc_auc",
                    "auc_auc_ci_low", "auc_auc_ci_high", "n"],
            aggfunc="first"
        )
        sub_pivot.to_csv(d_dir / "stat_grid_wide.csv")
    logger.info(f"Per-direction subfolders saved: "
                f"{sorted(long_df_dir['direction'].dropna().unique())}")

    return long_df, feas_df


def _hilo_matched_csv(cohort_mode, metric_low):
    """Path to the matched_features.csv produced by run_cross_matched.py for a
    given cohort_mode + metric (mmse/casi). Used as the input for design=
    'cross_matched' in build_cohort_ad_hi_lo."""
    return (OVERVIEW_DIR / cohort_name(cohort_mode) / "cross_matched" /
            f"{metric_low}_high_vs_low" / "matched_features.csv")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-mode",
                   choices=VALID_COHORT_CHOICES,
                   default="default")
    p.add_argument("--hc-source", choices=["ACS", "ACS_ext", "EACS"],
                   default="ACS",
                   help="ACS 群體組成：ACS=內部 91 人 (default)；"
                        "ACS_ext=內部+EACS；EACS=僅 EACS")
    p.add_argument("--eacs-sources", nargs="+", default=None,
                   help="EACS source 子集（例：UTKFace），只在 ACS_ext/EACS mode 有效")
    p.add_argument("--designs", nargs="+", choices=list(VALID_DESIGNS),
                   default=list(DESIGNS),
                   help="只跑指定 designs；其他在 grid 上標 n/a")
    p.add_argument("--modalities", nargs="+", default=None,
                   help="只算指定 modality_parent；其他標 skip_reason")
    p.add_argument("--output-suffix", default=None,
                   help="覆寫 subset 資料夾名，輸出到 stat_grid/subsets/<suffix>/")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="完全覆寫 OUTPUT_DIR（忽略 cohort/hc_source path 推導）")
    args = p.parse_args()

    if args.eacs_sources:
        os.environ["EACS_SOURCES"] = ",".join(args.eacs_sources)

    grid_root = OVERVIEW_DIR / cohort_name(args.cohort_mode) / "stat_grid"
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        is_subset = bool(
            args.eacs_sources or
            (set(args.designs) != set(DESIGNS)) or
            args.modalities or args.output_suffix
        )
        if is_subset:
            if args.output_suffix is not None:
                suffix = args.output_suffix
            else:
                parts = []
                if args.hc_source != "ACS":
                    parts.append(args.hc_source.lower())
                if args.eacs_sources:
                    parts.append("+".join(s.lower() for s in args.eacs_sources))
                if set(args.designs) != set(DESIGNS):
                    abbrev = {"cross_naive": "cn", "cross_matched": "cm",
                              "longitudinal_naive": "ln", "longitudinal_matched": "lm"}
                    parts.append("".join(abbrev[d] for d in sorted(args.designs)))
                if args.modalities:
                    parts.append("_".join(args.modalities)[:40])
                suffix = "_".join(parts)
            output_dir = grid_root / "subsets" / suffix
        else:
            output_dir = grid_root / args.hc_source.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"hc_source={args.hc_source}  cohort_mode={args.cohort_mode}  "
                f"output_dir={output_dir}")

    run_all(output_dir,
            cohort_mode=args.cohort_mode,
            hc_source_mode=args.hc_source,
            designs_to_run=set(args.designs),
            modality_keys=set(args.modalities) if args.modalities else None)


if __name__ == "__main__":
    main()
