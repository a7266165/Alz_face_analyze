"""
Embedding score classifier — forward / reverse strategies.

Forward strategy:
    1. Build full cohort for one of 5 partitions
       (ad_vs_hc / ad_vs_nad / ad_vs_acs / mmse_hilo / casi_hilo).
    2. Train 10-fold GroupKFold(base_id) classifier (LR or XGB) on the raw
       embedding 512-dim mean vector → out-of-fold prediction score per subject.
    3. Subset OOF scores to the age 1:1 matched cohort
       (matched count must equal arm_b counts: HC=180, NAD=169, ACS=29).
       Same predicted_age post-hoc filter as arm_b is applied — pairs with
       either side missing from predicted_ages.json are dropped, so the
       matched cohort here is identical to arm_b's matched cohort.
    4. Paired Wilcoxon signed-rank on score by pair_id +
       classification metrics (AUC + 95% bootstrap CI / BalAcc / MCC / F1 /
       Sens / Spec) on the matched subset.

Reverse strategy:
    1. Build matched cohort first (same matching as arm_b).
    2. 10-fold GroupKFold(base_id) on the matched cohort → 10 fold-models
       (each LR fold also fits its own StandardScaler).
    3. Each fold-model predicts proba over the *full* partition cohort →
       average across 10 folds = ensemble score.
    4. Classification metrics on full cohort (matched subset metrics also
       reported as training-domain reference).

Sweep axes: 5 partition × 3 embedding × 2 classifier × 2 strategy = 60 cells.

Usage:
    conda run -n Alz_face_main_analysis python scripts/experiments/run_fwd_rev_embedding.py \\
        --partition all --embedding all --classifier both --strategy both
    conda run -n Alz_face_main_analysis python scripts/experiments/run_fwd_rev_embedding.py \\
        --partition ad_vs_hc --embedding arcface --classifier logistic \\
        --strategy forward
"""
import argparse
import importlib.util
import itertools
import json
import logging
import os
import sys
from pathlib import Path

# Early sys.argv sniff: COHORT_MODE must be set in env BEFORE we import
# run_4arm_deep_dive (which reads it at module-load time, line 90).
_COHORT_MODES = {"default", "p_first_hc_all"}
for _i, _arg in enumerate(sys.argv):
    if _arg == "--cohort-mode" and _i + 1 < len(sys.argv):
        _v = sys.argv[_i + 1]
        if _v in _COHORT_MODES:
            os.environ["COHORT_MODE"] = "p_first_hc_all" if _v == "p_first_hc_all" else "default"
        break
    if _arg.startswith("--cohort-mode="):
        _v = _arg.split("=", 1)[1]
        if _v in _COHORT_MODES:
            os.environ["COHORT_MODE"] = "p_first_hc_all" if _v == "p_first_hc_all" else "default"
        break

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

ARMS_ROOT = PROJECT_ROOT / "workspace" / "arms_analysis"
EMBEDDING_FEAT_DIR = PROJECT_ROOT / "workspace" / "embedding" / "features"
ARM_B_DIR = ARMS_ROOT / "per_arm" / "arm_b"
AGES_FILE = (PROJECT_ROOT / "workspace" / "age" / "age_prediction" /
             "predicted_ages.json")

PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs", "mmse_hilo", "casi_hilo"]
EMBEDDINGS = ["arcface", "topofr", "dlib"]
CLASSIFIERS = ["logistic", "xgb"]
STRATEGIES = ["forward", "reverse"]
FEATURE_TYPES = [
    "original",
    "difference", "absolute_difference", "average",
    "relative_differences", "absolute_relative_differences",
]
VISIT_MODES = ["first", "all"]
PHOTO_MODES = ["mean", "all"]


# Module-level state (set by main); avoids threading params through every callsite.
_FEATURE_TYPE = "original"
_DROP_CORR_THRESHOLD = None  # None = disabled; float = enabled at this Pearson threshold
_DROP_CORR_METHOD = "pearson"
_PCA_COMPONENTS = None       # None = disabled; int = n_components; float<1 = variance ratio
_VISIT_MODE = "first"  # "first" = current behavior; "all" = include every qualifying visit per base_id
_PHOTO_MODE = "mean"   # "mean" = current behavior (mean over 10 photos); "all" = one row per photo
_COHORT_MODE = os.environ.get("COHORT_MODE", "default")  # "default"=p_first_hc_strict; "p_first_hc_all"
_LR_C = 1.0  # LogisticRegression C; ≠1.0 inserts `logistic_C{value}/` segment & drops cell-level classifier leaf


def _reducer_label():
    """Folder-name label for current reducer state. Mutually exclusive
    settings are validated at CLI parse time."""
    if _DROP_CORR_THRESHOLD is not None:
        return f"drop_{_DROP_CORR_THRESHOLD}"
    if _PCA_COMPONENTS is not None:
        return f"pca_{_PCA_COMPONENTS}"
    return "no_drop"


def output_dir_for(feature_type, drop_corr=None, visit_mode="first",
                    photo_mode="mean", pca_components=None,
                    cohort_mode="default", lr_C=1.0):
    """Layout (override the older 4-arg call sites at runtime via the module-
    level state). When called externally, callers can pass pca_components.

    cohort_mode='default' -> p_first_hc_strict/  (cohort default: visit=first,
                                                  photo=mean)
    cohort_mode='p_first_hc_all' -> p_first_hc_all/  (cohort default:
                                                       visit=all, photo=mean)

    Suffix `__visit_X[__photo_Y]` is only appended when (visit, photo)
    differ from the cohort's default — so p_first_hc_all + visit=all + photo=mean
    yields plain `pca_100`, not `pca_100__visit_all`.

    lr_C != 1.0 inserts `logistic_C{value}/` between cohort_dir and the
    `embedding_*classification` segment (and `cell_dir()` drops the trailing
    `logistic/` leaf). lr_C == 1.0 leaves the layout untouched.
    """
    if pca_components is not None and drop_corr is not None:
        raise ValueError("drop_corr and pca_components are mutually exclusive")
    if pca_components is not None:
        base = f"pca_{pca_components}"
    elif drop_corr is None:
        base = "no_drop"
    else:
        base = f"drop_{drop_corr}"
    if cohort_mode == "p_first_hc_all":
        default_visit, default_photo = "all", "mean"
    else:
        default_visit, default_photo = "first", "mean"
    suffix_parts = []
    if visit_mode != default_visit:
        suffix_parts.append(f"visit_{visit_mode}")
    if photo_mode != default_photo:
        suffix_parts.append(f"photo_{photo_mode}")
    suffix = ("__" + "_".join(suffix_parts)) if suffix_parts else ""
    drop_label = base + suffix
    cohort_dir = "p_first_hc_all" if cohort_mode == "p_first_hc_all" else "p_first_hc_strict"
    base_root = ARMS_ROOT / cohort_dir
    if lr_C != 1.0:
        base_root = base_root / f"logistic_C{lr_C:g}"
    if feature_type == "original":
        return base_root / "embedding_classification" / drop_label
    return (base_root / "embedding_asymmetry_classification" /
            drop_label / feature_type)


OUTPUT_DIR = output_dir_for(_FEATURE_TYPE, _DROP_CORR_THRESHOLD,
                              _VISIT_MODE, _PHOTO_MODE, _PCA_COMPONENTS,
                              _COHORT_MODE, _LR_C)


class _TorchGPUPCA:
    """sklearn-compatible GPU PCA via torch.pca_lowrank. Fixed-component only
    (n int) — variance-ratio mode falls back to sklearn (set externally).
    Stores mean / loadings on GPU; transform pulls back to numpy."""
    def __init__(self, n_components, niter=4):
        self.n_components = int(n_components)
        self.niter = niter

    def fit(self, X):
        import torch
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
        self.mean_ = Xt.mean(0, keepdim=True)
        # Clamp q to min(m, n) — sklearn PCA does this implicitly; torch.pca_lowrank errors.
        q = min(self.n_components, Xt.shape[0], Xt.shape[1])
        _, _, V = torch.pca_lowrank(Xt - self.mean_, q=q, niter=self.niter)
        self.V_ = V[:, :q].contiguous()
        self.n_components_ = q
        return self

    def transform(self, X):
        import torch
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
        return ((Xt - self.mean_) @ self.V_).cpu().numpy()


def _fit_reducer(X_train):
    """Fit the active feature reducer on X_train. Returns (reducer_or_None,
    n_kept). When both drop_corr and pca are disabled, returns (None, n_in).
    drop_corr and pca are mutually exclusive (validated at CLI parse).

    PCA backend: GPU torch.pca_lowrank for fixed n_components (int);
    sklearn fallback for variance-ratio mode (float<1).
    """
    if _DROP_CORR_THRESHOLD is not None:
        from feature_engine.selection import DropCorrelatedFeatures
        sel = DropCorrelatedFeatures(
            threshold=_DROP_CORR_THRESHOLD, method=_DROP_CORR_METHOD,
        )
        sel.fit(pd.DataFrame(X_train))
        n_kept = X_train.shape[1] - len(sel.features_to_drop_)
        return ("drop_corr", sel), n_kept
    if _PCA_COMPONENTS is not None:
        n = _PCA_COMPONENTS
        if isinstance(n, float) and 0 < n < 1:
            # variance-ratio mode → keep sklearn (torch needs explicit q)
            from sklearn.decomposition import PCA
            sel = PCA(n_components=float(n), random_state=0)
            sel.fit(X_train)
            return ("pca", sel), int(sel.n_components_)
        sel = _TorchGPUPCA(n_components=int(n))
        sel.fit(X_train)
        return ("pca", sel), int(sel.n_components_)
    return None, X_train.shape[1]


def _apply_reducer(reducer, X):
    """Apply a fitted reducer. No-op when reducer is None."""
    if reducer is None:
        return X
    kind, sel = reducer
    if kind == "drop_corr":
        return sel.transform(pd.DataFrame(X)).to_numpy()
    if kind == "pca":
        return sel.transform(X)
    raise ValueError(f"unknown reducer kind: {kind}")


# Backward-compat aliases (kept so older callers in this module — and the
# dropcorr sweep helper scripts — still resolve). They forward to the new
# generic reducer interface.
_fit_drop_correlated = _fit_reducer
_apply_drop_correlated = _apply_reducer

# Expected matched pair counts per arm_b (sanity check; raises if mismatch)
EXPECTED_PAIRS = {
    "ad_vs_hc": 180,
    "ad_vs_nad": 169,
    "ad_vs_acs": 29,
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Reuse helpers from existing arm scripts
# ============================================================

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_arm_a = _load_module("arm_a", PROJECT_ROOT / "scripts" / "experiments" /
                      "run_arm_a_ad_vs_hc.py")
_hilo = _load_module("mmse_hilo", PROJECT_ROOT / "scripts" / "experiments" /
                     "run_mmse_hilo_standalone.py")
_grid = _load_module("deep_dive", PROJECT_ROOT / "scripts" / "experiments" /
                     "run_4arm_deep_dive.py")

load_embedding_mean = _arm_a.load_embedding_mean  # legacy: original-only


def load_embedding(ids, model, feature_type="original", photo_mode=None):
    """Load per-subject embedding feature vectors from
    workspace/embedding/features/<model>/<feature_type>/<sid>.npy.

    photo_mode='mean' (default): mean-pool over the 10 photos → 1 row per ID.
    photo_mode='all'           : keep all 10 photos → 10 rows per ID, with
                                  an extra `photo_idx` column.

    The column `subject_id` always equals the source ID (visit-level).
    GroupKFold uses base_id (parsed from ID), so multiple rows per ID stay
    in the same fold automatically.
    """
    if photo_mode is None:
        photo_mode = _PHOTO_MODE
    emb_dir = EMBEDDING_FEAT_DIR / model / feature_type
    rows = []
    for sid in ids:
        npy = emb_dir / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy, allow_pickle=True)
        if a.dtype == object:
            a = list(a.item().values())[0]
        if a.ndim != 2:
            # Already a single 1-D vector, no per-photo dim available.
            rows.append({"subject_id": sid, "photo_idx": -1,
                         **{f"{model}__dim_{i}": float(a[i])
                            for i in range(a.shape[0])}})
            continue
        if photo_mode == "mean":
            vec = a.mean(axis=0)
            rows.append({"subject_id": sid, "photo_idx": -1,
                         **{f"{model}__dim_{i}": float(vec[i])
                            for i in range(vec.shape[0])}})
        else:  # photo_mode == "all"
            for k in range(a.shape[0]):
                rows.append({"subject_id": sid, "photo_idx": int(k),
                             **{f"{model}__dim_{i}": float(a[k, i])
                                for i in range(a.shape[1])}})
    return pd.DataFrame(rows) if rows else None
bootstrap_auc_ci = _arm_a.bootstrap_auc_ci
match_1to1 = _hilo.match_1to1
split_by_metric_median = _hilo.split_by_metric_median
load_p_demographics = _hilo.load_p_demographics
select_visit = _hilo.select_visit
build_cohort_ad_vs_HCgroup = _grid.build_cohort_ad_vs_HCgroup


# ============================================================
# Cohort builders (5 partitions)
# ============================================================

_PRED_AGES_CACHE = None


def _load_pred_ages():
    """Load predicted_ages.json (cached). Used for the matched-pair filter
    that mirrors arm_b's run_arm_b_ad_vs_hcgroups.py:143-147 — drop pairs
    whose either side lacks a predicted_age, so matched n_pairs aligns with
    arm_b's reference (HC=180, NAD=169, ACS=29 / hi-lo equivalent)."""
    global _PRED_AGES_CACHE
    if _PRED_AGES_CACHE is None:
        with open(AGES_FILE) as f:
            _PRED_AGES_CACHE = json.load(f)
    return _PRED_AGES_CACHE


def _filter_pairs_by_pred_age(matched, pairs):
    """Drop pairs where either minor_id or major_id lacks a predicted_age.
    Mirrors arm_b's run_arm_b_ad_vs_hcgroups.py:143-147."""
    if pairs is None or len(pairs) == 0:
        return matched, pairs
    pred_ages = _load_pred_ages()
    keep = pairs["minor_id"].isin(pred_ages) & pairs["major_id"].isin(pred_ages)
    pairs = pairs[keep].copy().reset_index(drop=True)
    kept_ids = set(pairs["minor_id"]).union(pairs["major_id"])
    matched = matched[matched["ID"].isin(kept_ids)].copy().reset_index(drop=True)
    return matched, pairs


def _expand_to_all_visits_ad_hc(full_first, hc_source):
    """For ad_vs_hc family: build an all-visits version of the full training
    cohort by re-applying per-visit filters (Global_CDR>=0.5 for AD; strict
    HC criteria for HC) and restricting to base_ids that are in `full_first`.

    Each visit row carries the same `label` / `group` as the first-visit row
    (the cohort's identity is fixed by the first-visit selection — we only
    add more visits as additional training samples).
    """
    import os as _os
    DEMOGRAPHICS_DIR = _grid.DEMOGRAPHICS_DIR
    HC_SOURCE_MODE = _grid.HC_SOURCE_MODE

    frames = []
    groups_to_load = ["P", "NAD"]
    if HC_SOURCE_MODE != "EACS":
        groups_to_load.append("ACS")
    for grp in groups_to_load:
        df = pd.read_csv(DEMOGRAPHICS_DIR / f"{grp}.csv")
        if "ID" not in df.columns:
            for col in df.columns:
                if col in ("ACS", "NAD"):
                    df = df.rename(columns={col: "ID"})
                    break
        df["group"] = grp
        df["Source"] = "internal"
        frames.append(df)
    if HC_SOURCE_MODE in ("ACS_ext", "EACS"):
        df_e = pd.read_csv(DEMOGRAPHICS_DIR / "EACS.csv")
        eacs_sources_env = _os.environ.get("EACS_SOURCES", "").strip()
        if eacs_sources_env:
            wanted = {s.strip() for s in eacs_sources_env.split(",") if s.strip()}
            if "Source" in df_e.columns:
                df_e = df_e[df_e["Source"].isin(wanted)].copy()
        df_e["group"] = "ACS"
        frames.append(df_e)
    demo = pd.concat(frames, ignore_index=True)
    if "Source" not in demo.columns:
        demo["Source"] = "internal"
    demo["Source"] = demo["Source"].fillna("internal")
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["Global_CDR"] = pd.to_numeric(demo.get("Global_CDR"), errors="coerce")
    demo["MMSE"] = pd.to_numeric(demo.get("MMSE"), errors="coerce")
    demo["base_id"] = demo["ID"].str.extract(r"^(.+)-\d+$")
    demo["visit"] = demo["ID"].str.extract(r"-(\d+)$").astype(float)

    ad_all = demo[(demo["group"] == "P") & (demo["Global_CDR"] >= 0.5)
                  & demo["Age"].notna()].copy()
    ad_all["label"] = 1
    hc_all = _grid._strict_hc_filter_all_visits(demo, hc_source)
    hc_all = hc_all[hc_all["Age"].notna()].copy()
    full_all = pd.concat([ad_all, hc_all], ignore_index=True)
    full_all["mmse_group"] = np.nan

    # Restrict to base_ids that are in the original (first-visit) cohort, and
    # propagate that cohort's `label` / `group` to every visit row of the same
    # base_id (in case an ad-side visit has CDR<0.5 etc., we keep the AD label).
    keep_base_ids = set(full_first["base_id"])
    full_all = full_all[full_all["base_id"].isin(keep_base_ids)].copy()
    bid_to_label = full_first.set_index("base_id")["label"].to_dict()
    bid_to_group = full_first.set_index("base_id")["group"].to_dict()
    full_all["label"] = full_all["base_id"].map(bid_to_label)
    full_all["group"] = full_all["base_id"].map(bid_to_group)
    full_all = full_all.dropna(subset=["label"]).copy()
    full_all["label"] = full_all["label"].astype(int)
    return full_all.reset_index(drop=True)


def _expand_matched_to_all_visits(matched_first, full_all):
    """Add all qualifying visits per base_id to the matched cohort, propagating
    `pair_id` (always) and `label` / `group` (if present) from the first-visit
    row. The first-visit IDs in `matched_first` are kept (they remain the
    canonical pair members for Wilcoxon); additional visits are appended for
    richer training signal.
    """
    matched_first = matched_first.copy()
    if "base_id" not in matched_first.columns:
        matched_first["base_id"] = matched_first["ID"].str.extract(
            r"^(.+)-\d+$"
        )[0]
    keep_base_ids = set(matched_first["base_id"])
    extra = full_all[full_all["base_id"].isin(keep_base_ids)].copy()
    # Drop visits that are already in matched_first (avoid duplicates).
    extra = extra[~extra["ID"].isin(set(matched_first["ID"]))].copy()
    extra["pair_id"] = extra["base_id"].map(
        matched_first.set_index("base_id")["pair_id"].to_dict()
    )
    if "label" in matched_first.columns:
        extra["label"] = extra["base_id"].map(
            matched_first.set_index("base_id")["label"].to_dict()
        ).astype(int)
    if "group" in matched_first.columns:
        extra["group"] = extra["base_id"].map(
            matched_first.set_index("base_id")["group"].to_dict()
        )
    common_cols = [c for c in matched_first.columns if c in extra.columns]
    expanded = pd.concat(
        [matched_first[common_cols], extra[common_cols]], ignore_index=True,
    )
    return expanded


def _build_ad_vs_hcgroup(hc_source):
    """ad_vs_hc / ad_vs_nad / ad_vs_acs.

    Returns (full_cohort, matched_cohort, pairs_df).
    full from arm='A' (no matching), matched from arm='B' (1:1 age) with
    the predicted_age post-hoc filter applied so n_pairs matches arm_b.

    When _VISIT_MODE == "all": full is expanded to all qualifying visits per
    base_id, and matched is similarly expanded (with pair_id/label/group
    propagated from the first-visit row). pairs_df is unchanged (still
    first-visit pair members for the paired Wilcoxon).
    """
    full_first, _ = build_cohort_ad_vs_HCgroup(hc_source, arm="A")
    matched, pairs = build_cohort_ad_vs_HCgroup(hc_source, arm="B")
    matched, pairs = _filter_pairs_by_pred_age(matched, pairs)
    if _VISIT_MODE == "all":
        full = _expand_to_all_visits_ad_hc(full_first, hc_source)
        matched = _expand_matched_to_all_visits(matched, full)
        return full, matched, pairs
    return full_first, matched, pairs


def _expand_hilo_to_all_visits(cohort_first, metric):
    """Expand a first-visit hi-lo cohort to all qualifying visits.
    Each visit row keeps the same `label` / hi-lo group from the first-visit
    row (the hi-lo split is defined on first-visit metric only)."""
    metric_low = metric.lower()
    group_col = f"{metric_low}_group"
    df = load_p_demographics()
    df = df[(df["Global_CDR"] >= 0.5)].copy()
    df_all = df.dropna(subset=[metric, "Age"]).copy()
    keep_base_ids = set(cohort_first["base_id"])
    df_all = df_all[df_all["base_id"].isin(keep_base_ids)].copy()
    bid_to_label = cohort_first.set_index("base_id")["label"].to_dict()
    bid_to_group = cohort_first.set_index("base_id")[group_col].to_dict()
    df_all["label"] = df_all["base_id"].map(bid_to_label).astype(int)
    df_all[group_col] = df_all["base_id"].map(bid_to_group)
    return df_all.reset_index(drop=True)


def _build_metric_hilo(metric):
    """mmse_hilo / casi_hilo (AD-only, P group with Global_CDR>=0.5).

    Splits by median, matches 1:1 by age (caliper 2y), then drops pairs
    where either side lacks predicted_age (parity with arm_b convention).
    Labels: high=1, low=0.

    When _VISIT_MODE == "all": cohort + matched are expanded to all qualifying
    visits per base_id (label / group propagated from first-visit row).
    """
    metric_low = metric.lower()
    group_col = f"{metric_low}_group"

    df = load_p_demographics()
    df = df[(df["Global_CDR"] >= 0.5)].copy()
    cohort = select_visit(df, visit_selection="first", metric=metric)
    cohort, _median = split_by_metric_median(
        cohort, metric=metric, group_col=group_col,
    )
    cohort["label"] = (cohort[group_col] == "high").astype(int)

    matched, pairs, _labels = match_1to1(
        cohort, caliper=2.0, seed=42, metric=metric, group_col=group_col,
    )
    # match_1to1 returns matched without label / base_id; merge them back
    extra = cohort[["ID", "base_id", "label", group_col]].drop_duplicates("ID")
    matched = matched.merge(extra, on="ID", how="left",
                            suffixes=("", "_p"))
    matched = matched.drop(columns=[c for c in matched.columns if c.endswith("_p")])
    matched, pairs = _filter_pairs_by_pred_age(matched, pairs)

    if _VISIT_MODE == "all":
        cohort_all = _expand_hilo_to_all_visits(cohort, metric)
        matched_all = _expand_matched_to_all_visits(matched, cohort_all)
        return cohort_all, matched_all, pairs
    return cohort, matched, pairs


def build_partition_cohort(partition):
    """Returns (full, matched, pairs, keep_groups).

    full / matched DataFrames must have at least: ID, base_id, label, group.
    matched must additionally have pair_id (from match_1to1).

    keep_groups is an optional set of group strings used at evaluation time
    to subset the cohort. ad_vs_nad and ad_vs_acs reuse the ad_vs_hc training
    cohort + matching, then filter the predictions by group:
        ad_vs_nad → keep {"P", "NAD"}
        ad_vs_acs → keep {"P", "ACS"}
    so the same model's predictions are evaluated on different sub-populations.
    """
    if partition == "ad_vs_hc":
        full, matched, pairs = _build_ad_vs_hcgroup("HC")
        return full, matched, pairs, None
    if partition == "ad_vs_nad":
        full, matched, pairs = _build_ad_vs_hcgroup("HC")
        return full, matched, pairs, {"P", "NAD"}
    if partition == "ad_vs_acs":
        full, matched, pairs = _build_ad_vs_hcgroup("HC")
        return full, matched, pairs, {"P", "ACS"}
    if partition == "mmse_hilo":
        return (*_build_metric_hilo("MMSE"), None)
    if partition == "casi_hilo":
        return (*_build_metric_hilo("CASI"), None)
    raise ValueError(f"unknown partition: {partition}")


# ============================================================
# Feature matrix
# ============================================================

def build_feature_matrix(cohort, embedding):
    """Returns (X, y, base_ids, ids). Drops subjects without .npy.

    Row count depends on (visit_mode, photo_mode):
      - first / mean: 1 row per base_id (current default)
      - first / all : 10 rows per base_id (one per photo)
      - all   / mean: 1 row per (base_id, visit)
      - all   / all : 10 rows per (base_id, visit)

    GroupKFold by `base_ids` keeps every row of the same subject in the same
    fold regardless of mode, so there's no leakage between train and val.
    """
    emb_df = load_embedding(cohort["ID"].tolist(), embedding,
                              feature_type=_FEATURE_TYPE)
    if emb_df is None or len(emb_df) == 0:
        raise RuntimeError(f"no embedding loaded for {embedding}")
    feat_cols = [c for c in emb_df.columns if c.startswith(f"{embedding}__dim_")]
    keep = ["ID", "base_id", "label"]
    merged = cohort[keep].merge(emb_df, left_on="ID", right_on="subject_id",
                                  how="inner")
    merged = merged.dropna(subset=feat_cols)
    X = merged[feat_cols].to_numpy(dtype=float)
    y = merged["label"].to_numpy(dtype=int)
    base_ids = merged["base_id"].to_numpy()
    ids = merged["ID"].to_numpy()
    n_unique_subjects_in = cohort["base_id"].nunique() \
        if "base_id" in cohort.columns else cohort["ID"].nunique()
    n_unique_subjects_kept = pd.unique(base_ids).size
    return X, y, base_ids, ids, n_unique_subjects_in - n_unique_subjects_kept


# ============================================================
# Classifier factory
# ============================================================

def make_classifier(name, seed=42):
    if name == "logistic":
        # class_weight balanced for imbalanced partitions; lbfgs default.
        # C read from module-level _LR_C (set by main() from --lr-C).
        return LogisticRegression(
            C=_LR_C, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, verbosity=0,
            tree_method="hist", device="cuda",
        )
    raise ValueError(f"unknown classifier: {name}")


def needs_scaler(classifier):
    return classifier == "logistic"


# ============================================================
# Metrics
# ============================================================

def compute_clf_metrics(y_true, y_score, threshold=0.5, n_bootstrap=1000,
                         seed=42):
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
    ci_low, ci_high = (bootstrap_auc_ci(y_true, y_score, n=n_bootstrap, seed=seed)
                       if len(np.unique(y_true)) > 1 else (float("nan"),
                                                           float("nan")))
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


def paired_wilcoxon_by_pair(matched_with_score):
    """matched_with_score must have [pair_id, label, y_score].

    Returns dict: W, p, n_pairs, mean_diff (label1 − label0).
    """
    pos = (matched_with_score[matched_with_score["label"] == 1]
           .sort_values("pair_id")["y_score"].to_numpy(dtype=float))
    neg = (matched_with_score[matched_with_score["label"] == 0]
           .sort_values("pair_id")["y_score"].to_numpy(dtype=float))
    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    if n < 2 or np.allclose(pos, neg):
        return {"W": float("nan"), "p": float("nan"),
                "n_pairs": int(n), "mean_diff": float("nan")}
    res = stats.wilcoxon(pos, neg, zero_method="wilcox", alternative="two-sided")
    return {
        "W": float(res.statistic), "p": float(res.pvalue),
        "n_pairs": int(n),
        "mean_diff": float(np.mean(pos - neg)),
    }


# ============================================================
# Forward strategy
# ============================================================

def _filter_pairs_complete(matched_df):
    """Drop pairs where either side was filtered out (so each kept pair has 2 rows)."""
    counts = matched_df.drop_duplicates(["pair_id", "label"])["pair_id"].value_counts()
    complete = counts[counts == 2].index
    return matched_df[matched_df["pair_id"].isin(complete)].copy()


def _aggregate_to_subject(scores_df, score_cols):
    """Collapse multi-row-per-(base_id) scores to one row per base_id (mean of
    score_cols, first of y_true). Idempotent when there's already one row per
    base_id (default first/mean mode), so safe to call unconditionally.
    """
    score_cols = [c for c in score_cols if c in scores_df.columns]
    if "base_id" not in scores_df.columns:
        scores_df = scores_df.copy()
        scores_df["base_id"] = (scores_df["ID"]
                                  .astype(str)
                                  .str.extract(r"^(.+)-\d+$")[0])
    agg_dict = {c: "mean" for c in score_cols}
    if "y_true" in scores_df.columns:
        agg_dict["y_true"] = "first"
    if "ID" in scores_df.columns:
        agg_dict["ID"] = "first"  # representative ID (first-seen visit per base_id)
    if "fold" in scores_df.columns:
        agg_dict["fold"] = "first"
    if "in_matched" in scores_df.columns:
        agg_dict["in_matched"] = "max"  # any row in_matched → subject in_matched
    return scores_df.groupby("base_id", as_index=False).agg(agg_dict)


def run_forward(full_cohort, matched_cohort, embedding, classifier,
                n_folds=10, seed=42, keep_groups=None):
    """Train on full_cohort 10-fold; evaluate on full + matched (optionally
    subset by group). When keep_groups is set, training is unchanged but
    the evaluation set is filtered to subjects whose `group` ∈ keep_groups
    (and matched pairs that lose a side are dropped)."""
    X, y, base_ids, ids, n_dropped = build_feature_matrix(full_cohort, embedding)
    n_groups = len(np.unique(base_ids))
    k = min(n_folds, n_groups)
    if k < 2:
        raise RuntimeError(f"too few groups ({n_groups}) for any fold")
    gkf = GroupKFold(n_splits=k)

    oof_score = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)
    n_kept_per_fold = []
    for fold_idx, (tri, tei) in enumerate(gkf.split(X, y, groups=base_ids)):
        if needs_scaler(classifier):
            scaler = StandardScaler().fit(X[tri])
            Xtr, Xte = scaler.transform(X[tri]), scaler.transform(X[tei])
        else:
            Xtr, Xte = X[tri], X[tei]
        sel, n_kept = _fit_drop_correlated(Xtr)
        Xtr = _apply_drop_correlated(sel, Xtr)
        Xte = _apply_drop_correlated(sel, Xte)
        n_kept_per_fold.append(n_kept)
        clf = make_classifier(classifier, seed=seed)
        clf.fit(Xtr, y[tri])
        oof_score[tei] = clf.predict_proba(Xte)[:, 1]
        oof_fold[tei] = fold_idx

    oof_df = pd.DataFrame({
        "ID": ids, "base_id": base_ids, "y_true": y,
        "y_score": oof_score, "fold": oof_fold,
    })

    # Aggregate row-level OOF (multiple rows per base_id when visit/photo mode
    # is "all") back to subject-level scores for matched-pair eval (matching
    # is defined at first-visit subject level so subject-level scores there).
    oof_subj = _aggregate_to_subject(oof_df, score_cols=["y_score"])

    # Full-cohort metrics: visit-level. Counts every visit row that entered
    # training (1 row per visit when photo=mean, 10 rows when photo=all).
    # `n_pos` / `n_neg` here count visits, matching the visit_level OOF used
    # to derive AUC/CM/etc.
    metrics_full = compute_clf_metrics(
        oof_df["y_true"].to_numpy(int),
        oof_df["y_score"].to_numpy(float),
        seed=seed,
    )

    # Matched-cohort eval: optionally subset by group (ad_vs_nad / ad_vs_acs).
    # Collapse the (possibly visit-expanded) matched cohort to one row per
    # base_id by keeping the first-visit row (canonical pair member).
    matched_first = matched_cohort.sort_values(
        ["base_id"] + (["visit"] if "visit" in matched_cohort.columns else [])
    ).drop_duplicates("base_id", keep="first")
    if keep_groups is not None:
        matched_eval = matched_first[matched_first["group"].isin(keep_groups)]
        matched_eval = _filter_pairs_complete(matched_eval)
    else:
        matched_eval = matched_first

    matched_min = matched_eval[["base_id", "ID", "pair_id", "label"]].copy()
    oof_matched = matched_min.merge(oof_subj[["base_id", "y_score"]],
                                      on="base_id", how="inner")
    metrics_matched = compute_clf_metrics(oof_matched["label"].to_numpy(int),
                                            oof_matched["y_score"].to_numpy(float),
                                            seed=seed)
    paired = paired_wilcoxon_by_pair(oof_matched)

    oof_df["in_matched"] = oof_df["base_id"].isin(matched_eval["base_id"]).astype(int)

    drop_corr_info = {
        "reducer": _reducer_label(),
        "drop_corr_threshold": _DROP_CORR_THRESHOLD,
        "drop_corr_method": _DROP_CORR_METHOD if _DROP_CORR_THRESHOLD else None,
        "pca_components": _PCA_COMPONENTS,
        "n_features_input": int(X.shape[1]),
        "n_features_kept_per_fold": [int(n) for n in n_kept_per_fold],
        "n_rows_input": int(X.shape[0]),
        "n_unique_subjects": int(pd.unique(base_ids).size),
        "visit_mode": _VISIT_MODE,
        "photo_mode": _PHOTO_MODE,
    }

    return (oof_df, oof_subj, metrics_full, metrics_matched, paired, n_folds,
            n_dropped, k, matched_eval, drop_corr_info)


def plot_paired_scatter(matched_with_score, partition, out_path):
    """Side-by-side stripplot + paired lines: high vs low."""
    pos = (matched_with_score[matched_with_score["label"] == 1]
           .sort_values("pair_id")["y_score"].to_numpy(float))
    neg = (matched_with_score[matched_with_score["label"] == 0]
           .sort_values("pair_id")["y_score"].to_numpy(float))
    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    fig, ax = plt.subplots(figsize=(5, 5))
    rng = np.random.RandomState(0)
    jx0 = rng.normal(0.0, 0.04, size=n)
    jx1 = rng.normal(1.0, 0.04, size=n)
    for i in range(n):
        ax.plot([jx0[i], jx1[i]], [neg[i], pos[i]],
                color="gray", alpha=0.25, linewidth=0.7)
    ax.scatter(jx0, neg, color="#4C72B0", s=18, label="label=0 (low/HC-side)")
    ax.scatter(jx1, pos, color="#C44E52", s=18, label="label=1 (high/AD-side)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["label=0", "label=1"])
    ax.set_ylabel("OOF prediction score")
    ax.set_title(f"Forward — paired prediction scores\n{partition}, n_pairs={n}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Reverse strategy
# ============================================================

def run_reverse(full_cohort, matched_cohort, embedding, classifier,
                n_folds=10, seed=42, keep_groups=None):
    """Reverse strategy with two model variants:

    (a) Ensemble: 10-fold GroupKFold on matched cohort → 10 fold-models, mean
        their predict_proba over the full cohort. (Existing.)
    (b) Single: train one model on the full matched cohort (no CV split),
        apply to the full cohort. Cleaner for unmatched evaluation since
        every unmatched subject is a true held-out test for it.

    When keep_groups is set, training is unchanged but evaluation cohorts
    (full + matched) are filtered to subjects whose `group` ∈ keep_groups.

    Reported metrics:
      - matched_oof:           ensemble OOF on matched (validation domain)
      - full_ensemble:         ensemble on full cohort (matched ∪ unmatched)
      - unmatched_ensemble:    ensemble on unmatched-only
      - unmatched_single:      single-model on unmatched-only
      - matched_single_train:  single-model on matched-only (training-domain
                                 reference, expected to be over-optimistic)
    """
    Xm, ym, gm, idsm, n_dropped_m = build_feature_matrix(matched_cohort, embedding)
    Xf, yf, _gf, idsf, n_dropped_f = build_feature_matrix(full_cohort, embedding)

    n_groups_m = len(np.unique(gm))
    k = min(n_folds, n_groups_m)
    if k < 2:
        raise RuntimeError(f"too few groups in matched ({n_groups_m})")
    gkf = GroupKFold(n_splits=k)

    # ---- (a) 10-fold ensemble ----
    oof_m = np.full(len(ym), np.nan, dtype=float)
    full_preds = []
    n_kept_per_fold = []
    for fold_idx, (tri, tei) in enumerate(gkf.split(Xm, ym, groups=gm)):
        if needs_scaler(classifier):
            scaler = StandardScaler().fit(Xm[tri])
            Xm_tr = scaler.transform(Xm[tri])
            Xm_te = scaler.transform(Xm[tei])
            Xf_t = scaler.transform(Xf)
        else:
            Xm_tr, Xm_te, Xf_t = Xm[tri], Xm[tei], Xf
        sel, n_kept = _fit_drop_correlated(Xm_tr)
        Xm_tr = _apply_drop_correlated(sel, Xm_tr)
        Xm_te = _apply_drop_correlated(sel, Xm_te)
        Xf_t = _apply_drop_correlated(sel, Xf_t)
        n_kept_per_fold.append(n_kept)
        clf = make_classifier(classifier, seed=seed)
        clf.fit(Xm_tr, ym[tri])
        oof_m[tei] = clf.predict_proba(Xm_te)[:, 1]
        full_preds.append(clf.predict_proba(Xf_t)[:, 1])

    full_score_ensemble = np.mean(np.stack(full_preds, axis=0), axis=0)

    # ---- (b) Single model trained on all matched ----
    if needs_scaler(classifier):
        scaler_single = StandardScaler().fit(Xm)
        Xm_all_t = scaler_single.transform(Xm)
        Xf_single_t = scaler_single.transform(Xf)
    else:
        Xm_all_t, Xf_single_t = Xm, Xf
    sel_single, n_kept_single = _fit_drop_correlated(Xm_all_t)
    Xm_all_t = _apply_drop_correlated(sel_single, Xm_all_t)
    Xf_single_t = _apply_drop_correlated(sel_single, Xf_single_t)
    clf_single = make_classifier(classifier, seed=seed)
    clf_single.fit(Xm_all_t, ym)
    full_score_single = clf_single.predict_proba(Xf_single_t)[:, 1]

    # ---- Aggregate row-level predictions to subject-level (one row per base_id).
    # No-op when each base_id already has one row (default first/mean mode).
    matched_oof_df = pd.DataFrame({
        "ID": idsm, "base_id": gm, "y_true": ym, "y_score": oof_m,
    })
    matched_oof_subj = _aggregate_to_subject(matched_oof_df, score_cols=["y_score"])

    full_df_rows = pd.DataFrame({
        "ID": idsf, "base_id": _gf, "y_true": yf,
        "score_ensemble": full_score_ensemble,
        "score_single": full_score_single,
    })
    full_subj = _aggregate_to_subject(
        full_df_rows, score_cols=["score_ensemble", "score_single"],
    )

    # ---- Subset masks ----
    matched_base_ids = set(pd.unique(gm))
    # Visit-level masks (on full_df_rows)
    full_df_rows["in_matched"] = full_df_rows["base_id"].isin(
        matched_base_ids
    ).astype(int)
    in_matched_visit = full_df_rows["in_matched"].to_numpy(bool)
    n_unmatched_visits = int((~in_matched_visit).sum())
    # Subject-level masks (kept for matched_single_train + scores CSV)
    full_subj["in_matched"] = full_subj["base_id"].isin(
        matched_base_ids
    ).astype(int)

    # ---- Optional eval-time group filter (derived views) ----
    if keep_groups is not None:
        bid_to_group_m = matched_cohort.set_index("base_id")["group"].to_dict()
        keep_m_subj = np.array([
            bid_to_group_m.get(b) in keep_groups
            for b in matched_oof_subj["base_id"]
        ])
    else:
        keep_m_subj = np.ones(len(matched_oof_subj), dtype=bool)
    eval_y_m_subj = matched_oof_subj["y_true"].to_numpy(int)[keep_m_subj]
    eval_score_m_subj = matched_oof_subj["y_score"].to_numpy(float)[keep_m_subj]

    # ---- Metrics ----
    # matched_oof: subject-level (matching is defined at subject level).
    metrics_matched_oof = (
        compute_clf_metrics(eval_y_m_subj, eval_score_m_subj, seed=seed)
        if len(eval_y_m_subj) >= 5 and len(np.unique(eval_y_m_subj)) > 1 else None
    )
    # full / unmatched: visit-level (count every visit row that entered training).
    yf_visit = full_df_rows["y_true"].to_numpy(int)
    full_score_ens_visit = full_df_rows["score_ensemble"].to_numpy(float)
    full_score_single_visit = full_df_rows["score_single"].to_numpy(float)
    metrics_full_ensemble = compute_clf_metrics(yf_visit, full_score_ens_visit,
                                                  seed=seed)

    metrics_unmatched_ensemble = None
    metrics_unmatched_single = None
    metrics_matched_single_train = None
    if n_unmatched_visits >= 5 and \
       len(np.unique(yf_visit[~in_matched_visit])) > 1:
        metrics_unmatched_ensemble = compute_clf_metrics(
            yf_visit[~in_matched_visit],
            full_score_ens_visit[~in_matched_visit], seed=seed,
        )
        metrics_unmatched_single = compute_clf_metrics(
            yf_visit[~in_matched_visit],
            full_score_single_visit[~in_matched_visit], seed=seed,
        )
    # matched_single_train: subject-level (parallel to matched_oof so the two
    # are directly comparable; matched eval population stays subject-level).
    yf_subj = full_subj["y_true"].to_numpy(int)
    full_score_single_subj = full_subj["score_single"].to_numpy(float)
    if keep_groups is not None:
        bid_to_group_all = matched_cohort.set_index("base_id")["group"].to_dict()
        matched_keep_base = {b for b in matched_base_ids
                              if bid_to_group_all.get(b) in keep_groups}
    else:
        matched_keep_base = matched_base_ids
    mask_matched_eval = np.array([
        b in matched_keep_base for b in full_subj["base_id"]
    ])
    if mask_matched_eval.sum() >= 5 and \
       len(np.unique(yf_subj[mask_matched_eval])) > 1:
        metrics_matched_single_train = compute_clf_metrics(
            yf_subj[mask_matched_eval],
            full_score_single_subj[mask_matched_eval], seed=seed,
        )

    # scores CSV: subject-level (in_matched reflects matched-eval population
    # if keep_groups was set; otherwise reflects whether the subject's base_id
    # is in the matched training pool).
    full_df = full_subj.copy()
    if keep_groups is not None:
        full_df["in_matched"] = full_df["base_id"].isin(matched_keep_base).astype(int)
    full_df = full_df[["ID", "y_true", "score_ensemble", "score_single",
                        "in_matched"]]

    drop_corr_info = {
        "reducer": _reducer_label(),
        "drop_corr_threshold": _DROP_CORR_THRESHOLD,
        "drop_corr_method": _DROP_CORR_METHOD if _DROP_CORR_THRESHOLD else None,
        "pca_components": _PCA_COMPONENTS,
        "n_features_input": int(Xm.shape[1]),
        "n_features_kept_per_fold": [int(n) for n in n_kept_per_fold],
        "n_features_kept_single": int(n_kept_single),
        "n_rows_matched": int(Xm.shape[0]),
        "n_rows_full": int(Xf.shape[0]),
        "n_unique_subjects_matched": int(pd.unique(gm).size),
        "n_unique_subjects_full": int(pd.unique(_gf).size),
        "visit_mode": _VISIT_MODE,
        "photo_mode": _PHOTO_MODE,
    }

    # Visit-level scores DataFrame (preserved so post-hoc visit-level metric
    # recomputation doesn't require re-running the reverse pipeline).
    full_df_visits = full_df_rows[
        ["ID", "base_id", "y_true", "score_ensemble", "score_single",
         "in_matched"]
    ].copy()

    return {
        "scores_df": full_df,
        "scores_df_visits": full_df_visits,
        "k_folds_used": k,
        "n_dropped_no_emb_matched": n_dropped_m,
        "n_dropped_no_emb_full": n_dropped_f,
        "n_unmatched": n_unmatched_visits,
        "metrics_matched_oof": metrics_matched_oof,
        "metrics_full_ensemble": metrics_full_ensemble,
        "metrics_unmatched_ensemble": metrics_unmatched_ensemble,
        "metrics_unmatched_single": metrics_unmatched_single,
        "metrics_matched_single_train": metrics_matched_single_train,
        "drop_corr_info": drop_corr_info,
    }


# ============================================================
# Per-cell driver
# ============================================================

def cell_dir(partition, embedding, classifier, strategy):
    """Layout: embedding_classification/{fwd,rev}/<partition>/<embedding>/<classifier>/.

    When `_LR_C != 1.0`, OUTPUT_DIR already encodes the LR C value via the
    `logistic_C{value}/` segment (see output_dir_for), so the trailing
    `<classifier>/` leaf is omitted (the whole subtree is LR by construction).
    """
    bucket = "fwd" if strategy == "forward" else "rev"
    base = OUTPUT_DIR / bucket / partition / embedding
    if _LR_C != 1.0:
        return base
    return base / classifier


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def assert_matched_size(partition, matched_cohort, is_derived=False):
    n_pairs = matched_cohort["pair_id"].nunique() \
        if "pair_id" in matched_cohort.columns else len(matched_cohort) // 2
    if is_derived:
        return n_pairs  # derived views reuse ad_vs_hc matching, skip check
    expected = EXPECTED_PAIRS.get(partition)
    if expected is not None and n_pairs != expected:
        logger.warning(
            f"{partition}: matched n_pairs={n_pairs} != expected {expected} "
            f"(arm_b reference). Continuing anyway."
        )
    return n_pairs


def _write_cohort_files(out, matched, pairs):
    """Write matched_cohort.csv and matched_pairs.csv in cell dir (idempotent)."""
    out.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out / "matched_cohort.csv", index=False)
    if pairs is not None and not (out / "matched_pairs.csv").exists():
        pairs.to_csv(out / "matched_pairs.csv", index=False)


def run_cell(partition, embedding, classifier, strategy, n_folds=10, seed=42):
    logger.info(f"=== {partition} / {embedding} / {classifier} / {strategy} ===")

    full, matched, pairs, keep_groups = build_partition_cohort(partition)
    n_pairs = assert_matched_size(partition, matched,
                                    is_derived=keep_groups is not None)
    n_subj_full = full["base_id"].nunique() if "base_id" in full.columns \
        else full["ID"].nunique()
    cohort_size_msg = f"full rows={len(full)} (n_subj={n_subj_full})"
    if keep_groups is not None:
        logger.info(f"  {cohort_size_msg}, matched n_pairs={n_pairs}, "
                    f"derived view (keep_groups={sorted(keep_groups)})")
    else:
        logger.info(f"  {cohort_size_msg}, matched n_pairs={n_pairs}")

    summary_rows = []

    if strategy in ("forward", "both"):
        out = cell_dir(partition, embedding, classifier, "forward")
        _write_cohort_files(out, matched, pairs)
        (oof_df, oof_subj, m_full, m_matched, paired, n_folds_used, n_dropped, k,
         matched_eval, drop_corr_info) = run_forward(
            full, matched, embedding, classifier,
            n_folds=n_folds, seed=seed, keep_groups=keep_groups,
        )
        oof_df.to_csv(out / "forward_oof_scores.csv", index=False)
        # Subject-level aggregated scores (mean across visit/photo rows of the
        # same base_id). Same as oof_df in the default first/mean mode.
        oof_subj.to_csv(out / "forward_oof_scores_subject.csv", index=False)
        write_json(out / "forward_matched_metrics.json", {
            "partition": partition, "embedding": embedding,
            "classifier": classifier, "strategy": "forward",
            **({"lr_C": _LR_C} if classifier == "logistic" else {}),
            "k_folds_used": k, "n_dropped_no_emb": n_dropped,
            "n_pairs_expected": EXPECTED_PAIRS.get(partition),
            "derived_view": (sorted(keep_groups)
                             if keep_groups is not None else None),
            "drop_corr_info": drop_corr_info,
            "metrics_full_cohort": m_full,
            "metrics_matched_subset": m_matched,
            "paired_wilcoxon": paired,
        })
        # Plot uses subject-level scores (one point per matched subject).
        matched_score = matched_eval[["base_id", "pair_id", "label"]].merge(
            oof_subj[["base_id", "y_score"]], on="base_id", how="inner",
        )
        plot_paired_scatter(matched_score, partition,
                            out / "forward_paired_scatter.png")
        logger.info(
            f"  forward: matched AUC={m_matched['auc']:.3f} "
            f"[CI {m_matched['auc_ci_low']:.3f}-{m_matched['auc_ci_high']:.3f}] "
            f"BalAcc={m_matched['balacc']:.3f} MCC={m_matched['mcc']:.3f} "
            f"Wilcoxon W={paired['W']}, p={paired['p']:.4g}, n={paired['n_pairs']}"
        )
        summary_rows.append({
            "partition": partition, "embedding": embedding,
            "classifier": classifier, "strategy": "forward",
            "scope": "matched",
            **{k: v for k, v in m_matched.items() if k != "confusion_matrix"},
            "wilcoxon_W": paired["W"], "wilcoxon_p": paired["p"],
            "n_pairs": paired["n_pairs"], "mean_diff": paired["mean_diff"],
        })
        summary_rows.append({
            "partition": partition, "embedding": embedding,
            "classifier": classifier, "strategy": "forward",
            "scope": "full",
            **{k: v for k, v in m_full.items() if k != "confusion_matrix"},
        })

    if strategy in ("reverse", "both"):
        out = cell_dir(partition, embedding, classifier, "reverse")
        _write_cohort_files(out, matched, pairs)
        rev = run_reverse(full, matched, embedding, classifier,
                          n_folds=n_folds, seed=seed,
                          keep_groups=keep_groups)
        scores_df = rev["scores_df"]
        common = {
            "partition": partition, "embedding": embedding,
            "classifier": classifier, "strategy": "reverse",
            **({"lr_C": _LR_C} if classifier == "logistic" else {}),
            "k_folds_used": rev["k_folds_used"],
            "n_dropped_no_emb_matched": rev["n_dropped_no_emb_matched"],
            "n_dropped_no_emb_full": rev["n_dropped_no_emb_full"],
            "n_unmatched": rev["n_unmatched"],
            "derived_view": (sorted(keep_groups)
                             if keep_groups is not None else None),
            "drop_corr_info": rev["drop_corr_info"],
        }

        scores_visits = rev["scores_df_visits"]

        # --- Method A: 10-fold ensemble ---
        ens_dir = out / "ensemble"
        ens_dir.mkdir(parents=True, exist_ok=True)
        scores_df[["ID", "y_true", "score_ensemble", "in_matched"]].rename(
            columns={"score_ensemble": "y_score"}
        ).to_csv(ens_dir / "scores.csv", index=False)
        scores_visits[["ID", "base_id", "y_true", "score_ensemble",
                       "in_matched"]].rename(
            columns={"score_ensemble": "y_score"}
        ).to_csv(ens_dir / "scores_visits.csv", index=False)
        write_json(ens_dir / "metrics.json", {
            **common, "method": "ensemble (10 fold-models)",
            "metrics_matched_oof": rev["metrics_matched_oof"],
            "metrics_full": rev["metrics_full_ensemble"],
            "metrics_unmatched": rev["metrics_unmatched_ensemble"],
        })

        # --- Method B: single model ---
        sgl_dir = out / "single"
        sgl_dir.mkdir(parents=True, exist_ok=True)
        scores_df[["ID", "y_true", "score_single", "in_matched"]].rename(
            columns={"score_single": "y_score"}
        ).to_csv(sgl_dir / "scores.csv", index=False)
        scores_visits[["ID", "base_id", "y_true", "score_single",
                       "in_matched"]].rename(
            columns={"score_single": "y_score"}
        ).to_csv(sgl_dir / "scores_visits.csv", index=False)
        write_json(sgl_dir / "metrics.json", {
            **common, "method": "single (1 model on all matched)",
            "metrics_matched_train": rev["metrics_matched_single_train"],
            "metrics_unmatched": rev["metrics_unmatched_single"],
        })

        m_oof = rev["metrics_matched_oof"]
        m_full = rev["metrics_full_ensemble"]
        m_un_ens = rev["metrics_unmatched_ensemble"]
        m_un_single = rev["metrics_unmatched_single"]
        msg = (f"  reverse: matched_oof AUC={m_oof['auc']:.3f}  "
               f"full_ensemble AUC={m_full['auc']:.3f}")
        if m_un_ens is not None:
            msg += f"  unmatched_ensemble AUC={m_un_ens['auc']:.3f}"
        if m_un_single is not None:
            msg += f"  unmatched_single AUC={m_un_single['auc']:.3f}"
        logger.info(msg)
        for scope_name, method, m in [
            ("matched_oof", "ensemble", m_oof),
            ("full", "ensemble", m_full),
            ("unmatched", "ensemble", m_un_ens),
            ("matched_train", "single", rev["metrics_matched_single_train"]),
            ("unmatched", "single", m_un_single),
        ]:
            if m is None:
                continue
            summary_rows.append({
                "partition": partition, "embedding": embedding,
                "classifier": classifier, "strategy": "reverse",
                "method": method, "scope": scope_name,
                **{k: v for k, v in m.items() if k != "confusion_matrix"},
            })

    return summary_rows


# ============================================================
# Sweep + summary
# ============================================================

def expand_axis(value, all_values):
    if value == "all" or value == "both":
        return all_values
    return [value]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--partition", default="all",
                        choices=PARTITIONS + ["all"])
    parser.add_argument("--embedding", default="all",
                        choices=EMBEDDINGS + ["all"])
    parser.add_argument("--classifier", default="both",
                        choices=CLASSIFIERS + ["both"])
    parser.add_argument("--strategy", default="both",
                        choices=STRATEGIES + ["both"])
    parser.add_argument("--feature-type", default="original",
                        choices=FEATURE_TYPES,
                        help="Embedding feature variant. 'original' writes "
                             "to embedding_classification/; asymmetry "
                             "variants write to "
                             "embedding_asymmetry_<variant>_classification/.")
    parser.add_argument("--drop-correlated-threshold", type=float, default=None,
                        help="Optional Pearson correlation threshold above "
                             "which one feature of each correlated pair is "
                             "dropped (fitted per-fold on training data only). "
                             "When set, output goes to "
                             "<base>/drop_<threshold>/ instead of "
                             "<base>/no_drop/. Mutually exclusive with "
                             "--pca-components.")
    parser.add_argument("--corr-method", default="pearson",
                        choices=["pearson", "spearman", "kendall"])
    parser.add_argument("--pca-components", default=None,
                        help="PCA n_components: int (fixed component count, "
                             "e.g. 50) or float<1 (variance ratio, e.g. 0.95). "
                             "Fitted per-fold on training data only. Output "
                             "goes to <base>/pca_<value>/. Mutually exclusive "
                             "with --drop-correlated-threshold.")
    parser.add_argument("--visit-mode", default="first", choices=VISIT_MODES,
                        help="'first' (default): one visit per base_id "
                             "(current behavior). 'all': include every "
                             "qualifying visit per base_id (more training "
                             "samples; GroupKFold(base_id) keeps same person "
                             "in same fold so no leakage).")
    parser.add_argument("--photo-mode", default="mean", choices=PHOTO_MODES,
                        help="'mean' (default): mean-pool the 10 photos per "
                             "subject into 1 feature vector (current "
                             "behavior). 'all': keep all 10 photos as "
                             "individual training rows.")
    parser.add_argument("--cohort-mode", default="default",
                        choices=["default", "p_first_hc_all"],
                        help="'default' (p_first_hc_strict) or "
                             "'p_first_hc_all' (HC = ALL NAD/ACS visits, no "
                             "strict filter). Output goes to "
                             "p_first_hc_<mode>/embedding_*classification/ . "
                             "Forwarded to run_4arm_deep_dive via env COHORT_MODE.")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-C", type=float, default=1.0,
                        help="LogisticRegression C (inverse regularization). "
                             "Default 1.0 keeps the original output layout. "
                             "Any other value inserts a 'logistic_C{value}/' "
                             "segment under cohort_dir/ and drops the trailing "
                             "'<classifier>/' leaf, so all cells live under "
                             "p_first_hc_*/logistic_C{value}/embedding_*/... "
                             "Must be combined with --classifier logistic.")
    args = parser.parse_args()

    global _FEATURE_TYPE, _DROP_CORR_THRESHOLD, _DROP_CORR_METHOD
    global _PCA_COMPONENTS, _VISIT_MODE, _PHOTO_MODE, _COHORT_MODE, _LR_C, OUTPUT_DIR
    if args.lr_C != 1.0 and args.classifier != "logistic":
        parser.error(
            "--lr-C != 1.0 requires --classifier logistic "
            "(the new path layout assumes the whole subtree is LR)."
        )
    if args.drop_correlated_threshold is not None and args.pca_components is not None:
        parser.error("--drop-correlated-threshold and --pca-components are "
                     "mutually exclusive")
    _FEATURE_TYPE = args.feature_type
    _DROP_CORR_THRESHOLD = args.drop_correlated_threshold
    _DROP_CORR_METHOD = args.corr_method
    if args.pca_components is None:
        _PCA_COMPONENTS = None
    else:
        # Parse "50" → int(50); "0.95" → float(0.95).
        s = str(args.pca_components)
        try:
            _PCA_COMPONENTS = int(s) if "." not in s else float(s)
        except ValueError:
            parser.error(f"--pca-components must be int or float, got {s!r}")
        if isinstance(_PCA_COMPONENTS, float) and not (0 < _PCA_COMPONENTS < 1):
            parser.error("--pca-components float must be in (0, 1) for "
                         "variance ratio")
    _VISIT_MODE = args.visit_mode
    _PHOTO_MODE = args.photo_mode
    _COHORT_MODE = args.cohort_mode
    _LR_C = args.lr_C
    # Verify the early sniff matched (the import-time read is the source of
    # truth for run_4arm_deep_dive).
    expected_env = "p_first_hc_all" if _COHORT_MODE == "p_first_hc_all" else "default"
    if os.environ.get("COHORT_MODE", "default") != expected_env:
        parser.error(
            f"COHORT_MODE env mismatch ({os.environ.get('COHORT_MODE')!r} "
            f"vs --cohort-mode {_COHORT_MODE!r}). Pass --cohort-mode "
            "before other flags or set COHORT_MODE env explicitly."
        )
    OUTPUT_DIR = output_dir_for(_FEATURE_TYPE, _DROP_CORR_THRESHOLD,
                                  _VISIT_MODE, _PHOTO_MODE,
                                  _PCA_COMPONENTS, _COHORT_MODE, _LR_C)

    partitions = expand_axis(args.partition, PARTITIONS)
    embeddings = expand_axis(args.embedding, EMBEDDINGS)
    classifiers = expand_axis(args.classifier, CLASSIFIERS)
    strategies = expand_axis(args.strategy, STRATEGIES)

    if _DROP_CORR_THRESHOLD is not None:
        reducer_label = f"drop_{_DROP_CORR_THRESHOLD} ({_DROP_CORR_METHOD})"
    elif _PCA_COMPONENTS is not None:
        reducer_label = f"pca_{_PCA_COMPONENTS}"
    else:
        reducer_label = "no_drop"
    logger.info(f"Feature type: {_FEATURE_TYPE}  ·  {reducer_label}  ·  "
                f"visit={_VISIT_MODE}  ·  photo={_PHOTO_MODE}  ·  "
                f"cohort={_COHORT_MODE}  ·  lr_C={_LR_C}  →  {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_dir = OUTPUT_DIR / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    n_cells = 0
    n_failed = 0
    for partition, embedding, classifier in itertools.product(
        partitions, embeddings, classifiers
    ):
        for strategy in strategies:
            n_cells += 1
            try:
                rows = run_cell(partition, embedding, classifier, strategy,
                                 n_folds=args.n_folds, seed=args.seed)
                all_rows.extend(rows)
            except Exception as e:
                n_failed += 1
                logger.exception(
                    f"FAILED {partition}/{embedding}/{classifier}/{strategy}: {e}"
                )

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        summary_path = summary_dir / "combined_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Wrote {summary_path} ({len(summary_df)} rows)")

    logger.info(
        f"Done: {n_cells} cells, {n_failed} failed. Outputs at {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
