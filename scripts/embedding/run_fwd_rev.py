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
import itertools
import json
import logging
import os
import sys
from pathlib import Path

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

from src.config import (
    EMBEDDING_CLASSIFICATION_DIR,
    EMBEDDING_FEATURES_DIR,
    EMBEDDING_ABTEST_CLASSIFICATION_DIR,
    FEATURES_DIR as EMBEDDING_ABTEST_FEAT_DIR,
    cohort_name,
)

EMBEDDING_FEAT_DIR = EMBEDDING_FEATURES_DIR
AGES_FILE = (PROJECT_ROOT / "workspace" / "age" / "predictions" /
             "p_first_hc_strict" / "predicted_ages.json")

PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs", "mmse_hilo", "casi_hilo"]
EMBEDDINGS = ["arcface", "topofr", "dlib"]
CLASSIFIERS = ["logistic", "xgb", "tabpfn"]
STRATEGIES = ["forward", "reverse"]
FEATURE_TYPES = [
    "original",
    "original_background",
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
_COHORT_MODE = os.environ.get("COHORT_MODE", "default")  # default=p_first_hc_strict / p_first_hc_all / p_all_hc_all
_LR_C = 1.0  # LogisticRegression C; encoded at cell-level leaf as logistic/C_<value>/
_RFE_DROP = None   # iterative RFE: drop N weakest features per iter (None/0 = off)
_RFE_ITERS = 5     # number of RFE iterations
_SAVE_OOF_PROB = False  # write pred_probability/<config>.csv for meta_analysis stacking


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
                    cohort_mode="default"):
    """Build per-cell output dir for the chosen feature_type / reducer /
    cohort / visit-photo mode combination.

    Tree layout:
      embedding/analysis/classification/<variant>/<cohort>/<reducer>/

    Variants: original / difference / absolute_difference / average /
              relative_differences / absolute_relative_differences

    cohort_mode='default'        -> p_first_hc_strict (default visit=first, photo=mean)
    cohort_mode='p_first_hc_all' -> p_first_hc_all    (default visit=all,   photo=mean)
    cohort_mode='p_all_hc_all'   -> p_all_hc_all      (default visit=all,   photo=mean)

    Reducer naming:
      drop_corr=X.X    → drop_feats/pearson_r_X.X
      pca_components=N → pca/n_components_N            (int N)
      pca_components=R → pca/var_ratio_R               (float R<1)
      neither          → no_drop

    Variant subdir `<visit_X>[_<photo_Y>]` is appended as a nested
    sub-directory under the reducer dir when (visit, photo) differ from the
    cohort's default — e.g. p_first_hc_strict + visit_mode=all yields
    `pca/n_components_100/visit_all`.

    Cell-leaf path: <output_dir>/<partition>/<fwd|rev>/<emb>/<clf>/[C_<value>/]
    LR `C` value is encoded at the cell-level leaf (logistic/C_<lr_C:g>/) by
    cell_dir(), not in this path.
    """
    if pca_components is not None and drop_corr is not None:
        raise ValueError("drop_corr and pca_components are mutually exclusive")
    if pca_components is not None:
        if isinstance(pca_components, float) and pca_components < 1:
            reducer = Path("pca") / f"var_ratio_{pca_components:g}"
        else:
            reducer = Path("pca") / f"n_components_{pca_components}"
    elif drop_corr is None:
        reducer = Path("no_drop")
    else:
        reducer = Path("drop_feats") / f"pearson_r_{drop_corr:g}"
    if cohort_mode in ("p_first_hc_all", "p_all_hc_all"):
        default_visit, default_photo = "all", "mean"
    else:
        default_visit, default_photo = "first", "mean"
    suffix_parts = []
    if visit_mode != default_visit:
        suffix_parts.append(f"visit_{visit_mode}")
    if photo_mode != default_photo:
        suffix_parts.append(f"photo_{photo_mode}")
    if suffix_parts:
        reducer = reducer / "_".join(suffix_parts)
    cohort_dir = cohort_name(cohort_mode)
    return EMBEDDING_CLASSIFICATION_DIR / feature_type / cohort_dir / reducer


OUTPUT_DIR = output_dir_for(_FEATURE_TYPE, _DROP_CORR_THRESHOLD,
                              _VISIT_MODE, _PHOTO_MODE, _PCA_COMPONENTS,
                              _COHORT_MODE)


class _TorchGPUPCA:
    """sklearn-compatible GPU PCA via torch.pca_lowrank. Fixed-component only
    (n int) — variance-ratio mode falls back to sklearn (set externally).
    Stores mean / loadings on GPU; transform pulls back to numpy."""
    def __init__(self, n_components, niter=4):
        self.n_components = int(n_components)
        self.niter = niter

    def fit(self, X):
        import torch
        # Seed torch RNG so randomized SVD in pca_lowrank is reproducible.
        # Without this, the same input produces slightly different V across
        # subprocess invocations (different y_score, different AUC).
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
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
# Direct imports from utilities (cohort + matching + stats)
# ============================================================

from scripts.utilities.cohort import (
    _strict_hc_filter_all_visits,
    build_cohort_ad_vs_HCgroup,
    DEMOGRAPHICS_DIR as _COHORT_DEMOGRAPHICS_DIR,
    load_p_demographics,
    match_1to1,
    select_visit,
    split_by_metric_median,
)
from scripts.utilities.feature_loaders import load_embedding_mean
from scripts.utilities.stats_helpers import bootstrap_auc_ci

# Module-level _COHORT_MODE is set in main() after argparse; cohort builders
# get it threaded as a parameter (no env-var-at-load globals).
_COHORT_MODE = "default"
_HC_SOURCE_MODE = os.environ.get("HC_SOURCE_MODE", "ACS")


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
# (bootstrap_auc_ci, match_1to1, split_by_metric_median, load_p_demographics,
# select_visit, build_cohort_ad_vs_HCgroup are imported above from
# scripts.utilities — no importlib reuse here.)


# ============================================================
# Cohort builders (5 partitions)
# ============================================================

_PRED_AGES_CACHE = None

# Training-output caches keyed by _train_cache_key(...). Lets the 3 ad_vs_*
# partitions share one trained model per (embedding, classifier, reducer,
# feature_type, ...) since they only differ in eval-time keep_groups filtering.
# Cleared implicitly per subprocess (driver spawns one process per outer combo).
# IMPORTANT: when adding a new module-level state that affects training, also
# add it to _train_cache_key below or risk false cache hits.
_FWD_TRAIN_CACHE = {}
_REV_TRAIN_CACHE = {}


def _train_cache_key(partition, embedding, classifier, n_folds, seed):
    """Tuple key uniquely identifying a training run. ad_vs_hc / ad_vs_nad /
    ad_vs_acs collapse to one family slot; mmse_hilo / casi_hilo each get their
    own slot (no false sharing — their cohorts are unique)."""
    family = ("ad_vs_hcgroup_HC"
              if partition in {"ad_vs_hc", "ad_vs_nad", "ad_vs_acs"}
              else partition)
    return (family, embedding, classifier,
            _FEATURE_TYPE, _DROP_CORR_THRESHOLD, _DROP_CORR_METHOD,
            _PCA_COMPONENTS, _VISIT_MODE, _PHOTO_MODE, _LR_C,
            n_folds, seed)


def _load_pred_ages():
    """Load predicted_ages.json (cached). Used for the matched-pair filter
    that mirrors run_cross_matched.run_hc_groups_comparison's predicted-age
    filter — drop pairs whose either side lacks a predicted_age, so matched
    n_pairs aligns with arm_b's reference (HC=180, NAD=169, ACS=29 / hi-lo
    equivalent)."""
    global _PRED_AGES_CACHE
    if _PRED_AGES_CACHE is None:
        with open(AGES_FILE) as f:
            _PRED_AGES_CACHE = json.load(f)
    return _PRED_AGES_CACHE


def _filter_pairs_by_pred_age(matched, pairs):
    """Drop pairs where either minor_id or major_id lacks a predicted_age."""
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
    DEMOGRAPHICS_DIR = _COHORT_DEMOGRAPHICS_DIR
    HC_SOURCE_MODE = _HC_SOURCE_MODE

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
    hc_all = _strict_hc_filter_all_visits(demo, hc_source, _COHORT_MODE)
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
    full from design='cross_naive' (no matching), matched from design=
    'cross_matched' (1:1 age) with the predicted_age post-hoc filter applied
    so n_pairs matches the cross_matched reference.

    visit_mode + cohort_mode interaction:
      cohort_mode='default' (p_first_hc_strict) picks one visit per subject on
      both sides; visit_mode='all' expands to all qualifying visits restricted
      to the same base_id set.

      cohort_mode in ('p_first_hc_all', 'p_all_hc_all') already controls visit
      selection at cohort-build time (HC unfiltered all-visits in both; AD
      first-visit in p_first_hc_all vs all-visits in p_all_hc_all). The
      expansion path is intentionally skipped — re-deriving from
      base_id-restricted demographics would collapse both modes onto the
      same expanded cohort, defeating the purpose of the cohort_mode flag.
    """
    full_first, _ = build_cohort_ad_vs_HCgroup(
        hc_source, design="cross_naive",
        cohort_mode=_COHORT_MODE, hc_source_mode=_HC_SOURCE_MODE,
    )
    matched, pairs = build_cohort_ad_vs_HCgroup(
        hc_source, design="cross_matched",
        cohort_mode=_COHORT_MODE, hc_source_mode=_HC_SOURCE_MODE,
    )
    matched, pairs = _filter_pairs_by_pred_age(matched, pairs)
    if _VISIT_MODE == "all" and _COHORT_MODE == "default":
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

    # See `_build_ad_vs_hcgroup` for cohort_mode × visit_mode interaction.
    # Expansion only meaningful for the legacy strict cohort (default); the
    # other two cohort_modes' visit-selection semantics are already controlled
    # at cohort-build time, so re-deriving via base_id-restrict would collapse
    # them onto a common expanded set.
    if _VISIT_MODE == "all" and _COHORT_MODE == "default":
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
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(
            ignore_pretraining_limits=True,
            random_state=seed,
        )
    raise ValueError(f"unknown classifier: {name}")


def needs_scaler(classifier):
    # TabPFN has its own internal normalization; xgb is tree-based.
    return classifier == "logistic"


def _fit_with_optional_rfe(X_train, y_train, classifier, seed):
    """Fit classifier; if module-level _RFE_DROP > 0 and classifier supports
    feature_importances_/coef_, do iterative RFE: drop _RFE_DROP weakest
    features per iteration for _RFE_ITERS rounds, then refit on the surviving
    feature subset.

    Returns (final_model, keep_mask). keep_mask is a bool array of length
    X_train.shape[1]; True means feature kept. When RFE is off or the
    classifier doesn't expose importance, keep_mask is all-True.
    """
    n_features = X_train.shape[1]
    keep_mask = np.ones(n_features, dtype=bool)
    if _RFE_DROP is None or _RFE_DROP <= 0 or classifier == "tabpfn":
        clf = make_classifier(classifier, seed=seed)
        clf.fit(X_train, y_train)
        return clf, keep_mask

    for _ in range(_RFE_ITERS):
        kept_idx = np.where(keep_mask)[0]
        if len(kept_idx) <= _RFE_DROP:
            break  # leave at least _RFE_DROP features alive
        clf = make_classifier(classifier, seed=seed)
        clf.fit(X_train[:, keep_mask], y_train)
        if hasattr(clf, "feature_importances_"):
            imp = np.asarray(clf.feature_importances_)
        elif hasattr(clf, "coef_"):
            imp = np.abs(np.asarray(clf.coef_).ravel())
        else:
            break
        # drop _RFE_DROP weakest within currently-kept indices
        weakest_within_kept = np.argsort(imp)[:_RFE_DROP]
        keep_mask[kept_idx[weakest_within_kept]] = False
    # final refit on surviving subset
    clf = make_classifier(classifier, seed=seed)
    clf.fit(X_train[:, keep_mask], y_train)
    return clf, keep_mask


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


def _forward_train(full_cohort, embedding, classifier, n_folds, seed):
    """Train-cohort-only side of forward: 10-fold OOF + union-level metrics.
    Output is independent of keep_groups (eval-time filter), so this is
    cacheable per (training cohort, embedding, classifier, reducer, ...)."""
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
        clf, rfe_mask = _fit_with_optional_rfe(Xtr, y[tri], classifier, seed)
        Xte_eval = Xte[:, rfe_mask] if rfe_mask is not None and not rfe_mask.all() else Xte
        oof_score[tei] = clf.predict_proba(Xte_eval)[:, 1]
        oof_fold[tei] = fold_idx

    oof_df_base = pd.DataFrame({
        "ID": ids, "base_id": base_ids, "y_true": y,
        "y_score": oof_score, "fold": oof_fold,
    })
    # Aggregate row-level OOF (multiple rows per base_id when visit/photo mode
    # is "all") back to subject-level for matched-pair eval (matching is
    # defined at first-visit subject level).
    oof_subj = _aggregate_to_subject(oof_df_base, score_cols=["y_score"])

    # Full-cohort (union) metrics: visit-level. Stays union-level by design —
    # the 3 ad_vs_* cells share this value via the cache.
    metrics_full = compute_clf_metrics(
        oof_df_base["y_true"].to_numpy(int),
        oof_df_base["y_score"].to_numpy(float),
        seed=seed,
    )

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

    return {
        "oof_df_base": oof_df_base,    # without in_matched col
        "oof_subj": oof_subj,
        "metrics_full": metrics_full,
        "drop_corr_info": drop_corr_info,
        "n_dropped": n_dropped,
        "k": k,
    }


def _forward_eval(train, matched_cohort, keep_groups, seed):
    """Eval-time (keep_groups) part: matched-subset metrics + per-cell oof_df
    with in_matched column. Always runs (cheap)."""
    matched_first = matched_cohort.sort_values(
        ["base_id"] + (["visit"] if "visit" in matched_cohort.columns else [])
    ).drop_duplicates("base_id", keep="first")
    if keep_groups is not None:
        matched_eval = matched_first[matched_first["group"].isin(keep_groups)]
        matched_eval = _filter_pairs_complete(matched_eval)
    else:
        matched_eval = matched_first

    matched_min = matched_eval[["base_id", "ID", "pair_id", "label"]].copy()
    oof_matched = matched_min.merge(
        train["oof_subj"][["base_id", "y_score"]],
        on="base_id", how="inner",
    )
    metrics_matched = compute_clf_metrics(
        oof_matched["label"].to_numpy(int),
        oof_matched["y_score"].to_numpy(float),
        seed=seed,
    )
    paired = paired_wilcoxon_by_pair(oof_matched)

    # Per-cell oof_df with partition-specific in_matched column.
    oof_df = train["oof_df_base"].copy()
    oof_df["in_matched"] = oof_df["base_id"].isin(
        matched_eval["base_id"]
    ).astype(int)

    return oof_df, metrics_matched, paired, matched_eval


def run_forward(full_cohort, matched_cohort, embedding, classifier,
                n_folds=10, seed=42, keep_groups=None, partition=None):
    """Train on full_cohort 10-fold; evaluate on full + matched (optionally
    subset by group). When keep_groups is set, training is unchanged but
    the evaluation set is filtered to subjects whose `group` ∈ keep_groups
    (and matched pairs that lose a side are dropped).

    Caches the training output keyed by `partition` family + module-level
    state so the 3 ad_vs_* partitions share one training pass."""
    cache_key = _train_cache_key(partition, embedding, classifier, n_folds, seed)
    train = _FWD_TRAIN_CACHE.get(cache_key)
    if train is None:
        train = _forward_train(full_cohort, embedding, classifier, n_folds, seed)
        _FWD_TRAIN_CACHE[cache_key] = train
        logger.info(f"  forward: trained from scratch (cache miss)")
    else:
        logger.info(f"  forward: reused cached training (cache hit)")

    oof_df, metrics_matched, paired, matched_eval = _forward_eval(
        train, matched_cohort, keep_groups, seed,
    )

    return (oof_df, train["oof_subj"], train["metrics_full"], metrics_matched,
            paired, n_folds, train["n_dropped"], train["k"], matched_eval,
            train["drop_corr_info"])


def plot_cm(cm, title, out_path, neg_label="0", pos_label="1"):
    """Render a 2×2 confusion matrix to PNG. cm = [[tn, fp], [fn, tp]]."""
    cm = np.asarray(cm, dtype=float)
    cmap = plt.get_cmap("Blues")
    vmin, vmax = float(cm.min()), float(cm.max())
    rng = vmax - vmin if vmax > vmin else 1.0
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {neg_label}", f"Pred {pos_label}"])
    ax.set_yticklabels([f"True {neg_label}", f"True {pos_label}"])
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            r, g, b, _ = cmap((v - vmin) / rng)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    color="white" if lum < 0.5 else "black",
                    fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.85, bottom=0.15)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Per-partition (neg, pos) labels used as CM tick labels.
CM_LABELS = {
    "ad_vs_hc":  ("HC",  "AD"),
    "ad_vs_nad": ("NAD", "AD"),
    "ad_vs_acs": ("ACS", "AD"),
    "mmse_hilo": ("MMSE-low", "MMSE-high"),
    "casi_hilo": ("CASI-low", "CASI-high"),
}


def _plot_cm_blocks(blocks, partition, out_dir, prefix):
    """Write one CM PNG per non-None block. blocks: list of (metric_dict,
    fname, scope) tuples; metric_dict carries `confusion_matrix` and `auc`."""
    neg, pos = CM_LABELS.get(partition, ("0", "1"))
    for m, fname, scope in blocks:
        if not m or not m.get("confusion_matrix"):
            continue
        title = (f"{partition} — {scope}\n"
                 f"n={m.get('n', '?')}  AUC={m.get('auc', float('nan')):.3f}")
        plot_cm(m["confusion_matrix"], title, out_dir / fname,
                 neg_label=neg, pos_label=pos)


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

def _reverse_train(full_cohort, matched_cohort, embedding, classifier,
                   n_folds, seed):
    """Train-cohort-only side of reverse: 10-fold GroupKFold ensemble on
    matched cohort, applied to the full cohort. Output is independent of
    keep_groups so it can be shared across the 3 ad_vs_* partitions —
    keep_groups-aware metrics (matched_oof + unmatched) are computed in
    _reverse_eval."""
    Xm, ym, gm, idsm, n_dropped_m = build_feature_matrix(matched_cohort, embedding)
    Xf, yf, _gf, idsf, n_dropped_f = build_feature_matrix(full_cohort, embedding)

    n_groups_m = len(np.unique(gm))
    k = min(n_folds, n_groups_m)
    if k < 2:
        raise RuntimeError(f"too few groups in matched ({n_groups_m})")
    gkf = GroupKFold(n_splits=k)

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
        clf, rfe_mask = _fit_with_optional_rfe(Xm_tr, ym[tri], classifier, seed)
        if rfe_mask is not None and not rfe_mask.all():
            Xm_te, Xf_t = Xm_te[:, rfe_mask], Xf_t[:, rfe_mask]
        oof_m[tei] = clf.predict_proba(Xm_te)[:, 1]
        full_preds.append(clf.predict_proba(Xf_t)[:, 1])

    full_score = np.mean(np.stack(full_preds, axis=0), axis=0)

    matched_oof_df = pd.DataFrame({
        "ID": idsm, "base_id": gm, "y_true": ym, "y_score": oof_m,
    })
    matched_oof_subj = _aggregate_to_subject(matched_oof_df, score_cols=["y_score"])

    # `group` column is only present for ad_vs_hcgroup family cohorts (used to
    # filter unmatched eval by keep_groups). Hi-lo partitions don't have it
    # and don't need it (their keep_groups is None).
    base_to_group = (full_cohort.set_index("base_id")["group"].to_dict()
                     if "group" in full_cohort.columns else {})

    full_df_rows = pd.DataFrame({
        "ID": idsf, "base_id": _gf, "y_true": yf,
        "y_score": full_score,
    })
    full_df_rows["group"] = full_df_rows["base_id"].map(base_to_group)
    full_subj = _aggregate_to_subject(full_df_rows, score_cols=["y_score"])
    full_subj["group"] = full_subj["base_id"].map(base_to_group)

    matched_base_ids = set(pd.unique(gm))
    full_df_rows["in_matched"] = full_df_rows["base_id"].isin(
        matched_base_ids
    ).astype(int)
    full_subj["in_matched"] = full_subj["base_id"].isin(
        matched_base_ids
    ).astype(int)

    drop_corr_info = {
        "reducer": _reducer_label(),
        "drop_corr_threshold": _DROP_CORR_THRESHOLD,
        "drop_corr_method": _DROP_CORR_METHOD if _DROP_CORR_THRESHOLD else None,
        "pca_components": _PCA_COMPONENTS,
        "n_features_input": int(Xm.shape[1]),
        "n_features_kept_per_fold": [int(n) for n in n_kept_per_fold],
        "n_rows_matched": int(Xm.shape[0]),
        "n_rows_full": int(Xf.shape[0]),
        "n_unique_subjects_matched": int(pd.unique(gm).size),
        "n_unique_subjects_full": int(pd.unique(_gf).size),
        "visit_mode": _VISIT_MODE,
        "photo_mode": _PHOTO_MODE,
    }

    full_df_visits = full_df_rows[
        ["ID", "base_id", "y_true", "y_score", "in_matched", "group"]
    ].copy()

    return {
        "matched_oof_subj": matched_oof_subj,
        "full_subj": full_subj,
        "full_df_visits": full_df_visits,
        "matched_base_ids": matched_base_ids,
        "drop_corr_info": drop_corr_info,
        "k": k,
        "n_dropped_m": n_dropped_m,
        "n_dropped_f": n_dropped_f,
    }


def _reverse_eval(train, matched_cohort, keep_groups, seed):
    """Eval-time (keep_groups) part: subject-level matched_oof + visit-level
    unmatched, both keep_groups-aware. Per-cell scores_df also gets
    keep_groups-filtered in_matched col."""
    matched_oof_subj = train["matched_oof_subj"]
    full_subj = train["full_subj"]
    full_df_visits = train["full_df_visits"]
    matched_base_ids = train["matched_base_ids"]

    # ---- matched_oof: subject-level, filtered by group ----
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
    metrics_matched_oof = (
        compute_clf_metrics(eval_y_m_subj, eval_score_m_subj, seed=seed)
        if len(eval_y_m_subj) >= 5 and len(np.unique(eval_y_m_subj)) > 1 else None
    )

    # ---- unmatched: visit-level, filtered by group ----
    in_matched_v = full_df_visits["in_matched"].to_numpy(bool)
    if keep_groups is not None:
        in_keep_v = full_df_visits["group"].isin(keep_groups).to_numpy()
    else:
        in_keep_v = np.ones(len(full_df_visits), dtype=bool)
    unmatched_mask = (~in_matched_v) & in_keep_v
    n_unmatched_visits = int(unmatched_mask.sum())
    yf_v = full_df_visits["y_true"].to_numpy(int)
    score_v = full_df_visits["y_score"].to_numpy(float)
    metrics_unmatched = None
    if n_unmatched_visits >= 5 and \
       len(np.unique(yf_v[unmatched_mask])) > 1:
        metrics_unmatched = compute_clf_metrics(
            yf_v[unmatched_mask], score_v[unmatched_mask], seed=seed,
        )

    # ---- scores_df (subject-level) keep_groups-aware ----
    if keep_groups is not None:
        bid_to_group_all = matched_cohort.set_index("base_id")["group"].to_dict()
        matched_keep_base = {b for b in matched_base_ids
                              if bid_to_group_all.get(b) in keep_groups}
    else:
        matched_keep_base = matched_base_ids
    full_df = full_subj.copy()
    if keep_groups is not None:
        full_df["in_matched"] = full_df["base_id"].isin(matched_keep_base).astype(int)
    full_df = full_df[["ID", "y_true", "y_score", "in_matched"]]

    # ---- scores_df_visits keep_groups-filtered (drop subjects outside keep_groups) ----
    if keep_groups is not None:
        scores_df_visits = full_df_visits[in_keep_v].copy()
    else:
        scores_df_visits = full_df_visits.copy()
    scores_df_visits = scores_df_visits[
        ["ID", "base_id", "y_true", "y_score", "in_matched"]
    ]

    return {
        "scores_df": full_df,
        "scores_df_visits": scores_df_visits,
        "metrics_matched_oof": metrics_matched_oof,
        "metrics_unmatched": metrics_unmatched,
        "n_unmatched_visits": n_unmatched_visits,
    }


def run_reverse(full_cohort, matched_cohort, embedding, classifier,
                n_folds=10, seed=42, keep_groups=None, partition=None):
    """Reverse strategy: 10-fold GroupKFold on matched cohort → 10 fold-models;
    mean their predict_proba over the full cohort. Reports:

      - matched_oof:  ensemble OOF on matched (subject-level, validation domain)
      - unmatched:    ensemble on unmatched-only (visit-level, held-out domain)

    When keep_groups is set, training is unchanged but the matched eval cohort
    (matched_oof + scores_df in_matched) is filtered to subjects whose
    `group` ∈ keep_groups.

    Caches training output keyed by `partition` family so the 3 ad_vs_*
    partitions share one training pass.
    """
    cache_key = _train_cache_key(partition, embedding, classifier, n_folds, seed)
    train = _REV_TRAIN_CACHE.get(cache_key)
    if train is None:
        train = _reverse_train(full_cohort, matched_cohort, embedding,
                                classifier, n_folds, seed)
        _REV_TRAIN_CACHE[cache_key] = train
        logger.info(f"  reverse: trained from scratch (cache miss)")
    else:
        logger.info(f"  reverse: reused cached training (cache hit)")

    ev = _reverse_eval(train, matched_cohort, keep_groups, seed)

    return {
        "scores_df": ev["scores_df"],
        "scores_df_visits": ev["scores_df_visits"],
        "k_folds_used": train["k"],
        "n_dropped_no_emb_matched": train["n_dropped_m"],
        "n_dropped_no_emb_full": train["n_dropped_f"],
        "n_unmatched": ev["n_unmatched_visits"],
        "metrics_matched_oof": ev["metrics_matched_oof"],
        "metrics_unmatched": ev["metrics_unmatched"],
        "drop_corr_info": train["drop_corr_info"],
    }


# ============================================================
# Per-cell driver
# ============================================================

def cell_dir(partition, embedding, classifier, strategy):
    """Layout: <reducer>/<partition>/{fwd,rev}/<embedding>/<classifier>[/C_<lr_C:g>]/

    LR cells always nest a `C_<lr_C:g>/` leaf under `logistic/` so different
    regularization strengths sit side by side (e.g. `logistic/C_1/`,
    `logistic/C_10/`). XGB cells stay flat (no C dimension).
    """
    bucket = "fwd" if strategy == "forward" else "rev"
    base = OUTPUT_DIR / partition / bucket / embedding / classifier
    if classifier == "logistic":
        return base / f"C_{_LR_C:g}"
    return base


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
            partition=partition,
        )
        oof_df.to_csv(out / "forward_oof_scores.csv", index=False)
        # Subject-level aggregated scores (mean across visit/photo rows of the
        # same base_id). Same as oof_df in the default first/mean mode.
        oof_subj.to_csv(out / "forward_oof_scores_subject.csv", index=False)
        if _SAVE_OOF_PROB:
            # Meta-stacking input: subject-level (one row per base_id) with
            # canonical column names that run_meta_analysis.py consumes.
            prob_dir = OUTPUT_DIR / "pred_probability"
            prob_dir.mkdir(parents=True, exist_ok=True)
            prob_df = oof_subj[["base_id", "y_true", "y_score"]].rename(
                columns={"y_true": "label", "y_score": "prob"})
            prob_df["partition"] = partition
            prob_df["embedding"] = embedding
            prob_df["classifier"] = classifier
            prob_df.to_csv(
                prob_dir / f"{partition}_{embedding}_{classifier}_forward.csv",
                index=False,
            )
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
        _plot_cm_blocks([
            (m_matched, "forward_cm_matched.png", "forward_matched"),
            (m_full,    "forward_cm_full.png",    "forward_full"),
        ], partition, out, prefix="forward")
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
                          keep_groups=keep_groups, partition=partition)
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

        scores_df.to_csv(out / "scores.csv", index=False)
        scores_visits.to_csv(out / "scores_visits.csv", index=False)
        write_json(out / "metrics.json", {
            **common,
            "metrics_matched_oof": rev["metrics_matched_oof"],
            "metrics_unmatched": rev["metrics_unmatched"],
        })

        m_oof = rev["metrics_matched_oof"]
        m_un = rev["metrics_unmatched"]
        _plot_cm_blocks([
            (m_oof, "cm_matched_oof.png", "matched_oof"),
            (m_un,  "cm_unmatched.png",   "unmatched"),
        ], partition, out, prefix="reverse")
        msg = f"  reverse: matched_oof AUC={m_oof['auc']:.3f}"
        if m_un is not None:
            msg += f"  unmatched AUC={m_un['auc']:.3f}"
        logger.info(msg)
        for scope_name, m in [("matched_oof", m_oof), ("unmatched", m_un)]:
            if m is None:
                continue
            summary_rows.append({
                "partition": partition, "embedding": embedding,
                "classifier": classifier, "strategy": "reverse",
                "scope": scope_name,
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
                        help="Embedding feature variant. Output goes to "
                             "embedding/analysis/classification/<variant>/.")
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
                        choices=["default", "p_first_hc_all", "p_all_hc_all"],
                        help="'default' (p_first_hc_strict), 'p_first_hc_all' "
                             "(first-visit P + ALL NAD/ACS), or 'p_all_hc_all' "
                             "(ALL P visits + ALL NAD/ACS). Output goes to "
                             "<cohort>/embedding_*classification/ . "
                             "Threaded as a function parameter to "
                             "scripts.utilities.cohort.build_cohort_*.")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-C", type=float, default=1.0,
                        help="LogisticRegression C (inverse regularization). "
                             "Encoded at the cell-level leaf as "
                             "logistic/C_<value>/ (e.g. C_1, C_10), so "
                             "different C values can coexist alongside xgb/. "
                             "Has no effect on xgb cells.")
    parser.add_argument("--embedding-abtest", action="store_true",
                        help="Read features from embedding_ABtest/features/ "
                             "and write outputs to "
                             "embedding_ABtest/analysis/classification/.  "
                             "Default: production embedding/ tree.")
    parser.add_argument("--save-oof-probabilities", action="store_true",
                        help="Additionally write pred_probability/<config>.csv "
                             "alongside per-cell outputs (consumed by "
                             "scripts/meta/run_meta_analysis.py as base-level "
                             "input for TabPFN stacking).")
    parser.add_argument("--rfe-drop", type=int, default=None,
                        help="Iterative RFE: drop N weakest features per "
                             "iteration after fitting; refit on the survivor "
                             "subset; repeat --rfe-iters times. Uses "
                             "feature_importances_ (xgb) or |coef_| "
                             "(logistic). TabPFN has neither, so RFE is "
                             "skipped for it.")
    parser.add_argument("--rfe-iters", type=int, default=5,
                        help="Number of RFE iterations (used with --rfe-drop).")
    args = parser.parse_args()

    global _FEATURE_TYPE, _DROP_CORR_THRESHOLD, _DROP_CORR_METHOD
    global _PCA_COMPONENTS, _VISIT_MODE, _PHOTO_MODE, _COHORT_MODE, _LR_C, OUTPUT_DIR
    global _RFE_DROP, _RFE_ITERS, _SAVE_OOF_PROB
    global EMBEDDING_CLASSIFICATION_DIR, EMBEDDING_FEAT_DIR
    if args.embedding_abtest:
        EMBEDDING_CLASSIFICATION_DIR = EMBEDDING_ABTEST_CLASSIFICATION_DIR
        EMBEDDING_FEAT_DIR = EMBEDDING_ABTEST_FEAT_DIR
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
    _RFE_DROP = args.rfe_drop
    _RFE_ITERS = args.rfe_iters
    _SAVE_OOF_PROB = args.save_oof_probabilities
    OUTPUT_DIR = output_dir_for(_FEATURE_TYPE, _DROP_CORR_THRESHOLD,
                                  _VISIT_MODE, _PHOTO_MODE,
                                  _PCA_COMPONENTS, _COHORT_MODE)

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
        # C-tagged filename so different LR C values don't clobber each other.
        # XGB-only runs still get a C tag (uses --lr-C default 1.0); harmless.
        summary_path = summary_dir / f"combined_summary_C{_LR_C:g}.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Wrote {summary_path} ({len(summary_df)} rows)")

    logger.info(
        f"Done: {n_cells} cells, {n_failed} failed. Outputs at {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
