"""
Single source of truth for cohort filtering, visit selection, matching,
and cohort building.

All dataset-filtering logic lives here:
  - CDR filters (per-visit and subject-level max)
  - HC strict HC cognitive filter (CDR==0 OR CDR.isna() AND MMSE>=26)
  - Predicted-age existence / threshold filter
  - Follow-up duration filter
  - Multi-visit filter
  - Feature-existence gate (expandable to all 14 modalities)
  - Visit selection (first / all, with feature-bearing fallback)
  - 1:1 age nearest-neighbor matching (cross-sec + longitudinal)
  - Cohort builders for AD-vs-HC and AD hi-lo designs
"""
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CohortSpec,
    DEFAULT_COHORT_MODE,
    EMO_AU_FEATURES_DIR,
    LONGITUDINAL_FEATURES_DIR,
    PREDICTED_AGES_FILE,
    PROJECT_ROOT,
    VALID_COHORT_CHOICES,
    cohort_spec_from_name,
)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
EMBEDDING_DIR = PROJECT_ROOT / "workspace" / "embedding" / "features"
LANDMARK_FEATURES_CSV = (
    PROJECT_ROOT / "workspace" / "asymmetry" / "features" / "pair_features.csv"
)

LONGITUDINAL_CSV = LONGITUDINAL_FEATURES_DIR / "patient_deltas.csv"
AD_DELTAS_CSV = LONGITUDINAL_FEATURES_DIR / "ad_patient_deltas.csv"
HC_LONGITUDINAL_CSV = LONGITUDINAL_FEATURES_DIR / "hc_patient_deltas.csv"
EACS_DELTAS_CSV = LONGITUDINAL_FEATURES_DIR / "eacs_patient_deltas.csv"

# ----------------------------------------------------------------------
# Constants & validators
# ----------------------------------------------------------------------
VALID_COHORT_MODES = tuple(VALID_COHORT_CHOICES)
VALID_HC_SOURCE_MODES = ("ACS", "ACS_ext", "EACS")
VALID_DESIGNS = (
    "cross_naive", "cross_matched",
    "longitudinal_naive", "longitudinal_matched",
)

EMBEDDING_MODELS = ["arcface", "topofr", "dlib"]
EMOTION_METHODS = [
    "openface", "libreface", "pyfeat", "dan",
    "hsemotion", "vit", "poster_pp", "fer",
]
ALL_MODALITIES = (
    [f"{m}/original" for m in EMBEDDING_MODELS]
    + [f"{m}/difference" for m in EMBEDDING_MODELS]
    + ["landmark", "predicted_age"]
    + EMOTION_METHODS
)

logger = logging.getLogger(__name__)


def _validate(cohort_mode, hc_source_mode):
    cohort_spec_from_name(cohort_mode)  # raises on invalid
    if hc_source_mode not in VALID_HC_SOURCE_MODES:
        raise ValueError(
            f"hc_source_mode must be one of {VALID_HC_SOURCE_MODES}, "
            f"got {hc_source_mode!r}"
        )


# ====================================================================
# Filters
# ====================================================================

def apply_p_cdr_filter(df, spec: CohortSpec):
    """Per-visit P-side CDR filter.  Keeps Global_CDR >= 0.5 unless
    spec.p_cdr == 'cdrall'."""
    if spec.p_cdr == "cdrall":
        return df
    cdr = pd.to_numeric(df.get("Global_CDR"), errors="coerce")
    return df[cdr >= 0.5]


def apply_hc_strict_filter(df, spec: CohortSpec):
    """HC HC cognitive filter.  Keeps CDR==0 OR (CDR.isna() AND MMSE>=26)
    when spec.hc_strict is True; returns df unmodified otherwise."""
    if not spec.hc_strict:
        return df
    cdr = pd.to_numeric(df.get("Global_CDR"), errors="coerce")
    mmse = pd.to_numeric(df.get("MMSE"), errors="coerce")
    keep = (cdr == 0) | (cdr.isna() & (mmse >= 26))
    return df[keep]


def apply_max_cdr_filter(df, min_cdr=0.5, group_col="base_id",
                         cdr_col="Global_CDR"):
    """Subject-level CDR filter: keep subjects whose max CDR across visits
    >= *min_cdr*.  Different from ``apply_p_cdr_filter`` which is per-visit."""
    cdr = pd.to_numeric(df[cdr_col], errors="coerce")
    max_cdr = cdr.groupby(df[group_col]).max()
    keep = max_cdr[max_cdr >= min_cdr].index
    return df[df[group_col].isin(keep)].copy()


_PRED_AGES_CACHE = None


def _load_predicted_ages():
    global _PRED_AGES_CACHE
    if _PRED_AGES_CACHE is None:
        with open(PREDICTED_AGES_FILE) as f:
            _PRED_AGES_CACHE = json.load(f)
    return _PRED_AGES_CACHE


def apply_predicted_age_filter(df, pred_ages=None, min_age=None,
                               id_col="ID"):
    """Keep rows whose *id_col* exists in *pred_ages*.  Optionally enforce
    predicted_age >= *min_age*."""
    if pred_ages is None:
        pred_ages = _load_predicted_ages()
    mask = df[id_col].astype(str).isin(pred_ages)
    if min_age is not None:
        mask = mask & df[id_col].astype(str).apply(
            lambda x: pred_ages.get(x, -1) >= min_age
        )
    return df[mask].copy()


def filter_pairs_by_predicted_age(matched, pairs, pred_ages=None):
    """Drop matched pairs where either side lacks a predicted_age entry.
    Returns (matched, pairs)."""
    if pairs is None or len(pairs) == 0:
        return matched, pairs
    if pred_ages is None:
        pred_ages = _load_predicted_ages()
    keep = (pairs["minor_id"].astype(str).isin(pred_ages)
            & pairs["major_id"].astype(str).isin(pred_ages))
    pairs = pairs[keep].copy().reset_index(drop=True)
    kept_ids = set(pairs["minor_id"]).union(pairs["major_id"])
    id_col = "ID" if "ID" in matched.columns else "base_id"
    matched = matched[matched[id_col].isin(kept_ids)].copy().reset_index(
        drop=True)
    return matched, pairs


def apply_followup_filter(df, min_days=180, col="follow_up_days"):
    """Keep rows where follow-up duration >= *min_days*."""
    return df[df[col] >= min_days].copy()


def apply_multivisit_filter(df, min_visits=2, group_col="base_id",
                            visit_col="visit"):
    """Keep rows from subjects with >= *min_visits* visits."""
    vcnt = df.groupby(group_col)[visit_col].count()
    keep = vcnt[vcnt >= min_visits].index
    return df[df[group_col].isin(keep)].copy()


# ====================================================================
# Feature-existence gate
# ====================================================================

_FEATURE_IDS_CACHE: dict = {}


def _ids_for_modality(modality: str) -> set:
    """Return set of visit-level IDs that have *modality* extracted.
    Results are lazily cached per modality key."""
    if modality in _FEATURE_IDS_CACHE:
        return _FEATURE_IDS_CACHE[modality]

    ids: set = set()
    if modality == "landmark":
        if LANDMARK_FEATURES_CSV.exists():
            lmk = pd.read_csv(LANDMARK_FEATURES_CSV)
            ids = set(lmk["subject_id"].astype(str))
    elif modality == "predicted_age":
        if PREDICTED_AGES_FILE.exists():
            with open(PREDICTED_AGES_FILE) as f:
                ids = set(json.load(f).keys())
    elif modality in EMOTION_METHODS:
        emo_dir = EMO_AU_FEATURES_DIR / modality
        if emo_dir.exists():
            ids = {p.stem for p in emo_dir.glob("*.npy")}
    elif "/" in modality:
        model, variant = modality.split("/", 1)
        d = EMBEDDING_DIR / model / variant
        if d.exists():
            ids = {p.stem for p in d.glob("*.npy")}
    else:
        d = EMBEDDING_DIR / modality / "original"
        if d.exists():
            ids = {p.stem for p in d.glob("*.npy")}

    _FEATURE_IDS_CACHE[modality] = ids
    return ids


def _ids_with_features(modalities=None):
    """Return set of visit-level IDs that have ALL specified modalities.

    modalities=None  -> backward compat: arcface/original AND landmark.
    modalities="all" -> intersection of all 14 modalities.
    modalities=[...] -> explicit list of modality keys.
    """
    if modalities is None:
        modalities = ["arcface/original", "landmark"]
    elif modalities == "all":
        modalities = list(ALL_MODALITIES)

    if not modalities:
        return set()

    result = _ids_for_modality(modalities[0])
    for m in modalities[1:]:
        result = result & _ids_for_modality(m)
    return result


def visits_with_features(include_emotion_proxy=False):
    """Return set of visit IDs with ArcFace original .npy extracted.

    When *include_emotion_proxy* is True, also include EACS subjects that
    have emotion CSVs but no embedding (allows longitudinal build to retain
    EACS rows).
    """
    ids = set()
    arcface_dir = EMBEDDING_DIR / "arcface" / "original"
    if arcface_dir.exists():
        ids.update(p.stem for p in arcface_dir.glob("*.npy"))
    if include_emotion_proxy:
        for tool in ("dan", "openface", "pyfeat", "vit"):
            emo_raw = (PROJECT_ROOT / "workspace" / "emo_au" / "features"
                       / "raw" / tool)
            if emo_raw.exists():
                ids.update(p.stem for p in emo_raw.glob("EACS_*.csv"))
    return ids


def _pick_first_visit_with_features(df_visits):
    """Per base_id, pick earliest visit with features; fall back to
    earliest visit if none has features."""
    good_ids = _ids_with_features()
    picked = []
    for _bid, g in df_visits.groupby("base_id", as_index=False, sort=False):
        g = g.sort_values("visit")
        with_feat = g[g["ID"].astype(str).isin(good_ids)]
        if len(with_feat) > 0:
            picked.append(with_feat.iloc[0])
        else:
            picked.append(g.iloc[0])
    return pd.DataFrame(picked).reset_index(drop=True)


def _keep_visits_with_features(df_visits):
    """Keep ALL feature-bearing visits (no per-subject pick)."""
    good_ids = _ids_with_features()
    return df_visits[
        df_visits["ID"].astype(str).isin(good_ids)
    ].reset_index(drop=True)


# ====================================================================
# Demographic loaders
# ====================================================================

def load_p_demographics():
    """Load P.csv with parsed numeric / base_id / visit columns."""
    df = pd.read_csv(DEMOGRAPHICS_DIR / "P.csv")
    df["MMSE"] = pd.to_numeric(df["MMSE"], errors="coerce")
    df["CASI"] = pd.to_numeric(df.get("CASI"), errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Global_CDR"] = pd.to_numeric(df.get("Global_CDR"), errors="coerce")
    df["base_id"] = df["ID"].str.extract(r"^([A-Za-z]+\d+)")
    df["visit"] = df["ID"].str.extract(r"-(\d+)$").astype(float)
    return df


def select_visit(df, visit_selection="first", metric="MMSE"):
    """Pick one visit per base_id where *metric* and Age are present."""
    elig = df.dropna(subset=[metric, "Age"]).copy()
    elig = elig.sort_values(["base_id", "visit"])
    if visit_selection == "first":
        return elig.groupby("base_id", as_index=False).first()
    return elig.groupby("base_id", as_index=False).last()


def split_by_metric_median(cohort, tiebreak="high", metric="MMSE",
                           group_col=None):
    """Median split of cohort by metric.  Adds <metric_low>_group col."""
    if group_col is None:
        group_col = f"{metric.lower()}_group"
    median = cohort[metric].median()
    if tiebreak == "high":
        cohort[group_col] = np.where(cohort[metric] >= median, "high", "low")
    else:
        cohort[group_col] = np.where(cohort[metric] > median, "high", "low")
    return cohort, float(median)


# ====================================================================
# HC visit selection + optional HC cognitive strict filter
# ====================================================================

def _select_hc_visits(demo, hc_source, spec: CohortSpec,
                      visit_selection="all"):
    """HC group-source filter with optional first-visit dedup."""
    if hc_source == "HC":
        mask = demo["group"].isin(["NAD", "ACS"])
    else:
        mask = demo["group"] == hc_source
    sub = demo[mask].copy()
    sub["Global_CDR"] = pd.to_numeric(sub.get("Global_CDR"), errors="coerce")
    sub["MMSE"] = pd.to_numeric(sub.get("MMSE"), errors="coerce")
    sub = apply_hc_strict_filter(sub, spec)
    sub = sub.sort_values(["base_id", "visit"])
    if visit_selection == "first":
        sub = sub.groupby("base_id", as_index=False).first()
    sub["label"] = 0
    return sub


# ====================================================================
# 1:1 age nearest-neighbor matching
# ====================================================================

def match_1to1(cohort, caliper=2.0, seed=42, metric="MMSE", group_col=None,
               match_mode="visit", priority_groups=None, id_col="ID",
               match_level="subject"):
    """1:1 age-optimal match; cohort must have *group_col* 'high'/'low'.

    Uses ``scipy.optimize.linear_sum_assignment`` to maximise the number
    of matched pairs within *caliper* while minimising total age difference.

    When *priority_groups* is set (e.g. ``["ACS"]``), subjects in those
    groups are matched first in a dedicated optimal-assignment round;
    remaining subjects are matched in a second round against the leftover
    major pool.

    match_level="subject": dedup to one row per base_id before matching.
    match_level="visit": each visit is an independent candidate; same
        subject can appear in multiple pairs (different visits).

    Returns (matched_df, pairs_df, (minor_label, major_label)).
    """
    from scipy.optimize import linear_sum_assignment

    if group_col is None:
        group_col = f"{metric.lower()}_group"
    metric_low = metric.lower() if metric else None
    rng = np.random.RandomState(seed)

    high = cohort[cohort[group_col] == "high"].copy()
    low = cohort[cohort[group_col] == "low"].copy()
    if len(low) <= len(high):
        minor, major = low, high
        minor_label, major_label = "low", "high"
    else:
        minor, major = high, low
        minor_label, major_label = "high", "low"

    subject_level = (match_level == "subject") and ("base_id" in cohort.columns)
    bid_col = "base_id" if subject_level else id_col

    def _to_subject_df(df):
        return (df.sort_values([bid_col, "Age"])
                .drop_duplicates(bid_col, keep="first")
                .reset_index(drop=True))

    if subject_level:
        minor_subj = _to_subject_df(minor)
        major_subj = _to_subject_df(major)
    else:
        minor_subj = minor.reset_index(drop=True)
        major_subj = major.reset_index(drop=True)

    def _optimal_assign(mi, ma):
        if len(mi) == 0 or len(ma) == 0:
            return []
        age_mi = mi["Age"].to_numpy(float)
        age_ma = ma["Age"].to_numpy(float)
        cost = np.abs(age_mi[:, None] - age_ma[None, :])
        cost[cost > caliper] = 1e9
        ri, ci = linear_sum_assignment(cost)
        valid = cost[ri, ci] <= caliper
        return list(zip(ri[valid], ci[valid]))

    pairs_records = []

    def _record_pairs(mi_df, ma_df, assignments):
        for ri, ci in assignments:
            mi_row = mi_df.iloc[ri]
            ma_row = ma_df.iloc[ci]
            rec = {
                "pair_id": None,
                "minor_id": mi_row[id_col], "minor_age": mi_row["Age"],
                "major_id": ma_row[id_col], "major_age": ma_row["Age"],
                "age_diff": ma_row["Age"] - mi_row["Age"],
            }
            if metric_low:
                rec[f"minor_{metric_low}"] = mi_row[metric]
                rec[f"major_{metric_low}"] = ma_row[metric]
            pairs_records.append(rec)

    _pg_set = set(priority_groups) if priority_groups else set()
    _has_pg = bool(_pg_set) and "group" in cohort.columns
    if _has_pg:
        grp_on_minor = bool(_pg_set & set(minor_subj["group"].unique()))
        used_minor_bids = set()
        used_major_bids = set()
        for grp in priority_groups:
            if grp_on_minor:
                mi_arg = minor_subj[
                    (minor_subj["group"] == grp)
                    & ~minor_subj[bid_col].isin(used_minor_bids)
                ].reset_index(drop=True)
                ma_arg = major_subj[
                    ~major_subj[bid_col].isin(used_major_bids)
                ].reset_index(drop=True)
            else:
                mi_arg = minor_subj[
                    ~minor_subj[bid_col].isin(used_minor_bids)
                ].reset_index(drop=True)
                ma_arg = major_subj[
                    (major_subj["group"] == grp)
                    & ~major_subj[bid_col].isin(used_major_bids)
                ].reset_index(drop=True)
            assignments = _optimal_assign(mi_arg, ma_arg)
            _record_pairs(mi_arg, ma_arg, assignments)
            used_minor_bids.update(
                mi_arg.iloc[ri][bid_col] for ri, _ in assignments)
            used_major_bids.update(
                ma_arg.iloc[ci][bid_col] for _, ci in assignments)

        mi_rest = minor_subj[
            ~minor_subj[bid_col].isin(used_minor_bids)
        ].reset_index(drop=True)
        ma_rest = major_subj[
            ~major_subj[bid_col].isin(used_major_bids)
        ].reset_index(drop=True)
        assignments = _optimal_assign(mi_rest, ma_rest)
        _record_pairs(mi_rest, ma_rest, assignments)
    else:
        assignments = _optimal_assign(minor_subj, major_subj)
        _record_pairs(minor_subj, major_subj, assignments)

    matched_records = []
    for i, rec in enumerate(pairs_records):
        rec["pair_id"] = i
        minor_rec = {
            "pair_id": i, id_col: rec["minor_id"],
            "Age": rec["minor_age"], group_col: minor_label,
        }
        major_rec = {
            "pair_id": i, id_col: rec["major_id"],
            "Age": rec["major_age"], group_col: major_label,
        }
        if metric_low:
            minor_rec[metric] = rec[f"minor_{metric_low}"]
            major_rec[metric] = rec[f"major_{metric_low}"]
        matched_records.append(minor_rec)
        matched_records.append(major_rec)
    pairs_df = pd.DataFrame(pairs_records)
    matched = pd.DataFrame(matched_records)
    return matched, pairs_df, (minor_label, major_label)



def build_caliper_group(full_cohort, matched_cohort, matched_pairs,
                        keep_groups=None, caliper=1.0,
                        ttest_threshold=0.05):
    """1:N balanced matching built on top of a 1:1 matched cohort.

    Starting from the 1:1 pairs, round-robin adds P subjects to each HC
    subject (one per HC per round, within *caliper*).  After each round a
    Welch t-test checks age balance; the process stops (and rolls back the
    last round) when the t-test drops below *ttest_threshold*.

    Parameters
    ----------
    full_cohort : DataFrame  — full (unmatched) cohort with base_id, Age,
        label, group.
    matched_cohort : DataFrame — 1:1 matched cohort with pair_id, base_id,
        Age, label, group.
    matched_pairs : DataFrame — pairs table from ``match_1to1`` (minor_id,
        major_id columns).
    keep_groups : optional set  — e.g. ``{"P", "ACS"}``.
    caliper : float — max |age_diff| for added P subjects.
    ttest_threshold : float — stop adding when p drops below this.

    Returns
    -------
    expanded_cohort : DataFrame (base_id, Age, label, group)
    age_balance : dict
    """
    from scipy import stats as _st

    cols = ["base_id", "Age", "label", "group"]
    mc = matched_cohort.drop_duplicates("base_id")

    # --- HC side (fixed) ---
    if keep_groups is not None:
        hc_groups = set(keep_groups) - {"P"}
        hc_subj = mc[mc["group"].isin(hc_groups)][cols].copy()
    else:
        hc_subj = mc[mc["label"] == 0][cols].copy()

    # --- P already in 1:1 ---
    matched_p_bids = set(mc[mc["label"] == 1]["base_id"])
    if keep_groups is not None:
        hc_pair_ids = set(
            matched_cohort[matched_cohort["group"].isin(hc_groups)]["pair_id"])
        matched_p_in_scope = set(
            matched_cohort[
                (matched_cohort["pair_id"].isin(hc_pair_ids))
                & (matched_cohort["label"] == 1)
            ]["base_id"])
    else:
        matched_p_in_scope = matched_p_bids
    p_matched = (full_cohort[full_cohort["base_id"].isin(matched_p_in_scope)]
                 .drop_duplicates("base_id")[cols].copy())

    # --- candidate P pool (not yet matched) ---
    all_p = (full_cohort[full_cohort["label"] == 1]
             .drop_duplicates("base_id"))
    p_candidates = all_p[~all_p["base_id"].isin(matched_p_bids)].copy()

    # --- build global (HC, P_candidate) pairs sorted by |age_diff| ---
    hc_records = hc_subj[["base_id", "Age"]].to_dict("records")
    cand_records = p_candidates[["base_id", "Age"]].to_dict("records")
    cand_age_map = {r["base_id"]: r["Age"] for r in cand_records}

    all_pairs = []
    for hc in hc_records:
        for c in cand_records:
            d = abs(c["Age"] - hc["Age"])
            if d <= caliper:
                all_pairs.append((d, c["base_id"], c["Age"]))
    all_pairs.sort()

    # --- greedy addition: add one P at a time, check t-test ---
    added_p_bids = []
    added_p_ages = []
    used_p = set()

    hc_ages = hc_subj["Age"].to_numpy(float)
    p_ages_base = p_matched["Age"].to_numpy(float)
    n_base = len(p_ages_base)

    for _, p_bid, p_age in all_pairs:
        if p_bid in used_p:
            continue

        trial_ages = np.concatenate([
            p_ages_base,
            np.array(added_p_ages + [p_age], dtype=float),
        ])
        _, pval = _st.ttest_ind(trial_ages, hc_ages, equal_var=False)

        if pval < ttest_threshold:
            continue

        used_p.add(p_bid)
        added_p_bids.append(p_bid)
        added_p_ages.append(p_age)

    # --- assemble final cohort ---
    added_p_df = (all_p[all_p["base_id"].isin(set(added_p_bids))]
                  [cols].copy())
    expanded = pd.concat([hc_subj, p_matched, added_p_df],
                         ignore_index=True)

    all_p_in = expanded[expanded["label"] == 1]
    p_arr_final = all_p_in["Age"].to_numpy(float)
    if len(p_arr_final) >= 2 and len(hc_ages) >= 2:
        t_stat, t_pval = _st.ttest_ind(p_arr_final, hc_ages, equal_var=False)
    else:
        t_stat, t_pval = float("nan"), float("nan")

    age_balance = {
        "caliper": caliper,
        "ttest_threshold": ttest_threshold,
        "n_hc": len(hc_subj),
        "n_p_matched_1to1": len(p_matched),
        "n_p_added": len(added_p_bids),
        "n_p_total": len(all_p_in),
        "n_p_pool": len(all_p),
        "hc_age_mean": float(hc_ages.mean()),
        "p_age_mean": float(p_arr_final.mean()) if len(p_arr_final) else None,
        "ttest_t": float(t_stat),
        "ttest_p": float(t_pval),
    }
    return expanded, age_balance


# ====================================================================
# Demographics loading for AD-vs-HC-group cohorts
# ====================================================================

def _load_combined_demographics(hc_source_mode):
    """Load + concat demographics across P/NAD/ACS/EACS per
    hc_source_mode."""
    frames = []
    groups_to_load = ["P", "NAD"]
    if hc_source_mode != "EACS":
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
    if hc_source_mode in ("ACS_ext", "EACS"):
        df_e = pd.read_csv(DEMOGRAPHICS_DIR / "EACS.csv")
        eacs_sources_env = os.environ.get("EACS_SOURCES", "").strip()
        if eacs_sources_env:
            wanted = {s.strip()
                      for s in eacs_sources_env.split(",") if s.strip()}
            if "Source" in df_e.columns:
                df_e = df_e[df_e["Source"].isin(wanted)].copy()
                logger.info(
                    f"filtered EACS to sources {wanted}: {len(df_e)} rows")
        df_e["group"] = "ACS"
        frames.append(df_e)
    demo = pd.concat(frames, ignore_index=True)
    if "Source" not in demo.columns:
        demo["Source"] = "internal"
    demo["Source"] = demo["Source"].fillna("internal")
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["Global_CDR"] = pd.to_numeric(
        demo.get("Global_CDR"), errors="coerce")
    demo["MMSE"] = pd.to_numeric(demo.get("MMSE"), errors="coerce")
    demo["base_id"] = demo["ID"].str.extract(r"^(.+)-\d+$")
    demo["visit"] = demo["ID"].str.extract(r"-(\d+)$").astype(float)
    return demo


# ====================================================================
# Main: AD vs HC-group cohort builder
# ====================================================================

def build_cohort_ad_vs_HCgroup(
    hc_source,
    design,
    cohort_mode=DEFAULT_COHORT_MODE,
    hc_source_mode="ACS",
    caliper=1.0,
    seed=42,
    priority_groups=None,
    match_level="subject",
):
    """Build cohort for AD vs {HC, NAD, ACS}.  Returns (cohort_df,
    pairs_df)."""
    _validate(cohort_mode, hc_source_mode)
    if design not in VALID_DESIGNS:
        raise ValueError(
            f"design must be one of {VALID_DESIGNS}, got {design!r}")

    spec = cohort_spec_from_name(cohort_mode)
    demo = _load_combined_demographics(hc_source_mode)

    # AD side
    ad_all = demo[(demo["group"] == "P") & demo["Age"].notna()].copy()
    ad_all = apply_p_cdr_filter(ad_all, spec)
    ad_all = ad_all.sort_values(["base_id", "visit"])
    if spec.p_visit == "all":
        ad = _keep_visits_with_features(ad_all)
    else:
        ad = _pick_first_visit_with_features(ad_all)
    ad["label"] = 1

    # HC side
    hc_all = _select_hc_visits(demo, hc_source, spec, visit_selection="all")
    hc_all = hc_all[hc_all["Age"].notna()].copy()
    if spec.hc_visit == "all":
        hc = hc_all.reset_index(drop=True)
    else:
        hc = _pick_first_visit_with_features(hc_all)

    if design == "cross_naive":
        cohort = pd.concat([ad, hc], ignore_index=True)
        cohort["mmse_group"] = np.nan
        return cohort, None

    if design == "cross_matched":
        prep = pd.concat([ad, hc], ignore_index=True)
        prep["mmse_group"] = np.where(prep["label"] == 1, "high", "low")
        prep["MMSE"] = prep["MMSE"].fillna(999)
        matched, pairs, _ = match_1to1(
            prep, caliper=caliper, seed=seed, metric="MMSE",
            group_col="mmse_group",
            priority_groups=priority_groups,
            match_level=match_level,
        )
        cohort = matched.merge(
            prep[["ID", "base_id", "group", "Age", "MMSE", "Global_CDR",
                  "label"]].drop_duplicates("ID"),
            on="ID", how="left", suffixes=("", "_p"),
        )
        cohort = cohort.drop(
            columns=[c for c in cohort.columns if c.endswith("_p")])
        return cohort, pairs

    # longitudinal_naive | longitudinal_matched
    do_match = (design == "longitudinal_matched")
    return _build_longitudinal_cohort_hc(
        hc_source, demo, hc_source_mode, spec, caliper, seed,
        do_match=do_match)


# ====================================================================
# Longitudinal cohort builder
# ====================================================================

def _build_longitudinal_cohort_hc(hc_source, demo, hc_source_mode,
                                  spec: CohortSpec, caliper, seed,
                                  do_match=True):
    """Longitudinal AD vs {HC/NAD/ACS} cohort."""
    if AD_DELTAS_CSV.exists():
        ad_delta = pd.read_csv(AD_DELTAS_CSV)
        ad_delta = ad_delta[
            (ad_delta["follow_up_days"] >= 180)
            & ad_delta["first_age"].notna()
        ].copy()
    else:
        ad_delta = pd.read_csv(LONGITUDINAL_CSV)
        ad_delta = ad_delta[
            (ad_delta["follow_up_days"] >= 180)
            & ad_delta["first_age"].notna()
        ].copy()
        ad_delta = _annualize_patient_deltas(ad_delta)
    ad_delta["label"] = 1
    if "group" not in ad_delta.columns:
        ad_delta["group"] = "P"

    if not HC_LONGITUDINAL_CSV.exists():
        raise FileNotFoundError(
            f"Missing {HC_LONGITUDINAL_CSV}; run "
            f"build_longitudinal_hc_and_vectors.py first")
    hc_delta_all = pd.read_csv(HC_LONGITUDINAL_CSV)

    if hc_source_mode in ("ACS_ext", "EACS") and EACS_DELTAS_CSV.exists():
        eacs_delta_all = pd.read_csv(EACS_DELTAS_CSV)
        eacs_delta_all["group"] = "ACS"
        if hc_source_mode == "EACS":
            hc_delta_all = eacs_delta_all
        else:
            hc_delta_all = pd.concat(
                [hc_delta_all, eacs_delta_all],
                ignore_index=True, sort=False)

    hc_pool = _select_hc_visits(demo, hc_source, spec, visit_selection="first")
    allowed_base = set(hc_pool["base_id"])
    hc_delta = hc_delta_all[
        hc_delta_all["base_id"].isin(allowed_base)].copy()
    if hc_source in ("NAD", "ACS"):
        hc_delta = hc_delta[hc_delta["group"] == hc_source]
    hc_delta["label"] = 0

    if hc_delta.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_delta = pd.concat([ad_delta, hc_delta], ignore_index=True,
                          sort=False)

    if not do_match:
        all_delta["mmse_group"] = np.where(
            all_delta["label"] == 1, "high", "low")
        return all_delta, None

    ad_delta["mmse_group"] = "high"
    hc_delta["mmse_group"] = "low"
    prep = pd.concat([
        ad_delta[["base_id", "first_age", "mmse_group",
                  "follow_up_years", "follow_up_days", "label"]].rename(
            columns={"first_age": "Age"}),
        hc_delta[["base_id", "first_age", "mmse_group",
                  "follow_up_years", "follow_up_days", "label"]].rename(
            columns={"first_age": "Age"}),
    ], ignore_index=True)
    matched, pairs_df, _ = match_1to1(
        prep, caliper=caliper, seed=seed, metric=None,
        group_col="mmse_group", id_col="base_id")
    cohort = matched.merge(all_delta, on="base_id", how="left",
                           suffixes=("", "_p"))
    return cohort, pairs_df


def _annualize_patient_deltas(df):
    """Add ann_* columns to patient_deltas-style DataFrame."""
    df = df.copy()
    y = df["follow_up_years"].replace(0, np.nan)
    for col in list(df.columns):
        if (col.startswith("delta_") or col.startswith("emb_cosine_dist_")
                or col == "emb_cosine_dist"):
            df[f"ann_{col}"] = df[col] / y
    if "ann_emb_cosine_dist_arcface" in df.columns:
        df["ann_emb_cosine_dist"] = df["ann_emb_cosine_dist_arcface"]
    return df


# ====================================================================
# AD-internal hi-lo cohort builder
# ====================================================================

def build_cohort_ad_hi_lo(
    design,
    cohort_mode=DEFAULT_COHORT_MODE,
    caliper=2.0,
    seed=42,
    metric="MMSE",
    matched_features_csv=None,
):
    """AD-internal hi-lo cohort.  metric in {'MMSE', 'CASI'}."""
    if design not in VALID_DESIGNS:
        raise ValueError(
            f"design must be one of {VALID_DESIGNS}, got {design!r}")
    spec = cohort_spec_from_name(cohort_mode)

    metric_low = metric.lower()
    group_col = f"{metric_low}_group"
    first_metric = f"first_{metric}"

    if design == "cross_naive":
        demo = load_p_demographics()
        demo["Global_CDR"] = pd.to_numeric(
            demo.get("Global_CDR"), errors="coerce")
        if spec.p_visit == "all":
            ad_all = demo[
                demo[metric].notna() & demo["Age"].notna()].copy()
            ad_all = apply_p_cdr_filter(ad_all, spec)
            ad_all = ad_all.sort_values(["base_id", "visit"])
            cohort = _keep_visits_with_features(ad_all)
        else:
            cohort = demo.dropna(subset=[metric, "Age"]).copy()
            cohort = cohort.sort_values(
                ["base_id", "visit"]).groupby(
                "base_id", as_index=False).first()
            cohort = apply_p_cdr_filter(cohort, spec).copy()
        med = cohort[metric].median()
        cohort[group_col] = np.where(
            cohort[metric] >= med, "high", "low")
        cohort["label"] = (cohort[group_col] == "high").astype(int)
        return cohort, None

    if design == "cross_matched":
        if matched_features_csv is None:
            raise ValueError(
                "design='cross_matched' for hi-lo requires "
                "matched_features_csv")
        if not Path(matched_features_csv).exists():
            raise FileNotFoundError(
                f"Run run_cross_matched.py --comparison "
                f"{metric_low}_hilo first; "
                f"missing: {matched_features_csv}")
        cohort = pd.read_csv(matched_features_csv)
        cohort["label"] = (cohort[group_col] == "high").astype(int)
        p_df = pd.read_csv(DEMOGRAPHICS_DIR / "P.csv")
        keep_cog = [c for c in ["MMSE", "CASI", "Global_CDR", "CDR_SB"]
                    if c in p_df.columns and c not in cohort.columns]
        if keep_cog:
            cohort = cohort.merge(
                p_df[["ID"] + keep_cog].drop_duplicates("ID"),
                on="ID", how="left")
            for c in keep_cog:
                cohort[c] = pd.to_numeric(cohort[c], errors="coerce")
        return cohort, None

    if design == "longitudinal_naive":
        if not AD_DELTAS_CSV.exists():
            raise FileNotFoundError(
                f"Missing {AD_DELTAS_CSV}; run "
                f"build_longitudinal_hc_and_vectors.py first")
        ad_d = pd.read_csv(AD_DELTAS_CSV)
        ad_d = ad_d[ad_d["follow_up_days"] >= 180].copy()
        ad_d = ad_d[
            ad_d[first_metric].notna() & ad_d["first_age"].notna()]
        med = ad_d[first_metric].median()
        ad_d[group_col] = np.where(
            ad_d[first_metric] >= med, "high", "low")
        ad_d["label"] = (ad_d[group_col] == "high").astype(int)
        return ad_d, None

    # design == 'longitudinal_matched'
    if not AD_DELTAS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {AD_DELTAS_CSV}; run "
            f"build_longitudinal_hc_and_vectors.py first")
    ad_d = pd.read_csv(AD_DELTAS_CSV)
    ad_d = ad_d[ad_d["follow_up_days"] >= 180].copy()
    ad_d = ad_d[ad_d[first_metric].notna() & ad_d["first_age"].notna()]
    med = ad_d[first_metric].median()
    ad_d[group_col] = np.where(
        ad_d[first_metric] >= med, "high", "low")
    prep = ad_d.copy()
    prep["Age"] = prep["first_age"]
    matched, pairs_df, _ = match_1to1(
        prep, caliper=caliper, seed=seed, metric=None,
        group_col=group_col, id_col="base_id")
    cohort = matched.merge(ad_d, on="base_id", how="left",
                           suffixes=("", "_p"))
    cohort["label"] = (cohort[group_col] == "high").astype(int)
    return cohort, pairs_df
