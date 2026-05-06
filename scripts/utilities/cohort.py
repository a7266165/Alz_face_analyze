"""
Cohort builders + 1:1 matching helpers for AD vs HC-group and AD hi-lo
(MMSE/CASI median split) analyses. Single source of truth for cohort_mode /
hc_source_mode handling.

Consumed by:
  - scripts/overview/run_cross_naive.py
  - scripts/overview/run_cross_matched.py
  - scripts/overview/run_stat_grid.py
  - scripts/age/run_classifiers.py
  - scripts/embedding/run_fwd_rev.py

Design parameter `design` replaces the legacy `arm` letter:
  cross_naive          (was arm A)  — AD ∪ HC-group, no matching
  cross_matched        (was arm B)  — 1:1 age NN, caliper=2y
  longitudinal_naive   (was arm C)  — longitudinal Δ, no matching
  longitudinal_matched (was arm D)  — longitudinal Δ + 1:1 baseline-age NN

cohort_mode: 'default' | 'p_first_hc_all' | 'p_all_hc_all' (see cohort_name in src.config)
hc_source_mode: 'ACS' | 'ACS_ext' | 'EACS' (controls ACS-group composition)

Migration note: cohort_mode and hc_source_mode were previously module-level
globals read from os.environ at import time (legacy `run_4arm_deep_dive.py`,
since split into `scripts/overview/run_cross_naive.py`,
`scripts/overview/run_cross_matched.py`, `scripts/overview/run_stat_grid.py`).
They are now explicit function parameters; this module reads no env vars.
EACS_SOURCES env (sub-source filter) remains read inside the function for
parity with the legacy behaviour.
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import LONGITUDINAL_FEATURES_DIR

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
EMBEDDING_DIR = PROJECT_ROOT / "workspace" / "embedding" / "features"
LANDMARK_FEATURES_CSV = PROJECT_ROOT / "workspace" / "asymmetry" / "features" / "pair_features.csv"

LONGITUDINAL_CSV = LONGITUDINAL_FEATURES_DIR / "patient_deltas.csv"
AD_DELTAS_CSV = LONGITUDINAL_FEATURES_DIR / "ad_patient_deltas.csv"
HC_LONGITUDINAL_CSV = LONGITUDINAL_FEATURES_DIR / "hc_patient_deltas.csv"
EACS_DELTAS_CSV = LONGITUDINAL_FEATURES_DIR / "eacs_patient_deltas.csv"

# ----------------------------------------------------------------------
# Constants & validators
# ----------------------------------------------------------------------
VALID_COHORT_MODES = ("default", "p_first_hc_all", "p_all_hc_all")
VALID_HC_SOURCE_MODES = ("ACS", "ACS_ext", "EACS")
VALID_DESIGNS = (
    "cross_naive", "cross_matched",
    "longitudinal_naive", "longitudinal_matched",
)

logger = logging.getLogger(__name__)


def _validate(cohort_mode, hc_source_mode):
    if cohort_mode not in VALID_COHORT_MODES:
        raise ValueError(f"cohort_mode must be one of {VALID_COHORT_MODES}, got {cohort_mode!r}")
    if hc_source_mode not in VALID_HC_SOURCE_MODES:
        raise ValueError(f"hc_source_mode must be one of {VALID_HC_SOURCE_MODES}, got {hc_source_mode!r}")


# ============================================================
# Demographic loaders
# ============================================================

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
    """Pick one visit per base_id where <metric> and Age are present."""
    elig = df.dropna(subset=[metric, "Age"]).copy()
    elig = elig.sort_values(["base_id", "visit"])
    if visit_selection == "first":
        return elig.groupby("base_id", as_index=False).first()
    return elig.groupby("base_id", as_index=False).last()


def split_by_metric_median(cohort, tiebreak="high", metric="MMSE", group_col=None):
    """Median split of cohort by metric. Adds <metric_low>_group col."""
    if group_col is None:
        group_col = f"{metric.lower()}_group"
    median = cohort[metric].median()
    if tiebreak == "high":
        cohort[group_col] = np.where(cohort[metric] >= median, "high", "low")
    else:
        cohort[group_col] = np.where(cohort[metric] > median, "high", "low")
    return cohort, float(median)


# ============================================================
# Feature-existence cache (drives "first-visit-with-features" logic)
# ============================================================

_FEATURE_IDS_CACHE = None


def _ids_with_features():
    """Return set of visit-level IDs that have BOTH arcface embedding +
    landmark features. Used to fall back to a feature-bearing visit when the
    earliest visit lacks one or the other (recovers AD subjects whose visit-1
    is missing from the raw dataset)."""
    global _FEATURE_IDS_CACHE
    if _FEATURE_IDS_CACHE is None:
        emb_dir = EMBEDDING_DIR / "arcface" / "original"
        emb_ids = {p.stem for p in emb_dir.glob("*.npy")}
        lmk = pd.read_csv(LANDMARK_FEATURES_CSV)
        lmk_ids = set(lmk["subject_id"].astype(str))
        _FEATURE_IDS_CACHE = emb_ids & lmk_ids
    return _FEATURE_IDS_CACHE


def _pick_first_visit_with_features(df_visits):
    """Per base_id, pick earliest visit with both embedding + landmark
    features; fall back to earliest visit if none does."""
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
    """Like _pick_first_visit_with_features but keeps ALL feature-bearing
    visits (no per-subject pick). Used for p_all_hc_all where we don't pick
    first-visit on either side."""
    good_ids = _ids_with_features()
    return df_visits[df_visits["ID"].astype(str).isin(good_ids)].reset_index(drop=True)


# ============================================================
# HC filter (group-source only; cognitive filter dropped 2026-05-06)
# ============================================================
# Function names retain the `_strict_` prefix for backward call-site
# compatibility, but the strict cognitive criterion (CDR=0 or MMSE>=26)
# has been removed — HC is now defined purely by group membership
# (NAD / ACS / etc) plus visit selection.

def _strict_hc_filter_all_visits(demo, hc_source, cohort_mode):
    """HC group-source filter, all qualifying visits kept.

    Previously also enforced strict cognitive criteria (CDR=0 OR
    (CDR=NaN AND MMSE>=26)); that filter was dropped 2026-05-06.
    `cohort_mode` is now informational — all modes share the same
    group-only filter.
    """
    if hc_source == "HC":
        mask = demo["group"].isin(["NAD", "ACS"])
    else:
        mask = demo["group"] == hc_source
    sub = demo[mask].copy()
    sub["Global_CDR"] = pd.to_numeric(sub.get("Global_CDR"), errors="coerce")
    sub["MMSE"] = pd.to_numeric(sub.get("MMSE"), errors="coerce")
    sub = sub.sort_values(["base_id", "visit"])
    sub["label"] = 0
    return sub


def _strict_hc_filter(demo, hc_source):
    """HC group-source filter, first visit per base_id.

    Used by longitudinal cohorts to determine the allowed HC pool.
    Previously also enforced strict cognitive criteria; that filter was
    dropped 2026-05-06.
    """
    if hc_source == "HC":
        mask = demo["group"].isin(["NAD", "ACS"])
    else:
        mask = demo["group"] == hc_source
    sub = demo[mask].copy()
    sub["Global_CDR"] = pd.to_numeric(sub.get("Global_CDR"), errors="coerce")
    sub["MMSE"] = pd.to_numeric(sub.get("MMSE"), errors="coerce")
    sub = sub.sort_values(["base_id", "visit"]).groupby("base_id", as_index=False).first()
    sub["label"] = 0
    return sub


# ============================================================
# 1:1 age nearest-neighbor matching
# ============================================================

def match_1to1(cohort, caliper=2.0, seed=42, metric="MMSE", group_col=None,
               match_mode="visit"):
    """1:1 age NN match; cohort must have <group_col> set to 'high'/'low'.

    match_mode:
      'visit'         each visit can match once. Same major subject can appear
                      multiple times via different visits.
      'subject_first' two-pass: pass 1 limits each major base_id to ≤1 use
                      (subject-level 1:1); pass 2 picks up unmatched minor rows
                      and lets them match remaining visits from already-used
                      major subjects (visit-level fallback). Requires `base_id`
                      column. Pairs gain `match_pass` ∈ {'subject', 'visit_fallback'}.

    Returns (matched_df, pairs_df, (minor_label, major_label)).
    """
    if group_col is None:
        group_col = f"{metric.lower()}_group"
    if match_mode not in ("visit", "subject_first"):
        raise ValueError(f"match_mode must be 'visit' or 'subject_first', got {match_mode!r}")
    if match_mode == "subject_first" and "base_id" not in cohort.columns:
        raise ValueError("match_mode='subject_first' requires base_id column in cohort")
    metric_low = metric.lower()
    rng = np.random.RandomState(seed)
    high = cohort[cohort[group_col] == "high"].copy()
    low = cohort[cohort[group_col] == "low"].copy()

    if len(low) <= len(high):
        minor, major = low, high
        minor_label, major_label = "low", "high"
    else:
        minor, major = high, low
        minor_label, major_label = "high", "low"

    minor_order = minor.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    available = major.copy().reset_index(drop=True)

    def _make_pair(minor_row, picked_row, pass_label):
        rec = {
            "pair_id": None,
            "minor_id": minor_row["ID"], "minor_age": minor_row["Age"],
            f"minor_{metric_low}": minor_row[metric],
            "major_id": picked_row["ID"], "major_age": picked_row["Age"],
            f"major_{metric_low}": picked_row[metric],
            "age_diff": picked_row["Age"] - minor_row["Age"],
        }
        if match_mode == "subject_first":
            rec["match_pass"] = pass_label
        return rec

    def _pick_nearest(cand_pool, age):
        diffs = (cand_pool["Age"] - age).abs()
        md = diffs.min()
        if pd.isna(md) or md > caliper:
            return None, None
        cands = cand_pool[diffs == md].sort_values("ID")
        return cands.iloc[0], md

    pairs_records = []

    if match_mode == "visit":
        for _, row in minor_order.iterrows():
            if len(available) == 0:
                break
            picked, _ = _pick_nearest(available, row["Age"])
            if picked is None:
                continue
            pairs_records.append(_make_pair(row, picked, "visit"))
            available = available.drop(picked.name).reset_index(drop=True)
    else:
        # Pass 1: subject-level 1:1
        used_major_subjects = set()
        used_minor_subjects = set()
        matched_minor_ids = set()
        for _, row in minor_order.iterrows():
            if row["base_id"] in used_minor_subjects:
                continue
            cand_pool = available[~available["base_id"].isin(used_major_subjects)]
            if len(cand_pool) == 0:
                break
            picked, _ = _pick_nearest(cand_pool, row["Age"])
            if picked is None:
                continue
            pairs_records.append(_make_pair(row, picked, "subject"))
            used_major_subjects.add(picked["base_id"])
            used_minor_subjects.add(row["base_id"])
            matched_minor_ids.add(row["ID"])
            available = available.drop(picked.name).reset_index(drop=True)
        # Pass 2: visit-level fallback
        unmatched_minor = minor_order[~minor_order["ID"].isin(matched_minor_ids)]
        for _, row in unmatched_minor.iterrows():
            if len(available) == 0:
                break
            picked, _ = _pick_nearest(available, row["Age"])
            if picked is None:
                continue
            pairs_records.append(_make_pair(row, picked, "visit_fallback"))
            available = available.drop(picked.name).reset_index(drop=True)

    for i, rec in enumerate(pairs_records):
        rec["pair_id"] = i
    pairs_df = pd.DataFrame(pairs_records)

    records = []
    for _, p in pairs_df.iterrows():
        records.append({
            "pair_id": p["pair_id"], "ID": p["minor_id"],
            "Age": p["minor_age"], metric: p[f"minor_{metric_low}"],
            group_col: minor_label,
        })
        records.append({
            "pair_id": p["pair_id"], "ID": p["major_id"],
            "Age": p["major_age"], metric: p[f"major_{metric_low}"],
            group_col: major_label,
        })
    matched = pd.DataFrame(records)
    return matched, pairs_df, (minor_label, major_label)


# ============================================================
# Demographics loading for AD-vs-HC-group cohorts
# ============================================================

def _load_combined_demographics(hc_source_mode):
    """Load + concat demographics across P/NAD/ACS/EACS per hc_source_mode."""
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
            wanted = {s.strip() for s in eacs_sources_env.split(",") if s.strip()}
            if "Source" in df_e.columns:
                df_e = df_e[df_e["Source"].isin(wanted)].copy()
                logger.info(f"filtered EACS to sources {wanted}: {len(df_e)} rows")
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
    return demo


# ============================================================
# Main: AD vs HC-group cohort builder
# ============================================================

def build_cohort_ad_vs_HCgroup(
    hc_source,
    design,
    cohort_mode="default",
    hc_source_mode="ACS",
    caliper=2.0,
    seed=42,
):
    """Build cohort for AD vs {HC, NAD, ACS}. Returns (cohort_df, pairs_df).

    hc_source: 'HC' (NAD ∪ ACS) | 'NAD' | 'ACS'
    design: 'cross_naive' | 'cross_matched' | 'longitudinal_naive' | 'longitudinal_matched'
    cohort_mode: 'default' | 'p_first_hc_all' | 'p_all_hc_all'
    hc_source_mode: 'ACS' | 'ACS_ext' | 'EACS'
    """
    _validate(cohort_mode, hc_source_mode)
    if design not in VALID_DESIGNS:
        raise ValueError(f"design must be one of {VALID_DESIGNS}, got {design!r}")

    demo = _load_combined_demographics(hc_source_mode)

    # AD side — first-visit with feature fallback (or all visits in p_all_hc_all).
    ad_all = demo[(demo["group"] == "P") & (demo["Global_CDR"] >= 0.5) &
                  demo["Age"].notna()].copy()
    ad_all = ad_all.sort_values(["base_id", "visit"])
    if cohort_mode == "p_all_hc_all":
        ad = _keep_visits_with_features(ad_all)
    else:
        ad = _pick_first_visit_with_features(ad_all)
    ad["label"] = 1

    # HC side — strict filter or unfiltered, depending on cohort_mode.
    hc_all = _strict_hc_filter_all_visits(demo, hc_source, cohort_mode)
    hc_all = hc_all[hc_all["Age"].notna()].copy()
    if cohort_mode in ("p_first_hc_all", "p_all_hc_all"):
        hc = hc_all.reset_index(drop=True).copy()
    else:
        hc = _pick_first_visit_with_features(hc_all)

    if design == "cross_naive":
        cohort = pd.concat([ad, hc], ignore_index=True)
        cohort["mmse_group"] = np.nan
        return cohort, None

    if design == "cross_matched":
        prep = pd.concat([ad, hc], ignore_index=True)
        prep["mmse_group"] = np.where(prep["label"] == 1, "high", "low")
        prep["MMSE"] = prep["MMSE"].fillna(999)  # match_1to1 reads Age only
        match_mode = ("subject_first"
                      if cohort_mode in ("p_first_hc_all", "p_all_hc_all")
                      else "visit")
        matched, pairs, _ = match_1to1(
            prep, caliper=caliper, seed=seed, metric="MMSE",
            group_col="mmse_group", match_mode=match_mode,
        )
        cohort = matched.merge(
            prep[["ID", "base_id", "group", "Age", "MMSE", "Global_CDR", "label"]]
            .drop_duplicates("ID"),
            on="ID", how="left", suffixes=("", "_p"),
        )
        cohort = cohort.drop(columns=[c for c in cohort.columns if c.endswith("_p")])
        return cohort, pairs

    # longitudinal_naive | longitudinal_matched
    do_match = (design == "longitudinal_matched")
    return _build_longitudinal_cohort_hc(hc_source, demo, hc_source_mode,
                                         caliper, seed, do_match=do_match)


# ============================================================
# Longitudinal cohort builder
# ============================================================

def _build_longitudinal_cohort_hc(hc_source, demo, hc_source_mode,
                                   caliper, seed, do_match=True):
    """Longitudinal AD vs {HC/NAD/ACS} cohort.

    do_match=True  → longitudinal_matched (1:1 NN on baseline_age)
    do_match=False → longitudinal_naive  (no matching)

    Inputs (must exist):
      workspace/longitudinal/features/ad_patient_deltas.csv
      workspace/longitudinal/features/hc_patient_deltas.csv
      workspace/longitudinal/features/eacs_patient_deltas.csv  (only if hc_source_mode != "ACS")
    Both have ann_* annualized Δ columns.
    """
    if AD_DELTAS_CSV.exists():
        ad_delta = pd.read_csv(AD_DELTAS_CSV)
        ad_delta = ad_delta[(ad_delta["follow_up_days"] >= 180) &
                            ad_delta["first_age"].notna()].copy()
    else:
        ad_delta = pd.read_csv(LONGITUDINAL_CSV)
        ad_delta = ad_delta[(ad_delta["follow_up_days"] >= 180) &
                            ad_delta["first_age"].notna()].copy()
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
            hc_delta_all = pd.concat([hc_delta_all, eacs_delta_all],
                                     ignore_index=True, sort=False)

    hc_strict = _strict_hc_filter(demo, hc_source)
    allowed_base = set(hc_strict["base_id"])
    hc_delta = hc_delta_all[hc_delta_all["base_id"].isin(allowed_base)].copy()
    if hc_source in ("NAD", "ACS"):
        hc_delta = hc_delta[hc_delta["group"] == hc_source]
    hc_delta["label"] = 0

    if hc_delta.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_delta = pd.concat([ad_delta, hc_delta], ignore_index=True, sort=False)

    if not do_match:
        cohort = all_delta.copy()
        cohort["mmse_group"] = np.where(cohort["label"] == 1, "high", "low")
        return cohort, None

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
    matched, pairs_df = _match_longitudinal_1to1(prep, caliper, seed)
    cohort = matched.merge(all_delta, on="base_id", how="left",
                           suffixes=("", "_p"))
    return cohort, pairs_df


def _annualize_patient_deltas(df):
    """Add ann_* columns to AD patient_deltas-style DataFrame."""
    df = df.copy()
    y = df["follow_up_years"].replace(0, np.nan)
    for col in list(df.columns):
        if (col.startswith("delta_") or col.startswith("emb_cosine_dist_")
                or col == "emb_cosine_dist"):
            df[f"ann_{col}"] = df[col] / y
    if "ann_emb_cosine_dist_arcface" in df.columns:
        df["ann_emb_cosine_dist"] = df["ann_emb_cosine_dist_arcface"]
    return df


def _match_longitudinal_1to1(prep, caliper, seed):
    """1:1 age NN match using 'Age' (= baseline first_age) on 'mmse_group'."""
    rng = np.random.RandomState(seed)
    g_hi = prep[prep["mmse_group"] == "high"].copy()
    g_lo = prep[prep["mmse_group"] == "low"].copy()
    if len(g_lo) <= len(g_hi):
        minor, major = g_lo, g_hi
        minor_lbl, major_lbl = "low", "high"
    else:
        minor, major = g_hi, g_lo
        minor_lbl, major_lbl = "high", "low"
    minor_order = minor.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    available = major.copy().reset_index(drop=True)

    pairs = []
    for _, row in minor_order.iterrows():
        if len(available) == 0:
            break
        age = row["Age"]
        diffs = (available["Age"] - age).abs()
        md = diffs.min()
        if md > caliper:
            continue
        cands = available[diffs == md].sort_values("base_id")
        pk = cands.iloc[0]
        pairs.append({
            "pair_id": len(pairs),
            "minor_id": row["base_id"], "minor_age": age,
            "major_id": pk["base_id"], "major_age": pk["Age"],
            "age_diff": pk["Age"] - age,
        })
        available = available.drop(pk.name).reset_index(drop=True)
    pairs_df = pd.DataFrame(pairs)

    records = []
    for _, p in pairs_df.iterrows():
        records.append({"pair_id": p["pair_id"], "base_id": p["minor_id"],
                        "mmse_group": minor_lbl})
        records.append({"pair_id": p["pair_id"], "base_id": p["major_id"],
                        "mmse_group": major_lbl})
    return pd.DataFrame(records), pairs_df


# ============================================================
# AD-internal hi-lo cohort builder
# ============================================================

def build_cohort_ad_hi_lo(
    design,
    cohort_mode="default",
    caliper=2.0,
    seed=42,
    metric="MMSE",
    matched_features_csv=None,
):
    """AD-internal hi-lo cohort. metric ∈ {'MMSE', 'CASI'}.

    For design='cross_matched', reads pre-built matched cohort from
    `matched_features_csv` (typically overview/<cohort>/cross_matched/
    <metric_low>_high_vs_low/matched_features.csv) — must be supplied.
    """
    if design not in VALID_DESIGNS:
        raise ValueError(f"design must be one of {VALID_DESIGNS}, got {design!r}")
    if cohort_mode not in VALID_COHORT_MODES:
        raise ValueError(f"cohort_mode must be one of {VALID_COHORT_MODES}, got {cohort_mode!r}")

    metric_low = metric.lower()
    group_col = f"{metric_low}_group"
    first_metric = f"first_{metric}"

    if design == "cross_naive":
        demo = load_p_demographics()
        demo["Global_CDR"] = pd.to_numeric(demo.get("Global_CDR"), errors="coerce")
        if cohort_mode in ("p_first_hc_all", "p_all_hc_all"):
            ad_all = demo[(demo["Global_CDR"] >= 0.5) &
                          demo[metric].notna() &
                          demo["Age"].notna()].copy()
            ad_all = ad_all.sort_values(["base_id", "visit"])
            if cohort_mode == "p_all_hc_all":
                cohort = _keep_visits_with_features(ad_all)
            else:
                cohort = _pick_first_visit_with_features(ad_all)
        else:
            cohort = demo.dropna(subset=[metric, "Age"]).copy()
            cohort = cohort.sort_values(["base_id", "visit"]).groupby(
                "base_id", as_index=False).first()
            cohort = cohort[cohort["Global_CDR"] >= 0.5].copy()
        med = cohort[metric].median()
        cohort[group_col] = np.where(cohort[metric] >= med, "high", "low")
        cohort["label"] = (cohort[group_col] == "high").astype(int)
        return cohort, None

    if design == "cross_matched":
        if matched_features_csv is None:
            raise ValueError(
                "design='cross_matched' for hi-lo requires matched_features_csv "
                "(typically overview/<cohort>/cross_matched/<metric_low>_high_vs_low/"
                "matched_features.csv from run_cross_matched.py --comparison "
                f"{metric_low}_hilo)")
        if not Path(matched_features_csv).exists():
            raise FileNotFoundError(
                f"Run run_cross_matched.py --comparison {metric_low}_hilo first; "
                f"missing: {matched_features_csv}")
        cohort = pd.read_csv(matched_features_csv)
        cohort["label"] = (cohort[group_col] == "high").astype(int)
        p_df = pd.read_csv(DEMOGRAPHICS_DIR / "P.csv")
        keep_cog = [c for c in ["MMSE", "CASI", "Global_CDR", "CDR_SB"]
                    if c in p_df.columns and c not in cohort.columns]
        if keep_cog:
            cohort = cohort.merge(p_df[["ID"] + keep_cog].drop_duplicates("ID"),
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
        ad_d = ad_d[ad_d[first_metric].notna() & ad_d["first_age"].notna()]
        med = ad_d[first_metric].median()
        ad_d[group_col] = np.where(ad_d[first_metric] >= med, "high", "low")
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
    ad_d[group_col] = np.where(ad_d[first_metric] >= med, "high", "low")
    prep = ad_d.copy()
    prep["Age"] = prep["first_age"]
    if group_col != "mmse_group":
        prep["mmse_group"] = prep[group_col]
    matched, pairs_df = _match_longitudinal_1to1(prep, caliper, seed)
    if "mmse_group" in matched.columns and group_col != "mmse_group":
        matched = matched.rename(columns={"mmse_group": group_col})
    cohort = matched.merge(ad_d, on="base_id", how="left", suffixes=("", "_p"))
    cohort["label"] = (cohort[group_col] == "high").astype(int)
    return cohort, pairs_df
