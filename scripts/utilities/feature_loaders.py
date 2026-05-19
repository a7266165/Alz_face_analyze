"""
Per-modality feature loaders shared across cross_naive / cross_matched /
stat_grid / age_classifiers / fwd_rev_embedding analyses.

All loaders return pandas DataFrames with `subject_id` as the merge key
(or sometimes `ID`, see docstrings). NaN-handling is downstream.
"""
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
EMBEDDING_DIR = PROJECT_ROOT / "workspace" / "embedding" / "features"
LANDMARK_FEATURES_CSV = PROJECT_ROOT / "workspace" / "asymmetry" / "features" / "pair_features.csv"
# Default cohort = V2.2 canonical (p_first_cdr05_hc_first_cdrall_or_mmseall).
from src.config import PREDICTED_AGES_FILE  # noqa: E402

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
EMOTION_METHODS = ["openface", "libreface", "pyfeat", "dan",
                   "hsemotion", "vit", "poster_pp", "fer"]
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
EMOTION_STATS = ["mean", "std", "range", "entropy"]
LANDMARK_REGIONS = ["eye", "nose", "mouth", "face_oval"]
EMBEDDING_MODELS = ["arcface", "topofr", "dlib"]

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Lazy emotion_loader (file-path importlib — module not on sys.path as package)
# ----------------------------------------------------------------------
_emotion_loader = None


def _get_emotion_loader():
    global _emotion_loader
    if _emotion_loader is None:
        spec = importlib.util.spec_from_file_location(
            "emotion_loader",
            Path(__file__).parent / "emotion_loader.py",
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _emotion_loader = m
    return _emotion_loader


# ============================================================
# Age error (real − predicted age)
# ============================================================

def load_age_error_pairs(ids):
    """Return DataFrame with columns [ID, predicted_age] for the requested ids
    that have a predicted age. Used by run_cross_naive."""
    with open(PREDICTED_AGES_FILE) as f:
        pred = json.load(f)
    rows = [(sid, pred.get(sid)) for sid in ids]
    return pd.DataFrame(rows, columns=["ID", "predicted_age"])


def load_age_error(subject_ids, demo):
    """Return DataFrame with [subject_id, age_error, abs_age_error]. Used by
    run_cross_matched. `demo` must have ID + Age columns."""
    with open(PREDICTED_AGES_FILE) as f:
        pred = json.load(f)
    rows = []
    for sid in subject_ids:
        real_row = demo[demo["ID"] == sid]
        if len(real_row) == 0:
            continue
        real_age = real_row.iloc[0]["Age"]
        pred_age = pred.get(sid)
        if pd.isna(real_age) or pred_age is None:
            continue
        rows.append({
            "subject_id": sid,
            "age_error": real_age - pred_age,
            "abs_age_error": abs(real_age - pred_age),
        })
    return pd.DataFrame(rows)


# ============================================================
# Embedding (mean-pool over photos → 1 vector per subject)
# ============================================================

def load_embedding_mean(ids, model):
    """Per-subject mean-pooled embedding vector (512 or 128-d).
    Returns DataFrame with subject_id + {model}__dim_{i} cols, or None if no rows."""
    emb_dir = EMBEDDING_DIR / model / "original"
    rows = []
    for sid in ids:
        npy = emb_dir / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy, allow_pickle=True)
        if a.dtype == object:
            a = list(a.item().values())[0]
        vec = a.mean(axis=0) if a.ndim == 2 else a
        rows.append({"subject_id": sid,
                     **{f"{model}__dim_{i}": float(vec[i]) for i in range(vec.shape[0])}})
    return pd.DataFrame(rows) if rows else None


# ============================================================
# Embedding asymmetry (mean-pool over photos of difference vector)
# ============================================================

def load_embedding_asymmetry(subject_ids):
    """Per-subject embedding-asymmetry L2 scalar per model.
    Returns DataFrame with subject_id + embasym__{model}_l2 (3 cols)."""
    rows = []
    for sid in subject_ids:
        out = {"subject_id": sid}
        for model in EMBEDDING_MODELS:
            npy = EMBEDDING_DIR / model / "difference" / f"{sid}.npy"
            if not npy.exists():
                out[f"embasym__{model}_l2"] = np.nan
                continue
            a = np.load(npy, allow_pickle=True)
            if a.dtype == object:
                a = list(a.item().values())[0]
            md = a.mean(axis=0) if a.ndim == 2 else a
            out[f"embasym__{model}_l2"] = float(np.linalg.norm(md))
        rows.append(out)
    return pd.DataFrame(rows)


def load_embedding_asymmetry_vec(ids, model):
    """Per-subject mean-pooled difference vector (full dim, 512 or 128).
    Returns DataFrame with subject_id + embasym_vec_{model}_{i} cols."""
    rows = []
    for sid in ids:
        npy = EMBEDDING_DIR / model / "difference" / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy, allow_pickle=True)
        if a.dtype == object:
            a = list(a.item().values())[0]
        vec = a.mean(axis=0) if a.ndim == 2 else a
        rows.append({"subject_id": sid,
                     **{f"embasym_vec_{model}_{i}": float(vec[i])
                        for i in range(vec.shape[0])}})
    return pd.DataFrame(rows) if rows else None


# ============================================================
# Landmark asymmetry (xy-pair diffs + area_diffs)
# ============================================================

def load_landmark_matrix(ids):
    """Per-subject landmark-asymmetry summary scalars:
      landmark__{region}_l2       (4)
      landmark__{region}_area_diff (4)
      landmark__total_l2          (1)
    Returns DataFrame with subject_id + 9 numeric cols.
    """
    df = pd.read_csv(LANDMARK_FEATURES_CSV)
    df = df[df["subject_id"].isin(ids)].copy()
    rows = []
    for _, r in df.iterrows():
        out = {"subject_id": r["subject_id"]}
        total_sq = 0.0
        for region in LANDMARK_REGIONS:
            xy_cols = [c for c in df.columns
                       if c.startswith(f"{region}_") and "area_diff" not in c]
            vec = r[xy_cols].to_numpy(dtype=float)
            l2 = float(np.sqrt(np.nansum(vec * vec)))
            out[f"landmark__{region}_l2"] = l2
            out[f"landmark__{region}_area_diff"] = float(r[f"{region}_area_diff"])
            total_sq += np.nansum(vec * vec)
        out["landmark__total_l2"] = float(np.sqrt(total_sq))
        rows.append(out)
    return pd.DataFrame(rows)


# Backward-compat alias used by run_cross_matched.py
load_landmark_asymmetry = load_landmark_matrix


def load_landmark_raw(ids):
    """Per-subject 130-d raw xy-pair diff vector (area_diff excluded).

    area_diff is observed via the 4-d per-region L2 sub-row; this row only
    contains xy-only raw pairs to avoid pixel vs pixel² scale mismatch
    dominating Euclidean PERMANOVA.
    """
    df = pd.read_csv(LANDMARK_FEATURES_CSV)
    df = df[df["subject_id"].isin(ids)].copy()
    feat_cols = []
    for region in LANDMARK_REGIONS:
        feat_cols += [c for c in df.columns
                      if c.startswith(f"{region}_") and "area_diff" not in c]
    keep = ["subject_id"] + feat_cols
    return df[keep].copy()


def load_landmark_region_l2(ids):
    """Per-subject 4-d per-region L2 (no global summary)."""
    df = load_landmark_matrix(ids)
    if df is None or df.empty:
        return df
    cols = ["subject_id"] + [f"landmark__{r}_l2" for r in LANDMARK_REGIONS]
    return df[[c for c in cols if c in df.columns]].copy()


# ============================================================
# Emotion features (8 methods × 7 emotions × 4 stats)
# ============================================================

def load_emotion_matrix(ids):
    """Wrap emotion_loader.load_emotion_matrix — returns single wide df."""
    return _get_emotion_loader().load_emotion_matrix(ids, methods=EMOTION_METHODS)


def load_emotion_features():
    """Return dict method → DataFrame with subject_id + (method-prefixed)
    7 emotions × 4 stats columns. Used by run_cross_matched.

    Loader iterates files for each method; cohort filtering happens downstream.
    """
    el = _get_emotion_loader()
    emotion_cols = [f"{emo}_{stat}" for emo in EMOTIONS for stat in EMOTION_STATS]
    method_dirs = el.EMO_AU_FEATURES_DIR
    all_ids = set()
    for m in EMOTION_METHODS:
        d = method_dirs / m
        if d.is_dir():
            all_ids.update(p.stem for p in d.glob("*.npy"))
    out = {}
    for method in EMOTION_METHODS:
        try:
            df = el.load_emotion(method, sorted(all_ids))
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"emotion {method}: {e}; skipping")
            continue
        if df.empty:
            continue
        keep = ["subject_id"] + [c for c in emotion_cols if c in df.columns]
        df = df[keep].copy()
        rename = {c: f"{method}__{c}" for c in keep if c != "subject_id"}
        df = df.rename(columns=rename)
        out[method] = df
    return out
