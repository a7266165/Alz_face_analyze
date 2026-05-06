"""
Emotion / AU feature loader.

Reads per-subject-visit npy files under
    workspace/emo_au/features/<method>/<subject>-<visit>.npy
plus the global column schema at
    workspace/emo_au/features/_schema.json
and aggregates frame-level rows to subject-level statistics
(mean / std / range / entropy) on demand.

Other scripts load this module via
    importlib.util.spec_from_file_location(...)
to avoid touching `src/`.

Public API:
    load_emotion(method, ids) -> pd.DataFrame
        rows × (subject_id + <method-prefixed feature_stat columns>)
    load_emotion_matrix(ids, methods=None) -> pd.DataFrame
        wide table merging all methods, columns named `<method>__<col>` —
        backward-compatible with the previous loader signatures used in
        run_cross_naive / run_stat_grid / run_cross_matched /
        run_age_window_classifier / build_longitudinal_dataset /
        build_longitudinal_hc_and_vectors.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMO_AU_FEATURES_DIR = PROJECT_ROOT / "workspace" / "emo_au" / "features"
SCHEMA_FILE = EMO_AU_FEATURES_DIR / "_schema.json"

EMOTION_METHODS: List[str] = [
    "openface", "libreface", "pyfeat", "dan",
    "hsemotion", "vit", "poster_pp", "fer",
]
EMOTIONS: List[str] = [
    "anger", "disgust", "fear", "happiness",
    "sadness", "surprise", "neutral",
]
STATS: List[str] = ["mean", "std", "range", "entropy"]


def _safe_entropy(values: np.ndarray, n_bins: int = 10) -> float:
    """Histogram entropy (base-2). Mirrors
    src/extractor/features/emotion/postprocess/aggregator.py:_safe_entropy.
    """
    values = values[~np.isnan(values)]
    if values.size < 2:
        return 0.0
    hist, _ = np.histogram(values, bins=n_bins, density=True)
    hist = hist + 1e-10
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def _aggregate_stats(values: np.ndarray, min_frames: int = 3) -> dict:
    """Compute mean/std/range/entropy of a 1-D values array.

    `entropy` falls back to 0 when frame count < min_frames (matches the
    extractor's `TemporalAggregator`).
    """
    values = values[~np.isnan(values)]
    n = values.size
    if n == 0:
        return {s: 0.0 for s in STATS}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)) if n > 1 else 0.0,
        "range": float(np.max(values) - np.min(values)),
        "entropy": _safe_entropy(values) if n >= min_frames else 0.0,
    }


_SCHEMA_CACHE: Optional[dict] = None


def _load_schema() -> dict:
    """Read & cache `_schema.json`. Schema file is created by the migration
    script; before migration runs, this raises a clear FileNotFoundError."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        if not SCHEMA_FILE.exists():
            raise FileNotFoundError(
                f"emo_au schema not found: {SCHEMA_FILE}.\n"
                "Run the workspace migration script first."
            )
        _SCHEMA_CACHE = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE


def _method_columns(method: str) -> List[str]:
    schema = _load_schema()
    methods = schema.get("methods", {})
    if method not in methods:
        raise KeyError(f"method '{method}' not in _schema.json")
    return list(methods[method]["columns"])


def load_emotion(method: str, ids: Iterable[str]) -> pd.DataFrame:
    """Aggregate frame-level npy data for `method` over the supplied subject-
    visit ids.

    Returns DataFrame with `subject_id` + one column per (col, stat) where
    `col` is anything in the method's column list (emotions + AUs + extras
    depending on method) and `stat` ∈ {mean, std, range, entropy}.

    Missing files / unreadable npy → row is skipped.
    """
    columns = _method_columns(method)
    method_dir = EMO_AU_FEATURES_DIR / method

    rows = []
    for sid in ids:
        npy_path = method_dir / f"{sid}.npy"
        if not npy_path.exists():
            continue
        try:
            arr = np.load(npy_path)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] != len(columns):
            # Misshapen file — skip to avoid silent column-meaning corruption.
            continue
        out = {"subject_id": sid}
        for j, col in enumerate(columns):
            stats = _aggregate_stats(arr[:, j])
            for s, v in stats.items():
                out[f"{col}_{s}"] = v
        rows.append(out)

    if not rows:
        # Honour the legacy contract: all expected columns present, 0 rows.
        cols = ["subject_id"] + [f"{c}_{s}" for c in columns for s in STATS]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def load_emotion_matrix(
    ids: Iterable[str],
    methods: Optional[Iterable[str]] = None,
) -> Optional[pd.DataFrame]:
    """Wide-format multi-method emotion matrix.

    For each method, picks emotion columns only (matching the legacy
    `load_emotion_matrix` API) and prefixes them with `<method>__`.
    Merges across methods on `subject_id`.

    Returns:
        DataFrame with `subject_id` + 8 method × 7 emotion × 4 stat = 224
        columns (when all methods present). Rows where no method had data
        are dropped. Returns None when zero methods produced data.
    """
    if methods is None:
        methods = EMOTION_METHODS
    emo_cols = [f"{e}_{s}" for e in EMOTIONS for s in STATS]
    frames = []
    for method in methods:
        try:
            df = load_emotion(method, ids)
        except (FileNotFoundError, KeyError):
            continue
        if df.empty:
            continue
        keep = ["subject_id"] + [c for c in emo_cols if c in df.columns]
        df = df[keep].copy()
        rename = {c: f"{method}__{c}" for c in keep if c != "subject_id"}
        df = df.rename(columns=rename)
        frames.append(df)
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="subject_id", how="outer")
    return out[out["subject_id"].isin(set(ids))]
