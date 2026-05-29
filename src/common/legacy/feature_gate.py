"""[LEGACY] Feature-existence gate — 從 src/cohort.py 隔離。

判斷某個 visit-level ID 是否已抽出特定 modality 的特徵（embedding /
landmark / emotion / predicted_age）。這屬於「特徵可用性」，不是人口學族群
挑選；原本它被耦合進 cohort 的 visit 選擇（「挑第一個*有特徵的* visit」），
讓族群挑選依賴下游抽特徵的結果。暫存於此待後續清理。
"""
import json

import pandas as pd

from src.config import (
    ASYMMETRY_PAIR_FEATURES_FILE,
    EMBEDDING_FEATURES_DIR,
    EMO_AU_FEATURES_DIR,
    PREDICTED_AGES_FILE,
)

EMBEDDING_DIR = EMBEDDING_FEATURES_DIR
LANDMARK_FEATURES_CSV = ASYMMETRY_PAIR_FEATURES_FILE

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
        for bg in ("no_background", "background", ""):
            d = EMBEDDING_DIR / model / bg / variant if bg else EMBEDDING_DIR / model / variant
            if d.exists():
                ids = {p.stem for p in d.glob("*.npy")}
                break
    else:
        for bg in ("no_background", "background", ""):
            d = EMBEDDING_DIR / modality / bg / "original" if bg else EMBEDDING_DIR / modality / "original"
            if d.exists():
                ids = {p.stem for p in d.glob("*.npy")}
                break

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
            emo_raw = EMO_AU_FEATURES_DIR / "raw" / tool
            if emo_raw.exists():
                ids.update(p.stem for p in emo_raw.glob("EACS_*.csv"))
    return ids


def pick_first_visit_with_features(df_visits):
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


def keep_visits_with_features(df_visits):
    """Keep ALL feature-bearing visits (no per-subject pick)."""
    good_ids = _ids_with_features()
    return df_visits[
        df_visits["ID"].astype(str).isin(good_ids)
    ].reset_index(drop=True)
