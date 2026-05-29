"""[LEGACY] Predicted-age 篩選 — 從 src/cohort.py 隔離。

predicted_age 是 age modality 的輸出（下游產物）。用它來篩 cohort 會讓族群
挑選依賴下游結果，違反 cohort 的單一職責。暫存待清理。
"""
import json

from src.config import PREDICTED_AGES_FILE

_PRED_AGES_CACHE = None


def load_predicted_ages():
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
        pred_ages = load_predicted_ages()
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
        pred_ages = load_predicted_ages()
    keep = (pairs["minor_id"].astype(str).isin(pred_ages)
            & pairs["major_id"].astype(str).isin(pred_ages))
    pairs = pairs[keep].copy().reset_index(drop=True)
    kept_ids = set(pairs["minor_id"]).union(pairs["major_id"])
    id_col = "ID" if "ID" in matched.columns else "base_id"
    matched = matched[matched[id_col].isin(kept_ids)].copy().reset_index(
        drop=True)
    return matched, pairs
