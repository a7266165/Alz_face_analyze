"""Meta：單一 session 層級的跨 modality stacking(embedding/asymmetry LR 分數 + 年齡 + BMI +
認知分數 → meta stacker),統一 feature combo 見 META_FEATURE_SETS、stacker 見 META_CLASSIFIERS。"""
from src.meta.classifier import META_CLASSIFIERS, make_meta_clf, make_tabpfn_v3
from src.meta.train import (
    ALL_FEATURE_COLS,
    ASYM_VARIANTS,
    META_FEATURE_SETS,
    OOF_FEATURE_COLS,
    base_oof,
    feature_set_needs_oof,
    meta_oof,
    oof_from_table,
    session_feature_table,
    session_oof,
)

__all__ = [
    "ASYM_VARIANTS", "META_FEATURE_SETS", "META_CLASSIFIERS",
    "ALL_FEATURE_COLS", "OOF_FEATURE_COLS",
    "feature_set_needs_oof", "base_oof",
    "session_feature_table", "oof_from_table", "session_oof",
    "make_meta_clf", "make_tabpfn_v3", "meta_oof",
]
