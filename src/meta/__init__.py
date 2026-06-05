"""Meta：精簡的跨 modality stacking（embedding-original + asymmetry OOF + 人口學 → TabPFN v3）。

舊的重型 stacking 套件見 src/meta/legacy/。
"""
from src.meta.tabpfn_meta import (
    ASYM_METHODS,
    ASYM_VARIANTS,
    FEATURE_COLS,
    base_oof,
    build_feature_table,
    make_tabpfn_v3,
    subject_demographics,
    sweep,
    tabpfn_oof,
    to_subject,
)

__all__ = [
    "ASYM_VARIANTS", "ASYM_METHODS", "FEATURE_COLS",
    "base_oof", "to_subject", "subject_demographics",
    "build_feature_table", "make_tabpfn_v3", "tabpfn_oof", "sweep",
]
