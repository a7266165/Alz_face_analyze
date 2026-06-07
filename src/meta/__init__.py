"""Meta：精簡的跨 modality stacking（embedding-original + asymmetry OOF + 人口學 → TabPFN v3）。"""
from src.meta.classifier import make_tabpfn_v3
from src.meta.train import (
    ASYM_METHODS,
    ASYM_VARIANTS,
    FEATURE_SETS,
    add_asym,
    base_oof,
    build_base_table,
    slice_xy,
    subject_demographics,
    sweep,
    tabpfn_oof,
    to_subject,
)

__all__ = [
    "ASYM_VARIANTS", "ASYM_METHODS", "FEATURE_SETS",
    "base_oof", "to_subject", "subject_demographics",
    "build_base_table", "add_asym", "slice_xy",
    "make_tabpfn_v3", "tabpfn_oof", "sweep",
]
