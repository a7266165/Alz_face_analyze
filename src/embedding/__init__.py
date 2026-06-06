"""
特徵提取模組

提供多種深度學習特徵提取器的統一介面
"""

from .asymmetry import calculate_differences
from .extractor import (
    EmbeddingExtractor,
    DlibExtractor,
    ArcFaceExtractor,
    TopoFRExtractor,
    VGGFaceExtractor,
    EXTRACTORS,
    get_extractor,
    available_extractors,
)

__all__ = [
    # 契約
    "EmbeddingExtractor",
    # 具體提取器
    "DlibExtractor",
    "ArcFaceExtractor",
    "TopoFRExtractor",
    "VGGFaceExtractor",
    # 註冊表 / 取得器
    "EXTRACTORS",
    "get_extractor",
    "available_extractors",
    # 鏡射不對稱特徵
    "calculate_differences",
]
