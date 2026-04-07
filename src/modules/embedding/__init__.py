"""
特徵提取模組

提供多種深度學習特徵提取器的統一介面
"""

from .base import BaseExtractor
from .feature_extractor import FeatureExtractor
from .feature_ops import calculate_differences, add_demographics

# 導入所有提取器以觸發註冊
from .dlib_extractor import DlibExtractor
from .arcface_extractor import ArcFaceExtractor
from .topofr_extractor import TopoFRExtractor
from .vggface_extractor import VGGFaceExtractor

__all__ = [
    # 核心類別
    "BaseExtractor",
    "FeatureExtractor",
    # 特徵操作
    "calculate_differences",
    "add_demographics",
    # 具體提取器
    "DlibExtractor",
    "ArcFaceExtractor",
    "TopoFRExtractor",
    "VGGFaceExtractor",
]
