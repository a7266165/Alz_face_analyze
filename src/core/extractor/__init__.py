"""
特徵提取模組

提供多種深度學習特徵提取器的統一介面
"""

from .base import BaseExtractor
from .registry import ExtractorRegistry

# 導入所有提取器以觸發註冊
from .dlib_extractor import DlibExtractor
from .arcface_extractor import ArcFaceExtractor
from .topofr_extractor import TopoFRExtractor
from .vggface_extractor import VGGFaceExtractor

from typing import Dict, List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    統一的特徵提取器介面

    向後兼容原有的 FeatureExtractor 類別
    """

    def __init__(self):
        """初始化特徵提取器"""
        self._registry = ExtractorRegistry()
        self._registry.report_status()

    @property
    def available_models(self) -> List[str]:
        """取得可用的模型列表"""
        return self._registry.get_available()

    def extract_features(
        self,
        images: List[np.ndarray],
        models: Union[str, List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, List[Optional[np.ndarray]]]:
        """
        批次提取特徵

        Args:
            images: 影像列表（BGR 格式）
            models: 模型名稱或列表
                - None: 使用所有可用模型
                - "dlib": 單一模型
                - ["dlib", "arcface"]: 多個模型
            verbose: 是否顯示進度

        Returns:
            {model_name: [features1, features2, ...]}
        """
        # 統一轉成列表
        if models is None:
            models = self.available_models
        elif isinstance(models, str):
            models = [models]

        # 驗證模型名稱
        invalid = set(models) - set(self.available_models)
        if invalid:
            logger.warning(f"以下模型不可用，將跳過: {invalid}")

        valid_models = [m for m in models if m in self.available_models]

        if not valid_models:
            logger.error("沒有可用的模型")
            return {}

        # 提取特徵
        results = {}
        for model_name in valid_models:
            extractor = self._registry.get(model_name)
            if extractor:
                results[model_name] = extractor.extract_batch(images, verbose)

        return results

    def get_available_models(self) -> List[str]:
        """取得可用的模型列表"""
        return self.available_models

    def get_feature_dim(self, model_name: str) -> Optional[int]:
        """取得模型輸出維度"""
        extractor = self._registry.get(model_name)
        if extractor:
            return extractor.feature_dim
        return None

    def calculate_differences(
        self,
        left_features: np.ndarray,
        right_features: np.ndarray,
        methods: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        計算左右特徵差異

        Args:
            left_features: 左臉特徵
            right_features: 右臉特徵
            methods: 計算方法列表

        Returns:
            {method_name: result}
        """
        if methods is None:
            raise ValueError("必須明確指定 methods 參數")

        valid_methods = {
            "differences", "absolute_differences", "averages",
            "relative_differences", "absolute_relative_differences"
        }
        if invalid := set(methods) - valid_methods:
            raise ValueError(f"未知的方法: {invalid}")

        results = {}

        if "differences" in methods:
            diff = left_features - right_features
            results["embedding_differences"] = diff.astype(np.float32)

        if "absolute_differences" in methods:
            abs_diff = np.abs(left_features - right_features)
            results["embedding_absolute_differences"] = abs_diff.astype(np.float32)

        if "averages" in methods:
            results["embedding_averages"] = (
                (left_features + right_features) / 2
            ).astype(np.float32)

        if "relative_differences" in methods:
            diff = left_features - right_features
            norm = np.sqrt(left_features**2 + right_features**2)
            rel_diff = np.zeros_like(diff)
            mask = norm > 1e-8
            rel_diff[mask] = diff[mask] / norm[mask]
            results["embedding_relative_differences"] = rel_diff.astype(np.float32)

        if "absolute_relative_differences" in methods:
            abs_diff = np.abs(left_features - right_features)
            norm = np.sqrt(left_features**2 + right_features**2)
            rel_abs_diff = np.zeros_like(abs_diff)
            mask = norm > 1e-8
            rel_abs_diff[mask] = abs_diff[mask] / norm[mask]
            results["embedding_absolute_relative_differences"] = rel_abs_diff.astype(np.float32)

        return results

    def add_demographics(
        self,
        features_list: List[np.ndarray],
        ages: List[float],
        genders: List[float]
    ) -> List[np.ndarray]:
        """
        加入人口學特徵

        Args:
            features_list: 特徵向量列表
            ages: 年齡列表
            genders: 性別列表 (0=女, 1=男)

        Returns:
            結合後的特徵向量列表
        """
        if not (len(features_list) == len(ages) == len(genders)):
            raise ValueError(
                f"長度不一致: features={len(features_list)}, "
                f"ages={len(ages)}, genders={len(genders)}"
            )

        return [
            np.concatenate([feat, np.array([age, gender], dtype=np.float32)])
            for feat, age, gender in zip(features_list, ages, genders)
        ]


__all__ = [
    # 核心類別
    "BaseExtractor",
    "ExtractorRegistry",
    "FeatureExtractor",
    # 具體提取器
    "DlibExtractor",
    "ArcFaceExtractor",
    "TopoFRExtractor",
    "VGGFaceExtractor",
]
