"""
統一的特徵提取器介面

提供模型註冊、懶載入與批次特徵提取功能
"""

from typing import Callable, Dict, List, Optional, Type, Union
import numpy as np
import logging

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    統一的特徵提取器介面

    整合模型註冊（decorator）、懶載入與批次提取功能。
    使用 singleton 模式確保全域只有一個實例。

    使用方式:
        # 註冊（各 extractor 檔案中）
        @FeatureExtractor.register("arcface")
        class ArcFaceExtractor(BaseExtractor):
            ...

        # 使用
        fe = FeatureExtractor()
        results = fe.extract_features(images, models=["arcface"])
    """

    # --- class-level registry ---
    _registry: Dict[str, Type[BaseExtractor]] = {}

    # --- singleton ---
    _instance: Optional['FeatureExtractor'] = None

    def __new__(cls) -> 'FeatureExtractor':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._extractors: Dict[str, BaseExtractor] = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._extractors: Dict[str, BaseExtractor] = {}
            self._initialized = True
            self._report_status()

    # ==================== 註冊機制 ====================

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        註冊提取器的裝飾器

        使用方式:
            @FeatureExtractor.register("dlib")
            class DlibExtractor(BaseExtractor):
                ...
        """
        def decorator(extractor_cls: Type[BaseExtractor]) -> Type[BaseExtractor]:
            cls._registry[name] = extractor_cls
            logger.debug(f"已註冊提取器: {name}")
            return extractor_cls
        return decorator

    # ==================== 模型管理 ====================

    def _get_extractor(self, name: str) -> Optional[BaseExtractor]:
        """取得提取器實例（懶載入）"""
        if name in self._extractors:
            return self._extractors[name]

        if name not in self._registry:
            logger.warning(f"未知的提取器: {name}")
            return None

        try:
            extractor = self._registry[name]()
            if extractor.is_available():
                self._extractors[name] = extractor
                logger.info(f"✓ {name} 載入成功")
                return extractor
            else:
                logger.warning(f"✗ {name} 不可用")
                return None
        except Exception as e:
            logger.warning(f"✗ {name} 初始化失敗: {e}")
            return None

    @property
    def available_models(self) -> List[str]:
        """取得可用的模型列表"""
        available = []
        for name in self._registry:
            if self._get_extractor(name) is not None:
                available.append(name)
        return available

    def get_available_models(self) -> List[str]:
        """取得可用的模型列表"""
        return self.available_models

    def get_feature_dim(self, model_name: str) -> Optional[int]:
        """取得模型輸出維度"""
        extractor = self._get_extractor(model_name)
        if extractor:
            return extractor.feature_dim
        return None

    # ==================== 特徵提取 ====================

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
        if models is None:
            models = self.available_models
        elif isinstance(models, str):
            models = [models]

        invalid = set(models) - set(self.available_models)
        if invalid:
            logger.warning(f"以下模型不可用，將跳過: {invalid}")

        valid_models = [m for m in models if m in self.available_models]

        if not valid_models:
            logger.error("沒有可用的模型")
            return {}

        results = {}
        for model_name in valid_models:
            extractor = self._get_extractor(model_name)
            if extractor:
                results[model_name] = extractor.extract_batch(images, verbose)

        return results

    # ==================== 狀態報告 ====================

    def _report_status(self):
        """報告所有提取器的狀態"""
        available = self.available_models
        if available:
            logger.info("=" * 50)
            logger.info("特徵提取器狀態")
            logger.info(f"可用模型: {', '.join(available)}")
            for name in available:
                extractor = self._extractors[name]
                logger.info(f"  - {name}: {extractor.feature_dim} 維")
            logger.info("=" * 50)
        else:
            logger.warning("警告：沒有任何特徵提取模型可用")
