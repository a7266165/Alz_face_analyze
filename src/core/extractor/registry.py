"""
特徵提取器註冊表

提供懶載入和統一的提取器管理
"""

from typing import Dict, List, Optional, Type, Callable
import logging

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """
    特徵提取器註冊表

    使用單例模式和懶載入，只在需要時初始化提取器
    """

    _instance: Optional['ExtractorRegistry'] = None
    _registry: Dict[str, Type[BaseExtractor]] = {}

    def __new__(cls) -> 'ExtractorRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._extractors = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._extractors: Dict[str, BaseExtractor] = {}
            self._initialized = True

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        註冊提取器的裝飾器

        使用方式:
            @ExtractorRegistry.register("dlib")
            class DlibExtractor(BaseExtractor):
                ...
        """
        def decorator(extractor_cls: Type[BaseExtractor]) -> Type[BaseExtractor]:
            cls._registry[name] = extractor_cls
            logger.debug(f"已註冊提取器: {name}")
            return extractor_cls
        return decorator

    def get(self, name: str) -> Optional[BaseExtractor]:
        """
        取得提取器實例（懶載入）

        Args:
            name: 提取器名稱 (dlib, arcface, topofr, vggface)

        Returns:
            提取器實例，如果不可用則返回 None
        """
        # 如果已經初始化過，直接返回
        if name in self._extractors:
            return self._extractors[name]

        # 檢查是否已註冊
        if name not in self._registry:
            logger.warning(f"未知的提取器: {name}")
            return None

        # 嘗試初始化
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

    def get_available(self) -> List[str]:
        """取得所有可用的提取器名稱"""
        available = []
        for name in self._registry:
            extractor = self.get(name)
            if extractor is not None:
                available.append(name)
        return available

    def get_all_extractors(self) -> Dict[str, BaseExtractor]:
        """取得所有已載入的提取器"""
        # 確保所有提取器都已嘗試載入
        for name in self._registry:
            self.get(name)
        return {k: v for k, v in self._extractors.items()}

    @classmethod
    def registered_names(cls) -> List[str]:
        """取得所有已註冊的提取器名稱"""
        return list(cls._registry.keys())

    def report_status(self):
        """報告所有提取器的狀態"""
        available = self.get_available()
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
