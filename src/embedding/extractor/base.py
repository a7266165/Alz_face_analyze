"""
特徵提取器抽象基礎類別
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class EmbeddingExtractor(ABC):
    """
    特徵提取器抽象基礎類別
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名稱"""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """特徵維度"""
        ...

    @abstractmethod
    def initialize(self) -> None:
        """載入模型權重等資源"""
        ...

    @abstractmethod
    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        從影像提取特徵

        Args:
            image: BGR 格式的影像

        Returns:
            特徵向量 (float32)，若提取失敗返回 None
        """
        ...

    def is_available(self) -> bool:
        """檢查提取器"""
        return True
