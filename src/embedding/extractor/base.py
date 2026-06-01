"""
特徵提取器基礎類別

定義所有特徵提取器的共同介面。純函數 (已讀好的 image: ndarray) → embedding;
讀檔、遍歷、批次迴圈是 producer 的工作（見 src/common/image_io）。與
EmoAUExtractor / BasePredictor 同屬「單張進、無 I/O、無批次」的契約。
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class EmbeddingExtractor(ABC):
    """
    特徵提取器抽象基礎類別

    所有特徵提取器都應繼承此類並實現 extract 方法
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
    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        從單一影像中提取特徵

        Args:
            image: BGR 格式的影像

        Returns:
            特徵向量 (float32)，若提取失敗返回 None
        """
        ...

    def is_available(self) -> bool:
        """檢查提取器是否可用"""
        return True
