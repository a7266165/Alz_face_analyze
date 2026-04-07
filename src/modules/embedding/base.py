"""
特徵提取器基礎類別

定義所有特徵提取器的共同介面
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
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
        從影像中提取特徵

        Args:
            image: BGR 格式的影像

        Returns:
            特徵向量 (float32)，如果提取失敗則返回 None
        """
        ...

    def extract_batch(
        self,
        images: List[np.ndarray],
        verbose: bool = False
    ) -> List[Optional[np.ndarray]]:
        """
        批次提取特徵

        Args:
            images: BGR 格式的影像列表
            verbose: 是否顯示進度

        Returns:
            特徵向量列表
        """
        results = []
        for i, image in enumerate(images):
            if verbose and i % 10 == 0:
                logger.info(f"{self.model_name} 處理進度: {i}/{len(images)}")
            try:
                features = self.extract(image)
                results.append(features)
            except Exception as e:
                logger.error(f"{self.model_name} 提取失敗: {e}")
                results.append(None)

        # 統計成功率
        success = sum(1 for f in results if f is not None)
        logger.info(f"{self.model_name}: {success}/{len(images)} 成功")

        return results

    def is_available(self) -> bool:
        """檢查提取器是否可用"""
        return True
