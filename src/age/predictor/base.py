"""
年齡預測器基礎類別。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BasePredictor(ABC):
    """年齡預測器抽象基底。"""

    @abstractmethod
    def initialize(self) -> None:
        """載入模型權重等資源"""
        ...

    @abstractmethod
    def predict_single(self, image: np.ndarray) -> Optional[float]:
        """單張影像 → 預測年齡；偵測/推論失敗回 None。"""
        ...

    def predict(self, images: List[np.ndarray]) -> List[float]:
        """逐張預測，僅保留成功（非 None）的結果。"""
        return [a for img in images
                if (a := self.predict_single(img)) is not None]
