"""
年齡預測器基礎類別

定義所有年齡預測器的共同介面。各模型 wrapper（MiVOLO / InsightFace / DeepFace /
FairFace / OpenCV DNN）繼承此類，各自實作 initialize()（載入模型）與
predict_single()（單張影像 → 年齡）；逐張的 predict() 由本基底統一提供。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BasePredictor(ABC):
    """年齡預測器抽象基底。

    所有年齡模型 wrapper 繼承此類，共同介面：
      initialize()         載入模型權重等資源（昂貴，呼叫一次）
      predict_single(img)  單張 BGR 影像 → 預測年齡；偵測/推論失敗回 None
    """

    @abstractmethod
    def initialize(self) -> None:
        """載入模型權重等資源（昂貴，呼叫一次）。"""
        ...

    @abstractmethod
    def predict_single(self, image: np.ndarray) -> Optional[float]:
        """單張 BGR 影像 → 預測年齡；偵測/推論失敗回 None。"""
        ...

    def predict(self, images: List[np.ndarray]) -> List[float]:
        """逐張預測，僅保留成功（非 None）的結果。"""
        return [a for img in images
                if (a := self.predict_single(img)) is not None]
