"""
年齡預測器基礎類別。
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BasePredictor(ABC):
    """年齡預測器抽象基底。

    純函數 (已讀好的 image: ndarray) → 年齡;讀檔、遍歷、批次迴圈是 producer 的工作
    （見 scripts/age/predict.py 與 src/common/image_io）。與 EmbeddingExtractor /
    EmoAUExtractor 同屬「單張進、無 I/O、無批次」的契約。

    initialize() 為 eager 載入 hook（producer 取得 predictor 後呼叫）;is_available()
    供 registry（get_predictor）篩掉在當前 env 跑不起來的模型，與兩個 extractor 家族對齊。
    """

    @abstractmethod
    def initialize(self) -> None:
        """載入模型權重等資源"""
        ...

    @abstractmethod
    def predict_single(self, image: np.ndarray) -> Optional[float]:
        """單張影像 → 預測年齡；偵測/推論失敗回 None。"""
        ...

    def is_available(self) -> bool:
        """檢查此預測器是否可用（依賴已安裝、權重存在）。registry 據此篩選；預設可用。"""
        return True
