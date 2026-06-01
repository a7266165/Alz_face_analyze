"""
特徵提取器基礎類別

定義所有特徵提取器的共同介面。純函數 (已讀好的 image: ndarray) → embedding;
讀檔、遍歷、批次迴圈是 producer 的工作（見 src/common/image_io）。與
EmoAUExtractor / BasePredictor 同屬「單張進、無 I/O、無批次」的契約。

生命週期（三家族一致）:
  建構（__init__，瑣碎、不載入）→ is_available()（便宜探測:依賴可 import?權重在?）
  → initialize()（eager 載入權重，由 producer 取得後顯式呼叫一次）→ extract()（純推論）。
這讓「載入」是 producer 可控的顯式步驟:載入錯誤在進迴圈前就爆、載入與逐張推論的
計時乾淨分開（benchmark 重點），extract 維持無分支的純函數。
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
    def initialize(self) -> None:
        """eager 載入模型權重等資源（producer 取得 extractor 後呼叫一次）。

        應為冪等（重複呼叫安全）。is_available() 已先行篩掉跑不起來的模型，故此處
        遇到真正的載入失敗應直接 raise（fail-fast），不要吞掉。
        """
        ...

    @abstractmethod
    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        從單一影像中提取特徵（假設已 initialize()）

        Args:
            image: BGR 格式的影像

        Returns:
            特徵向量 (float32)，若提取失敗返回 None
        """
        ...

    def is_available(self) -> bool:
        """檢查提取器是否可用（依賴已安裝、權重存在）。registry 據此篩選;便宜探測、不載入。"""
        return True
