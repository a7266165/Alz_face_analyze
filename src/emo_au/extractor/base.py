"""
emotion / AU 特徵提取器基礎類別

定義所有 emotion / AU 提取器的共同介面。與 src/embedding/extractor 的
EmbeddingExtractor、src/age/predictor 的 BasePredictor 互為鏡像:皆為
「純函數 (已讀好的 image: ndarray) → result」—— 讀檔、遍歷、批次迴圈都是
producer 的工作（見 src/common/image_io），extractor 只負責對單張臉做事。

  EmbeddingExtractor.extract(img) -> ndarray      EmoAUExtractor.extract(img) -> dict

唯一刻意不對稱(honest asymmetry):embedding 用單一 int `feature_dim` 描述 schema；
emo_au 特徵具名且跨工具變長，故用單一有序 `output_columns`（raw CSV 欄序的唯一真實來源）。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmoAUExtractor(ABC):
    """
    emotion / AU 特徵提取器抽象基底類別

    所有提取器（OpenFace, Py-Feat, LibreFace, POSTER++, DAN, EmoNet, FER,
    HSEmotion, ViT）繼承此類。
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """工具名稱識別符"""
        ...

    @property
    @abstractmethod
    def output_columns(self) -> List[str]:
        """此工具 raw CSV 的有序欄位（不含 frame）。

        為 schema 的唯一真實來源:producer 以此 reindex 落地，保證欄序穩定。
        必須等於該工具實際輸出的欄序（含 AU / emotion / 額外欄如 gaze、valence）。
        """
        ...

    @abstractmethod
    def extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        從單一影像中提取特徵

        Args:
            image: BGR 格式的影像

        Returns:
            特徵字典 {column_name: value}，若提取失敗返回 None
        """
        ...

    def is_available(self) -> bool:
        """檢查此提取器是否可用（依賴已安裝）"""
        return True
