"""
FER (justinshenk/fer) Emotion 特徵提取器

使用 FER 套件（TensorFlow-based）提取 7-class emotion probability
支援 MTCNN 或 Haar cascade 人臉偵測

Reference:
  https://github.com/justinshenk/fer
"""

from typing import Dict, List, Optional

import numpy as np
import logging

from .base import EmoAUExtractor
from src.emo_au.extractor.au_config import HARMONIZED_EMOTIONS

logger = logging.getLogger(__name__)

# FER 輸出 key → harmonized 名稱
FER_LABEL_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}


class FERExtractor(EmoAUExtractor):
    """
    FER Emotion 提取器

    - Input: BGR numpy array（FER 自動偵測人臉）
    - Output: 7-class emotion probability
    - 無 AU 輸出
    - 需要 fer pip package（TensorFlow 環境）
    """

    def __init__(self, mtcnn: bool = True, device: str = "cuda"):
        # device 為對齊工廠統一簽名而收;FER 的 TF detector 不吃 device。
        self._mtcnn = mtcnn
        self._detector = None
        self._available = None

    @property
    def model_name(self) -> str:
        return "fer"

    @property
    def output_columns(self) -> List[str]:
        # 只輸出 7 情緒（harmonized 名稱、無 AU）；extract() 回 name→prob dict。
        return list(HARMONIZED_EMOTIONS)

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from fer.fer import FER  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("FER 套件未安裝。請在 fer_env 環境中執行: pip install fer")
            self._available = False
        return self._available

    def initialize(self) -> None:
        """建立 FER detector（MTCNN 或 Haar cascade）。"""
        if self._detector is not None:
            return
        from fer.fer import FER
        self._detector = FER(mtcnn=self._mtcnn)
        logger.info(f"FER detector 初始化完成 (MTCNN={self._mtcnn})")

    def extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        try:
            detections = self._detector.detect_emotions(image)
            if not detections:
                return None
            # aligned face 通常只有一張臉，取第一個
            emo_scores = detections[0]["emotions"]
            return {
                FER_LABEL_MAP[k]: float(v)
                for k, v in emo_scores.items()
                if k in FER_LABEL_MAP
            }
        except Exception as e:
            logger.debug(f"  FER 提取失敗: {e}")
            return None
