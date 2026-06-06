"""FER (justinshenk/fer) emotion 提取器:TensorFlow-based、MTCNN 或 Haar cascade 偵測，輸出 7-class emotion 機率。

Reference: https://github.com/justinshenk/fer
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
    """輸入 BGR 影像（FER 自動偵測人臉），輸出 7-class emotion 機率（無 AU）；需 fer 套件（TensorFlow 環境）。"""

    def __init__(self, mtcnn: bool = True, device: str = "cuda"):
        # device 為對齊工廠統一簽名而收;FER 的 TF detector 不吃 device。
        self._mtcnn = mtcnn
        self._detector = None

    @property
    def model_name(self) -> str:
        return "fer"

    @property
    def output_columns(self) -> List[str]:
        # 只輸出 7 情緒（harmonized 名稱、無 AU）；extract() 回 name→prob dict。
        return list(HARMONIZED_EMOTIONS)

    def _probe(self) -> bool:
        return self._probe_import("fer.fer", "在 fer_env 環境執行: pip install fer")

    def _load(self) -> None:
        """建立 FER detector（MTCNN 或 Haar cascade）。"""
        from fer.fer import FER
        self._detector = FER(mtcnn=self._mtcnn)
        logger.info(f"FER detector 初始化完成 (MTCNN={self._mtcnn})")

    def _extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
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
