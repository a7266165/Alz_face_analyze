"""HSEmotion emotion 提取器:EfficientNet-based，輸出 7-class emotion 機率。

Reference: Savchenko, "Facial expression and attributes recognition based on
multi-task learning of lightweight neural networks", 2021.
https://github.com/HSE-asavchenko/face-emotion-recognition
"""

from typing import Dict, List, Optional

import cv2
import numpy as np
import logging

from .base import EmoAUExtractor
from src.emo_au.extractor.au_config import HARMONIZED_EMOTIONS

logger = logging.getLogger(__name__)

# HSEmotion enet_b2_7 output order:
# Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
HSEMOTION_LABEL_ORDER = [
    "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise",
]


class HSEmotionExtractor(EmoAUExtractor):
    """輸入 BGR aligned 臉，輸出 7-class emotion 機率（無 AU）；需 hsemotion 套件。"""

    def __init__(self, device: str = "cuda", backbone: str = "enet_b2_7"):
        self._device = device
        self._backbone = backbone
        self._recognizer = None

    @property
    def model_name(self) -> str:
        return "hsemotion"

    @property
    def output_columns(self) -> List[str]:
        # 只輸出 7 情緒（harmonized 名稱、無 AU）；extract() 回 name→prob dict。
        return list(HARMONIZED_EMOTIONS)

    def _probe(self) -> bool:
        return self._probe_import("hsemotion.facial_emotions", "pip install hsemotion")

    def _load(self) -> None:
        """載入 HSEmotion recognizer。"""
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self._recognizer = HSEmotionRecognizer(
            model_name=self._backbone, device=self._device,
        )
        logger.info(f"HSEmotion 模型載入完成 (model={self._backbone}, device={self._device})")

    def _extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, scores = self._recognizer.predict_emotions(rgb, logits=False)
        return {
            name: float(scores[i])
            for i, name in enumerate(HSEMOTION_LABEL_ORDER)
        }
