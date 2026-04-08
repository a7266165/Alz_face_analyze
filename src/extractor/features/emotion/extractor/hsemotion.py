"""
HSEmotion Emotion 特徵提取器

使用 HSEmotion (EfficientNet-based) 模型提取 7-class emotion probability

Reference:
  Savchenko, A.V., "Facial expression and attributes recognition based
  on multi-task learning of lightweight neural networks", 2021
  https://github.com/HSE-asavchenko/face-emotion-recognition
"""

from typing import Dict, List, Optional

import cv2
import numpy as np
import logging

from .base import BaseAUExtractor
from src.extractor.features.emotion.extractor.au_config import HARMONIZED_EMOTIONS

logger = logging.getLogger(__name__)

# HSEmotion enet_b2_7 output order:
# Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
HSEMOTION_LABEL_ORDER = [
    "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise",
]


class HSEmotionExtractor(BaseAUExtractor):
    """
    HSEmotion Emotion 提取器

    - Input: BGR aligned face image
    - Output: 7-class emotion probability
    - 無 AU 輸出
    - 需要 hsemotion pip package
    """

    def __init__(self, device: str = "cuda", model_name: str = "enet_b2_7"):
        self._device = device
        self._model_name = model_name
        self._fer = None
        self._available = None

    @property
    def tool_name(self) -> str:
        return "hsemotion"

    @property
    def au_columns(self) -> List[str]:
        return []

    @property
    def emotion_columns(self) -> List[str]:
        return list(HARMONIZED_EMOTIONS)

    @property
    def extra_columns(self) -> List[str]:
        return []

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning(
                "HSEmotion 套件未安裝。請執行: pip install hsemotion"
            )
            self._available = False
        return self._available

    def _init_model(self):
        if self._fer is not None:
            return
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self._fer = HSEmotionRecognizer(
            model_name=self._model_name, device=self._device,
        )
        logger.info(f"HSEmotion 模型載入完成 (model={self._model_name}, device={self._device})")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        self._init_model()
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _, scores = self._fer.predict_emotions(rgb, logits=False)
            return {
                HSEMOTION_LABEL_ORDER[i]: float(scores[i])
                for i in range(7)
            }
        except Exception as e:
            logger.debug(f"  HSEmotion 提取失敗: {e}")
            return None
