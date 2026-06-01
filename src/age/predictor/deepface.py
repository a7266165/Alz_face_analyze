"""
DeepFace 年齡預測器
"""

import logging
from typing import Optional

import numpy as np

from .base import BasePredictor

logger = logging.getLogger(__name__)


class DeepFacePredictor(BasePredictor):
    """DeepFace 年齡預測器"""

    def __init__(self):
        self._deepface = None

    def is_available(self) -> bool:
        try:
            import deepface  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self):
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
        except ImportError:
            raise RuntimeError("deepface 未安裝")

        try:
            # DeepFace 第一個參數（img_path）也接受 BGR numpy array，直接傳。
            test_img = np.zeros((224, 224, 3), dtype=np.uint8)
            self._deepface.analyze(
                test_img, actions=['age'], enforce_detection=False, silent=True,
            )
            logger.info("DeepFace 初始化完成")
        except Exception as e:
            raise RuntimeError(f"DeepFace 初始化失敗: {e}")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        try:
            # DeepFace 第一個參數（img_path）也接受 BGR numpy array，直接傳。
            result = self._deepface.analyze(
                image, actions=['age'], enforce_detection=False, silent=True,
            )
            if result:
                return float(result[0]['age'])
        except Exception as e:
            logger.debug(f"DeepFace 預測失敗: {e}")
        return None
