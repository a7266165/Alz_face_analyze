"""
InsightFace (buffalo_l) 年齡預測器
"""

import logging
from typing import Optional

import numpy as np

from .base import BasePredictor

logger = logging.getLogger(__name__)


class InsightFacePredictor(BasePredictor):
    """InsightFace (buffalo_l) 年齡預測器"""

    def __init__(self):
        self._app = None

    def initialize(self):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise RuntimeError("insightface 未安裝")

        try:
            self._app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace 初始化完成")
        except Exception as e:
            raise RuntimeError(f"InsightFace 初始化失敗: {e}")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        try:
            # InsightFace 期待 BGR：其內部 blobFromImage 已設 swapRB=True
            # （見 model_zoo/attribute.py），會自行轉 RGB。不要在這裡先轉，
            # 否則會雙重交換導致 R/B 通道顛倒。
            faces = self._app.get(image)
            if faces:
                return float(faces[0].age)
        except Exception as e:
            logger.debug(f"InsightFace 預測失敗: {e}")
        return None
