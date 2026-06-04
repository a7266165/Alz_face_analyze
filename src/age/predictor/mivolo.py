"""
MiVOLO v2 年齡預測器
"""

import logging
from typing import Optional

import numpy as np

from .base import BasePredictor

logger = logging.getLogger(__name__)


class MiVOLOPredictor(BasePredictor):
    """MiVOLO v2 年齡預測器。
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self):
        """載入模型"""
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            import torch

            use_cuda = torch.cuda.is_available()
            self._device = "cuda" if use_cuda else "cpu"
            dtype = torch.float16 if use_cuda else torch.float32

            self._model = AutoModelForImageClassification.from_pretrained(
                "iitolstykh/mivolo_v2",
                trust_remote_code=True,
                dtype=dtype
            )
            self._processor = AutoImageProcessor.from_pretrained(
                "iitolstykh/mivolo_v2",
                trust_remote_code=True
            )

            self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"✓ MiVOLO 初始化完成 ({self._device.upper()})")

        except Exception as e:
            raise RuntimeError(f"MiVOLO 初始化失敗: {e}")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        """單張已裁切人臉 → 年齡；推論失敗回 None。"""
        import torch

        try:
            inputs = self._processor(images=[image])["pixel_values"]
            inputs = inputs.to(dtype=self._model.dtype, device=self._model.device)

            with torch.no_grad():
                outputs = self._model(faces_input=inputs, body_input=inputs)

            if hasattr(outputs, "age_output"):
                return outputs.age_output[0].item()
            logger.debug("MiVOLO output missing age_output attribute (schema mismatch?)")

        except Exception as e:
            logger.debug(f"預測失敗: {e}")

        return None
