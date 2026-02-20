"""
VGGFace 特徵提取器

使用 DeepFace 提取 4096 維人臉特徵
"""

import tempfile
from typing import Optional
import numpy as np
import logging

from .base import BaseExtractor
from .registry import ExtractorRegistry

logger = logging.getLogger(__name__)


@ExtractorRegistry.register("vggface")
class VGGFaceExtractor(BaseExtractor):
    """
    VGGFace 特徵提取器

    使用 DeepFace 的 VGG-Face 模型提取 4096 維特徵
    """

    def __init__(self):
        self._available = False
        self._deepface = None
        self._init_vggface()

    @property
    def model_name(self) -> str:
        return "vggface"

    @property
    def feature_dim(self) -> int:
        return 4096

    def is_available(self) -> bool:
        return self._available

    def _init_vggface(self):
        """初始化 VGGFace"""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
        except ImportError:
            logger.debug("deepface 未安裝")
            return

        try:
            import cv2
            import os

            # 測試 VGGFace 模型是否可用（會自動下載模型）
            test_img = np.zeros((224, 224, 3), dtype=np.uint8)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                cv2.imwrite(f.name, test_img)
                temp_path = f.name

            try:
                self._deepface.represent(
                    img_path=temp_path,
                    model_name='VGG-Face',
                    enforce_detection=False
                )
            finally:
                os.unlink(temp_path)

            self._available = True

        except Exception as e:
            logger.warning(f"VGGFace 初始化失敗: {e}")

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取 VGGFace 特徵 (4096維)

        Args:
            image: BGR 格式的影像

        Returns:
            4096 維特徵向量
        """
        result = self._deepface.represent(
            img_path=image,
            model_name='VGG-Face',
            enforce_detection=False
        )

        if result and len(result) > 0:
            embedding = np.array(result[0]['embedding'], dtype=np.float32)
            return embedding

        return None
