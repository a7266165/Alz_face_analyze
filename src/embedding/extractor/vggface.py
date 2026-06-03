"""
VGGFace 特徵提取器（DeepFace VGG-Face，輸出 4096 維）。
"""

from typing import Optional
import numpy as np
import logging

from .base import EmbeddingExtractor

logger = logging.getLogger(__name__)


class VGGFaceExtractor(EmbeddingExtractor):
    """
    VGGFace 4096 維人臉特徵提取器
    """

    def __init__(self):
        self._deepface = None

    @property
    def model_name(self) -> str:
        return "vggface"

    @property
    def feature_dim(self) -> int:
        return 4096

    def is_available(self) -> bool:
        try:
            from deepface import DeepFace  # noqa: F401

            return True
        except ImportError:
            logger.debug("deepface 未安裝")
            return False

    def initialize(self) -> None:
        """載入 DeepFace VGG-Face 模型，並以空圖預熱一次。"""
        if self._deepface is not None:
            return
        from deepface import DeepFace

        # 預熱模型
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.represent(
            test_img,
            model_name="VGG-Face",
            enforce_detection=False,
        )
        self._deepface = DeepFace

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取 VGGFace 特徵 (4096維)

        Args:
            image: BGR 格式的影像

        Returns:
            4096 維特徵向量
        """
        result = self._deepface.represent(
            image,
            model_name="VGG-Face",
            enforce_detection=False,
        )

        if result and len(result) > 0:
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            return embedding

        return None
