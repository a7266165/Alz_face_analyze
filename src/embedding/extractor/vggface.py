"""
VGGFace 特徵提取器

使用 DeepFace 提取 4096 維人臉特徵
"""

from typing import Optional
import numpy as np
import logging

from .base import EmbeddingExtractor

logger = logging.getLogger(__name__)


class VGGFaceExtractor(EmbeddingExtractor):
    """
    VGGFace 特徵提取器

    使用 DeepFace 的 VGG-Face 模型提取 4096 維特徵
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
        """載入 DeepFace VGG-Face。以一張空白影像做自我測試 + warm-up（首次會自動下載權重）。"""
        if self._deepface is not None:
            return
        from deepface import DeepFace
        # DeepFace 第一個參數雖名為 img_path，但也接受 BGR numpy array，
        # 直接傳空白影像即可觸發下載與 warm-up，免暫存檔。
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.represent(
            test_img, model_name='VGG-Face', enforce_detection=False,
        )
        self._deepface = DeepFace

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """提取 VGGFace 特徵 (4096維)。接受 BGR numpy array。"""
        # DeepFace 第一個參數（img_path）也接受 BGR numpy array，直接傳。
        result = self._deepface.represent(
            image, model_name='VGG-Face', enforce_detection=False,
        )

        if result and len(result) > 0:
            embedding = np.array(result[0]['embedding'], dtype=np.float32)
            return embedding

        return None
