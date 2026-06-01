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
            # 自我測試（會自動下載模型）。DeepFace 第一個參數雖名為 img_path，
            # 但也接受 BGR numpy array，直接傳空白影像即可，免暫存檔。
            test_img = np.zeros((224, 224, 3), dtype=np.uint8)
            self._deepface.represent(
                test_img, model_name='VGG-Face', enforce_detection=False,
            )
            self._available = True

        except Exception as e:
            logger.warning(f"VGGFace 初始化失敗: {e}")

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
