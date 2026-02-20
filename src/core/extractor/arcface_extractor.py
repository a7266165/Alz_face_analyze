"""
ArcFace 特徵提取器

使用 InsightFace 提取 512 維人臉特徵
"""

from typing import Optional
import numpy as np
import logging

from .base import BaseExtractor
from .registry import ExtractorRegistry

logger = logging.getLogger(__name__)


@ExtractorRegistry.register("arcface")
class ArcFaceExtractor(BaseExtractor):
    """
    ArcFace 特徵提取器

    使用 InsightFace 的 buffalo_l 模型提取 512 維特徵
    """

    def __init__(self):
        self._available = False
        self._app = None
        self._init_arcface()

    @property
    def model_name(self) -> str:
        return "arcface"

    @property
    def feature_dim(self) -> int:
        return 512

    def is_available(self) -> bool:
        return self._available

    def _init_arcface(self):
        """初始化 ArcFace"""
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.debug("InsightFace 未安裝")
            return

        try:
            self._app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))

            self._available = True

        except Exception as e:
            logger.warning(f"ArcFace 初始化失敗: {e}")

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取 ArcFace 特徵 (512維)

        Args:
            image: BGR 格式的影像

        Returns:
            512 維特徵向量
        """
        import cv2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self._app.get(image_rgb)

        if not faces:
            # 偵測失敗時，直接用 recognition model 處理整張圖
            logger.debug("ArcFace 未檢測到人臉，使用整張圖")

            img_resized = cv2.resize(image_rgb, (112, 112))
            img_input = np.transpose(img_resized, (2, 0, 1))[np.newaxis, ...]
            img_input = (img_input - 127.5) / 127.5
            img_input = img_input.astype(np.float32)
            embedding = self._app.models['recognition'].forward(img_input)
            return embedding.flatten().astype(np.float32)

        return faces[0].embedding.astype(np.float32)
