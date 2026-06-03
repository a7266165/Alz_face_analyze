"""
Dlib 特徵提取器（ResNet 人臉辨識模型，輸出 128 維）。
"""

from typing import Optional
import numpy as np
import logging

from .base import EmbeddingExtractor
from src.config import EXTERNAL_DIR

logger = logging.getLogger(__name__)


class DlibExtractor(EmbeddingExtractor):
    """
    Dlib 128 維人臉特徵提取器
    """

    # 模型檔案（external/embedding/dlib/ 下）
    _PREDICTOR_FILE = "shape_predictor_68_face_landmarks.dat"
    _FACE_REC_FILE = "dlib_face_recognition_resnet_model_v1.dat"

    def __init__(self):
        self._dlib = None
        self._detector = None
        self._predictor = None
        self._face_rec = None

    @property
    def model_name(self) -> str:
        return "dlib"

    @property
    def feature_dim(self) -> int:
        return 128

    def is_available(self) -> bool:
        try:
            import dlib  # noqa: F401
        except ImportError:
            logger.debug("Dlib 未安裝")
            return False
        dlib_dir = EXTERNAL_DIR / "embedding" / "dlib"
        return (dlib_dir / self._PREDICTOR_FILE).exists() and (
            dlib_dir / self._FACE_REC_FILE
        ).exists()

    def initialize(self) -> None:
        """載入 Dlib 偵測器 + 68 landmark + ResNet 人臉辨識模型。"""
        if self._face_rec is not None:
            return
        import dlib

        self._dlib = dlib
        self._detector = dlib.get_frontal_face_detector()

        dlib_dir = EXTERNAL_DIR / "embedding" / "dlib"
        predictor_path = dlib_dir / self._PREDICTOR_FILE
        face_rec_path = dlib_dir / self._FACE_REC_FILE
        if not predictor_path.exists() or not face_rec_path.exists():
            raise FileNotFoundError(
                f"Dlib 模型檔案缺失於 {dlib_dir}/（需 {self._PREDICTOR_FILE} 與 "
                f"{self._FACE_REC_FILE}），請下載並放置到 external/embedding/dlib/"
            )

        self._predictor = dlib.shape_predictor(str(predictor_path))
        self._face_rec = dlib.face_recognition_model_v1(str(face_rec_path))

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取 Dlib 特徵 (128維)

        Args:
            image: BGR 格式的影像

        Returns:
            128 維特徵向量
        """
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detector(gray, 1)

        if not faces:
            h, w = gray.shape[:2]
            faces = [self._dlib.rectangle(0, 0, w, h)]
            logger.debug("Dlib 未檢測到人臉，使用整張圖")

        shape = self._predictor(gray, faces[0])
        descriptor = self._face_rec.compute_face_descriptor(image, shape)

        return np.array(descriptor, dtype=np.float32)
