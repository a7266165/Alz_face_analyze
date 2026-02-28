"""
Dlib 特徵提取器

提取 128 維人臉特徵
"""

from typing import Optional
import numpy as np
import logging

from .base import BaseExtractor
from .feature_extractor import FeatureExtractor
from src.config import EXTERNAL_DIR

logger = logging.getLogger(__name__)


@FeatureExtractor.register("dlib")
class DlibExtractor(BaseExtractor):
    """
    Dlib 特徵提取器

    使用 dlib 的 ResNet 模型提取 128 維人臉特徵
    """

    def __init__(self):
        self._available = False
        self._init_dlib()

    @property
    def model_name(self) -> str:
        return "dlib"

    @property
    def feature_dim(self) -> int:
        return 128

    def is_available(self) -> bool:
        return self._available

    def _init_dlib(self):
        """初始化 Dlib"""
        try:
            import dlib
            self._dlib = dlib
        except ImportError:
            logger.debug("Dlib 未安裝")
            return

        try:
            # 人臉檢測器
            self._detector = dlib.get_frontal_face_detector()

            # 檢查模型檔案
            dlib_dir = EXTERNAL_DIR / "dlib"
            predictor_path = dlib_dir / "shape_predictor_68_face_landmarks.dat"
            face_rec_path = dlib_dir / "dlib_face_recognition_resnet_model_v1.dat"

            missing_files = []
            if not predictor_path.exists():
                missing_files.append(str(predictor_path))
            if not face_rec_path.exists():
                missing_files.append(str(face_rec_path))

            if missing_files:
                logger.warning(
                    f"Dlib 模型檔案缺失:\n" +
                    "\n".join(f"  - {f}" for f in missing_files) +
                    "\n請下載並放置到 external/dlib/ 目錄"
                )
                return

            # 載入模型
            self._predictor = dlib.shape_predictor(str(predictor_path))
            self._face_rec = dlib.face_recognition_model_v1(str(face_rec_path))

            self._available = True

        except Exception as e:
            logger.warning(f"Dlib 初始化失敗: {e}")

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
