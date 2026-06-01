"""
OpenCV DNN (Caffe) 年齡預測器
"""

import logging
from typing import Optional

import cv2
import numpy as np

from src.config import EXTERNAL_DIR
from .base import BasePredictor

logger = logging.getLogger(__name__)


class OpenCVDNNPredictor(BasePredictor):
    """OpenCV DNN (Caffe) 年齡預測器"""

    AGE_BINS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
                "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    AGE_MIDPOINTS = [1.0, 5.0, 10.0, 17.5,
                     28.5, 40.5, 50.5, 80.0]
    MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
    PROTO_URL = ("https://raw.githubusercontent.com/spmallick/"
                 "learnopencv/master/AgeGender/age_deploy.prototxt")
    MODEL_URL = ("https://github.com/spmallick/learnopencv/raw/"
                 "master/AgeGender/age_net.caffemodel")

    def __init__(self):
        self._net = None
        self._face_detector = None

    def is_available(self) -> bool:
        model_dir = EXTERNAL_DIR / "age" / "opencv_age"
        return ((model_dir / "age_deploy.prototxt").exists()
                and (model_dir / "age_net.caffemodel").exists())

    def initialize(self):
        model_dir = EXTERNAL_DIR / "age" / "opencv_age"
        proto_path = model_dir / "age_deploy.prototxt"
        model_path = model_dir / "age_net.caffemodel"

        if not proto_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"OpenCV DNN 模型不存在: {model_dir}\n"
                f"請下載 age_deploy.prototxt 和 age_net.caffemodel 到 {model_dir}/"
            )

        self._net = cv2.dnn.readNet(str(model_path), str(proto_path))
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logger.info("OpenCV DNN 初始化完成")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(faces) == 0:
                face_crop = image
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                margin = int(max(w, h) * 0.2)
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
                face_crop = image[y1:y2, x1:x2]

            blob = cv2.dnn.blobFromImage(
                face_crop, 1.0, (227, 227), self.MODEL_MEAN, swapRB=False,
            )
            self._net.setInput(blob)
            preds = self._net.forward()
            probs = preds[0]
            midpoints = np.array(self.AGE_MIDPOINTS)
            return float((probs * midpoints).sum())
        except Exception as e:
            logger.debug(f"OpenCV DNN 預測失敗: {e}")
        return None
