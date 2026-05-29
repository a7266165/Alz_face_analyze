"""
年齡預測器

5 個年齡模型 wrapper，共用 BasePredictor 介面（initialize / predict_single；
逐張 predict 由基底提供）。各 wrapper 在 initialize() 內才延遲 import 重套件
（torch / transformers / insightface / deepface / torchvision），故 import 本套件
本身僅需 cv2 / numpy，可在無這些重依賴的環境安全 import。
"""

import os
# transformers / deepface 預設會嘗試載入 TensorFlow；強制走 PyTorch 後端並壓低 TF log。
# 必須在任何 predictor.initialize() 觸發重套件 import 之前設定，故置於套件最上方。
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .base import BasePredictor
from .mivolo import MiVOLOPredictor
from .insightface import InsightFacePredictor
from .deepface import DeepFacePredictor
from .fairface import FairFacePredictor
from .opencv_dnn import OpenCVDNNPredictor

PREDICTOR_MAP = {
    "mivolo": MiVOLOPredictor,
    "insightface": InsightFacePredictor,
    "deepface": DeepFacePredictor,
    "fairface": FairFacePredictor,
    "opencv_dnn": OpenCVDNNPredictor,
}

BENCHMARK_DIR_NAMES = {
    "mivolo": "1_MiVOLO",
    "insightface": "2_InsightFace",
    "deepface": "3_DeepFace",
    "fairface": "4_FairFace",
    "opencv_dnn": "5_OpenCV_DNN",
}

__all__ = [
    "BasePredictor",
    "MiVOLOPredictor",
    "InsightFacePredictor",
    "DeepFacePredictor",
    "FairFacePredictor",
    "OpenCVDNNPredictor",
    "PREDICTOR_MAP",
    "BENCHMARK_DIR_NAMES",
]
