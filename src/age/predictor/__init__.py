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

import logging
from typing import Dict, List, Optional, Type

from .base import BasePredictor
from .mivolo import MiVOLOPredictor
from .insightface import InsightFacePredictor
from .deepface import DeepFacePredictor
from .fairface import FairFacePredictor
from .opencv_dnn import OpenCVDNNPredictor

logger = logging.getLogger(__name__)

# 名稱 → 類別（加新模型只要在這裡多一行）。各 wrapper 把重依賴延遲到 initialize()，
# 故 eager import class 很便宜（與 EmbeddingExtractor 的 registry 同風格）。
PREDICTORS: Dict[str, Type[BasePredictor]] = {
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

_cache: Dict[str, Optional[BasePredictor]] = {}


def get_predictor(name: str) -> Optional[BasePredictor]:
    """取得預測器（建構 + is_available 篩選 + 快取）。未知/不可用一律回 None。

    回傳的是「尚未載入權重」的 predictor;呼叫端需自行 initialize()（eager 載入）。
    """
    if name in _cache:
        return _cache[name]
    if name not in PREDICTORS:
        logger.warning(f"未知的預測器: {name}")
        _cache[name] = None
        return None
    try:
        p: Optional[BasePredictor] = PREDICTORS[name]()
        if not p.is_available():
            logger.warning(f"✗ {name} 不可用（依賴未安裝或權重缺失）")
            p = None
    except Exception as e:
        logger.warning(f"✗ {name} 建立失敗: {e}")
        p = None
    _cache[name] = p
    return p


def available_predictors(names: Optional[List[str]] = None) -> List[str]:
    """實際可用的預測器名稱（會觸發 is_available 探測）。"""
    return [n for n in (names or PREDICTORS) if get_predictor(n) is not None]


__all__ = [
    "BasePredictor",
    "MiVOLOPredictor",
    "InsightFacePredictor",
    "DeepFacePredictor",
    "FairFacePredictor",
    "OpenCVDNNPredictor",
    "PREDICTORS",
    "BENCHMARK_DIR_NAMES",
    "get_predictor",
    "available_predictors",
]
