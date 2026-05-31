"""年齡模組。

``load_predicted_ages`` 是輕量工具（僅依賴標準函式庫），可在無 cv2/torch 的環境
（例如 meta 分析）直接 import。預測器類別依賴 cv2 等重套件，採延遲載入，只有在
真正存取時才會 import ``predictor``。
"""

from .utils import load_age_error, load_predicted_ages

__all__ = [
    "load_age_error",
    "load_predicted_ages",
    "BasePredictor",
    "MiVOLOPredictor",
    "InsightFacePredictor",
    "DeepFacePredictor",
    "FairFacePredictor",
    "OpenCVDNNPredictor",
    "PREDICTOR_MAP",
    "BENCHMARK_DIR_NAMES",
]

_PREDICTOR_EXPORTS = {
    "BasePredictor",
    "MiVOLOPredictor",
    "InsightFacePredictor",
    "DeepFacePredictor",
    "FairFacePredictor",
    "OpenCVDNNPredictor",
    "PREDICTOR_MAP",
    "BENCHMARK_DIR_NAMES",
}


def __getattr__(name):
    if name in _PREDICTOR_EXPORTS:
        from . import predictor
        return getattr(predictor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
