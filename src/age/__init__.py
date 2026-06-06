"""年齡模組。

``load_predicted_ages`` / ``build_cohort_with_age_error`` / ``calculate_age_error`` 可在無
cv2/torch 的環境（例如 meta 分析）直接 import；import 本套件會連帶拉進 pandas + cohort，但不涉及 cv2/torch。
預測器類別依賴 cv2 等重套件，採延遲載入，只有在真正存取時才會 import ``predictor``。
"""

from .utils import (
    build_cohort_with_age_error,
    calculate_age_error,
    load_predicted_ages,
)

__all__ = [
    "build_cohort_with_age_error",
    "calculate_age_error",
    "load_predicted_ages",
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

_PREDICTOR_EXPORTS = {
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
}


def __getattr__(name):
    if name in _PREDICTOR_EXPORTS:
        from . import predictor
        return getattr(predictor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
