"""
共用模組

提供指標計算、人口學載入、MediaPipe 常數等跨模組共用工具。

不在 __init__ 做 eager re-export，避免 sklearn 等重 dep 強制 load 進
不需要它的 modality env（rotation / asymmetry）。Callers 直接 import
submodule：

    from src.common.metrics import MetricsCalculator
    from src.common.demographics import DemographicsLoader
    from src.common.mediapipe_utils import MIDLINE_INDICES
"""
