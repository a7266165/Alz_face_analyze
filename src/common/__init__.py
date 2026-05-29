"""
共用模組

不在 __init__ 做 eager re-export，避免 sklearn 等重 dep 強制 load 進
不需要它的 modality env（rotation / asymmetry）。Callers 直接 import
submodule，例如：

    from src.common.cohort import cohort_list
    from src.common.mediapipe_utils import MIDLINE_POINTS
"""
