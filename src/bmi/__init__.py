"""人臉 → BMI 迴歸。

輕量核心（core / embedding，僅依賴 numpy/sklearn）於此 eager 匯出；影像 CNN 路線
（image_data / image，依賴 torch）請直接 import 子模組，避免無 torch 環境載入即失敗。
"""

from .core import encode_groups, load_bmi_subjects, regression_metrics
from .embedding import (
    build_embedding_dataset,
    cross_validate,
    load_arcface_features,
    make_regressor,
    train_final,
)

__all__ = [
    # core
    "regression_metrics",
    "load_bmi_subjects",
    "encode_groups",
    # embedding 路線（ArcFace / MeFEm）
    "load_arcface_features",
    "build_embedding_dataset",
    "make_regressor",
    "cross_validate",
    "train_final",
]
