"""
Meta-level stacking 模組

整合多來源特徵，訓練 meta-model (TabPFN / LR / XGBoost)
"""

from .config import MetaConfig
from .pipeline import MetaPipeline
from .trainer import (
    BaseMetaTrainer,
    TabPFNMetaTrainer,
    LogisticMetaTrainer,
    XGBoostMetaTrainer,
    TrainResult,
    create_trainer,
)
from .evaluator import MetaEvaluator

__all__ = [
    "MetaConfig",
    "MetaPipeline",
    "BaseMetaTrainer",
    "TabPFNMetaTrainer",
    "LogisticMetaTrainer",
    "XGBoostMetaTrainer",
    "TrainResult",
    "create_trainer",
    "MetaEvaluator",
]
