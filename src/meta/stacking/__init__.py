"""
Meta-level stacking 模組

整合多模組特徵，訓練 TabPFN meta-model
"""

from .config import MetaConfig
from .pipeline import MetaPipeline
from .trainer import TabPFNMetaTrainer
from .evaluator import MetaEvaluator

__all__ = [
    "MetaConfig",
    "MetaPipeline",
    "TabPFNMetaTrainer",
    "MetaEvaluator",
]
