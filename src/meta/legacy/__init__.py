"""
分析模組

整合 base-level classifiers 和 meta-level stacking，
提供資料載入、分類、評估的完整 pipeline。
"""

from src.meta.legacy.loader import DataLoader, Dataset
from src.meta.legacy.classifier import XGBoostAnalyzer, create_analyzer
from src.meta.legacy.evaluation.plotter import ResultPlotter
from src.meta.legacy.stacking import MetaConfig, MetaPipeline, MetaEvaluator
from src.meta.legacy.stacking.trainer import (
    BaseMetaTrainer, TabPFNMetaTrainer,
    LogisticMetaTrainer, XGBoostMetaTrainer,
    TrainResult, create_trainer,
)
from src.meta.legacy.loader import FoldData, MetaDataset, MetaDataLoader

__all__ = [
    # Loader
    "DataLoader",
    "Dataset",
    "FoldData",
    "MetaDataset",
    "MetaDataLoader",
    # Classifier (legacy)
    "XGBoostAnalyzer",
    "create_analyzer",
    # Stacking
    "MetaConfig",
    "MetaPipeline",
    "BaseMetaTrainer",
    "TabPFNMetaTrainer",
    "LogisticMetaTrainer",
    "XGBoostMetaTrainer",
    "TrainResult",
    "create_trainer",
    "MetaEvaluator",
    # Evaluation
    "ResultPlotter",
]
