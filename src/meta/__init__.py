"""
分析模組

整合 base-level classifiers 和 meta-level stacking，
提供資料載入、分類、評估的完整 pipeline。
"""

from src.meta.loader import DataLoader, Dataset
from src.meta.classifier import XGBoostAnalyzer, create_analyzer
from src.meta.evaluation.plotter import ResultPlotter
from src.meta.stacking import MetaConfig, MetaPipeline, MetaEvaluator
from src.meta.stacking.trainer import (
    BaseMetaTrainer, TabPFNMetaTrainer,
    LogisticMetaTrainer, XGBoostMetaTrainer,
    TrainResult, create_trainer,
)
from src.meta.loader import FoldData, MetaDataset, MetaDataLoader

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
