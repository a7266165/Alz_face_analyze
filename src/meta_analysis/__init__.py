"""
Meta Analysis 模組

整合 14 個特徵（2 LR 分數 + 年齡誤差 + 真實年齡 + 10 情緒），
訓練 TabPFN meta-model 進行二元分類。
"""

from src.meta_analysis.config import MetaConfig
from src.meta_analysis.pipeline import MetaPipeline
from src.meta_analysis.data import FoldData, MetaDataset, MetaDataLoader
from src.meta_analysis.model import TabPFNMetaTrainer, MetaEvaluator

__all__ = [
    "MetaConfig",
    "MetaPipeline",
    "FoldData",
    "MetaDataset",
    "MetaDataLoader",
    "TabPFNMetaTrainer",
    "MetaEvaluator",
]
