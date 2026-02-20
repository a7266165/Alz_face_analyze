"""
共用模組

提供 Protocol 定義、統一指標計算、和其他共用工具
"""

from src.common.types import (
    Extractor,
    Analyzer,
    DataLoader as DataLoaderProtocol,
    TrainingResult,
    FoldResult,
)
from src.common.metrics import MetricsCalculator, calculate_metrics

__all__ = [
    # Types
    "Extractor",
    "Analyzer",
    "DataLoaderProtocol",
    "TrainingResult",
    "FoldResult",
    # Metrics
    "MetricsCalculator",
    "calculate_metrics",
]
