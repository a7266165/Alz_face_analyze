"""
分析器模組

提供 XGBoost、Logistic Regression 等分析器
"""

from pathlib import Path
from typing import Literal, Optional, Union

from .base import BaseAnalyzer
from .logistic_analyzer import LogisticAnalyzer
from .tabpfn_analyzer import TabPFNAnalyzer
from .xgboost_analyzer import XGBoostAnalyzer

# 類型別名
AnalyzerType = Literal["xgboost", "logistic", "tabpfn"]

# 分析器註冊表
ANALYZER_REGISTRY = {
    "xgboost": XGBoostAnalyzer,
    "logistic": LogisticAnalyzer,
    "tabpfn": TabPFNAnalyzer,
}


def create_analyzer(
    analyzer_type: AnalyzerType,
    models_dir: Path = None,
    reports_dir: Path = None,
    pred_prob_dir: Path = None,
    n_folds: int = 5,
    n_drop_features: int = 5,
    random_seed: int = 42,
    **kwargs,
) -> Union[XGBoostAnalyzer, LogisticAnalyzer]:
    """
    工廠函數：建立分析器實例

    Args:
        analyzer_type: 分析器類型 ("xgboost" 或 "logistic")
        models_dir: 模型儲存目錄
        reports_dir: 報告儲存目錄
        pred_prob_dir: 預測機率儲存目錄
        n_folds: K-fold CV 折數
        n_drop_features: 每次迭代丟棄的特徵數量
        random_seed: 隨機種子
        **kwargs: 傳遞給特定分析器的額外參數
            - xgb_params: XGBoost 專用參數
            - lr_params: Logistic Regression 專用參數

    Returns:
        分析器實例

    Raises:
        ValueError: 如果 analyzer_type 不支援
    """
    if analyzer_type not in ANALYZER_REGISTRY:
        raise ValueError(
            f"不支援的分析器類型: {analyzer_type}. "
            f"可用選項: {list(ANALYZER_REGISTRY.keys())}"
        )

    analyzer_class = ANALYZER_REGISTRY[analyzer_type]

    # 過濾掉 TabPFN 不支援的參數
    if analyzer_type == "tabpfn":
        kwargs.pop("lr_params", None)
        kwargs.pop("xgb_params", None)
        kwargs.pop("importance_ratio", None)

    return analyzer_class(
        models_dir=models_dir,
        reports_dir=reports_dir,
        pred_prob_dir=pred_prob_dir,
        n_folds=n_folds,
        n_drop_features=n_drop_features,
        random_seed=random_seed,
        **kwargs,
    )


def get_available_analyzers() -> list:
    """取得所有可用的分析器類型"""
    return list(ANALYZER_REGISTRY.keys())


__all__ = [
    "BaseAnalyzer",
    "LogisticAnalyzer",
    "TabPFNAnalyzer",
    "XGBoostAnalyzer",
    "create_analyzer",
    "get_available_analyzers",
    "AnalyzerType",
]
