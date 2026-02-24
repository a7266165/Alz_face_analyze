"""
Meta Analysis 設定
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MetaConfig:
    """
    Meta 分析設定

    Attributes:
        cdr_threshold: CDR 閾值篩選
        n_folds: 交叉驗證折數
        random_seed: 隨機種子
        save_models: 是否儲存模型
        save_predictions: 是否儲存預測結果
        save_reports: 是否儲存報告
        models: 要分析的嵌入模型列表 (預設: ["arcface", "topofr"])
        asymmetry_method: 不對稱性特徵方法名稱
        n_features_list: 要分析的 n_features 列表 (None 表示自動發現全部)
        demographics_dir: 人口學資料目錄
        predicted_ages_file: 預測年齡 JSON 檔案路徑
    """

    # 資料設定
    cdr_threshold: float = 0

    # 訓練設定
    n_folds: int = 10
    random_seed: int = 42

    # 輸出設定
    save_models: bool = True
    save_predictions: bool = True
    save_reports: bool = True

    # 分析範圍設定
    models: List[str] = field(default_factory=lambda: ["arcface", "topofr"])
    asymmetry_method: str = "absolute_relative_differences"
    n_features_list: Optional[List[int]] = None  # None = 自動發現全部

    # 資料路徑
    demographics_dir: Optional[Path] = None
    predicted_ages_file: Optional[Path] = None

    def __post_init__(self):
        """驗證設定"""
        valid_models = {"arcface", "topofr", "dlib"}
        for m in self.models:
            if m not in valid_models:
                raise ValueError(f"無效的模型: {m}，必須是 {valid_models}")

        valid_methods = {
            "average", "difference", "absolute_difference",
            "relative_differences", "absolute_relative_differences",
        }
        if self.asymmetry_method not in valid_methods:
            raise ValueError(
                f"無效的不對稱方法: {self.asymmetry_method}，"
                f"必須是 {valid_methods}"
            )
