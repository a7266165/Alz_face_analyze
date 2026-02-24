"""
Meta Analysis 資料結構定義
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class MetaDataset:
    """
    Meta 分析資料集

    整合 14 個特徵的資料結構：
      - age_error, real_age
      - lr_score_original, lr_score_asymmetry
      - 8 個表情 (Anger~Surprise) + Valence + Arousal

    Attributes:
        X: 特徵矩陣 (n_samples, 14)
        y: 標籤陣列 (0=健康, 1=患者)
        subject_ids: 完整個案編號列表 (如 ["P1-1", "P1-2", "ACS10-1"])
        base_ids: 基礎個案編號列表 (如 ["P1", "P1", "ACS10"])，用於 GroupKFold
        fold_assignments: 各樣本的 fold 分配 (從 LR test CSV 的 fold 欄位)
        feature_names: 特徵欄位名稱
        metadata: 額外資訊 (n_features, model 等)
    """

    X: np.ndarray
    y: np.ndarray
    subject_ids: List[str]
    base_ids: List[str]
    fold_assignments: np.ndarray
    feature_names: List[str]
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """驗證資料一致性"""
        n_samples = len(self.X)
        assert len(self.y) == n_samples, "y 長度與 X 不一致"
        assert len(self.subject_ids) == n_samples, "subject_ids 長度與 X 不一致"
        assert len(self.base_ids) == n_samples, "base_ids 長度與 X 不一致"
        assert len(self.fold_assignments) == n_samples, "fold_assignments 長度與 X 不一致"
        assert self.X.shape[1] == len(self.feature_names), "feature_names 長度與特徵數不一致"

    @property
    def n_samples(self) -> int:
        """樣本數"""
        return len(self.X)

    @property
    def n_features(self) -> int:
        """特徵數"""
        return self.X.shape[1]

    @property
    def n_folds(self) -> int:
        """Fold 數量"""
        return len(np.unique(self.fold_assignments))

    @property
    def class_distribution(self) -> Dict[int, int]:
        """類別分布"""
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def __repr__(self) -> str:
        return (
            f"MetaDataset(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"n_folds={self.n_folds}, class_dist={self.class_distribution})"
        )
