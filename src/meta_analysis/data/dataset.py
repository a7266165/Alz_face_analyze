"""
Meta Analysis 資料結構定義

支援折疊對齊 (fold-aligned) 的 stacking 架構：
每個 fold 包含對應的訓練集和測試集資料。
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class FoldData:
    """
    單一折疊的訓練/測試資料

    Attributes:
        X_train: 訓練特徵矩陣 (n_train, 14)
        y_train: 訓練標籤 (n_train,)
        X_test: 測試特徵矩陣 (n_test, 14)
        y_test: 測試標籤 (n_test,)
        train_subject_ids: 訓練集個案編號
        test_subject_ids: 測試集個案編號
        train_base_ids: 訓練集基礎個案編號 (GroupKFold 用)
        test_base_ids: 測試集基礎個案編號
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_subject_ids: List[str]
    test_subject_ids: List[str]
    train_base_ids: List[str]
    test_base_ids: List[str]

    def __post_init__(self):
        """驗證資料一致性"""
        assert len(self.X_train) == len(self.y_train), "X_train 與 y_train 長度不一致"
        assert len(self.X_test) == len(self.y_test), "X_test 與 y_test 長度不一致"
        assert len(self.train_subject_ids) == len(self.X_train), "train_subject_ids 長度不一致"
        assert len(self.test_subject_ids) == len(self.X_test), "test_subject_ids 長度不一致"

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


@dataclass
class MetaDataset:
    """
    Meta 分析資料集（折疊對齊版）

    每個 fold 的訓練/測試分割與 base model 的 K-fold CV 對齊。

    Attributes:
        fold_data: fold_i → FoldData 的對應
        feature_names: 特徵欄位名稱 (14 個)
        metadata: 額外資訊 (n_features, model 等)
    """

    fold_data: Dict[int, FoldData]
    feature_names: List[str]
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """驗證資料一致性"""
        if self.fold_data:
            n_feat = len(self.feature_names)
            for fold_idx, fd in self.fold_data.items():
                assert fd.X_train.shape[1] == n_feat, (
                    f"Fold {fold_idx} X_train 特徵數 {fd.X_train.shape[1]} != {n_feat}"
                )
                assert fd.X_test.shape[1] == n_feat, (
                    f"Fold {fold_idx} X_test 特徵數 {fd.X_test.shape[1]} != {n_feat}"
                )

    @property
    def n_folds(self) -> int:
        """折數"""
        return len(self.fold_data)

    @property
    def n_features(self) -> int:
        """特徵數"""
        return len(self.feature_names)

    @property
    def n_samples(self) -> int:
        """總測試樣本數（各折測試集加總）"""
        return sum(fd.n_test for fd in self.fold_data.values())

    @property
    def class_distribution(self) -> Dict[int, int]:
        """類別分布（從所有折的測試集匯總）"""
        all_y = np.concatenate([fd.y_test for fd in self.fold_data.values()])
        unique, counts = np.unique(all_y, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def __repr__(self) -> str:
        return (
            f"MetaDataset(n_folds={self.n_folds}, n_features={self.n_features}, "
            f"n_samples={self.n_samples}, class_dist={self.class_distribution})"
        )
