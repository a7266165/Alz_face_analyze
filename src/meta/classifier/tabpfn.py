"""
TabPFN 分析器

支援兩種資料格式：
- averaged: 每個個案一個平均特徵向量
- per_image: 每個個案多張相片，訓練時展開，測試時聚合

不做特徵篩選（n_drop_features=0），單次評估。
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tabpfn import TabPFNClassifier

from .base import BaseAnalyzer

logger = logging.getLogger(__name__)


class TabPFNAnalyzer(BaseAnalyzer):
    """TabPFN 分析器"""

    def __init__(
        self,
        models_dir: Path = None,
        reports_dir: Path = None,
        pred_prob_dir: Path = None,
        n_folds: int = 5,
        n_drop_features: int = 0,
        random_seed: int = 42,
    ):
        super().__init__(
            n_folds=n_folds,
            n_drop_features=0,  # TabPFN 不做特徵篩選
            random_seed=random_seed,
            models_dir=models_dir,
            reports_dir=reports_dir,
            pred_prob_dir=pred_prob_dir,
        )

        logger.info("TabPFN 分析器初始化完成")
        logger.info(f"CV 折數: {self.n_folds}")

    @property
    def model_name(self) -> str:
        return "TabPFN"

    def _train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        model = TabPFNClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)

        return {
            "model": model,
            "y_pred": model.predict(X_test),
            "y_prob": model.predict_proba(X_test)[:, 1],
            "y_pred_train": model.predict(X_train),
            "y_prob_train": model.predict_proba(X_train)[:, 1],
            "feature_importance": None,
        }
