"""
Logistic Regression 分析器

支援兩種資料格式：
- averaged: 每個個案一個平均特徵向量
- per_image: 每個個案多張相片，訓練時展開，測試時聚合
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseAnalyzer

logger = logging.getLogger(__name__)


class LogisticAnalyzer(BaseAnalyzer):
    """Logistic Regression 分析器"""

    def __init__(
        self,
        models_dir: Path = None,
        reports_dir: Path = None,
        pred_prob_dir: Path = None,
        n_folds: int = 5,
        n_drop_features: int = 5,
        random_seed: int = 42,
        lr_params: Optional[Dict] = None,
    ):
        super().__init__(
            n_folds=n_folds,
            n_drop_features=n_drop_features,
            random_seed=random_seed,
            models_dir=models_dir,
            reports_dir=reports_dir,
            pred_prob_dir=pred_prob_dir,
        )

        self.lr_params = lr_params or {
            "C": 1,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": random_seed,
            "n_jobs": -1,
        }

        logger.info("Logistic Regression 分析器初始化完成")
        logger.info(
            f"CV 折數: {self.n_folds}, 每次捨棄特徵數: {self.n_drop_features}"
        )

    @property
    def model_name(self) -> str:
        return "LogisticRegression"

    def _train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        model = LogisticRegression(**self.lr_params)
        model.fit(X_train, y_train)

        return {
            "model": model,
            "y_pred": model.predict(X_test),
            "y_prob": model.predict_proba(X_test)[:, 1],
            "y_pred_train": model.predict(X_train),
            "y_prob_train": model.predict_proba(X_train)[:, 1],
            "feature_importance": np.abs(model.coef_[0]),
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        }

    def _save_model_file(self, model: Any, path: Path):
        joblib.dump(model, path)

    def _get_model_suffix(self) -> str:
        return ".joblib"
