"""
XGBoost 分析器

支援兩種資料格式：
- averaged: 每個個案一個平均特徵向量
- per_image: 每個個案多張相片，訓練時展開，測試時聚合
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb

from .base import BaseAnalyzer

logger = logging.getLogger(__name__)


class XGBoostAnalyzer(BaseAnalyzer):
    """XGBoost 分析器"""

    def __init__(
        self,
        models_dir: Path = None,
        reports_dir: Path = None,
        pred_prob_dir: Path = None,
        n_folds: int = 5,
        n_drop_features: int = 5,
        random_seed: int = 42,
        xgb_params: Optional[Dict] = None,
        importance_ratio: float = 0.8,
    ):
        super().__init__(
            n_folds=n_folds,
            n_drop_features=n_drop_features,
            random_seed=random_seed,
            models_dir=models_dir,
            reports_dir=reports_dir,
            pred_prob_dir=pred_prob_dir,
        )

        self.importance_ratio = importance_ratio
        self.xgb_params = xgb_params or {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": random_seed,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        logger.info("XGBoost 分析器初始化完成")
        logger.info(
            f"CV 折數: {self.n_folds}, 每次捨棄特徵數: {self.n_drop_features}"
        )

    @property
    def model_name(self) -> str:
        return "XGBoost"

    def _train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        fold_xgb_params = self.xgb_params.copy()
        fold_xgb_params["scale_pos_weight"] = scale_pos_weight

        model = xgb.XGBClassifier(**fold_xgb_params)
        model.fit(X_train, y_train)

        return {
            "model": model,
            "y_pred": model.predict(X_test),
            "y_prob": model.predict_proba(X_test)[:, 1],
            "y_pred_train": model.predict(X_train),
            "y_prob_train": model.predict_proba(X_train)[:, 1],
            "feature_importance": model.feature_importances_,
            "avg_scale_pos_weight": scale_pos_weight,
        }

    def _save_model_file(self, model: Any, path: Path):
        model.save_model(str(path))

    def _get_model_suffix(self) -> str:
        return ".json"
