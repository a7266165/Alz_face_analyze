"""
Logistic Regression 分析器

基於 BaseAnalyzer 的 Logistic Regression 實現
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseAnalyzer

logger = logging.getLogger(__name__)


class LogisticAnalyzer(BaseAnalyzer):
    """Logistic Regression 分析器"""

    def __init__(
        self,
        n_folds: int = 5,
        random_seed: int = 42,
        models_dir: Optional[str] = None,
        reports_dir: Optional[str] = None,
        pred_prob_dir: Optional[str] = None,
        lr_params: Optional[Dict] = None,
    ):
        """
        初始化 Logistic Regression 分析器

        Args:
            n_folds: K-fold CV 的折數
            random_seed: 隨機種子
            models_dir: 模型儲存目錄
            reports_dir: 報告儲存目錄
            pred_prob_dir: 預測機率儲存目錄
            lr_params: LogisticRegression 參數
        """
        super().__init__(
            n_folds=n_folds,
            random_seed=random_seed,
            models_dir=models_dir,
            reports_dir=reports_dir,
            pred_prob_dir=pred_prob_dir,
        )

        self.lr_params = lr_params or {
            "max_iter": 1000,
            "random_state": random_seed,
            "solver": "lbfgs",
            "n_jobs": -1,
        }

        logger.info("Logistic Regression 分析器初始化完成")
        logger.info(f"CV 折數: {self.n_folds}")

    @property
    def model_name(self) -> str:
        """模型名稱"""
        return "LogisticRegression"

    def _train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        訓練單一 fold

        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            X_test: 測試特徵
            y_test: 測試標籤
            **kwargs: 額外參數

        Returns:
            包含模型、預測結果等的字典
        """
        # 計算類別權重
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)

        if pos_count > 0 and neg_count > 0:
            class_weight = {0: 1.0, 1: neg_count / pos_count}
        else:
            class_weight = "balanced"

        # 建立模型
        fold_lr_params = self.lr_params.copy()
        fold_lr_params["class_weight"] = class_weight

        model = LogisticRegression(**fold_lr_params)
        model.fit(X_train, y_train)

        # 預測
        y_pred_train = model.predict(X_train)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 特徵重要性（使用係數絕對值）
        feature_importance = np.abs(model.coef_[0])

        return {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_pred_train": y_pred_train,
            "y_prob_train": y_prob_train,
            "feature_importance": feature_importance,
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        }
