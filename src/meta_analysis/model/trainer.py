"""
TabPFN Meta Trainer

使用預定義的 fold 分配進行 K-fold 交叉驗證訓練。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from tabpfn import TabPFNClassifier

from src.meta_analysis.data.dataset import MetaDataset
from src.meta_analysis.model.evaluator import MetaEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """訓練結果"""

    # 聚合指標
    test_metrics: Dict[str, Any]
    train_metrics: Dict[str, Any]

    # 各 fold 詳細
    fold_metrics: List[Dict[str, Any]]

    # 特徵重要性
    feature_importance: Dict[str, float]

    # 預測結果
    predictions: pd.DataFrame

    # 模型 (TabPFN instances)
    models: List[Any]

    # 元資訊
    metadata: Dict[str, Any] = field(default_factory=dict)


class TabPFNMetaTrainer:
    """
    TabPFN Meta 訓練器

    使用 14 個特徵 (2 LR 分數 + age_error + real_age + 10 emotion)
    訓練 TabPFN 進行最終分類。
    """

    def __init__(self, random_seed: int = 42):
        """
        初始化訓練器

        Args:
            random_seed: 隨機種子
        """
        self.random_seed = random_seed

    def train(self, dataset: MetaDataset) -> TrainResult:
        """
        使用預定義的 fold 分配進行 K-fold CV 訓練

        Args:
            dataset: MetaDataset 物件

        Returns:
            TrainResult 物件
        """
        X = dataset.X
        y = dataset.y
        subject_ids = np.array(dataset.subject_ids)
        fold_assignments = dataset.fold_assignments
        feature_names = dataset.feature_names

        unique_folds = sorted(np.unique(fold_assignments))
        n_folds = len(unique_folds)

        logger.info(f"開始 {n_folds}-Fold CV 訓練 (TabPFN)")
        logger.info(f"樣本數: {len(X)}, 特徵數: {X.shape[1]}")

        fold_metrics = []
        all_predictions = []
        all_importances = []
        models = []

        for fold in unique_folds:
            # 分割資料
            test_mask = fold_assignments == fold
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            train_ids = subject_ids[train_mask]
            test_ids = subject_ids[test_mask]

            # 訓練 TabPFN
            model = TabPFNClassifier(random_state=self.random_seed)
            model.fit(X_train, y_train)
            models.append(model)

            # 預測
            y_pred_train = model.predict(X_train)
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_pred_test = model.predict(X_test)
            y_prob_test = model.predict_proba(X_test)[:, 1]

            # 計算指標
            train_metrics = MetaEvaluator.calculate(y_train, y_pred_train, y_prob_train)
            test_metrics = MetaEvaluator.calculate(y_test, y_pred_test, y_prob_test)

            fold_metrics.append({
                "fold": fold,
                "train": train_metrics,
                "test": test_metrics,
                "n_train": len(X_train),
                "n_test": len(X_test),
            })

            # 特徵重要性 (permutation importance)
            perm_result = permutation_importance(
                model, X_test, y_test,
                n_repeats=10,
                random_state=self.random_seed,
                scoring="accuracy",
            )
            all_importances.append(perm_result.importances_mean)

            # 收集預測
            for sid, prob in zip(train_ids, y_prob_train):
                all_predictions.append({
                    "subject_id": sid,
                    "pred_score": float(prob),
                    "fold": fold,
                    "split": "train",
                })
            for sid, prob in zip(test_ids, y_prob_test):
                all_predictions.append({
                    "subject_id": sid,
                    "pred_score": float(prob),
                    "fold": fold,
                    "split": "test",
                })

            logger.info(
                f"  Fold {fold}: Test Acc={test_metrics['accuracy']:.4f}, "
                f"MCC={test_metrics['mcc']:.4f}, "
                f"AUC={test_metrics.get('auc', 'N/A')}"
            )

        # 聚合指標
        test_fold_metrics = [f["test"] for f in fold_metrics]
        train_fold_metrics = [f["train"] for f in fold_metrics]
        agg_test_metrics = MetaEvaluator.aggregate_fold_metrics(test_fold_metrics)
        agg_train_metrics = MetaEvaluator.aggregate_fold_metrics(train_fold_metrics)

        # 平均特徵重要性
        avg_importance = np.mean(all_importances, axis=0)
        feature_importance = dict(zip(feature_names, avg_importance.tolist()))

        # 建立預測 DataFrame
        predictions_df = pd.DataFrame(all_predictions)

        logger.info(
            f"訓練完成: Test Acc={agg_test_metrics['accuracy']:.4f}, "
            f"MCC={agg_test_metrics['mcc']:.4f}"
        )

        return TrainResult(
            test_metrics=agg_test_metrics,
            train_metrics=agg_train_metrics,
            fold_metrics=fold_metrics,
            feature_importance=feature_importance,
            predictions=predictions_df,
            models=models,
            metadata={
                "n_folds": n_folds,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "feature_names": feature_names,
                "classifier": "TabPFN",
                "timestamp": datetime.now().isoformat(),
            },
        )
