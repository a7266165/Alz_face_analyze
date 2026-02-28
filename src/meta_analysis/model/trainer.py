"""
TabPFN Meta Trainer（折疊對齊版）

使用 MetaDataset 的 per-fold 分割進行訓練，
折疊直接對應 base model 的 K-fold CV。
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
    TabPFN Meta 訓練器（折疊對齊版）

    使用 MetaDataset 的 per-fold 分割，
    折疊直接對應 base model 的 K-fold CV。
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def train(self, dataset: MetaDataset) -> TrainResult:
        """
        使用折疊對齊的分割進行訓練

        Args:
            dataset: MetaDataset（per-fold 結構）

        Returns:
            TrainResult
        """
        n_folds = dataset.n_folds
        feature_names = dataset.feature_names

        logger.info(f"開始 {n_folds}-Fold CV 訓練 (TabPFN, 折疊對齊)")
        logger.info(f"特徵數: {dataset.n_features}")

        fold_metrics = []
        all_predictions = []
        all_importances = []
        models = []

        for fold_idx, fold_d in sorted(dataset.fold_data.items()):
            X_train, y_train = fold_d.X_train, fold_d.y_train
            X_test, y_test = fold_d.X_test, fold_d.y_test

            logger.info(
                f"  Fold {fold_idx}: train={len(X_train)}, test={len(X_test)}"
            )

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
                "fold": fold_idx,
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
            for sid, prob in zip(fold_d.train_subject_ids, y_prob_train):
                all_predictions.append({
                    "subject_id": sid,
                    "pred_score": float(prob),
                    "fold": fold_idx,
                    "split": "train",
                })
            for sid, prob in zip(fold_d.test_subject_ids, y_prob_test):
                all_predictions.append({
                    "subject_id": sid,
                    "pred_score": float(prob),
                    "fold": fold_idx,
                    "split": "test",
                })

            logger.info(
                f"  Fold {fold_idx}: Test Acc={test_metrics['accuracy']:.4f}, "
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
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
                "feature_names": feature_names,
                "classifier": "TabPFN",
                "fold_aligned": True,
                "timestamp": datetime.now().isoformat(),
            },
        )
