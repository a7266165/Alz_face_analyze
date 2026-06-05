"""
Meta Trainer（折疊對齊版 v2）

支援三種 classifier：TabPFN、Logistic Regression、XGBoost。
使用 MetaDataset 的 per-fold 分割進行訓練，
折疊直接對應 base model 的 K-fold CV。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.meta.legacy.loader.dataset import MetaDataset
from src.meta.legacy.stacking.evaluator import MetaEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    test_metrics: Dict[str, Any]
    train_metrics: Dict[str, Any]
    fold_metrics: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    predictions: pd.DataFrame
    models: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMetaTrainer(ABC):
    """共用 fold-aligned CV 訓練邏輯。"""

    def __init__(self, random_seed: int = 42, normalize: str = None):
        self.random_seed = random_seed
        self.normalize = normalize

    @abstractmethod
    def _create_model(self) -> Any:
        ...

    @property
    @abstractmethod
    def classifier_name(self) -> str:
        ...

    def train(self, dataset: MetaDataset) -> TrainResult:
        n_folds = dataset.n_folds
        feature_names = dataset.feature_names

        logger.info(f"開始 {n_folds}-Fold CV 訓練 ({self.classifier_name}, 折疊對齊)")
        logger.info(f"特徵數: {dataset.n_features}")

        fold_metrics = []
        all_predictions = []
        all_importances = []
        models = []

        for fold_idx, fold_d in sorted(dataset.fold_data.items()):
            X_train, y_train = fold_d.X_train.copy(), fold_d.y_train
            X_test, y_test = fold_d.X_test.copy(), fold_d.y_test

            if self.normalize == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            elif self.normalize == "standard":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            logger.info(f"  Fold {fold_idx}: train={len(X_train)}, test={len(X_test)}")

            model = self._create_model()
            model.fit(X_train, y_train)
            models.append(model)

            y_pred_train = model.predict(X_train)
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_pred_test = model.predict(X_test)
            y_prob_test = model.predict_proba(X_test)[:, 1]

            train_m = MetaEvaluator.calculate(y_train, y_pred_train, y_prob_train)
            test_m = MetaEvaluator.calculate(y_test, y_pred_test, y_prob_test)

            fold_metrics.append({
                "fold": fold_idx,
                "train": train_m,
                "test": test_m,
                "n_train": len(X_train),
                "n_test": len(X_test),
            })

            perm_result = permutation_importance(
                model, X_test, y_test,
                n_repeats=10,
                random_state=self.random_seed,
                scoring="accuracy",
            )
            all_importances.append(perm_result.importances_mean)

            for sid, prob in zip(fold_d.train_subject_ids, y_prob_train):
                all_predictions.append({
                    "subject_id": sid, "pred_score": float(prob),
                    "fold": fold_idx, "split": "train",
                })
            for sid, prob in zip(fold_d.test_subject_ids, y_prob_test):
                all_predictions.append({
                    "subject_id": sid, "pred_score": float(prob),
                    "fold": fold_idx, "split": "test",
                })

            logger.info(
                f"  Fold {fold_idx}: Test Acc={test_m['accuracy']:.4f}, "
                f"MCC={test_m['mcc']:.4f}, AUC={test_m.get('auc', 'N/A')}"
            )

        test_fold_metrics = [f["test"] for f in fold_metrics]
        train_fold_metrics = [f["train"] for f in fold_metrics]
        agg_test = MetaEvaluator.aggregate_fold_metrics(test_fold_metrics)
        agg_train = MetaEvaluator.aggregate_fold_metrics(train_fold_metrics)

        avg_importance = np.mean(all_importances, axis=0)
        feature_importance = dict(zip(feature_names, avg_importance.tolist()))

        predictions_df = pd.DataFrame(all_predictions)

        logger.info(
            f"訓練完成: Test Acc={agg_test['accuracy']:.4f}, "
            f"MCC={agg_test['mcc']:.4f}"
        )

        return TrainResult(
            test_metrics=agg_test,
            train_metrics=agg_train,
            fold_metrics=fold_metrics,
            feature_importance=feature_importance,
            predictions=predictions_df,
            models=models,
            metadata={
                "n_folds": n_folds,
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
                "feature_names": feature_names,
                "classifier": self.classifier_name,
                "fold_aligned": True,
                "timestamp": datetime.now().isoformat(),
            },
        )


class TabPFNMetaTrainer(BaseMetaTrainer):
    _V3_PATH = Path.home() / "AppData" / "Roaming" / "tabpfn" / "tabpfn-v3-classifier-v3_default.ckpt"

    @property
    def classifier_name(self) -> str:
        return "TabPFN"

    def _create_model(self):
        from tabpfn import TabPFNClassifier
        if self._V3_PATH.exists():
            return TabPFNClassifier(model_path=str(self._V3_PATH), random_state=self.random_seed)
        return TabPFNClassifier(random_state=self.random_seed)


class LogisticMetaTrainer(BaseMetaTrainer):
    def __init__(self, random_seed: int = 42, C: float = 1.0, normalize: str = None):
        super().__init__(random_seed, normalize=normalize)
        self.C = C

    @property
    def classifier_name(self) -> str:
        return "LogisticRegression"

    def _create_model(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=self.C, random_state=self.random_seed,
            max_iter=1000, solver="lbfgs",
        )


class XGBoostMetaTrainer(BaseMetaTrainer):
    def __init__(self, random_seed: int = 42, normalize: str = None, **xgb_params):
        super().__init__(random_seed, normalize=normalize)
        self.xgb_params = xgb_params

    @property
    def classifier_name(self) -> str:
        return "XGBoost"

    def _create_model(self):
        from xgboost import XGBClassifier
        defaults = dict(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            random_state=self.random_seed, eval_metric="logloss",
            verbosity=0,
        )
        defaults.update(self.xgb_params)
        return XGBClassifier(**defaults)


def create_trainer(name: str, random_seed: int = 42,
                   normalize: str = None) -> BaseMetaTrainer:
    if name == "tabpfn":
        return TabPFNMetaTrainer(random_seed=random_seed, normalize=normalize)
    elif name == "logistic":
        return LogisticMetaTrainer(random_seed=random_seed, normalize=normalize)
    elif name == "xgboost":
        return XGBoostMetaTrainer(random_seed=random_seed, normalize=normalize)
    else:
        raise ValueError(f"未知的 classifier: {name}")
