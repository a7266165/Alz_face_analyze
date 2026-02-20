"""
分析器基礎類別

提供 K-fold CV 框架和共用邏輯
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold
import logging

from src.common.metrics import MetricsCalculator, calculate_corrected_metrics

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """
    分析器抽象基礎類別

    提供 K-fold CV 框架，子類只需實現 _train_fold 方法
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_seed: int = 42,
        models_dir: Optional[str] = None,
        reports_dir: Optional[str] = None,
        pred_prob_dir: Optional[str] = None,
    ):
        """
        初始化分析器

        Args:
            n_folds: K-fold CV 的折數
            random_seed: 隨機種子
            models_dir: 模型儲存目錄
            reports_dir: 報告儲存目錄
            pred_prob_dir: 預測機率儲存目錄
        """
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.pred_prob_dir = pred_prob_dir

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名稱"""
        ...

    @abstractmethod
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
        ...

    def run_kfold_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        base_ids: np.ndarray,
        filter_stats: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """
        執行 K-fold CV

        使用 GroupKFold 確保同一人的不同訪視在同一 fold

        Args:
            X: 特徵矩陣
            y: 標籤
            subject_ids: 受試者 ID
            base_ids: 基底 ID（用於分組）
            filter_stats: 篩選統計（用於校正指標）
            **kwargs: 傳遞給 _train_fold 的額外參數

        Returns:
            各 fold 的結果列表
        """
        gkf = GroupKFold(n_splits=self.n_folds)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=base_ids)):
            logger.debug(f"訓練 Fold {fold_idx + 1}/{self.n_folds}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_subject_ids = subject_ids[train_idx]
            test_subject_ids = subject_ids[test_idx]

            # 調用子類實現的訓練方法
            fold_result = self._train_fold(
                X_train, y_train,
                X_test, y_test,
                fold_idx=fold_idx,
                train_subject_ids=train_subject_ids,
                test_subject_ids=test_subject_ids,
                **kwargs
            )

            # 計算指標
            y_pred = fold_result.get('y_pred')
            y_prob = fold_result.get('y_prob')
            y_pred_train = fold_result.get('y_pred_train')
            y_prob_train = fold_result.get('y_prob_train')

            test_metrics = MetricsCalculator.calculate(y_test, y_pred, y_prob)

            train_metrics = None
            if y_pred_train is not None:
                train_metrics = MetricsCalculator.calculate(y_train, y_pred_train, y_prob_train)

            # 計算校正後指標
            corrected_metrics = None
            if filter_stats:
                predicted_ages = filter_stats.get('predicted_ages', {})
                min_age = filter_stats.get('min_predicted_age', 65.0)
                if predicted_ages:
                    corrected_metrics = calculate_corrected_metrics(
                        y_test, y_pred, test_subject_ids,
                        predicted_ages, min_age
                    )

            fold_results.append({
                'model': fold_result.get('model'),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'corrected_metrics': corrected_metrics,
                'feature_importance': fold_result.get('feature_importance'),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_train_subjects': len(np.unique(train_subject_ids)),
                'n_test_subjects': len(np.unique(test_subject_ids)),
                **{k: v for k, v in fold_result.items()
                   if k not in ['model', 'y_pred', 'y_prob', 'y_pred_train',
                                'y_prob_train', 'feature_importance']}
            })

        return fold_results

    def run_kfold_cv_per_image(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        base_ids: np.ndarray,
        sample_groups: np.ndarray,
        filter_stats: Optional[Dict] = None,
        **kwargs
    ) -> List[Dict]:
        """
        執行 K-fold CV（per-image 格式，測試時聚合）

        Args:
            X: 特徵矩陣（每張影像一筆）
            y: 標籤
            subject_ids: 受試者 ID
            base_ids: 基底 ID
            sample_groups: 樣本群組索引
            filter_stats: 篩選統計
            **kwargs: 額外參數

        Returns:
            各 fold 的結果列表
        """
        gkf = GroupKFold(n_splits=self.n_folds)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=base_ids)):
            logger.debug(f"訓練 Fold {fold_idx + 1}/{self.n_folds} (per-image)")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            subject_ids_train = subject_ids[train_idx]
            subject_ids_test = subject_ids[test_idx]

            # 調用子類實現的訓練方法
            fold_result = self._train_fold(
                X_train, y_train,
                X_test, y_test,
                fold_idx=fold_idx,
                train_subject_ids=subject_ids_train,
                test_subject_ids=subject_ids_test,
                **kwargs
            )

            # 獲取預測結果
            y_prob = fold_result.get('y_prob')

            # 聚合預測（同一 subject 的多張影像）
            y_agg, y_prob_agg, subject_ids_agg = self._aggregate_predictions(
                y_test, y_prob, subject_ids_test
            )

            y_pred_agg = (y_prob_agg >= 0.5).astype(int)

            # 計算指標
            test_metrics = MetricsCalculator.calculate(y_agg, y_pred_agg, y_prob_agg)

            # 計算校正後指標
            corrected_metrics = None
            if filter_stats:
                predicted_ages = filter_stats.get('predicted_ages', {})
                min_age = filter_stats.get('min_predicted_age', 65.0)
                if predicted_ages:
                    corrected_metrics = calculate_corrected_metrics(
                        y_agg, y_pred_agg, subject_ids_agg,
                        predicted_ages, min_age
                    )

            fold_results.append({
                'model': fold_result.get('model'),
                'train_metrics': None,  # per-image 不計算訓練指標
                'test_metrics': test_metrics,
                'corrected_metrics': corrected_metrics,
                'feature_importance': fold_result.get('feature_importance'),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_train_subjects': len(np.unique(subject_ids_train)),
                'n_test_subjects': len(np.unique(subject_ids_agg)),
            })

        return fold_results

    def _aggregate_predictions(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        subject_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        聚合同一 subject 的多張影像預測

        Args:
            y_true: 真實標籤
            y_prob: 預測機率
            subject_ids: 受試者 ID

        Returns:
            (聚合後的標籤, 聚合後的機率, 唯一的 subject_ids)
        """
        unique_subjects = np.unique(subject_ids)
        y_agg = []
        y_prob_agg = []

        for subject in unique_subjects:
            mask = subject_ids == subject
            # 標籤取第一個（同一 subject 應該相同）
            y_agg.append(y_true[mask][0])
            # 機率取平均
            y_prob_agg.append(np.mean(y_prob[mask]))

        return (
            np.array(y_agg),
            np.array(y_prob_agg),
            unique_subjects
        )

    def aggregate_fold_results(
        self,
        fold_results: List[Dict],
        metric_key: str = 'test_metrics'
    ) -> Dict[str, Any]:
        """
        聚合所有 fold 的結果

        Args:
            fold_results: 各 fold 的結果列表
            metric_key: 要聚合的指標鍵

        Returns:
            聚合後的結果
        """
        if not fold_results:
            return {}

        # 收集所有 fold 的指標
        all_metrics = [f[metric_key] for f in fold_results if f.get(metric_key)]

        # 使用 MetricsCalculator 聚合
        aggregated = MetricsCalculator.aggregate_fold_metrics(all_metrics)

        # 添加訓練/測試樣本數統計
        aggregated['n_train_total'] = sum(f.get('n_train', 0) for f in fold_results)
        aggregated['n_test_total'] = sum(f.get('n_test', 0) for f in fold_results)
        aggregated['n_folds'] = len(fold_results)

        return aggregated
