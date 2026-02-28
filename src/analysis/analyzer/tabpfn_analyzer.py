"""
TabPFN 分析器

支援兩種資料格式：
- averaged: 每個個案一個平均特徵向量
- per_image: 每個個案多張相片，訓練時展開，測試時聚合
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNClassifier

from src.analysis.loader import Dataset

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
        n_drop_features: int = 5,
        random_seed: int = 42,
    ):
        super().__init__(
            n_folds=n_folds,
            random_seed=random_seed,
            models_dir=models_dir,
            reports_dir=reports_dir,
            pred_prob_dir=pred_prob_dir,
        )

        self.models_dir = Path(models_dir) if models_dir else None
        self.reports_dir = Path(reports_dir) if reports_dir else None
        self.pred_prob_dir = Path(pred_prob_dir) if pred_prob_dir else None

        logger.info("TabPFN 分析器初始化完成")
        logger.info(f"CV 折數: {self.n_folds}")

    @property
    def model_name(self) -> str:
        return "TabPFN"

    # ========== 主要分析方法 ==========

    def analyze(self, datasets: List[Dataset], filter_stats: Dict = None) -> Dict:
        """分析所有資料集（不做特徵篩選）"""
        logger.info(f"開始分析 {len(datasets)} 個資料集")

        all_results = {}

        for i, dataset in enumerate(datasets, 1):
            meta = dataset.metadata
            dataset_key = (
                f"{meta['model']}_{meta['feature_type']}_cdr{meta['cdr_threshold']}"
            )

            logger.info(f"\n[{i}/{len(datasets)}] 分析: {dataset_key}")
            logger.info(f"資料格式: {dataset.data_format}")
            logger.info("-" * 50)

            try:
                X, y = dataset.X, dataset.y
                subject_ids = np.array(dataset.subject_ids)
                base_ids = np.array(dataset.base_ids)
                sample_groups = dataset.sample_groups
                n_features = X.shape[1]

                logger.info(f"特徵數: {n_features}")

                all_predictions = []

                if dataset.data_format == "per_image":
                    fold_results = self._run_kfold_cv_per_image(
                        X, y, subject_ids, base_ids, sample_groups,
                        filter_stats, all_predictions,
                    )
                else:
                    fold_results = self._run_kfold_cv(
                        X, y, subject_ids, base_ids,
                        filter_stats, all_predictions,
                    )

                self._save_all_predictions(dataset_key, n_features, all_predictions)

                result = self._aggregate_fold_results(fold_results, filter_stats)
                result["metadata"] = meta
                result["selected_indices"] = list(range(n_features))
                result["original_n_features"] = n_features
                result["current_n_features"] = n_features
                result["filter_stats"] = filter_stats
                result["data_format"] = dataset.data_format
                result["timestamp"] = datetime.now().isoformat()

                self._save_result(dataset_key, n_features, result)

                logger.info(
                    f"  → Acc: {result['test_metrics']['accuracy']:.3f}, "
                    f"MCC: {result['test_metrics']['mcc']:.3f}"
                )

                all_results[dataset_key] = {n_features: result}

            except Exception as e:
                logger.error(f"✗ {dataset_key}: {e}")
                import traceback

                traceback.print_exc()

        return all_results

    # ========== K-fold CV（格式一：averaged）==========

    def _run_kfold_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        base_ids: np.ndarray,
        filter_stats: Dict = None,
        all_predictions: List[Dict] = None,
    ) -> List[Dict]:
        """執行 K-fold CV（格式一）"""
        gkf = GroupKFold(n_splits=self.n_folds)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups=base_ids)
        ):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_subject_ids = subject_ids[train_idx]
            test_subject_ids = subject_ids[test_idx]

            model = TabPFNClassifier(random_state=self.random_seed)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            test_metrics = self._calculate_metrics(y_test, y_pred, y_prob)
            train_metrics = self._calculate_metrics(y_train, y_pred_train)

            # 收集預測
            if all_predictions is not None:
                self._collect_predictions(
                    all_predictions, fold_idx + 1,
                    train_subject_ids, y_prob_train, "train"
                )
                self._collect_predictions(
                    all_predictions, fold_idx + 1,
                    test_subject_ids, y_prob, "test"
                )

            fold_results.append(
                {
                    "model": model,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "feature_importance": None,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
            )

        return fold_results

    # ========== K-fold CV（格式二：per_image）==========

    def _run_kfold_cv_per_image(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        base_ids: np.ndarray,
        sample_groups: np.ndarray,
        filter_stats: Dict = None,
        all_predictions: List[Dict] = None,
    ) -> List[Dict]:
        """執行 K-fold CV（格式二）- 測試時聚合"""
        gkf = GroupKFold(n_splits=self.n_folds)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups=base_ids)
        ):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            subject_ids_train = subject_ids[train_idx]
            subject_ids_test = subject_ids[test_idx]

            n_train_subjects = len(set(subject_ids_train))
            n_test_subjects = len(set(subject_ids_test))

            model = TabPFNClassifier(random_state=self.random_seed)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, y_pred_train)

            y_prob_train_all = model.predict_proba(X_train)[:, 1]
            train_unique_ids, train_prob_agg = self._aggregate_predictions_with_ids(
                subject_ids_train, y_prob_train_all
            )

            y_prob_all = model.predict_proba(X_test)[:, 1]
            y_test_agg, y_pred_agg, y_prob_agg = self._aggregate_predictions(
                subject_ids_test, y_test, y_prob_all
            )

            test_unique_ids, _ = self._aggregate_predictions_with_ids(
                subject_ids_test, y_prob_all
            )

            test_metrics = self._calculate_metrics(y_test_agg, y_pred_agg, y_prob_agg)

            # 收集預測
            if all_predictions is not None:
                self._collect_predictions(
                    all_predictions, fold_idx + 1,
                    train_unique_ids, train_prob_agg, "train"
                )
                self._collect_predictions(
                    all_predictions, fold_idx + 1,
                    test_unique_ids, y_prob_agg, "test"
                )

            fold_results.append(
                {
                    "model": model,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "feature_importance": None,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_train_subjects": n_train_subjects,
                    "n_test_subjects": n_test_subjects,
                }
            )

        return fold_results

    def _aggregate_predictions(
        self,
        subject_ids: np.ndarray,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按個案聚合預測結果"""
        unique_subjects = []
        seen = set()
        for sid in subject_ids:
            if sid not in seen:
                unique_subjects.append(sid)
                seen.add(sid)

        y_true_agg = []
        y_prob_agg = []

        for sid in unique_subjects:
            mask = subject_ids == sid
            y_true_agg.append(y_true[mask][0])
            y_prob_agg.append(np.mean(y_prob[mask]))

        y_true_agg = np.array(y_true_agg)
        y_prob_agg = np.array(y_prob_agg)
        y_pred_agg = (y_prob_agg >= 0.5).astype(int)

        return y_true_agg, y_pred_agg, y_prob_agg

    def _aggregate_predictions_with_ids(
        self, subject_ids: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按個案聚合預測分數，回傳唯一 ID 和聚合後的分數"""
        unique_subjects = []
        seen = set()
        for sid in subject_ids:
            if sid not in seen:
                unique_subjects.append(sid)
                seen.add(sid)

        y_prob_agg = []
        for sid in unique_subjects:
            mask = subject_ids == sid
            y_prob_agg.append(np.mean(y_prob[mask]))

        return np.array(unique_subjects), np.array(y_prob_agg)

    # ========== 預測分數收集與儲存 ==========

    def _collect_predictions(
        self,
        all_predictions: List[Dict],
        fold_idx: int,
        subject_ids: np.ndarray,
        y_prob: np.ndarray,
        split: str,
    ):
        """收集預測到列表"""
        for sid, prob in zip(subject_ids, y_prob):
            all_predictions.append({
                "個案編號": sid,
                "預測分數": float(prob),
                "fold": fold_idx,
                "split": split,
            })

    def _save_all_predictions(
        self,
        dataset_key: str,
        n_features: int,
        all_predictions: List[Dict],
    ):
        """儲存所有 fold 的預測分數"""
        if self.pred_prob_dir is None or not all_predictions:
            return

        feature_suffix = f"n_features_{n_features}"
        output_dir = self.pred_prob_dir / feature_suffix
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(all_predictions)

        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()

        if not test_df.empty:
            test_output = test_df[["個案編號", "預測分數", "fold"]]
            test_output = test_output.sort_values(["fold", "個案編號"])
            test_path = output_dir / f"{dataset_key}_test.csv"
            test_output.to_csv(test_path, index=False, encoding="utf-8-sig")
            logger.debug(f"測試集預測已儲存: {test_path}")

        # 儲存 train.csv（保留 fold 欄位，供 meta-learner 連動使用）
        if not train_df.empty:
            train_output = train_df[["個案編號", "預測分數", "fold"]]
            train_output = train_output.sort_values(["fold", "個案編號"])
            train_path = output_dir / f"{dataset_key}_train.csv"
            train_output.to_csv(train_path, index=False, encoding="utf-8-sig")
            logger.debug(f"訓練集預測已儲存: {train_path}")

    # ========== 結果彙整 ==========

    def _aggregate_fold_results(
        self, fold_results: List[Dict], filter_stats: Dict = None
    ) -> Dict:
        """彙整各 fold 結果"""
        metric_keys = [
            "accuracy", "mcc", "sensitivity", "specificity",
            "precision", "recall", "f1", "auc",
        ]

        train_metrics = {}
        for key in metric_keys:
            values = [
                f["train_metrics"].get(key)
                for f in fold_results
                if f["train_metrics"].get(key) is not None
            ]
            train_metrics[key] = float(np.mean(values)) if values else None
        train_cms = [
            np.array(f["train_metrics"]["confusion_matrix"]) for f in fold_results
        ]
        train_metrics["confusion_matrix"] = np.mean(train_cms, axis=0).tolist()

        test_metrics = {}
        for key in metric_keys:
            values = [
                f["test_metrics"].get(key)
                for f in fold_results
                if f["test_metrics"].get(key) is not None
            ]
            test_metrics[key] = float(np.mean(values)) if values else None
        cms = [np.array(f["test_metrics"]["confusion_matrix"]) for f in fold_results]
        test_metrics["confusion_matrix"] = np.mean(cms, axis=0).tolist()

        # 校正指標：五折 CM 加總後，再加回被篩掉的人
        corrected_metrics = None
        if filter_stats:
            health_filtered_out = filter_stats.get("health_filtered_out", 0)
            patient_filtered_out = filter_stats.get("patient_filtered_out", 0)

            if health_filtered_out > 0 or patient_filtered_out > 0:
                summed_cm = np.sum(cms, axis=0)
                tn, fp, fn, tp = summed_cm.ravel().astype(int)

                tn += health_filtered_out
                fn += patient_filtered_out

                total = tn + fp + fn + tp
                accuracy = (tn + tp) / total if total > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_val = sensitivity
                f1_val = (
                    2 * precision_val * recall_val / (precision_val + recall_val)
                    if (precision_val + recall_val) > 0
                    else 0
                )
                numerator = (tp * tn) - (fp * fn)
                denominator = np.sqrt(
                    (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                )
                mcc_val = numerator / denominator if denominator > 0 else 0

                corrected_metrics = {
                    "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
                    "accuracy": float(accuracy),
                    "mcc": float(mcc_val),
                    "sensitivity": float(sensitivity),
                    "specificity": float(specificity),
                    "precision": float(precision_val),
                    "recall": float(recall_val),
                    "f1": float(f1_val),
                    "auc": None,
                }

        result = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "corrected_metrics": corrected_metrics,
            "feature_importance": None,
            "n_train": int(np.mean([f["n_train"] for f in fold_results])),
            "n_test": int(np.mean([f["n_test"] for f in fold_results])),
            "n_folds": len(fold_results),
            "fold_models": [f["model"] for f in fold_results],
        }

        if "n_train_subjects" in fold_results[0]:
            result["n_train_subjects"] = int(
                np.mean([f["n_train_subjects"] for f in fold_results])
            )
            result["n_test_subjects"] = int(
                np.mean([f["n_test_subjects"] for f in fold_results])
            )

        return result

    # ========== 評估指標 ==========

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict:
        """計算評估指標"""
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                tn = cm[0, 0] if y_true[0] == 0 else 0
                tp = cm[0, 0] if y_true[0] == 1 else 0

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "confusion_matrix": cm.tolist(),
        }

        if y_prob is not None:
            try:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                metrics["auc"] = None

        return metrics

    # ========== 儲存 ==========

    def _save_result(self, dataset_key: str, n_features: int, result: Dict):
        """儲存結果"""
        feature_suffix = f"n_features_{n_features}"

        if self.reports_dir:
            report_subdir = self.reports_dir / feature_suffix
            report_subdir.mkdir(parents=True, exist_ok=True)
            self._save_report(report_subdir, dataset_key, result)

    def _save_report(self, report_dir: Path, dataset_key: str, result: Dict):
        """儲存文字報告"""
        report_path = report_dir / f"{dataset_key}_report.txt"

        metric_order = [
            "accuracy", "mcc", "sensitivity", "specificity",
            "precision", "recall", "f1", "auc",
        ]
        data_format = result.get("data_format", "averaged")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"TabPFN 分析報告 ({self.n_folds}-Fold CV)\n")
            f.write("=" * 60 + "\n")
            f.write(f"資料集: {dataset_key}\n")
            f.write(f"資料格式: {data_format}\n")
            f.write(f"分析時間: {result['timestamp']}\n")
            f.write(
                f"特徵數: {result['current_n_features']} / {result['original_n_features']}\n"
            )
            f.write(f"CV 折數: {result['n_folds']}\n")

            if "n_train_subjects" in result:
                f.write(
                    f"平均訓練集: {result['n_train_subjects']} 個案, {result['n_train']} 樣本\n"
                )
                f.write(
                    f"平均測試集: {result['n_test_subjects']} 個案, {result['n_test']} 樣本\n"
                )
            else:
                f.write(f"平均訓練集: {result['n_train']} 樣本\n")
                f.write(f"平均測試集: {result['n_test']} 樣本\n")

            filter_stats = result.get("filter_stats")
            if filter_stats:
                f.write("\n年齡篩選統計:\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"  最低年齡閾值: {filter_stats.get('min_predicted_age', 'N/A')}\n"
                )
                f.write(
                    f"  整體篩除比例: {filter_stats.get('filtered_out_ratio', 0):.1%}\n"
                )
                f.write(
                    f"  健康組篩除: {filter_stats.get('health_filtered_out', 0)} / "
                    f"{filter_stats.get('health_original', 0)}\n"
                )
                f.write(
                    f"  病患組篩除: {filter_stats.get('patient_filtered_out', 0)} / "
                    f"{filter_stats.get('patient_original', 0)}\n"
                )

            f.write(f"\n測試集混淆矩陣 ({self.n_folds}-Fold 平均):\n")
            f.write("-" * 30 + "\n")
            cm = result["test_metrics"]["confusion_matrix"]
            f.write("         真實0  真實1\n")
            f.write(f"預測0   {cm[0][0]:5.1f}  {cm[1][0]:5.1f}\n")
            f.write(f"預測1   {cm[0][1]:5.1f}  {cm[1][1]:5.1f}\n")

            f.write(f"\n測試集效能 ({self.n_folds}-Fold 平均):\n")
            f.write("-" * 30 + "\n")
            for metric in metric_order:
                value = result["test_metrics"].get(metric)
                if value is not None:
                    f.write(f"  {metric}: {value:.4f}\n")

            if result.get("corrected_metrics"):
                f.write(f"\n校正後混淆矩陣 ({self.n_folds}-Fold 加總+校正):\n")
                f.write("-" * 30 + "\n")
                cm = result["corrected_metrics"]["confusion_matrix"]
                f.write("         真實0  真實1\n")
                f.write(f"預測0   {cm[0][0]:5d}  {cm[1][0]:5d}\n")
                f.write(f"預測1   {cm[0][1]:5d}  {cm[1][1]:5d}\n")

                f.write(f"\n校正後效能:\n")
                f.write("-" * 30 + "\n")
                for metric in metric_order:
                    value = result["corrected_metrics"].get(metric)
                    if value is not None:
                        f.write(f"  {metric}: {value:.4f}\n")

            if result.get("train_metrics"):
                f.write(f"\n訓練集混淆矩陣 ({self.n_folds}-Fold 平均):\n")
                f.write("-" * 30 + "\n")
                cm = result["train_metrics"]["confusion_matrix"]
                f.write("         真實0  真實1\n")
                f.write(f"預測0   {cm[0][0]:5.1f}  {cm[1][0]:5.1f}\n")
                f.write(f"預測1   {cm[0][1]:5.1f}  {cm[1][1]:5.1f}\n")

                f.write(f"\n訓練集效能 ({self.n_folds}-Fold 平均):\n")
                f.write("-" * 30 + "\n")
                for metric in metric_order:
                    value = result["train_metrics"].get(metric)
                    if value is not None:
                        f.write(f"  {metric}: {value:.4f}\n")

    # ========== 兼容 BaseAnalyzer 的抽象方法 ==========

    def _train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs,
    ) -> Dict:
        model = TabPFNClassifier(random_state=self.random_seed)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        perm_result = permutation_importance(
            model, X_test, y_test,
            n_repeats=self.n_perm_repeats,
            random_state=self.random_seed,
            scoring="accuracy",
        )

        return {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_pred_train": y_pred_train,
            "y_prob_train": y_prob_train,
            "feature_importance": perm_result.importances_mean,
        }
