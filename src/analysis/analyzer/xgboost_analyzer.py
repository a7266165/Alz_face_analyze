"""
XGBoost 分析器

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
import xgboost as xgb
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

from src.analysis.loader import Dataset

logger = logging.getLogger(__name__)


class XGBoostAnalyzer:
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
        self.models_dir = Path(models_dir) if models_dir else None
        self.reports_dir = Path(reports_dir) if reports_dir else None
        self.pred_prob_dir = Path(pred_prob_dir) if pred_prob_dir else None
        self.n_folds = n_folds
        self.n_drop_features = n_drop_features
        self.random_seed = random_seed
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

    # ========== 主要分析方法 ==========

    def analyze(self, datasets: List[Dataset], filter_stats: Dict = None) -> Dict:
        """分析所有資料集"""
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
                results_by_n_features = self._analyze_with_feature_reduction(
                    dataset, filter_stats
                )
                all_results[dataset_key] = results_by_n_features
            except Exception as e:
                logger.error(f"✗ {dataset_key}: {e}")
                import traceback

                traceback.print_exc()

        return all_results

    def _analyze_with_feature_reduction(
        self, dataset: Dataset, filter_stats: Dict = None
    ) -> Dict[int, Dict]:
        """遞減特徵選擇分析"""
        X, y = dataset.X, dataset.y
        subject_ids = np.array(dataset.subject_ids)
        base_ids = np.array(dataset.base_ids)
        sample_groups = dataset.sample_groups
        data_format = dataset.data_format
        meta = dataset.metadata
        dataset_key = (
            f"{meta['model']}_{meta['feature_type']}_cdr{meta['cdr_threshold']}"
        )

        n_features = X.shape[1]
        selected_indices = list(range(n_features))
        results = {}

        while len(selected_indices) >= 5:
            current_n_features = len(selected_indices)
            X_selected = X[:, selected_indices]

            logger.info(f"特徵數: {current_n_features}")

            # 收集所有 fold 的預測
            all_predictions = []

            if data_format == "per_image":
                fold_results = self._run_kfold_cv_per_image(
                    X_selected,
                    y,
                    subject_ids,
                    base_ids,
                    sample_groups,
                    filter_stats,
                    all_predictions,
                )
            else:
                fold_results = self._run_kfold_cv(
                    X_selected,
                    y,
                    subject_ids,
                    base_ids,
                    filter_stats,
                    all_predictions,
                )

            # 統一儲存所有 fold 的預測
            self._save_all_predictions(
                dataset_key, current_n_features, all_predictions
            )

            result = self._aggregate_fold_results(fold_results, filter_stats)
            result["metadata"] = meta
            result["selected_indices"] = selected_indices.copy()
            result["original_n_features"] = n_features
            result["current_n_features"] = current_n_features
            result["filter_stats"] = filter_stats
            result["data_format"] = data_format
            result["timestamp"] = datetime.now().isoformat()

            self._save_result(dataset_key, current_n_features, result)
            results[current_n_features] = result

            logger.info(
                f"  → Acc: {result['test_metrics']['accuracy']:.3f}, "
                f"MCC: {result['test_metrics']['mcc']:.3f}"
            )

            avg_importance = result["feature_importance"]
            if avg_importance is None or len(selected_indices) <= self.n_drop_features:
                break

            sorted_idx = np.argsort(avg_importance)
            drop_positions = sorted_idx[: self.n_drop_features]
            selected_indices = [
                idx
                for i, idx in enumerate(selected_indices)
                if i not in drop_positions
            ]

        return results

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

            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            fold_xgb_params = self.xgb_params.copy()
            fold_xgb_params["scale_pos_weight"] = scale_pos_weight

            model = xgb.XGBClassifier(**fold_xgb_params)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            test_metrics = self._calculate_metrics(y_test, y_pred, y_prob)
            train_metrics = self._calculate_metrics(y_train, y_pred_train)

            corrected_metrics = None
            if filter_stats:
                corrected_metrics = self._calculate_corrected_metrics_fold(
                    y_test, y_pred, filter_stats, test_subject_ids
                )

            # 收集預測（不立即儲存）
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
                    "corrected_metrics": corrected_metrics,
                    "feature_importance": model.feature_importances_,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "avg_scale_pos_weight": scale_pos_weight,
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

            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            n_train_subjects = len(set(subject_ids_train))
            n_test_subjects = len(set(subject_ids_test))

            fold_xgb_params = self.xgb_params.copy()
            fold_xgb_params["scale_pos_weight"] = scale_pos_weight

            model = xgb.XGBClassifier(**fold_xgb_params)
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

            corrected_metrics = None
            if filter_stats:
                corrected_metrics = self._calculate_corrected_metrics_fold(
                    y_test_agg, y_pred_agg, filter_stats, np.array(test_unique_ids)
                )

            # 收集預測（不立即儲存）
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
                    "corrected_metrics": corrected_metrics,
                    "feature_importance": model.feature_importances_,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_train_subjects": n_train_subjects,
                    "n_test_subjects": n_test_subjects,
                    "avg_scale_pos_weight": scale_pos_weight,
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

        # 分離 train 和 test
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()

        # 儲存 test.csv（保留 fold 欄位）
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

        corrected_metrics = None
        if filter_stats and fold_results[0].get("corrected_metrics"):
            corrected_metrics = {}
            for key in metric_keys:
                values = [
                    f["corrected_metrics"].get(key)
                    for f in fold_results
                    if f["corrected_metrics"]
                    and f["corrected_metrics"].get(key) is not None
                ]
                corrected_metrics[key] = float(np.mean(values)) if values else None
            ccms = [
                np.array(f["corrected_metrics"]["confusion_matrix"])
                for f in fold_results
                if f["corrected_metrics"]
            ]
            corrected_metrics["confusion_matrix"] = np.mean(ccms, axis=0).tolist()

        importances = [f["feature_importance"] for f in fold_results]
        avg_importance = np.mean(importances, axis=0).tolist()

        result = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "corrected_metrics": corrected_metrics,
            "feature_importance": avg_importance,
            "n_train": int(np.mean([f["n_train"] for f in fold_results])),
            "n_test": int(np.mean([f["n_test"] for f in fold_results])),
            "n_folds": len(fold_results),
            "fold_models": [f["model"] for f in fold_results],
            "avg_scale_pos_weight": float(
                np.mean([f["avg_scale_pos_weight"] for f in fold_results])
            ),
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

    def _calculate_corrected_metrics_fold(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        filter_stats: Dict,
        test_subject_ids: np.ndarray = None,
    ) -> Dict:
        """計算單一 fold 的校正後指標"""
        min_age = filter_stats.get("min_predicted_age", 65.0)
        predicted_ages = filter_stats.get("predicted_ages", {})

        if test_subject_ids is not None and predicted_ages:
            y_pred_corrected = y_pred.copy()
            for i, subject_id in enumerate(test_subject_ids):
                age = predicted_ages.get(subject_id, min_age)
                if age < min_age:
                    y_pred_corrected[i] = 0
        else:
            y_pred_corrected = y_pred

        cm = confusion_matrix(y_test, y_pred_corrected)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        corrected_cm = [[int(tn), int(fp)], [int(fn), int(tp)]]

        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / denominator if denominator > 0 else 0

        return {
            "confusion_matrix": corrected_cm,
            "accuracy": float(accuracy),
            "mcc": float(mcc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ========== 儲存 ==========

    def _save_result(self, dataset_key: str, n_features: int, result: Dict):
        """儲存結果"""
        feature_suffix = f"n_features_{n_features}"

        if self.models_dir:
            model_subdir = self.models_dir / feature_suffix
            model_subdir.mkdir(parents=True, exist_ok=True)

            model_path = model_subdir / f"{dataset_key}.json"
            result["fold_models"][0].save_model(str(model_path))

            feature_info = {
                "selected_indices": result["selected_indices"],
                "original_dim": result["original_n_features"],
                "selected_dim": result["current_n_features"],
            }
            feature_path = model_subdir / f"{dataset_key}_features.json"
            with open(feature_path, "w", encoding="utf-8") as f:
                json.dump(feature_info, f, indent=2)

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
            f.write(f"XGBoost 分析報告 ({self.n_folds}-Fold CV)\n")
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
                f.write(f"\n校正後混淆矩陣 ({self.n_folds}-Fold 平均):\n")
                f.write("-" * 30 + "\n")
                cm = result["corrected_metrics"]["confusion_matrix"]
                f.write("         真實0  真實1\n")
                f.write(f"預測0   {cm[0][0]:5.1f}  {cm[1][0]:5.1f}\n")
                f.write(f"預測1   {cm[0][1]:5.1f}  {cm[1][1]:5.1f}\n")

                f.write(f"\n校正後效能 ({self.n_folds}-Fold 平均):\n")
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
