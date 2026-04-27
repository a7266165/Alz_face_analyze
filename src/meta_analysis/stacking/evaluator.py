"""
Meta Analysis 評估指標計算器
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetaEvaluator:
    """
    Meta 分析評估器

    計算二元分類的各種評估指標。
    """

    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        計算完整的評估指標

        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            y_prob: 預測機率（可選）

        Returns:
            包含所有指標的字典
        """
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                tn = int(cm[0, 0]) if y_true[0] == 0 else 0
                tp = int(cm[0, 0]) if y_true[0] == 1 else 0

        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "confusion_matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

        if y_prob is not None:
            metrics["auc"] = MetaEvaluator._calculate_auc(y_true, y_prob)

        return metrics

    @staticmethod
    def aggregate_fold_metrics(
        fold_metrics: List[Dict[str, Any]],
        include_std: bool = True
    ) -> Dict[str, Any]:
        """
        聚合多個 fold 的指標

        Args:
            fold_metrics: 各 fold 的指標字典列表
            include_std: 是否包含標準差

        Returns:
            聚合後的指標
        """
        if not fold_metrics:
            return {}

        # 需要聚合的數值指標
        numeric_keys = [
            "accuracy", "precision", "recall", "f1", "mcc",
            "sensitivity", "specificity", "auc"
        ]

        result = {}

        for key in numeric_keys:
            values = [m.get(key) for m in fold_metrics if m.get(key) is not None]
            if values:
                result[key] = float(np.mean(values))
                if include_std:
                    result[f"{key}_std"] = float(np.std(values))

        # 混淆矩陣：累加
        tn_sum = sum(m.get("tn", 0) for m in fold_metrics)
        fp_sum = sum(m.get("fp", 0) for m in fold_metrics)
        fn_sum = sum(m.get("fn", 0) for m in fold_metrics)
        tp_sum = sum(m.get("tp", 0) for m in fold_metrics)

        result["confusion_matrix"] = [[tn_sum, fp_sum], [fn_sum, tp_sum]]
        result["tn"] = tn_sum
        result["fp"] = fp_sum
        result["fn"] = fn_sum
        result["tp"] = tp_sum

        return result

    @staticmethod
    def _calculate_auc(
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Optional[float]:
        """計算 AUC，處理異常情況"""
        try:
            if len(np.unique(y_true)) < 2:
                return None
            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            return None

    @staticmethod
    def format_metrics_report(metrics: Dict[str, Any], title: str = "") -> str:
        """
        格式化指標報告

        Args:
            metrics: 指標字典
            title: 報告標題

        Returns:
            格式化的報告字串
        """
        lines = []
        if title:
            lines.append(title)
            lines.append("-" * len(title))

        metric_order = [
            "accuracy", "mcc", "sensitivity", "specificity",
            "precision", "recall", "f1", "auc"
        ]

        for key in metric_order:
            value = metrics.get(key)
            std = metrics.get(f"{key}_std")
            if value is not None:
                if std is not None:
                    lines.append(f"  {key}: {value:.4f} (+/- {std:.4f})")
                else:
                    lines.append(f"  {key}: {value:.4f}")

        return "\n".join(lines)
