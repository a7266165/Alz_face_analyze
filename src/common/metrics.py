"""
統一的指標計算模組

提供二元分類的評估指標計算，避免代碼重複
"""

from typing import Dict, Optional, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)


class MetricsCalculator:
    """
    統一的指標計算器

    用於計算二元分類的各種評估指標，包含：
    - 基本指標：accuracy, precision, recall, f1, mcc
    - 醫學指標：sensitivity, specificity
    - 進階指標：auc（如果提供機率）
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
        tn, fp, fn, tp = MetricsCalculator._extract_confusion_values(cm, y_true)

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'mcc': float(matthews_corrcoef(y_true, y_pred)),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'confusion_matrix': cm.tolist(),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        }

        if y_prob is not None:
            metrics['auc'] = MetricsCalculator._calculate_auc(y_true, y_prob)

        return metrics

    @staticmethod
    def calculate_from_confusion_matrix(
        tn: int, fp: int, fn: int, tp: int
    ) -> Dict[str, float]:
        """
        從混淆矩陣值計算指標

        Args:
            tn, fp, fn, tp: 混淆矩陣的四個值

        Returns:
            指標字典
        """
        total = tn + fp + fn + tp

        accuracy = (tn + tp) / total if total > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = sensitivity
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # MCC 計算
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        mcc = numerator / denominator if denominator > 0 else 0.0

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'mcc': float(mcc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        }

    @staticmethod
    def aggregate_fold_metrics(
        fold_metrics: List[Dict[str, Any]],
        method: str = "mean"
    ) -> Dict[str, Any]:
        """
        聚合多個 fold 的指標

        Args:
            fold_metrics: 各 fold 的指標字典列表
            method: 聚合方法 ("mean" 或 "sum")

        Returns:
            聚合後的指標
        """
        if not fold_metrics:
            return {}

        # 需要聚合的數值指標
        numeric_keys = [
            'accuracy', 'precision', 'recall', 'f1', 'mcc',
            'sensitivity', 'specificity', 'auc'
        ]

        result = {}

        for key in numeric_keys:
            values = [m.get(key) for m in fold_metrics if m.get(key) is not None]
            if values:
                if method == "mean":
                    result[key] = float(np.mean(values))
                    result[f'{key}_std'] = float(np.std(values))
                else:
                    result[key] = float(np.sum(values))

        # 混淆矩陣的累加
        cm_sum = [[0, 0], [0, 0]]
        for m in fold_metrics:
            cm = m.get('confusion_matrix')
            if cm and len(cm) == 2:
                for i in range(2):
                    for j in range(2):
                        cm_sum[i][j] += cm[i][j]

        result['confusion_matrix'] = cm_sum
        result['tn'] = cm_sum[0][0]
        result['fp'] = cm_sum[0][1]
        result['fn'] = cm_sum[1][0]
        result['tp'] = cm_sum[1][1]

        return result

    @staticmethod
    def _extract_confusion_values(
        cm: np.ndarray,
        y_true: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """從混淆矩陣提取 tn, fp, fn, tp"""
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                tn = cm[0, 0] if y_true[0] == 0 else 0
                tp = cm[0, 0] if y_true[0] == 1 else 0

        return int(tn), int(fp), int(fn), int(tp)

    @staticmethod
    def _calculate_auc(
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Optional[float]:
        """計算 AUC，處理異常情況"""
        try:
            # 確保有兩個類別
            if len(np.unique(y_true)) < 2:
                return None
            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            return None


# ========== 便捷函式 ==========

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    計算評估指標的便捷函式

    這是 MetricsCalculator.calculate() 的簡化呼叫方式
    """
    return MetricsCalculator.calculate(y_true, y_pred, y_prob)


def calculate_corrected_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
    predicted_ages: Dict[str, float],
    min_predicted_age: float
) -> Dict[str, Any]:
    """
    計算校正後的指標

    校正邏輯：將預測年齡 < min_predicted_age 的樣本，
    預測結果強制改成 0（陰性/健康）

    Args:
        y_true: 真實標籤
        y_pred: 原始預測標籤
        subject_ids: 樣本的 subject ID
        predicted_ages: subject_id -> 預測年齡的映射
        min_predicted_age: 年齡閾值

    Returns:
        校正後的指標
    """
    y_pred_corrected = y_pred.copy()

    for i, subject_id in enumerate(subject_ids):
        age = predicted_ages.get(str(subject_id), min_predicted_age)
        if age < min_predicted_age:
            y_pred_corrected[i] = 0

    return MetricsCalculator.calculate(y_true, y_pred_corrected)
