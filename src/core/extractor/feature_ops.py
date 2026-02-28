"""
特徵後處理操作

提供特徵差異計算與人口學特徵合併等純函式
"""

from typing import Dict, List
import numpy as np


def calculate_differences(
    left_features: np.ndarray,
    right_features: np.ndarray,
    methods: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    計算左右特徵差異

    Args:
        left_features: 左臉特徵
        right_features: 右臉特徵
        methods: 計算方法列表

    Returns:
        {method_name: result}
    """
    if methods is None:
        raise ValueError("必須明確指定 methods 參數")

    valid_methods = {
        "differences", "absolute_differences", "averages",
        "relative_differences", "absolute_relative_differences"
    }
    if invalid := set(methods) - valid_methods:
        raise ValueError(f"未知的方法: {invalid}")

    results = {}

    if "differences" in methods:
        diff = left_features - right_features
        results["embedding_differences"] = diff.astype(np.float32)

    if "absolute_differences" in methods:
        abs_diff = np.abs(left_features - right_features)
        results["embedding_absolute_differences"] = abs_diff.astype(np.float32)

    if "averages" in methods:
        results["embedding_averages"] = (
            (left_features + right_features) / 2
        ).astype(np.float32)

    if "relative_differences" in methods:
        diff = left_features - right_features
        norm = np.sqrt(left_features**2 + right_features**2)
        rel_diff = np.zeros_like(diff)
        mask = norm > 1e-8
        rel_diff[mask] = diff[mask] / norm[mask]
        results["embedding_relative_differences"] = rel_diff.astype(np.float32)

    if "absolute_relative_differences" in methods:
        abs_diff = np.abs(left_features - right_features)
        norm = np.sqrt(left_features**2 + right_features**2)
        rel_abs_diff = np.zeros_like(abs_diff)
        mask = norm > 1e-8
        rel_abs_diff[mask] = abs_diff[mask] / norm[mask]
        results["embedding_absolute_relative_differences"] = rel_abs_diff.astype(np.float32)

    return results


def add_demographics(
    features_list: List[np.ndarray],
    ages: List[float],
    genders: List[float]
) -> List[np.ndarray]:
    """
    加入人口學特徵

    Args:
        features_list: 特徵向量列表
        ages: 年齡列表
        genders: 性別列表 (0=女, 1=男)

    Returns:
        結合後的特徵向量列表
    """
    if not (len(features_list) == len(ages) == len(genders)):
        raise ValueError(
            f"長度不一致: features={len(features_list)}, "
            f"ages={len(ages)}, genders={len(genders)}"
        )

    return [
        np.concatenate([feat, np.array([age, gender], dtype=np.float32)])
        for feat, age, gender in zip(features_list, ages, genders)
    ]
