"""鏡射人臉 embedding 的左右不對稱特徵。

把左/右鏡射影像各自抽出的 embedding 相減，得四種不對稱表徵供下游分類。
這是 embedding 模態的不對稱，與 src/landmark（landmark 幾何不對稱）不同，勿混淆。
"""

from typing import Dict, List, Optional

import numpy as np

ASYMMETRY_METHODS = (
    "differences",
    "absolute_differences",
    "relative_differences",
    "absolute_relative_differences",
)


def calculate_differences(
    left_features: np.ndarray,
    right_features: np.ndarray,
    methods: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """左右 embedding 的不對稱差，逐 method 一個 (n, dim) 陣列。

    Args:
        left_features, right_features: 同形狀 (n, dim) 的左/右 embedding。
        methods: ASYMMETRY_METHODS 的子集，必填。relative 系列以
                 sqrt(l² + r²) 正規化、分母近 0 處留 0。

    Returns:
        {f"embedding_{method}": float32 array}。
    """
    if methods is None:
        raise ValueError("必須明確指定 methods 參數")
    if invalid := set(methods) - set(ASYMMETRY_METHODS):
        raise ValueError(f"未知的方法: {invalid}")

    diff = left_features - right_features
    norm = np.sqrt(left_features**2 + right_features**2)
    mask = norm > 1e-8

    def relative(numerator):
        out = np.zeros_like(numerator)
        out[mask] = numerator[mask] / norm[mask]
        return out

    builders = {
        "differences": lambda: diff,
        "absolute_differences": lambda: np.abs(diff),
        "relative_differences": lambda: relative(diff),
        "absolute_relative_differences": lambda: relative(np.abs(diff)),
    }
    return {f"embedding_{m}": builders[m]().astype(np.float32) for m in methods}
