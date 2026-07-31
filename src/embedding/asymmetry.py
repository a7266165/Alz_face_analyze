"""鏡射人臉 embedding 的左右不對稱特徵。

把左/右鏡射影像各自抽出的 embedding 相減，得四種不對稱表徵供下游分類。
這是 embedding 模態的不對稱，與 src/landmark（landmark 幾何不對稱）不同，勿混淆。
"""

from typing import Dict, List, Optional

import numpy as np

from src.config import NO_NORMALIZE, NORMALIZE_MODES, NORMALIZE_ORD

ASYMMETRY_METHODS = (
    "differences",
    "absolute_differences",
    "relative_differences",
    "absolute_relative_differences",
)


def normalize_embeddings(z: np.ndarray, normalize: str) -> np.ndarray:
    """PDF §4:把每個 embedding 除以自己的範數 → 單位向量(ẑ = z/‖z‖ₚ)。

    normalize = no_normalize 時原樣回傳。ArcFace 回傳未正規化 embedding(‖z‖₂ ≈ 23.7)，
    而模長與年齡/性別/診斷共變，故正規化與否會實質改變下游特徵，非單純換單位。
    """
    if normalize == NO_NORMALIZE:
        return z
    if normalize not in NORMALIZE_ORD:
        raise ValueError(f"未知的 normalize: {normalize!r}(可用 {NORMALIZE_MODES}）")
    n = np.linalg.norm(z, ord=NORMALIZE_ORD[normalize], axis=-1, keepdims=True)
    return np.divide(z, n, out=np.zeros_like(z, dtype=np.float64), where=n > 1e-8)


def calculate_differences(
    left_features: np.ndarray,
    right_features: np.ndarray,
    methods: Optional[List[str]] = None,
    normalize: str = NO_NORMALIZE,
) -> Dict[str, np.ndarray]:
    """左右 embedding 的不對稱差，逐 method 一個 (n, dim) 陣列。

    Args:
        left_features, right_features: 同形狀 (n, dim) 的左/右 embedding。
        methods: ASYMMETRY_METHODS 的子集，必填。relative 系列以
                 sqrt(l² + r²) 正規化、分母近 0 處留 0。
        normalize: no_normalize | l1_normalize | l2_normalize。非 no_normalize 時，
                   先把 left/right 各自除以自身範數再套下列公式(PDF §4)。

    Returns:
        {f"embedding_{method}": float32 array}。
    """
    if methods is None:
        raise ValueError("必須明確指定 methods 參數")
    if invalid := set(methods) - set(ASYMMETRY_METHODS):
        raise ValueError(f"未知的方法: {invalid}")

    left_features = normalize_embeddings(left_features, normalize)
    right_features = normalize_embeddings(right_features, normalize)

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
