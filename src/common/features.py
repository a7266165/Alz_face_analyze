"""
載入Embedding 特徵。
"""

from typing import List, Sequence, Tuple

import numpy as np

from src.config import EMBEDDING_FEATURES_DIR

VALID_PHOTO_MODES = ("mean", "all")


def load_feature_matrix(
    ids: Sequence[str],
    model: str,
    variant: str,
    bg_mode: str,
    photo_mode: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """讀取指定ID群的npy檔。

    Args:
        ids: 載入清單。
        model: arcface | dlib | topofr | vggface | ...
        variant: original | difference | absolute_difference | average |
                 relative_differences | absolute_relative_differences
        bg_mode: background | no_background
        photo_mode: mean | all

    Returns:
        (X, row_ids)
        - X : (m, embedding_dimension)
        if photo_mode == "mean", m = len(ids)
        else m = n * len(ids)

        - row_ids : X[i]對應的 ID。

        模式 mean 每 ID 佔一列，模式 all 每 ID 佔 n 列(ID 重複出現)。
    """
    if photo_mode not in VALID_PHOTO_MODES:
        raise ValueError(
            f"photo_mode must be one of {VALID_PHOTO_MODES}, got {photo_mode!r}"
        )

    feat_dir = EMBEDDING_FEATURES_DIR / model / bg_mode / variant
    vecs: List[np.ndarray] = []
    row_ids: List[str] = []

    for sid in ids:
        npy = feat_dir / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy)
        if a.ndim == 1:
            # 已是單一向量,無 per-photo 維度
            vecs.append(a.astype(np.float64))
            row_ids.append(sid)
        elif a.ndim == 2:
            if photo_mode == "mean":
                vecs.append(a.mean(axis=0).astype(np.float64))
                row_ids.append(sid)
            else:  # all
                for k in range(a.shape[0]):
                    vecs.append(a[k].astype(np.float64))
                    row_ids.append(sid)
        else:
            raise ValueError(f"{npy}: 預期 1D/2D 陣列,得到 shape {a.shape}")

    if not vecs:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=object)
    return np.vstack(vecs), np.asarray(row_ids, dtype=object)
