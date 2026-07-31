"""載入 Embedding 特徵。"""

from typing import List, Sequence, Tuple

import numpy as np

from src.config import EMBEDDING_FEATURES_DIR, NO_NORMALIZE, NORMALIZE_MODES

VALID_PHOTO_MODES = ("mean", "all")

_ASYMMETRY_VARIANTS = (
    "differences",
    "absolute_differences",
    "relative_differences",
    "absolute_relative_differences",
)


def load_feature_matrix(
    ids: Sequence[str],
    model: str,
    variant: str,
    bg_mode: str,
    photo_mode: str = "mean",
    normalize: str = NO_NORMALIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """讀取指定ID群的npy檔。

    Args:
        ids: 載入清單。
        model: arcface | dlib | topofr | vggface | ...
        variant: original | face_left | face_right | differences |
                 absolute_differences | relative_differences |
                 absolute_relative_differences
        bg_mode: background | no_background
        photo_mode: mean | all
        normalize: no_normalize | l1_normalize | l2_normalize。非 no_normalize 時，
                   每個 embedding 先各自除以自身範數再算 variant(PDF §4)。此時不可退回
                   讀預存的 variant 檔——那些是未正規化算出來的。

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
    if normalize not in NORMALIZE_MODES:
        raise ValueError(
            f"normalize must be one of {NORMALIZE_MODES}, got {normalize!r}"
        )

    root = EMBEDDING_FEATURES_DIR / model / bg_mode
    derive = variant in _ASYMMETRY_VARIANTS
    # 正規化須在「算不對稱」之前發生，故只能走 derive 路徑;預存的 variant 檔是未正規化
    # 的成品，正規化時不可拿來當 fallback（會靜默混入錯誤特徵）。
    normalizing = normalize != NO_NORMALIZE
    if derive:
        # 延遲匯入，避免輕量 common 層在載入時就拉進 embedding 套件的重相依。
        from src.embedding.asymmetry import calculate_differences
        left_dir = root / "face_left"
        right_dir = root / "face_right"
        legacy_dir = None if normalizing else root / variant
    else:
        from src.embedding.asymmetry import normalize_embeddings
        feat_dir = root / variant

    vecs: List[np.ndarray] = []
    row_ids: List[str] = []

    for sid in ids:
        if derive:
            lf = left_dir / f"{sid}.npy"
            rf = right_dir / f"{sid}.npy"
            a = None
            if lf.exists() and rf.exists():
                try:
                    a = calculate_differences(
                        np.load(lf), np.load(rf), methods=[variant],
                        normalize=normalize,
                    )[f"embedding_{variant}"]
                except (ValueError, OSError, EOFError):
                    a = None  # 檔案可能正被提取程序寫入(半寫)，退回 legacy
            if a is None:  # 缺檔或讀取失敗 → 過渡相容讀舊的預存 variant
                if legacy_dir is None:  # 正規化時無合法 fallback，寧可漏掉這筆
                    continue
                legacy = legacy_dir / f"{sid}.npy"
                if not legacy.exists():
                    continue
                a = np.load(legacy)
        else:
            npy = feat_dir / f"{sid}.npy"
            if not npy.exists():
                continue
            a = np.load(npy)
            if normalizing:  # original / face_left / face_right 也要能正規化(PDF §2 一致性)
                a = normalize_embeddings(a, normalize)

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
            raise ValueError(f"{sid}: 預期 1D/2D 陣列,得到 shape {a.shape}")

    if not vecs:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=object)
    return np.vstack(vecs), np.asarray(row_ids, dtype=object)
