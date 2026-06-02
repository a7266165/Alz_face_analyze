"""
Embedding 特徵載入 —— 下游 Stage-1 的最上游,cohort-free / fold-free 的純資料整形。

每個 .npy 是一個 ID(= session,如 ``ACS1-1``)的特徵堆疊:
  original   → (n_photos, dim)
  asymmetry  → (n_pairs, dim)   # difference / absolute_difference / ... 由
                                  scripts/embedding/extract_features.py 預先算好落地

photo_mode 決定列數:
  mean → 把該 ID 內的列平均成 1 條(特徵空間平均)→ 1 列/ID
  all  → 保留每一列(每張照片 / 每個 pair)→ n 列/ID

回傳純矩陣 + row_ids(每列對應的 ID,'all' 模式下 ID 會重複)。這層不碰 label、
不碰 cohort、不碰 fold —— 那些都在上層(producer / OOF 引擎)。Subject 由 ID 推得
(``extract_base_id``),也不在這層。
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
    """讀 ``EMBEDDING_FEATURES_DIR/<model>/<bg_mode>/<variant>/<id>.npy``,套 mean/all。

    Args:
        ids: 要載入的 ID 清單(= .npy 檔名,session 粒度,如 ``'ACS1-1'``)。輸入順序
             即輸出順序(GroupKFold 的折劃分依賴順序,故由呼叫端決定)。
        model: arcface | dlib | topofr | vggface | ...
        variant: original | difference | absolute_difference | average |
                 relative_differences | absolute_relative_differences
        bg_mode: background | no_background(對應目錄層)
        photo_mode: 'mean'(1 列/ID)| 'all'(n 列/ID)

    Returns:
        ``(X, row_ids)``:``X`` 形狀 ``(n_rows, dim)`` float64;``row_ids[i]`` 為
        ``X[i]`` 對應的 ID(mean → 每 ID 一列;all → 每 ID n 列、ID 重複)。
        沒有對應 .npy 的 ID 會被略過(不報錯,與 legacy 行為一致)。
    """
    if photo_mode not in VALID_PHOTO_MODES:
        raise ValueError(
            f"photo_mode must be one of {VALID_PHOTO_MODES}, got {photo_mode!r}")

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
