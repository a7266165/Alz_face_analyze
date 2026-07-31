"""共用折分工具（受試者分組 + 可選的 class 分層）。

現有流程一律 GroupKFold by base_id：同一受試者的所有 visit 落在同一折，避免 leakage。
`20260727-xLLM Training proeducre WORKFLOW.pdf` 的 Step 1 另外要求各折「大致保持 class
分佈」，那是 StratifiedGroupKFold。兩種都在此提供，由呼叫端明示；預設 "group" 以維持
既有落地結果不變。

seed 慣例沿用 embedding classification：
    seed == 0  → shuffle=False（確定性折，對應現有 seed_0 的結果）
    seed >= 1  → shuffle=True, random_state=seed（repeated-CV 的不同折分）
"""
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.common.cohort import base_id_of

__all__ = ["FOLD_KINDS", "subject_groups", "make_splitter", "fold_labels"]

FOLD_KINDS = ("group", "stratified_group")


def subject_groups(ids) -> np.ndarray:
    """ID 串 → subject base_id 串，如 'ACS1-1' → 'ACS1'。分組用。"""
    return np.array([base_id_of(i) for i in ids], dtype=object)


def make_splitter(kind: str = "group", *, n_splits: int = 10, seed: int = 0):
    """回 sklearn splitter。kind ∈ FOLD_KINDS；seed 慣例見模組 docstring。"""
    if kind not in FOLD_KINDS:
        raise ValueError(f"unknown fold kind: {kind!r} (expected one of {FOLD_KINDS})")
    cls = GroupKFold if kind == "group" else StratifiedGroupKFold
    return (cls(n_splits=n_splits, shuffle=True, random_state=seed) if seed
            else cls(n_splits=n_splits))


def fold_labels(y, groups, *, kind: str = "group", n_splits: int = 10,
                seed: int = 0) -> np.ndarray:
    """回長度同 y 的整數折號陣列（0..n_splits-1）。

    實際折數取 min(n_splits, 群組數)；群組數 < 2 時全部給 0（無法切折）。
    """
    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    k = min(n_splits, len(np.unique(groups)))
    if k < 2:
        return np.zeros(len(y), dtype=int)
    sp = make_splitter(kind, n_splits=k, seed=seed)
    out = np.full(len(y), -1, dtype=int)
    for i, (_, te) in enumerate(sp.split(np.zeros(len(y)), y, groups)):
        out[te] = i
    assert (out >= 0).all(), "有樣本沒被分到折"
    return out
