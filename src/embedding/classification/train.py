"""
訓練流程
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.common.cohort import base_id_of
from src.common.matching import match_by_age

_DEFAULT_MS = ("no_priority", "priority_acs", "priority_nad")
_MS_PRIORITY = {"no_priority": None, "priority_acs": ["ACS"], "priority_nad": ["NAD"]}


# ── 共用小工具(私有,只服務 OOF) ────────────────────────────────────────────
def _subject_of(ids) -> np.ndarray:
    """ID → subject base_id，如 'ACS1-1' → 'ACS1'(GroupKFold 分組用)。"""
    return np.array([base_id_of(i) for i in ids], dtype=object)


def _score(est, X, score_method: str) -> np.ndarray:
    """依 score_method 對個案評估分數。

    Args:
        score_method: predict_proba | decision_function。

    Returns:
        分數陣列 (n_samples,)。
    """
    if score_method == "predict_proba":
        return est.predict_proba(X)[:, 1]
    if score_method == "decision_function":
        return np.asarray(est.decision_function(X)).ravel()
    raise ValueError(f"unsupported score_method: {score_method!r} "
                     f"(expected 'predict_proba' | 'decision_function')")


def _pool_to_id(df: pd.DataFrame) -> pd.DataFrame:
    """'all' 模式一個 ID 多列 → 1 列/ID(y_score mean,fold first)。"""
    return df.groupby("ID", as_index=False).agg(
        y_true=("y_true", "first"), y_score=("y_score", "mean"), fold=("fold", "first"))


# ── 統一的 OOF 引擎(forward = pool only;reverse = pool + external target) ──
def _kfold(X_pool, ids_pool, y_pool, build_estimator, score_method, needs_cv,
               X_target=None, ids_target=None, y_target=None, n_splits=10, fold_seed=0):
    """K fold訓練

    Args:
        X_pool, ids_pool, y_pool: 特徵 / ID / label。
        build_estimator: estimator。
        score_method: 見 _score。
        needs_cv: False 時 score_method 為 norm 函數,直接套用(l1_norm/l2_norm),不進折迴圈。
        X_target, ids_target, y_target: 外部資料集(reverse 用),與 pool 不相交。
        n_splits: 訓練折數,預設 10。
        fold_seed: GroupKFold 折分 seed。0(預設)→ 確定性折(shuffle=False,現有結果);
            ≥1 → shuffle=True, random_state=fold_seed(repeated-CV 的不同折分)。

    Returns:
        DataFrame [ID, y_true, y_score, fold]。
    """
    ids_pool = np.asarray(ids_pool, dtype=object)
    y_pool = np.asarray(y_pool)
    has_target = X_target is not None

    def _frame(ids, yy, scores, fold):
        return pd.DataFrame({"ID": np.asarray(ids, dtype=object),
                             "y_true": np.asarray(yy).astype(int),
                             "y_score": scores, "fold": fold})

    if not needs_cv:                       # 純 norm scorer(l1_norm/l2_norm):score_method 即 norm 函數,無 fold
        norm_fn = score_method
        frames = [_frame(ids_pool, y_pool, norm_fn(X_pool), -1)]
        if has_target:
            frames.append(_frame(ids_target, y_target, norm_fn(X_target), -1))
        return _pool_to_id(pd.concat(frames, ignore_index=True))

    g = _subject_of(ids_pool)
    k = min(n_splits, len(np.unique(g)))
    if k < 2:
        raise RuntimeError(f"too few subjects ({len(np.unique(g))}) for CV")
    gkf = (GroupKFold(n_splits=k, shuffle=True, random_state=fold_seed)
           if fold_seed else GroupKFold(n_splits=k))

    oof = np.full(len(y_pool), np.nan)
    folds = np.full(len(y_pool), -1, dtype=int)
    accum = np.zeros(len(X_target)) if has_target else None
    nf = 0
    for f, (tri, tei) in enumerate(gkf.split(X_pool, y_pool, groups=g)):
        est = build_estimator()
        est.fit(X_pool[tri], y_pool[tri])
        oof[tei] = _score(est, X_pool[tei], score_method)
        folds[tei] = f
        if has_target:
            accum += _score(est, X_target, score_method)
        nf += 1

    frames = [_frame(ids_pool, y_pool, oof, folds)]
    if has_target:
        frames.append(_frame(ids_target, y_target, accum / nf, -1))
    return _pool_to_id(pd.concat(frames, ignore_index=True))


# ── 私有 worker ───────────────────────────────────────────────────────────
def _train_forward(X, row_ids, y, build_estimator, score_method, needs_cv, n_splits, fold_seed=0):
    """回傳單一 fold DataFrame [ID, y_true, y_score, fold]。"""
    return _kfold(X, row_ids, y, build_estimator, score_method, needs_cv,
                  n_splits=n_splits, fold_seed=fold_seed)


def _train_reverse(X_full, ids_full, y_full, build_estimator, score_method, needs_cv,
                   cohort, match_strategies, n_splits, fold_seed=0):
    """以matched cohort 作訓練池、unmatched 當 external target。
    Args:
        X_full, ids_full, y_full: full cohort 的特徵 / ID / label。
        build_estimator, score_method, needs_cv: 透傳給 _kfold。
        cohort: (p_visit, p_score, hc_visit, hc_score) 4-token,供 match_by_age 算 matched。
        match_strategies: 要跑的 priority 清單。
        n_splits: 折數上限,透傳給 _kfold。

    Returns:
        dict[match_strategy → oof DataFrame [ID, y_true, y_score, fold]]。
    """
    ids_full = np.asarray(ids_full, dtype=object)
    y_full = np.asarray(y_full)
    out = {}
    for ms in match_strategies:
        p_ids, hc_ids = match_by_age(*cohort, priority=_MS_PRIORITY[ms],
                                     level="subject", caliper=1.0)
        matched_ids = np.array(list(set(p_ids) | set(hc_ids)), dtype=object)
        pool = np.isin(ids_full, matched_ids)
        if pool.sum() == 0:
            continue
        target = ~pool
        out[ms] = _kfold(X_full[pool], ids_full[pool], y_full[pool], build_estimator,
                             score_method, needs_cv,
                             X_full[target], ids_full[target], y_full[target],
                             n_splits=n_splits, fold_seed=fold_seed)
    return out


# ── 對外入口 ──────────────────────────────────────────────────────────────
def train(X, row_ids, y, build_estimator, score_method, needs_cv, direction, *,
          cohort=None, match_strategies=None, n_splits=10, fold_seed=0):
    """訓練流程入口。

    Args:
        X, row_ids, y: full cohort 的特徵 / ID / label。
        build_estimator: 全新 estimator。
        score_method, needs_cv: 來自 producer 的 _build_estimator。
        direction: forward | reverse
        cohort: 僅 reverse 用, 供 match_by_age 當場算 matched 訓練池。
        match_strategies: 僅 reverse 用,no_priority | priority_acs | priority_nad
        n_splits: 折數上限,實際折數 = min(n_splits, 受試者數),預設 10。
        fold_seed: GroupKFold 折分 seed(0=確定性折/現有結果;≥1=repeated-CV 不同折分)。

    Returns:
        forward → DataFrame[ID, y_true, y_score, fold];reverse → dict[match_strategy → DataFrame]。
    """
    if direction == "forward":
        return _train_forward(X, row_ids, y, build_estimator, score_method, needs_cv,
                              n_splits, fold_seed=fold_seed)
    if direction == "reverse":
        return _train_reverse(X, row_ids, y, build_estimator, score_method, needs_cv,
                              cohort, match_strategies or list(_DEFAULT_MS), n_splits,
                              fold_seed=fold_seed)
    raise ValueError(f"unknown direction: {direction!r} (expected 'forward' | 'reverse')")
