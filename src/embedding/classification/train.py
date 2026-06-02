"""
Classification 下游 Stage-1 的「訓練」—— leakage-free OOF 分數生產。**對外只有一個
`train`(dispatcher),按 direction 分流到兩個私有 worker。不碰磁碟、不算指標、不標 label。**

  - forward:`_oof_kfold(full)` → full cohort 的 OOF(1 列/ID)。
  - reverse:每 match_strategy,`_oof_kfold(matched, target=external)` → 回 dict[ms → oof]。

`_oof_kfold` 是 leakage-free 引擎:pool 做 GroupKFold(Subject)held-out OOF;若給 target
(與 pool 不相交),每折模型 ensemble 預測 target → 每列分數都來自「沒訓練過它」的模型。
forward 與 reverse 只差「pool/target 怎麼給」。

reducer + classifier 串成的單一 Pipeline 由 producer 傳進來的 `build_est`(0-arg factory)
提供;`_oof_kfold` 每折 `build_est()` 拿全新 Pipeline、一次 fit、一次 score,所以 reducer
與 classifier 必用同一折的 train/test,reducer 在 fit 階段碰不到 test fold。

輸出的 oof DataFrame [ID, y_true, y_score, fold] 交給 report 做 label / 落地 / 指標。
"""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.common.matching import match_by_age

N_SPLITS = 10
_DEFAULT_MS = ("no_priority", "priority_acs", "priority_nad")
# reverse 訓練池的優先序映射(對齊 legacy 配對的 priority 設定)
_MS_PRIORITY = {"no_priority": None, "priority_acs": ["ACS"], "priority_nad": ["NAD"]}
_BASE_ID_RE = re.compile(r"^([A-Za-z]+\d+)")


# ----------------------------------------------------------------------------
# 共用小工具(私有,只服務 OOF)
# ----------------------------------------------------------------------------
def _subject_of(ids) -> np.ndarray:
    """ID → subject base_id(GroupKFold 分組用),如 'ACS1-1' → 'ACS1'。
    inline regex,對齊全庫慣例(同 extract_base_id / meta pipeline 那條 regex)。"""
    def bid(i):
        m = _BASE_ID_RE.match(str(i))
        return m.group(1) if m else str(i)
    return np.array([bid(i) for i in ids], dtype=object)


def _score(est, X, score_method: str) -> np.ndarray:
    if score_method == "predict_proba":
        return est.predict_proba(X)[:, 1]
    if score_method == "decision_function":
        return np.asarray(est.decision_function(X)).ravel()
    return np.asarray(est.predict(X)).ravel()


def _pool_to_id(df: pd.DataFrame) -> pd.DataFrame:
    """'all' 模式一個 ID 多列 → 1 列/ID(y_score mean,fold first)。"""
    return df.groupby("ID", as_index=False).agg(
        y_true=("y_true", "first"), y_score=("y_score", "mean"), fold=("fold", "first"))


# ----------------------------------------------------------------------------
# 統一的 OOF 引擎(forward = pool only;reverse = pool + external target)
# ----------------------------------------------------------------------------
def _oof_kfold(X_pool, ids_pool, y_pool, build_est, score_method, needs_cv,
               X_tgt=None, ids_tgt=None, y_tgt=None):
    """Leakage-free 10-fold OOF。pool 做 GroupKFold(Subject),每折 fit→score held-out
    (pool 列 fold>=0)。若給 target(必須與 pool 不相交),每折模型對 target 預測、折平均
    ensemble(target 列 fold=-1)。每列分數都來自沒訓練過它的模型。

    手寫折迴圈(非 cross_val_predict)—— 因為 reverse 要預測外部 target,且要當場記 fold。
    forward 無 target,結果與 cross_val_predict 數值等價(同折、同 fit、同 predict)。
    Returns: DataFrame [ID, y_true, y_score, fold],1 列/ID。
    """
    ids_pool = np.asarray(ids_pool, dtype=object)
    y_pool = np.asarray(y_pool)
    has_tgt = X_tgt is not None

    def _frame(ids, yy, scores, fold):
        return pd.DataFrame({"ID": np.asarray(ids, dtype=object),
                             "y_true": np.asarray(yy).astype(int),
                             "y_score": scores, "fold": fold})

    if not needs_cv:                       # l2_norm:純 norm,無 fold
        frames = [_frame(ids_pool, y_pool, np.linalg.norm(X_pool, axis=1), -1)]
        if has_tgt:
            frames.append(_frame(ids_tgt, y_tgt, np.linalg.norm(X_tgt, axis=1), -1))
        return _pool_to_id(pd.concat(frames, ignore_index=True))

    g = _subject_of(ids_pool)
    k = min(N_SPLITS, len(np.unique(g)))
    if k < 2:
        raise RuntimeError(f"too few subjects ({len(np.unique(g))}) for CV")
    gkf = GroupKFold(n_splits=k)

    oof = np.full(len(y_pool), np.nan)
    folds = np.full(len(y_pool), -1, dtype=int)
    accum = np.zeros(len(X_tgt)) if has_tgt else None
    nf = 0
    for f, (tri, tei) in enumerate(gkf.split(X_pool, y_pool, groups=g)):
        est = build_est()
        est.fit(X_pool[tri], y_pool[tri])
        oof[tei] = _score(est, X_pool[tei], score_method)
        folds[tei] = f
        if has_tgt:
            accum += _score(est, X_tgt, score_method)
        nf += 1

    frames = [_frame(ids_pool, y_pool, oof, folds)]
    if has_tgt:
        frames.append(_frame(ids_tgt, y_tgt, accum / nf, -1))
    return _pool_to_id(pd.concat(frames, ignore_index=True))


# ----------------------------------------------------------------------------
# 私有 worker
# ----------------------------------------------------------------------------
def _train_forward(X, row_ids, y, build_est, score_method, needs_cv):
    """full cohort 的 OOF。回傳單一 oof DataFrame [ID, y_true, y_score, fold]。"""
    return _oof_kfold(X, row_ids, y, build_est, score_method, needs_cv)


def _train_reverse(X_full, ids_full, y_full, build_est, score_method, needs_cv,
                   cohort, match_strategies):
    """每 match_strategy:matched 當訓練池、unmatched 當 external target(leakage-free)。
    回傳 dict[match_strategy → oof DataFrame]。matched 池由 match_by_age(common)當場算,
    matched ⊂ full → 切 X_full,免重載特徵。"""
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
        tgt = ~pool
        out[ms] = _oof_kfold(X_full[pool], ids_full[pool], y_full[pool], build_est,
                             score_method, needs_cv,
                             X_full[tgt], ids_full[tgt], y_full[tgt])
    return out


# ----------------------------------------------------------------------------
# 對外唯一入口:dispatcher
# ----------------------------------------------------------------------------
def train(X, row_ids, y, build_est, score_method, needs_cv, direction, *,
          cohort=None, match_strategies=None):
    """單一公開入口:算 OOF,按 direction 分流。**只產分數,不落地、不評估。**

    Args:
        X, row_ids, y: full cohort 的特徵 / ID / label(load_feature_matrix + producer 組)。
        build_est: 0-arg factory,每次回傳全新 estimator(l2_norm 為 None,不會被呼叫)。
        score_method, needs_cv: 來自 producer 的 _build_estimator。
        direction: 'forward'(回單一 oof df)| 'reverse'(回 dict[ms → oof df])。
        cohort: 僅 reverse 用 —— (p_visit, p_score, hc_visit, hc_score) 4-token,
                供 match_by_age 當場算 matched 訓練池。
        match_strategies: 僅 reverse 用,預設三種 priority。
    """
    if direction == "forward":
        return _train_forward(X, row_ids, y, build_est, score_method, needs_cv)
    if direction == "reverse":
        return _train_reverse(X, row_ids, y, build_est, score_method, needs_cv,
                              cohort, match_strategies or list(_DEFAULT_MS))
    raise ValueError(f"unknown direction: {direction!r} (expected 'forward' | 'reverse')")
