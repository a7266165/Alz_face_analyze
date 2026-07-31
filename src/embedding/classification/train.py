"""
訓練流程
"""
import numpy as np
import pandas as pd

from src.common.cohort import base_id_of
from src.common.folds import make_splitter
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


def _inner_to_id(df: pd.DataFrame) -> pd.DataFrame:
    """內折表的收合:鍵是 (ID, outer_fold),**不能**重用 _pool_to_id。

    一個 ID 在 10 個外折裡有 9 個屬於訓練集,故有 9 列;只 groupby ID 會把這 9 列
    靜默平均成 1 列。inner_fold 取 first 是安全的:同一 ID 的所有 photo 列共用一個
    base_id,內層 splitter 以 base_id 分組,必然落在同一個內折。
    """
    out = df.groupby(["ID", "outer_fold"], as_index=False).agg(
        y_true=("y_true", "first"), y_score=("y_score", "mean"),
        inner_fold=("inner_fold", "first"))
    return out[["ID", "y_true", "y_score", "outer_fold", "inner_fold"]]


def _inner_kfold(X_tr, y_tr, g_tr, build_estimator, score_method, *,
                 n_inner, fold_kind, fold_seed) -> np.ndarray:
    """在一個外折的訓練集上再切 n_inner 折,回傳該訓練集每一列的 inner-OOF 分數。

    回傳 (scores, inner_fold);受試者數不足以切 n_inner 折時回傳 (None, None)。
    """
    ki = min(n_inner, len(np.unique(g_tr)))
    if ki < 2:
        return None, None
    splitter = make_splitter(fold_kind, n_splits=ki, seed=fold_seed)
    sc = np.full(len(y_tr), np.nan)
    which = np.full(len(y_tr), -1, dtype=int)
    for j, (itr, ite) in enumerate(splitter.split(X_tr, y_tr, groups=g_tr)):
        est = build_estimator()
        est.fit(X_tr[itr], y_tr[itr])
        sc[ite] = _score(est, X_tr[ite], score_method)
        which[ite] = j
    return sc, which


# ── 統一的 OOF 引擎(forward = pool only;reverse = pool + external target) ──
def _kfold(X_pool, ids_pool, y_pool, build_estimator, score_method, needs_cv,
               X_target=None, ids_target=None, y_target=None, n_splits=10, fold_seed=0,
               fold_kind="group", n_inner=0):
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
        fold_kind: group(預設,GroupKFold=現有結果)| stratified_group
            (StratifiedGroupKFold,受試者分組再依 class 分層;PDF Step 1.2)。
        n_inner: >0 時,每個外折的訓練集再切 n_inner 折,額外回傳內折 OOF 表
            (stacking 的 meta 訓練列;見 inner_scores.csv)。0(預設)= 完全不做,
            回傳型別與行為與加這個參數之前逐格相同。

    Returns:
        n_inner=0 → DataFrame [ID, y_true, y_score, fold]。
        n_inner>0 → (上面那張表, DataFrame [ID, y_true, y_score, outer_fold, inner_fold])。

    註:外層那次 fit(在完整外折訓練集上)本來就存在,它就是 nested CV 裡「用完整
    outer-train 重訓後預測 outer-test」的那一步,不是額外成本;新增的只有內層的
    n_inner 次 fit,故每折從 1 次變成 n_inner+1 次。
    """
    ids_pool = np.asarray(ids_pool, dtype=object)
    y_pool = np.asarray(y_pool)
    has_target = X_target is not None

    def _frame(ids, yy, scores, fold):
        return pd.DataFrame({"ID": np.asarray(ids, dtype=object),
                             "y_true": np.asarray(yy).astype(int),
                             "y_score": scores, "fold": fold})

    if not needs_cv:                       # 純 norm scorer(l1_norm/l2_norm):score_method 即 norm 函數,無 fold
        if n_inner:
            raise ValueError("n_inner>0 需要折迴圈,但這個 scorer 是 needs_cv=False "
                             "(l1_norm / l2_norm 直接套範數函數,沒有可切的訓練集)")
        norm_fn = score_method
        frames = [_frame(ids_pool, y_pool, norm_fn(X_pool), -1)]
        if has_target:
            frames.append(_frame(ids_target, y_target, norm_fn(X_target), -1))
        return _pool_to_id(pd.concat(frames, ignore_index=True))

    g = _subject_of(ids_pool)
    k = min(n_splits, len(np.unique(g)))
    if k < 2:
        raise RuntimeError(f"too few subjects ({len(np.unique(g))}) for CV")
    gkf = make_splitter(fold_kind, n_splits=k, seed=fold_seed)

    oof = np.full(len(y_pool), np.nan)
    folds = np.full(len(y_pool), -1, dtype=int)
    accum = np.zeros(len(X_target)) if has_target else None
    inner_frames = []
    nf = 0
    for f, (tri, tei) in enumerate(gkf.split(X_pool, y_pool, groups=g)):
        if n_inner:                        # 內層先做:此折訓練集再切 n_inner 折
            X_tr = X_pool[tri]
            sc, which = _inner_kfold(X_tr, y_pool[tri], g[tri], build_estimator,
                                     score_method, n_inner=n_inner, fold_kind=fold_kind,
                                     fold_seed=fold_seed)
            if sc is not None:
                inner_frames.append(pd.DataFrame({
                    "ID": ids_pool[tri], "y_true": y_pool[tri].astype(int),
                    "y_score": sc, "outer_fold": f, "inner_fold": which}))
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
    out = _pool_to_id(pd.concat(frames, ignore_index=True))
    if not n_inner:
        return out
    inner = (_inner_to_id(pd.concat(inner_frames, ignore_index=True))
             if inner_frames else None)
    return out, inner


# ── 私有 worker ───────────────────────────────────────────────────────────
def _train_forward(X, row_ids, y, build_estimator, score_method, needs_cv, n_splits,
                   fold_seed=0, fold_kind="group", n_inner=0):
    """回傳單一 fold DataFrame [ID, y_true, y_score, fold];n_inner>0 時多回傳內折表。"""
    return _kfold(X, row_ids, y, build_estimator, score_method, needs_cv,
                  n_splits=n_splits, fold_seed=fold_seed, fold_kind=fold_kind,
                  n_inner=n_inner)


def _train_reverse(X_full, ids_full, y_full, build_estimator, score_method, needs_cv,
                   cohort, match_strategies, n_splits, fold_seed=0, fold_kind="group"):
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
                             n_splits=n_splits, fold_seed=fold_seed, fold_kind=fold_kind)
    return out


# ── 對外入口 ──────────────────────────────────────────────────────────────
def train(X, row_ids, y, build_estimator, score_method, needs_cv, direction, *,
          cohort=None, match_strategies=None, n_splits=10, fold_seed=0,
          fold_kind="group", n_inner=0):
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
        fold_kind: group(預設=現有結果)| stratified_group(見 src/common/folds.py)。
        n_inner: >0 時每個外折的訓練集再切 n_inner 折,回傳型別改成 tuple(見 Returns)。
            目前僅 forward 支援;reverse 的訓練池只是年齡配對後的子集,語意不同,先擋掉。

    Returns:
        n_inner=0(預設)→ forward: DataFrame[ID, y_true, y_score, fold];
                          reverse: dict[match_strategy → DataFrame]。
        n_inner>0      → (forward 的那張表, 內折表[ID, y_true, y_score,
                          outer_fold, inner_fold])。

    註:回傳型別隨參數改變,與本函式既有的 direction 慣例一致(forward 回 DataFrame、
    reverse 回 dict)。這樣寫是為了讓 n_inner=0 的既有呼叫端(含正在跑的 sweep)
    完全不受影響。
    """
    if direction == "forward":
        return _train_forward(X, row_ids, y, build_estimator, score_method, needs_cv,
                              n_splits, fold_seed=fold_seed, fold_kind=fold_kind,
                              n_inner=n_inner)
    if direction == "reverse" and n_inner:
        raise NotImplementedError(
            "reverse 尚未支援 n_inner:它的訓練池是年齡配對後的子集(約 40% 的 ID),"
            "external target 那些列又是十個模型的平均分,與 forward 的 nested 語意不同")
    if direction == "reverse":
        return _train_reverse(X, row_ids, y, build_estimator, score_method, needs_cv,
                              cohort, match_strategies or list(_DEFAULT_MS), n_splits,
                              fold_seed=fold_seed, fold_kind=fold_kind)
    raise ValueError(f"unknown direction: {direction!r} (expected 'forward' | 'reverse')")
