"""OOF 落地路徑:寫端(report)與讀端(meta / sweep skip-if-exists)共用的單一真相。"""
from src.config import (
    EMBEDDING_CLASSIFICATION_DIR, NO_NORMALIZE, embedding_classification_path,
)

from .classifier import CLASSIFIERS, DEFAULT_XGB_PARAMS
from .reducer import reducer_label
from .scorer import NORM_SCORERS

MATCH_STRATEGIES = ("no_priority", "priority_acs", "priority_nad")


def clf_param_label(model, lr_C=1.0, xgb_params=None):
    """classifier 的 hyperparameter 路徑子層標籤(scorer 無 → None)。
    logistic → C_<value>;xgb → ne_<n>_md_<d>_lr_<lr>。"""
    if model == "logistic":
        return f"C_{lr_C}"
    if model == "xgb":
        p = xgb_params or DEFAULT_XGB_PARAMS
        return f"ne_{p['n_estimators']}_md_{p['max_depth']}_lr_{p['learning_rate']}"
    return None


def oof_dir(cohort, bg_mode, embedding, variant, photo_mode, reducer, model,
            direction, *, pca_components=None, drop_corr_threshold=None,
            lr_C=1.0, xgb_params=None, seed=0, normalize=NO_NORMALIZE,
            fold_kind="group", root=None):
    """這格 OOF 的輸出目錄。寫端、讀端共用同一組 rlabel / clf_param / dir_seg 規則
    (forward 純 norm scorer(l1/l2_norm) 無 fwd/rev 段、其餘 fwd;reverse 一律 rev),確保寫哪讀哪一致。

    seed:repeated-CV 的折分維度(seed_<N> 緊接 classifier);預設 0(= 確定性折,現有結果)。
    normalize:embedding 正規化(no_normalize | l1_normalize | l2_normalize),路徑落在
    emb 與 variant 之間;預設 no_normalize(= 原始 embedding,現有結果)。

    fold_kind:折分方式(group=GroupKFold,現有結果;stratified_group=StratifiedGroupKFold,
    PDF Step 1.2)。非 group 時路徑多一層 folds_<kind>,兩種折分不互相覆蓋。"""
    root = root or EMBEDDING_CLASSIFICATION_DIR
    is_classify = model in CLASSIFIERS
    rlabel = (reducer_label(reducer, pca_components=pca_components,
                            drop_corr_threshold=drop_corr_threshold)
              if is_classify else "no_drop")
    clf_param = clf_param_label(model, lr_C, xgb_params) if is_classify else None
    dir_seg = ("rev" if direction == "reverse"
               else (None if model in NORM_SCORERS else "fwd"))
    return embedding_classification_path(
        *cohort, bg_mode, embedding, variant, photo_mode, rlabel, normalize=normalize,
        clf=model, clf_param=clf_param, direction=dir_seg, seed=seed,
        fold_kind=fold_kind, root=root)


def oof_paths(cohort, bg_mode, embedding, variant, photo_mode, reducer, model,
              direction, *, pca_components=None, drop_corr_threshold=None,
              lr_C=1.0, xgb_params=None, seed=0, normalize=NO_NORMALIZE,
              fold_kind="group", match_strategies=MATCH_STRATEGIES, root=None):
    """這格預期寫出/讀入的 oof_scores.csv(forward 1 個;reverse 每 match_strategy 一個)。"""
    out_dir = oof_dir(cohort, bg_mode, embedding, variant, photo_mode, reducer, model,
                      direction, pca_components=pca_components,
                      drop_corr_threshold=drop_corr_threshold,
                      lr_C=lr_C, xgb_params=xgb_params, seed=seed,
                      normalize=normalize, fold_kind=fold_kind, root=root)
    if direction == "reverse":
        return [out_dir / ms / "oof_scores.csv" for ms in match_strategies]
    return [out_dir / "oof_scores.csv"]


def inner_path(cohort, bg_mode, embedding, variant, photo_mode, reducer, model,
               direction, *, pca_components=None, drop_corr_threshold=None,
               lr_C=1.0, xgb_params=None, seed=0, normalize=NO_NORMALIZE,
               fold_kind="group", root=None):
    """這格的 inner_scores.csv(內折 OOF,stacking 的 meta 訓練列)。

    與 oof_scores.csv 同層。只有 forward 有:reverse 的訓練池是年齡配對後的子集,
    語意不同(見 train.train 的 NotImplementedError)。
    """
    if direction != "forward":
        raise ValueError(f"inner_scores.csv 只有 forward 有,收到 direction={direction!r}")
    out_dir = oof_dir(cohort, bg_mode, embedding, variant, photo_mode, reducer, model,
                      direction, pca_components=pca_components,
                      drop_corr_threshold=drop_corr_threshold,
                      lr_C=lr_C, xgb_params=xgb_params, seed=seed,
                      normalize=normalize, fold_kind=fold_kind, root=root)
    return out_dir / "inner_scores.csv"
