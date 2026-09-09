"""age_error 單獨分類器:視次層 apparent-age discrepancy → 單變量 LR。

協定對齊 meta 主結果(embedding_only / asymmetry_only 的 score_avg 主結果):同 cohort、
keep_nan 全母體、repeated StratifiedGroupKFold(10-fold × N reps,受試者分組再依 class 分層)、
跨 rep 平均每視次的 OOF 機率(score-avg)、subject 層評估、0.5 threshold。差別只有兩點
(依老師 20260818 說明):
  1. age_error 是純量共變數,不經 base 壓縮,直接當單一特徵;
  2. stacker 換成單變量 LR(make_meta_clf('lr') = SimpleImputer+StandardScaler+LR(C=1,
     class_weight=balanced),即 meta 的 'lr' stacker),不用 TabPFN。

對照老師的逐步規格:
  Use the outer training participants only / Fit the univariate LR on their visit-level
  apparent-age discrepancies / Predict P(AD) for the held-out visits          → meta_oof 逐折
  Repeat until every visit has an OOF probability                              → 一個 rep 的 10 折
  Repeat the complete 10-fold process 100 times                               → n_reps 個不同折分
  Average the 100 OOF probabilities for each visit                            → np.nanmean(reps)
  For participants with repeated visits, average their visit probabilities    → evaluate 的 eval_by_subject
  AUC / balanced accuracy / MCC / sensitivity / specificity @ 0.5 threshold   → compute_clf_metrics

本模組刻意不掛在 src.age.__init__(它會拉進 src.meta.train,而 train 於頂層 import src.age,
掛上去會成 import cycle);請以 `from src.age.classify import ...` 明確載入。
"""
import numpy as np
import pandas as pd

from src.common.folds import fold_labels, subject_groups
from src.meta.train import meta_oof, session_feature_table

FEATURE = "age_error"
# 折分預設 = meta 主結果(score_avg)實際採用的 StratifiedGroupKFold(PDF Step 1.2:各折保持
# class 分佈)。age_error 要與 embedding_only/asymmetry_only/core3 同表比較,就得用同一種折分;
# 已核對:那三欄的落地數字皆帶 fold_kind=stratified_group,且此 cohort 無 GroupKFold 主結果樹。
# 注意這與 src/common/folds.py 的 kind='group' 那個 default 不同——後者是 PDF 前舊結果樹的
# 向後相容 default,不是本 PPT 報的主結果。
DEFAULT_FOLD_KIND = "stratified_group"


def age_error_table(cohort, *, feature=FEATURE, root=None):
    """取 keep_nan 全母體的 [ID, y_true, <feature>](沿用 session_feature_table 的 cohort/母體)。

    complete_case=False → 不丟認知缺值列,母體與 meta 的 keep_nan 主結果一致(headline
    cohort = 2070 視次 / 1610 人);age_error 本身無缺值。其餘 base/variant 參數對 age_error
    無影響(它是共變數),沿用預設即可。
    """
    t = session_feature_table(cohort, complete_case=False, root=root)
    out = t[["ID", "y_true", feature]].reset_index(drop=True)
    if out[feature].isna().any():
        n = int(out[feature].isna().sum())
        raise ValueError(f"{feature} 有 {n} 個缺值——age_error 應為全母體皆有,請檢查預測年齡來源")
    return out


def age_error_oof(table, *, feature=FEATURE, meta_clf="lr", n_reps=100, n_folds=10,
                  fold_kind=DEFAULT_FOLD_KIND, seed0=0, return_reps=False):
    """repeated group 10-fold OOF、跨 rep 平均 → 每視次一個 y_score。

    每個 rep r:依 fold_kind 切 n_folds 折(受試者分組;stratified_group 另依 class 分層,
    seed=seed0+r),meta_oof 逐折在訓練視次上 fit LR、預測留出視次 → 該 rep 的每視次 OOF 機率;
    N 個 rep 的機率取平均(= 老師說的 average the OOF probabilities for each visit)。seed 慣例
    同 embedding 主結果(seed 0 為確定性折 shuffle=False,seed≥1 為 shuffle+random_state=seed)。

    Args:
        table: age_error_table 的產物(ID / y_true / <feature>)。
        feature: 要當單一特徵的欄名(預設 age_error)。
        meta_clf: 分類器,預設 'lr'(單變量 LR);理論上可換,但本流程設計給 LR。
        n_reps / n_folds: 重複次數與折數(預設 100 × 10,同 embedding score_avg)。
        fold_kind: 折分方式('stratified_group' | 'group',見 src/common/folds.py FOLD_KINDS)。
            預設 DEFAULT_FOLD_KIND=stratified_group,對齊 meta 主結果;實測換成 group 對 age_error
            的指標影響 ≤0.001(1610 人 10 折下兩者近乎等價)。
        seed0: 第一個 rep 的 fold seed(reps 用 seed0..seed0+n_reps-1)。
        return_reps: 為真時另回 (n_visit × n_reps) 的逐 rep 機率矩陣。

    Returns:
        oof(ID, y_true, y_score, fold=-1);y_score = 跨 rep 平均的每視次機率。
        fold 恆為 -1(分數已跨折分平均,fold 對 forward 評估無意義,只為 evaluate 的 schema)。
    """
    X = table[[feature]].to_numpy(dtype=float)
    y = table["y_true"].to_numpy(dtype=int)
    groups = subject_groups(table["ID"].to_numpy())
    reps = np.full((len(table), n_reps), np.nan)
    for r in range(n_reps):
        fold = fold_labels(y, groups, kind=fold_kind, n_splits=n_folds,
                           seed=seed0 + r)
        reps[:, r] = meta_oof(X, y, fold, meta_clf=meta_clf)
    y_score = np.nanmean(reps, axis=1)
    oof = pd.DataFrame({"ID": table["ID"].to_numpy(), "y_true": y,
                        "y_score": y_score, "fold": -1})
    return (oof, reps) if return_reps else oof
