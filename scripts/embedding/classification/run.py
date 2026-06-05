"""
Embedding 下游 classification producer(重構版)—— **只產 OOF 分數,不做評估**。

流程(每個 cell):
  build cohort(ID+label)→ load_feature_matrix → _build_estimator(model, reducer)
  → train(X, ids, y, build_estimator, ..., direction)  → oof
  → report(oof, out_dir, direction)               → 寫 oof_scores.csv
      forward:full cohort OOF → oof_scores.csv
      reverse:每 match_strategy 的 matched OOF + external ensemble → <ms>/oof_scores.csv

producer 只負責「組 cohort + 載特徵 + 組裝 estimator + 算 out_dir + 呼叫 train→report」。
**配對評估(年齡配對 × partition × 指標)是獨立的下游步驟**(吃 oof_scores.csv),不在這裡;
故本 producer 對 src/meta 零依賴,只用 common + embedding。輸出寫到
``workspace_refactor``(EMBEDDING_CLASSIFICATION_REFACTOR_DIR),**絕不與 legacy 混**。

正交軸(互不干涉):
  - cohort 4-token(``--p-visit/--p-score/--hc-visit/--hc-score``):決定族群,順序同 cohort_list。
  - ``--variant``:feature type(original / absolute_differences …),只決定載哪個 X。
  - ``--model``:logistic/xgb → reducer+classifier;l2/centroid/lda → scorer。routing key 在此。
  - ``--reducer``:只在 model ∈ CLASSIFIERS 時有意義;scorer 帶 reducer 旗標會被 ap.error 擋掉。

reverse 的 matched 訓練池由 ``match_by_age``(common)當場算,無需 matching cache。

用法:
    python scripts/embedding/classification/run.py \\
        --p-visit p_first --p-score p_cdr05 --hc-visit hc_all --hc-score hc_cdrall_or_mmseall \\
        --bg-mode background --embedding arcface --variant absolute_differences \\
        --model centroid_dist
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    embedding_classification_path,
    EMBEDDING_CLASSIFICATION_REFACTOR_DIR,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.common.cohort import cohort_list
from src.common.features import load_feature_matrix
from src.embedding.classification import (
    ALL_METHODS, CLASSIFIERS, DIM_REDUCERS, DEFAULT_XGB_PARAMS,
    build_reducer, build_classifier, build_scorer, reducer_label, train, report,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# reverse 訓練池的配對策略(forward 不用配對)
MATCH_STRATEGIES = ["no_priority", "priority_acs", "priority_nad"]

# grid search 範圍(對齊 overview 圖 / legacy run_fwd_rev 的 _DEFAULT_*_GRID)
LR_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
XGB_GRID = [
    {"n_estimators": ne, "max_depth": md, "learning_rate": lr}
    for ne in (200, 500, 1000)
    for md in (3, 6, 9)
    for lr in (0.05, 0.1, 0.2)
]


def _clf_param_label(model, lr_C, xgb_params):
    """classifier 的 hyperparameter 路徑子層標籤(scorer 無 → None)。
    logistic → C_<value>;xgb → ne_<n>_md_<d>_lr_<lr>(對齊 legacy 命名)。"""
    if model == "logistic":
        return f"C_{lr_C}"
    if model == "xgb":
        p = xgb_params or DEFAULT_XGB_PARAMS
        return f"ne_{p['n_estimators']}_md_{p['max_depth']}_lr_{p['learning_rate']}"
    return None


def build_ad_full_cohort(cohort):
    """完整 P-vs-HC 訓練 cohort(label:P=1,其餘=0)。AD 三 partition 共用此 OOF。
    cohort = (p_visit, p_score, hc_visit, hc_score) 4-token,順序同 cohort_list。"""
    full = cohort_list(*cohort)
    full["label"] = (full["Group"] == "P").astype(int)
    return full


def _eparams(reducer, pca_components, drop_corr_threshold, lr_C, xgb_params):
    return dict(reducer_type=reducer, pca_components=pca_components,
                drop_corr_threshold=drop_corr_threshold,
                lr_C=lr_C, xgb_params=xgb_params)


def _build_estimator(model, ep):
    """唯一 router(_assemble 收進來,key 在 model)→ (estimator, score_method, needs_cv)。

    model ∉ CLASSIFIERS → scorer(l2/centroid/lda;暫固定 no_drop,併入 reducer 之事先保留)。
    model ∈ CLASSIFIERS → reducer + classifier 串成單一 Pipeline(scaler?/reducer?/clf 同次
                          fit 防 leakage);feature type 由 variant 決定、與此無關。
    """
    if model not in CLASSIFIERS:
        return build_scorer(model)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    rd = build_reducer(ep["reducer_type"], pca_components=ep["pca_components"],
                       drop_corr_threshold=ep["drop_corr_threshold"])
    clf, needs_scaler = build_classifier(model, lr_C=ep["lr_C"],
                                         xgb_params=ep["xgb_params"])
    steps = []
    if needs_scaler:
        steps.append(("scaler", StandardScaler()))
    if rd != "passthrough":
        steps.append(("reducer", rd))
    steps.append(("classifier", clf))
    return Pipeline(steps), "predict_proba", True


def run_cell(cohort, bg_mode, embedding, variant, photo_mode, reducer,
             model, direction, *, pca_components=None, drop_corr_threshold=None,
             lr_C=1.0, xgb_params=None, output_root=None, match_strategies=None):
    match_strategies = match_strategies or MATCH_STRATEGIES
    root = output_root or EMBEDDING_CLASSIFICATION_REFACTOR_DIR
    is_classify = model in CLASSIFIERS
    rlabel = reducer_label(reducer, pca_components=pca_components,
                           drop_corr_threshold=drop_corr_threshold) \
        if is_classify else "no_drop"
    ep = _eparams(reducer, pca_components, drop_corr_threshold, lr_C, xgb_params)

    full = build_ad_full_cohort(cohort)
    label_map = dict(zip(full["ID"], full["label"]))
    X_full, ids_full = load_feature_matrix(
        full["ID"].tolist(), embedding, variant, bg_mode, photo_mode)
    if len(X_full) == 0:
        logger.warning(f"  [skip] no features: {embedding}/{bg_mode}/{variant}")
        return None
    y_full = np.array([label_map[i] for i in ids_full], dtype=int)

    # estimator factory(每折要全新 estimator,故傳 0-arg factory)
    _, score_method, needs_cv = _build_estimator(model, ep)
    build_estimator = lambda: _build_estimator(model, ep)[0]

    # 路徑:forward l2 無 fwd/rev 段、其餘 fwd;reverse 一律 rev。
    dir_seg = ("rev" if direction == "reverse"
               else (None if model == "l2_norm" else "fwd"))
    # classifier 多一層 hyperparameter 子層(C_<value> / ne_md_lr);scorer 為 None。
    clf_param = _clf_param_label(model, lr_C, xgb_params) if is_classify else None
    out_dir = embedding_classification_path(
        *cohort, bg_mode, embedding, variant, photo_mode, rlabel,
        clf=model, clf_param=clf_param, direction=dir_seg, root=root)

    # train → 只產 OOF;report → 只把 OOF 落地成 oof_scores.csv(評估是獨立下游步驟)。
    oof = train(X_full, ids_full, y_full, build_estimator, score_method, needs_cv, direction,
                cohort=cohort, match_strategies=match_strategies)
    paths = report(oof, out_dir, direction)
    logger.info(f"  [{direction}] wrote {len(paths)} oof_scores.csv")
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Cohort(4-token,正交軸;順序同 cohort_list)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS),
                    default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS),
                    default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS),
                    default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                    default=DEFAULT_COHORT_TOKENS[3])

    # 資料選擇(decide which X 被載入;與 model 正交)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--bg-mode", choices=["background", "no_background"],
                    default="no_background")
    ap.add_argument("--variant", default="original",
                    help="feature type:original / absolute_differences / …。"
                         "只決定載哪個特徵,與 --model 正交")
    ap.add_argument("--photo-mode", choices=["mean", "all"], default="mean")

    # 模型選擇(decide which 建構路徑;與 --variant 正交)
    ap.add_argument("--model", required=True, choices=list(ALL_METHODS),
                    help="logistic/xgb → reducer+classifier;"
                         "l2_norm/centroid_dist/lda_projection → scorer")
    ap.add_argument("--reducer", choices=list(DIM_REDUCERS), default="no_drop",
                    help="只在 --model ∈ {logistic,xgb} 時有意義;scorer 帶此旗標會報錯")
    ap.add_argument("--pca-components", type=float, default=None,
                    help="int(>=1)→GPU PCA;float(<1)→sklearn variance-ratio PCA")
    ap.add_argument("--drop-corr-threshold", type=float, default=None)
    ap.add_argument("--lr-C", type=float, default=1.0)
    ap.add_argument("--grid-search", action="store_true",
                    help="掃 hyperparameter grid:logistic→C∈{1e-3..1e2};"
                         "xgb→{200,500,1k}×{3,6,9}×{.05,.1,.2}。只支援 classifier")

    # 跑法設定(partition / threshold 等是「評估」旋鈕,屬獨立下游步驟,不在此 producer)
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="覆寫輸出根(預設 EMBEDDING_CLASSIFICATION_REFACTOR_DIR)")
    args = ap.parse_args()

    # 正交軸一致性硬擋:scorer 不可帶 reducer 旗標
    is_classify = args.model in CLASSIFIERS
    if not is_classify and (args.reducer != "no_drop"
                            or args.pca_components is not None
                            or args.drop_corr_threshold is not None):
        ap.error(f"--model={args.model} 是 scorer,不可帶 reducer 旗標"
                 f"(--reducer/--pca-components/--drop-corr-threshold)")
    if is_classify and args.reducer == "pca" and args.pca_components is None:
        ap.error("--reducer pca 需要 --pca-components")
    if is_classify and args.reducer == "drop_corr" and args.drop_corr_threshold is None:
        ap.error("--reducer drop_corr 需要 --drop-corr-threshold")

    # int 形式的 pca_components(50.0 → 50)交給 reducer 判斷;float(<1)保留
    pca = args.pca_components
    if pca is not None and pca >= 1:
        pca = int(pca)

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    logger.info(f"cohort={cohort}")

    # grid search:只掃 classifier 的 hyperparameter;否則單一配置。
    if args.grid_search:
        if not is_classify:
            ap.error(f"--grid-search 只支援 classifier(logistic/xgb),不適用 {args.model}")
        if args.model == "logistic":
            grid = [dict(lr_C=c, xgb_params=None) for c in LR_C_GRID]
        else:  # xgb
            grid = [dict(lr_C=1.0, xgb_params=p) for p in XGB_GRID]
    else:
        grid = [dict(lr_C=args.lr_C, xgb_params=None)]

    logger.info(f"cell: {args.embedding}/{args.bg_mode}/{args.variant}/"
                f"{args.photo_mode}/{args.model}"
                f"{('/' + args.reducer) if is_classify else ''}/{args.direction}"
                f"  ({len(grid)} param point(s))")
    for i, g in enumerate(grid, 1):
        if len(grid) > 1:
            logger.info(f"  [{i}/{len(grid)}] "
                        f"{_clf_param_label(args.model, g['lr_C'], g['xgb_params'])}")
        run_cell(cohort, args.bg_mode, args.embedding, args.variant,
                 args.photo_mode, args.reducer, args.model, args.direction,
                 pca_components=pca, drop_corr_threshold=args.drop_corr_threshold,
                 lr_C=g["lr_C"], xgb_params=g["xgb_params"],
                 output_root=args.output_root)
    logger.info("done.")


if __name__ == "__main__":
    main()
