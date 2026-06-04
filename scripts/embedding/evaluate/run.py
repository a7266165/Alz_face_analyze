"""
scripts/embedding/evaluate/run.py
Embedding 下游 evaluation 的 **單一 cell consumer** —— 仿 classification/run(producer)。

producer(run_cell)算出 out_dir、訓練、把 OOF 落地成 oof_scores.csv;本檔(eval_cell)用
**同一組 path helper / dir_seg 規則**反推那格的 oof_scores.csv 路徑,丟給
src.common.evaluate.evaluate 算出同資料夾的 metrics.csv。純讀 OOF + 算指標,不訓練、不碰特徵。

路徑對稱(producer 寫哪、consumer 就讀哪):
  forward → <out_dir>/oof_scores.csv                  → 同層 metrics.csv
  reverse → <out_dir>/<priority>/oof_scores.csv (×N)  → 各自同層 metrics.csv

評估的內部軸(matched_units × matching_priorities × eval_units × contrasts)由 evaluate
自行全掃,這裡只指定「哪一格」(= 與 run_cell 相同的 cell 身份旗標)。

用法:
    python scripts/embedding/evaluate/run.py \\
        --p-visit p_first --p-score p_cdr05 --hc-visit hc_all --hc-score hc_cdrall_or_mmseall \\
        --bg-mode background --embedding arcface --variant absolute_difference \\
        --model centroid_dist --direction forward
"""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    embedding_classification_path,
    EMBEDDING_CLASSIFICATION_REFACTOR_DIR,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.embedding.classification import ALL_METHODS, CLASSIFIERS, DIM_REDUCERS, reducer_label
from src.common.evaluate import evaluate
from scripts.embedding.classification.run import (
    _clf_param_label, MATCH_STRATEGIES, LR_C_GRID, XGB_GRID,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cell_oof_paths(cohort, bg_mode, embedding, variant, photo_mode, reducer,
                   model, direction, *, pca_components=None,
                   drop_corr_threshold=None, lr_C=1.0, xgb_params=None, root=None):
    """這格預期的 oof_scores.csv 路徑(forward 1 個、reverse 每 priority 一個)。

    與 run_cell 用同一組 path helper / dir_seg 規則(forward l2_norm 無 fwd/rev 段、
    其餘 fwd;reverse 一律 rev,且每 priority 一個子夾),確保 producer 寫哪、consumer 讀哪。
    """
    root = root or EMBEDDING_CLASSIFICATION_REFACTOR_DIR
    is_classify = model in CLASSIFIERS
    rlabel = (reducer_label(reducer, pca_components=pca_components,
                            drop_corr_threshold=drop_corr_threshold)
              if is_classify else "no_drop")
    clf_param = _clf_param_label(model, lr_C, xgb_params) if is_classify else None
    dir_seg = ("rev" if direction == "reverse"
               else (None if model == "l2_norm" else "fwd"))
    out_dir = embedding_classification_path(
        *cohort, bg_mode, embedding, variant, photo_mode, rlabel,
        clf=model, clf_param=clf_param, direction=dir_seg, root=root)
    if direction == "reverse":
        return [out_dir / mp / "oof_scores.csv" for mp in MATCH_STRATEGIES]
    return [out_dir / "oof_scores.csv"]


def eval_cell(cohort, bg_mode, embedding, variant, photo_mode, reducer,
              model, direction, *, pca_components=None, drop_corr_threshold=None,
              lr_C=1.0, xgb_params=None, output_root=None, write=True, seed=42):
    """評估單一 cell:對每個存在的 oof_scores.csv 算出同層 metrics.csv。

    回寫出的 metrics.csv 路徑 list;找不到 oof 的略過(producer 還沒產這格)。
    評估軸由 evaluate 全掃,reverse 的 matching_priority 由各 oof 的父目錄推得。
    """
    paths = cell_oof_paths(
        cohort, bg_mode, embedding, variant, photo_mode, reducer, model, direction,
        pca_components=pca_components, drop_corr_threshold=drop_corr_threshold,
        lr_C=lr_C, xgb_params=xgb_params, root=output_root)
    written = []
    for p in paths:
        if not p.exists():
            logger.warning(f"  [skip] no oof: {p}")
            continue
        evaluate(p, cohort, direction=direction, write=write, seed=seed)
        written.append(p.parent / "metrics.csv")
        logger.info(f"  [{direction}] metrics -> {p.parent / 'metrics.csv'}")
    return written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Cohort(4-token;順序同 cohort_list,與 run_classification 對齊)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])

    # cell 身份(決定讀哪個 oof,旗標與 run_classification 一致)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--bg-mode", choices=["background", "no_background"], default="no_background")
    ap.add_argument("--variant", default="original")
    ap.add_argument("--photo-mode", choices=["mean", "all"], default="mean")
    ap.add_argument("--model", required=True, choices=list(ALL_METHODS))
    ap.add_argument("--reducer", choices=list(DIM_REDUCERS), default="no_drop")
    ap.add_argument("--pca-components", type=float, default=None)
    ap.add_argument("--drop-corr-threshold", type=float, default=None)
    ap.add_argument("--lr-C", type=float, default=1.0)
    ap.add_argument("--grid-search", action="store_true",
                    help="評估整個 hyperparameter grid(對齊 producer 的 --grid-search)")
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")

    # 評估設定
    ap.add_argument("--no-write", dest="write", action="store_false",
                    help="只算不落檔(預設會寫同層 metrics.csv)")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="覆寫輸出根(預設 EMBEDDING_CLASSIFICATION_REFACTOR_DIR)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pca = args.pca_components
    if pca is not None and pca >= 1:
        pca = int(pca)
    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)

    # grid search:對齊 producer 子層,逐 param point 各評一次
    is_classify = args.model in CLASSIFIERS
    if args.grid_search:
        if not is_classify:
            ap.error(f"--grid-search 只支援 classifier(logistic/xgb),不適用 {args.model}")
        if args.model == "logistic":
            grid = [dict(lr_C=c, xgb_params=None) for c in LR_C_GRID]
        else:  # xgb
            grid = [dict(lr_C=1.0, xgb_params=p) for p in XGB_GRID]
    else:
        grid = [dict(lr_C=args.lr_C, xgb_params=None)]

    logger.info(f"cohort={cohort}  cell={args.embedding}/{args.bg_mode}/{args.variant}/"
                f"{args.photo_mode}/{args.model}/{args.direction}  ({len(grid)} param point(s))")
    total = 0
    for g in grid:
        total += len(eval_cell(
            cohort, args.bg_mode, args.embedding, args.variant, args.photo_mode,
            args.reducer, args.model, args.direction,
            pca_components=pca, drop_corr_threshold=args.drop_corr_threshold,
            lr_C=g["lr_C"], xgb_params=g["xgb_params"],
            output_root=args.output_root, write=args.write, seed=args.seed))
    logger.info(f"done. wrote {total} metrics.csv")


if __name__ == "__main__":
    main()
