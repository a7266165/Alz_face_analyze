"""計算各cell指標"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.embedding.classification import ALL_METHODS, CLASSIFIERS, DIM_REDUCERS
from src.common.evaluate import evaluate
from scripts.embedding.classification.run import cell_oof_paths, param_grid

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
    if args.grid_search and not is_classify:
        ap.error(f"--grid-search 只支援 classifier(logistic/xgb),不適用 {args.model}")
    grid = param_grid(args.model, args.grid_search, lr_C=args.lr_C)

    logger.info(f"cohort={cohort}  cell={args.embedding}/{args.bg_mode}/{args.variant}/"
                f"{args.photo_mode}/{args.model}/{args.direction}  ({len(grid)} param point(s))")
    total = 0
    for lr_C, xgb_params in grid:
        total += len(eval_cell(
            cohort, args.bg_mode, args.embedding, args.variant, args.photo_mode,
            args.reducer, args.model, args.direction,
            pca_components=pca, drop_corr_threshold=args.drop_corr_threshold,
            lr_C=lr_C, xgb_params=xgb_params,
            output_root=args.output_root, write=args.write, seed=args.seed))
    logger.info(f"done. wrote {total} metrics.csv")


if __name__ == "__main__":
    main()
