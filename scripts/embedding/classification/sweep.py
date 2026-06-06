"""Embedding classification 的 **in-process sweep** —— 窮舉(或縮小範圍)跑多組 cell。

設計:不重寫任何 cell 邏輯,直接 ``import classification.run.run_cell`` 在同一個 python
行程裡迴圈呼叫(in-process)。每格包 try/except(一格爆不拖垮整批)、skip-if-exists
(已有 oof_scores.csv 就跳過 → 可續跑)。「合法組合矩陣」與 orchestration 是這支的職責;
classification.run 維持「單一 cell」producer 不動。

軸(每軸可用旗標給一或多值,預設為合法全集):
  cohort(單一,4 旗標)× bg_mode × embedding × variant × photo_mode × model × reducer
  × direction;classifier(logistic/xgb)再 × hyperparameter grid(預設開,--no-grid-search 關)。
  scorer(l2/centroid/lda)自動只配 reducer=no_drop、無 grid。
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from scripts.embedding.classification.run import (
    run_cell, cell_oof_paths, param_grid, _clf_param_label,
)
from src.embedding.classification import CLASSIFIERS, ALL_METHODS
from src.config import (
    EMBEDDING_CLASSIFICATION_REFACTOR_DIR,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)

logger = logging.getLogger("classification_sweep")

# 各軸合法全集(sweep 預設)
EMBEDDINGS = ["arcface", "topofr", "dlib", "vggface"]
VARIANTS = ["original", "differences", "absolute_differences",
            "relative_differences", "absolute_relative_differences"]
BG_MODES = ["background", "no_background"]
PHOTO_MODES = ["mean", "all"]
DIRECTIONS = ["forward", "reverse"]

# 已知會硬 segfault 的 (model, emb) —— sweep 直接跳過(try/except 攔不到 segfault)。
# lda_projection × vggface:vggface 4096 維、rank 僅 ~1989(嚴重 rank-deficient + 40 全零欄),
# sklearn LDA 的 LAPACK 特徵分解在此奇異高維散布矩陣上崩潰。其餘 emb(≤512 維、滿秩)安全。
KNOWN_CRASH = {("lda_projection", "vggface")}


def _is_known_crash(cell):
    return (cell["model"], cell["emb"]) in KNOWN_CRASH


def oof_paths_for(cell, root):
    """cell dict(iter_cells 產出)→ 這格的 oof_scores.csv 路徑 list(skip-if-exists 用)。
    委派給 producer 的 cell_oof_paths,確保「檢查的格 == 產的格」。"""
    return cell_oof_paths(
        cell["cohort"], cell["bg"], cell["emb"], cell["variant"], cell["photo"],
        cell["reducer"], cell["model"], cell["direction"],
        lr_C=cell["lr_C"], xgb_params=cell["xgb_params"], root=root)


def iter_cells(args):
    """產生所有合法 cell dict。scorer 只配 reducer=no_drop;classifier 才展開 grid。"""
    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    for bg in args.bg_mode:
        for emb in args.embedding:
            for variant in args.variant:
                for photo in args.photo_mode:
                    for model in args.model:
                        reducers = (args.reducer if model in CLASSIFIERS
                                    else ["no_drop"])
                        for reducer in reducers:
                            for direction in args.direction:
                                for lr_C, xgb_params in param_grid(model, args.grid_search):
                                    yield dict(
                                        cohort=cohort, bg=bg, emb=emb,
                                        variant=variant, photo=photo,
                                        reducer=reducer, model=model,
                                        direction=direction,
                                        lr_C=lr_C, xgb_params=xgb_params)


def _label(c):
    g = (_clf_param_label(c["model"], c["lr_C"], c["xgb_params"])
         if c["model"] in CLASSIFIERS else "-")
    return (f"{c['bg']}/{c['emb']}/{c['variant']}/{c['photo']}/"
            f"{c['model']}/{c['reducer']}/{g}/{c['direction']}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # cohort(單一)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])
    # 其餘軸(可多值)
    ap.add_argument("--bg-mode", nargs="+", choices=BG_MODES, default=BG_MODES)
    ap.add_argument("--embedding", nargs="+", default=EMBEDDINGS)
    ap.add_argument("--variant", nargs="+", default=VARIANTS)
    ap.add_argument("--photo-mode", nargs="+", choices=PHOTO_MODES, default=PHOTO_MODES)
    ap.add_argument("--model", nargs="+", choices=list(ALL_METHODS), default=list(ALL_METHODS))
    ap.add_argument("--reducer", nargs="+", default=["no_drop"],
                    help="目前 sweep 只支援 no_drop(pca/drop_corr 需 param 清單,未做)")
    ap.add_argument("--direction", nargs="+", choices=DIRECTIONS, default=DIRECTIONS)
    ap.add_argument("--no-grid-search", dest="grid_search", action="store_false",
                    help="關掉 hyperparameter grid(classifier 改用單一預設)")
    ap.add_argument("--overwrite", action="store_true", help="已存在也重跑")
    ap.add_argument("--dry-run", action="store_true", help="只列計畫與計數,不跑")
    ap.add_argument("--output-root", type=Path, default=None)
    args = ap.parse_args()

    if any(r != "no_drop" for r in args.reducer):
        ap.error("sweep 目前只支援 --reducer no_drop(pca/drop_corr 的 param sweep 未實作)")

    root = args.output_root or EMBEDDING_CLASSIFICATION_REFACTOR_DIR
    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    cells = list(iter_cells(args))
    logger.info(f"cohort={cohort}")
    logger.info(f"planned cells: {len(cells)}  (grid={'on' if args.grid_search else 'off'}, "
                f"dry_run={args.dry_run}, overwrite={args.overwrite})")
    n_excluded = sum(1 for c in cells if _is_known_crash(c))
    if n_excluded:
        logger.info(f"excluding {n_excluded} cells (known LAPACK segfault): {sorted(KNOWN_CRASH)}")

    ran = skipped = no_feat = failed = excluded = 0
    failures = []
    for i, c in enumerate(cells, 1):
        desc = _label(c)
        if _is_known_crash(c):  # 硬 segfault,直接跳過(不進 try/except —— 攔不到)
            excluded += 1
            continue
        if not args.overwrite and all(p.exists() for p in oof_paths_for(c, root)):
            skipped += 1
            continue
        if args.dry_run:
            logger.info(f"[{i}/{len(cells)}] WOULD RUN  {desc}")
            ran += 1
            continue
        try:
            res = run_cell(c["cohort"], c["bg"], c["emb"], c["variant"], c["photo"],
                           c["reducer"], c["model"], c["direction"],
                           lr_C=c["lr_C"], xgb_params=c["xgb_params"], output_root=root)
            if res is None:
                no_feat += 1
                logger.warning(f"[{i}/{len(cells)}] no features  {desc}")
            else:
                ran += 1
                logger.info(f"[{i}/{len(cells)}] done  {desc}")
        except Exception as e:  # 一格爆不拖垮整批
            failed += 1
            failures.append((desc, repr(e)))
            logger.warning(f"[{i}/{len(cells)}] FAILED  {desc}: {e}")

    logger.info("=" * 60)
    logger.info(f"planned={len(cells)} ran={ran} skipped(exists)={skipped} "
                f"excluded(known-crash)={excluded} no_features={no_feat} failed={failed}")
    for desc, err in failures:
        logger.info(f"  FAIL {desc}: {err}")


if __name__ == "__main__":
    main()
