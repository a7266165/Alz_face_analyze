"""
scripts/embedding/evaluate/sweep.py
Embedding 下游 evaluation 的 **sweep orchestrator** —— 仿 classification/sweep(producer sweep)。

對 classification/sweep 產出的每一格(復用同一套 ``iter_cells``),呼叫 evaluate/run.eval_cell 算 metrics.csv。
只把「跑一格」從 run_cell 換成 eval_cell、skip-if-exists 從 oof_scores.csv 換成 metrics.csv;
合法組合矩陣(iter_cells)與 known-crash 名單直接復用 classification/sweep,確保「評的格 == 產的格」。

每格三種結局:
  - skip   :metrics.csv 已存在(可續跑;--overwrite 強制重算)
  - no_oof :producer 還沒產 oof_scores.csv(跳過,不算失敗)
  - ran    :算出 metrics.csv;一格爆 try/except 攔下不拖垮整批(failed)

用法:
    # 評估預設 cohort 的全集
    python scripts/embedding/evaluate/sweep.py
    # 縮小 + 先看計畫不算
    python scripts/embedding/evaluate/sweep.py --embedding arcface --variant original --dry-run
"""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_CLASSIFICATION_REFACTOR_DIR,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.embedding.classification import ALL_METHODS
from scripts.embedding.evaluate.run import eval_cell, cell_oof_paths
from scripts.embedding.classification.sweep import (
    iter_cells, _is_known_crash, _label,
    EMBEDDINGS, VARIANTS, BG_MODES, PHOTO_MODES, DIRECTIONS,
)

logger = logging.getLogger("evaluate_sweep")


def _cell_oof_paths(c, root):
    """cell dict(iter_cells 產出)→ 這格的 oof_scores.csv 路徑 list。"""
    return cell_oof_paths(
        c["cohort"], c["bg"], c["emb"], c["variant"], c["photo"],
        c["reducer"], c["model"], c["direction"],
        lr_C=c["lr_C"], xgb_params=c["xgb_params"], root=root)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # cohort(單一)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])
    # 其餘軸(可多值)—— 與 classification_sweep 對齊,決定要評估哪些已產出的格
    ap.add_argument("--bg-mode", nargs="+", choices=BG_MODES, default=BG_MODES)
    ap.add_argument("--embedding", nargs="+", default=EMBEDDINGS)
    ap.add_argument("--variant", nargs="+", default=VARIANTS)
    ap.add_argument("--photo-mode", nargs="+", choices=PHOTO_MODES, default=PHOTO_MODES)
    ap.add_argument("--model", nargs="+", choices=list(ALL_METHODS), default=list(ALL_METHODS))
    ap.add_argument("--reducer", nargs="+", default=["no_drop"],
                    help="目前 sweep 只支援 no_drop(對齊 classification_sweep)")
    ap.add_argument("--direction", nargs="+", choices=DIRECTIONS, default=DIRECTIONS)
    ap.add_argument("--no-grid-search", dest="grid_search", action="store_false",
                    help="關掉 hyperparameter grid(對齊 producer 的同名旗標)")
    ap.add_argument("--overwrite", action="store_true", help="metrics.csv 已存在也重算")
    ap.add_argument("--dry-run", action="store_true", help="只列計畫與計數,不算")
    ap.add_argument("--no-write", dest="write", action="store_false")
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if any(r != "no_drop" for r in args.reducer):
        ap.error("sweep 目前只支援 --reducer no_drop(對齊 classification_sweep)")

    root = args.output_root or EMBEDDING_CLASSIFICATION_REFACTOR_DIR
    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    cells = list(iter_cells(args))
    logger.info(f"cohort={cohort}")
    logger.info(f"planned cells: {len(cells)}  (grid={'on' if args.grid_search else 'off'}, "
                f"dry_run={args.dry_run}, overwrite={args.overwrite})")

    ran = skipped = no_oof = failed = excluded = 0
    failures = []
    for i, c in enumerate(cells, 1):
        desc = _label(c)
        if _is_known_crash(c):  # producer 硬 segfault → 從沒產 oof
            excluded += 1
            continue
        oof_paths = _cell_oof_paths(c, root)
        metrics_paths = [p.parent / "metrics.csv" for p in oof_paths]
        if not args.overwrite and all(m.exists() for m in metrics_paths):
            skipped += 1
            continue
        if not any(p.exists() for p in oof_paths):  # producer 還沒產這格
            no_oof += 1
            continue
        if args.dry_run:
            logger.info(f"[{i}/{len(cells)}] WOULD EVAL  {desc}")
            ran += 1
            continue
        try:
            written = eval_cell(
                c["cohort"], c["bg"], c["emb"], c["variant"], c["photo"],
                c["reducer"], c["model"], c["direction"],
                lr_C=c["lr_C"], xgb_params=c["xgb_params"],
                output_root=root, write=args.write, seed=args.seed)
            if written:
                ran += 1
                logger.info(f"[{i}/{len(cells)}] done ({len(written)})  {desc}")
            else:
                no_oof += 1
        except Exception as e:  # 一格爆不拖垮整批
            failed += 1
            failures.append((desc, repr(e)))
            logger.warning(f"[{i}/{len(cells)}] FAILED  {desc}: {e}")

    logger.info("=" * 60)
    logger.info(f"planned={len(cells)} ran={ran} skipped(metrics exist)={skipped} "
                f"excluded(known-crash)={excluded} no_oof={no_oof} failed={failed}")
    for desc, err in failures:
        logger.info(f"  FAIL {desc}: {err}")


if __name__ == "__main__":
    main()
