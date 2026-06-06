"""
依序掃過所有cell組合
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    EMBEDDING_CLASSIFICATION_REFACTOR_DIR,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.embedding.classification import ALL_METHODS
from scripts.embedding.evaluate.run import eval_cell
from scripts.embedding.classification.sweep import (
    iter_cells, oof_paths_for, _is_known_crash, _label,
    EMBEDDINGS, VARIANTS, BG_MODES, PHOTO_MODES, DIRECTIONS,
)

logger = logging.getLogger("evaluate_sweep")


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
        oof_paths = oof_paths_for(c, root)
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
