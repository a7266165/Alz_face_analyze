"""跑 age_error 單獨分類器並落地(對齊 meta 主結果協定;見 src/age/classify.py)。

流程:單變量 LR × repeated StratifiedGroupKFold(10-fold × N reps)→ 跨 rep 平均每視次機率
→ subject 層 evaluate(0.5 threshold)。刻意只用 age_error 一個特徵、不接 TabPFN。

輸出 workspace/age/<selection>/classify/<visit_dir>/<cdr_mmse_dir>/keep_nan/age_error_only/<clf>/:
  oof_scores.csv   跨 rep 平均後每視次的 OOF 機率(ID, y_true, y_score, fold=-1)
  all_metrics.csv  evaluate 全 eval combos(eval_by_subject)
  ppt_cells.csv    PPT 三張表要填的格(contrast × {full, 1:1 matched} × 8 指標)

PPT 對應:full = domain 'all';1:1 matched = domain '1by1' / matched_unit 'visit' /
matching_priority 'priority_acs'(與 embedding_only 等欄同一慣例,已用其數字反查確認)。

用法:
  python scripts/age/classify.py                 # headline cohort、100 reps、lr
  python scripts/age/classify.py --n-reps 100 --clf lr
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401  # sys.path 副作用

from src.age.classify import DEFAULT_FOLD_KIND, age_error_oof, age_error_table
from src.common.evaluate import evaluate
from src.common.folds import FOLD_KINDS
from src.config import AGE_DIR, cohort_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# headline cohort（同 meta score_avg 主結果的母體）
DEFAULT_COHORT = ("p_first", "p_cdrall", "hc_all", "hc_cdrall_or_mmseall")
CONTRASTS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
# PPT 「1:1 matched」欄的配對慣例（用 embedding_only 的落地數字反查得到）
PPT_MATCHED_UNIT = "visit"
PPT_MATCHING_PRIORITY = "priority_acs"
METRIC_COLS = ["auc", "balacc", "mcc", "f1", "sens", "spec", "ppv", "npv"]


def _ppt_cells(metrics: pd.DataFrame) -> pd.DataFrame:
    """從 evaluate 全表挑出 PPT 要填的格：每 contrast 的 full(all) 與 1:1(visit/priority_acs)。"""
    rows = []
    for con in CONTRASTS:
        full = metrics[(metrics.contrast == con) & (metrics.domain == "all")]
        m1 = metrics[(metrics.contrast == con) & (metrics.domain == "1by1")
                     & (metrics.matched_unit == PPT_MATCHED_UNIT)
                     & (metrics.matching_priority == PPT_MATCHING_PRIORITY)]
        for col_label, sub in (("full", full), ("1:1 matched", m1)):
            if not len(sub):
                logger.warning("缺 %s / %s 的指標列", con, col_label)
                continue
            r = sub.iloc[0]
            rows.append({"contrast": con, "column": col_label, "n": int(r.n),
                         **{c: float(r[c]) for c in METRIC_COLS}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="age_error 單獨分類器（對齊 meta 主結果協定）")
    ap.add_argument("--cohort", nargs=4, metavar=("P_VISIT", "P_SCORE", "HC_VISIT", "HC_SCORE"),
                    default=list(DEFAULT_COHORT))
    ap.add_argument("--clf", default="lr", help="分類器（預設 lr = 單變量 LR）")
    ap.add_argument("--fold-kind", default=DEFAULT_FOLD_KIND, choices=FOLD_KINDS,
                    help=f"折分方式（預設 {DEFAULT_FOLD_KIND}，對齊 meta 主結果）")
    ap.add_argument("--n-reps", type=int, default=100)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=None, help="覆寫輸出目錄（預設落 workspace/age/<selection>/classify/...）")
    args = ap.parse_args()

    cohort = tuple(args.cohort)
    visit_dir, cdr_mmse_dir = cohort_dirs(*cohort)
    out_dir = (Path(args.out) if args.out else
               AGE_DIR / "classify" / visit_dir / cdr_mmse_dir / "keep_nan"
               / f"folds_{args.fold_kind}" / "age_error_only" / args.clf)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("cohort=%s → %s/%s", cohort, visit_dir, cdr_mmse_dir)
    table = age_error_table(cohort)
    logger.info("table: %d 視次（AD=%d, HC=%d）", len(table),
                int((table.y_true == 1).sum()), int((table.y_true == 0).sum()))

    logger.info("age_error OOF: %s × %d reps × %d-fold %s ...",
                args.clf, args.n_reps, args.n_folds, args.fold_kind)
    oof = age_error_oof(table, meta_clf=args.clf, n_reps=args.n_reps,
                        n_folds=args.n_folds, fold_kind=args.fold_kind, seed0=args.seed0)
    oof_path = out_dir / "oof_scores.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8")

    metrics = evaluate(oof_path, cohort, direction="forward",
                       eval_units=["eval_by_subject"], write=False)
    metrics.to_csv(out_dir / "all_metrics.csv", index=False, encoding="utf-8")

    cells = _ppt_cells(metrics)
    cells.to_csv(out_dir / "ppt_cells.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\n=== age_error_only / {args.clf} → PPT cells ===")
    for con in CONTRASTS:
        print(f"\n[{con}]")
        sub = cells[cells.contrast == con]
        for _, r in sub.iterrows():
            print(f"  {r.column:12} n={int(r.n):5}  "
                  + "  ".join(f"{c.upper()}={r[c]:.4f}" for c in METRIC_COLS))
    print(f"\n落地: {out_dir}")


if __name__ == "__main__":
    main()
