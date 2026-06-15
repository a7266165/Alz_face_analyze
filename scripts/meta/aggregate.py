"""把 meta 各 cell 的 metrics.csv 彙整成 cohort 層一張 all_metrics.csv(仿 embedding aggregate)。

走訪 META_ANALYSIS_DIR/<cohort> 下所有 tabpfn_v3/metrics.csv(由 run.py 落地、已含 cell 身份欄),
直接 concat、依身份 + 評估軸排序 → <cohort>/all_metrics.csv。下游 plot 純讀此檔。

用法:
    python scripts/meta/aggregate.py --p-visit p_first --p-score p_cdrall \\
        --hc-visit hc_all --hc-score hc_cdrall_or_mmseall
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

import pandas as pd

from src.config import (
    META_ANALYSIS_DIR, cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
)
from src.meta import META_CLASSIFIERS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 排序鍵(存在才用):cell 身份 → 評估軸
_SORT_KEYS = ["feature_set", "variant", "base_clf", "clf_param", "meta_clf",
              "contrast", "eval_unit", "matched_unit", "matching_priority", "domain"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default="p_first")
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default="p_cdrall")
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default="hc_all")
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                    default="hc_cdrall_or_mmseall")
    ap.add_argument("--out", type=Path, default=None,
                    help="輸出 csv(預設 <cohort>/all_metrics.csv)")
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    base = META_ANALYSIS_DIR / cohort_path(*cohort)
    paths = sorted(p for p in base.rglob("metrics.csv") if p.parent.name in META_CLASSIFIERS)
    if not paths:
        logger.warning(f"在 {base} 下找不到任何 <meta_clf>/metrics.csv;請先跑 scripts/meta/run.py。")
        return

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    sort_keys = [c for c in _SORT_KEYS if c in df.columns]
    df = df.sort_values(sort_keys, na_position="last").reset_index(drop=True)

    out = args.out or (base / "all_metrics.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"collected {len(paths)} metrics.csv -> {len(df)} rows x {df.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
