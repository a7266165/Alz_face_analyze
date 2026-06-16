"""把 meta 各 cell 的 metrics.csv 彙整成 cohort 層兩張表(仿 embedding aggregate)。

走訪 META_ANALYSIS_DIR/<cohort> 下所有 <meta_clf>/metrics.csv(由 run.py 落地、已含 cell 身份欄),
依路徑的 seed_<N> 段標上 `seed` → 產兩檔:
  - all_metrics.csv      = seed_0 子集(單一 run 基準;下游 c_curve/confusion/bar 不開 --reps 讀此檔)。
  - all_metrics_reps.csv = 依 cell 身份鍵跨 seed groupby 的 mean/std + 95% CI(repeated-CV;bar --reps 讀此檔)。

用法:
    python scripts/meta/aggregate.py --p-visit p_first --p-score p_cdrall \\
        --hc-visit hc_all --hc-score hc_cdrall_or_mmseall
"""
import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

import numpy as np
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

# cross-seed 統計:cell 身份 + 評估軸(groupby 鍵)/ 要算 mean·std·CI 的指標
_IDENT_KEYS = ["p_visit", "p_score", "hc_visit", "hc_score", "bg", "emb", "photo",
               "reducer", "feature_set", "variant", "base_clf", "clf_param", "meta_clf",
               "direction", "contrast", "eval_unit", "matched_unit", "matching_priority", "domain"]
_REP_METRICS = ["auc", "balacc", "mcc", "sens", "spec", "f1", "n"]
_SEED_RE = re.compile(r"seed_(\d+)")


def _seed_of(path):
    """由 cell 路徑解析 seed_<N> 段 → int(無則 0)。"""
    m = next((_SEED_RE.fullmatch(part) for part in path.parts if _SEED_RE.fullmatch(part)), None)
    return int(m.group(1)) if m else 0


def _reps_summary(df):
    """跨 seed groupby cell 身份 → 各指標 mean/std + 95% CI(mean ± 1.96·std/√n_rep)+ n_rep。"""
    keys = [k for k in _IDENT_KEYS if k in df.columns]
    metrics = [m for m in _REP_METRICS if m in df.columns]
    rows = []
    for vals, g in df.groupby(keys, dropna=False):
        rec = dict(zip(keys, vals if isinstance(vals, tuple) else (vals,)))
        n_rep = len(g)
        rec["n_rep"] = n_rep
        for m in metrics:
            x = g[m].to_numpy(dtype=float)
            mean = float(np.nanmean(x))
            std = float(np.nanstd(x, ddof=1)) if n_rep > 1 else 0.0
            half = 1.96 * std / np.sqrt(n_rep) if n_rep > 1 else 0.0
            rec[f"{m}_mean"], rec[f"{m}_std"] = mean, std
            rec[f"{m}_ci_low"], rec[f"{m}_ci_high"] = mean - half, mean + half
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default="p_first")
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default="p_cdrall")
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default="hc_all")
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                    default="hc_cdrall_or_mmseall")
    ap.add_argument("--case-mode", choices=["no_nan", "keep_nan"], default="no_nan",
                    help="meta 母體子樹:no_nan(complete-case)/ keep_nan(full cohort)")
    ap.add_argument("--out", type=Path, default=None,
                    help="輸出 csv(預設 <cohort>/<case_mode>/all_metrics.csv)")
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    base = META_ANALYSIS_DIR / cohort_path(*cohort) / args.case_mode
    paths = sorted(p for p in base.rglob("metrics.csv") if p.parent.name in META_CLASSIFIERS)
    if not paths:
        logger.warning(f"在 {base} 下找不到任何 <meta_clf>/metrics.csv;請先跑 scripts/meta/run.py。")
        return

    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["seed"] = _seed_of(p)          # 路徑為準(舊 metrics 無 seed 欄亦可)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    sort_keys = [c for c in _SORT_KEYS if c in df.columns]
    df = df.sort_values(sort_keys, na_position="last").reset_index(drop=True)

    base.mkdir(parents=True, exist_ok=True)
    # all_metrics.csv = seed_0 單一 run 基準(下游 c_curve/confusion/bar 不開 --reps 讀此檔)
    out = args.out or (base / "all_metrics.csv")
    df0 = df[df["seed"] == 0].reset_index(drop=True)
    df0.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"collected {len(paths)} metrics.csv -> seed_0 {len(df0)} rows x {df0.shape[1]} cols -> {out}")

    # all_metrics_reps.csv = 跨 seed mean/std/CI(repeated-CV;bar --reps 讀此檔)
    reps = _reps_summary(df)
    reps_out = base / "all_metrics_reps.csv"
    reps = reps.sort_values([c for c in _SORT_KEYS if c in reps.columns],
                            na_position="last").reset_index(drop=True)
    reps.to_csv(reps_out, index=False, encoding="utf-8")
    n_seed = df["seed"].nunique()
    logger.info(f"cross-seed ({n_seed} seeds): {len(reps)} cells -> {reps_out}")


if __name__ == "__main__":
    main()
