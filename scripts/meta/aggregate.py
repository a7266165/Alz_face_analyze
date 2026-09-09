"""把 meta 各 cell 的 metrics.csv 彙整成 cohort 層三張表(仿 embedding aggregate)。

走訪 META_ANALYSIS_DIR/<cohort> 下所有 <meta_clf>/metrics.csv(由 run.py 落地、已含 cell 身份欄),
依路徑的 seed_<N> 段標上 `seed` → 產三檔:
  - all_metrics_score_avg.csv = **主要結果**。先對每個人平均各折分 seed 的分數,再算一次
    指標(PDF Step 10-11)。每格的平均分數落在 score_avg/<cell 去掉 seed 段>/。
  - all_metrics_reps.csv = 依 cell 身份鍵跨 seed groupby 的 mean/std + 95% CI。
    這是「先各 seed 算指標再平均」,與上面**不等價**(AUC 對分數非線性),留作敏感度分析。
  - all_metrics.csv      = seed_0 子集(單一 run 基準;下游 c_curve/confusion/bar 不開 --reps 讀此檔)。

用法:
    python scripts/meta/aggregate.py --p-visit p_first --p-score p_cdrall \\
        --hc-visit hc_all --hc-score hc_cdrall_or_mmseall
"""
import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

import numpy as np
import pandas as pd

from src.common.evaluate import evaluate
from src.common.folds import FOLD_KINDS
from src.config import (
    META_ANALYSIS_DIR, cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
)
from src.meta import META_CLASSIFIERS

# 平均分數的輸出子樹。它的 leaf 目錄名也是 <meta_clf>,故所有 rglob 都要排除這個
# 路徑段,否則會把自己的輸出當成 run.py 的 cell 再吃一次。
SCORE_AVG_DIR = "score_avg"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 排序鍵(存在才用):cell 身份 → 評估軸
_SORT_KEYS = ["feature_set", "variant", "base_clf", "clf_param", "meta_clf",
              "contrast", "eval_unit", "matched_unit", "matching_priority", "domain"]

# cross-seed 統計:cell 身份 + 評估軸(groupby 鍵)/ 要算 mean·std·CI 的指標
_IDENT_KEYS = ["p_visit", "p_score", "hc_visit", "hc_score", "bg", "emb", "photo",
               "reducer", "feature_set", "variant", "base_clf", "clf_param", "meta_clf",
               "fold_kind",
               "direction", "contrast", "eval_unit", "matched_unit", "matching_priority", "domain"]
_REP_METRICS = ["auc", "balacc", "mcc", "sens", "spec", "f1", "n"]
_SEED_RE = re.compile(r"seed_(\d+)")
_FOLDS_RE = re.compile(r"folds_(\w+)")


def _seed_of(path):
    """由 cell 路徑解析 seed_<N> 段 → int(無則 0)。"""
    m = next((_SEED_RE.fullmatch(part) for part in path.parts if _SEED_RE.fullmatch(part)), None)
    return int(m.group(1)) if m else 0


def _fold_kind_of(path):
    """由 cell 路徑解析 folds_<kind> 段 → str(無此段即 group,對應既有結果樹)。"""
    m = next((_FOLDS_RE.fullmatch(part) for part in path.parts if _FOLDS_RE.fullmatch(part)), None)
    return m.group(1) if m else "group"


def _cells(base, filename, fold_kind="group"):
    """base 底下該 fold_kind 的 <meta_clf>/<filename>(排除 score_avg 自己的輸出)。

    兩種折分共用同一棵 cohort 子樹(stratified 多一層 folds_<kind>),彙整時必須分開,
    不然 group 與 stratified 的格子會被混進同一張 all_metrics,cross-seed 統計也會
    把兩者當成同一格的不同 seed 平均掉。
    """
    return sorted(p for p in base.rglob(filename)
                  if p.parent.name in META_CLASSIFIERS and SCORE_AVG_DIR not in p.parts
                  and _fold_kind_of(p) == fold_kind)


def _identity_key(cell_dir, base):
    """cell 路徑去掉 seed_<N> 段 → 同一格不同折分 seed 的共同鍵。"""
    return tuple(p for p in cell_dir.relative_to(base).parts if not _SEED_RE.fullmatch(p))


def _score_average(base, cohort, fold_kind="group"):
    """PDF Step 10-11:先對每個人平均各 seed 的分數,再算一次指標。

    與 _reps_summary(先各 seed 算指標再平均)不等價 —— AUC 對分數是非線性的,
    兩者的差就是「換彙整方式」的效果,把它與「換方法」的效果分開。
    """
    groups = defaultdict(list)
    for p in _cells(base, "oof_scores.csv", fold_kind):
        groups[_identity_key(p.parent, base)].append(p)

    frames = []
    for key, paths in sorted(groups.items()):
        oof = pd.concat([pd.read_csv(p)[["ID", "y_true", "y_score"]] for p in paths],
                        ignore_index=True)
        mean = oof.groupby("ID", as_index=False).agg(
            y_true=("y_true", "first"), y_score=("y_score", "mean"),
            n_seeds=("y_score", "size"))
        # fold 已無意義(分數是跨折分平均的結果),但 evaluate 的 schema 需要這欄
        mean["fold"] = -1
        out_dir = base / SCORE_AVG_DIR / Path(*key)
        out_dir.mkdir(parents=True, exist_ok=True)
        oof_path = out_dir / "oof_scores.csv"
        mean.to_csv(oof_path, index=False, encoding="utf-8")

        # 身份欄沿用任一 seed 的 metrics.csv(除了 seed 以外,同一格的身份欄都一樣)
        ident = pd.read_csv(paths[0].parent / "metrics.csv").iloc[0]
        met = evaluate(oof_path, cohort, direction="forward",
                       eval_units=["eval_by_subject"], write=False)
        for c in _IDENT_KEYS:
            if c in ident.index and c not in met.columns:
                met[c] = ident[c]
        met["seed"] = "score_avg"
        met["fold_kind"] = fold_kind          # 舊 cell 的 metrics.csv 沒有這欄,由路徑補
        met["n_seeds_used"] = len(paths)
        cols = [c for c in _IDENT_KEYS if c in met.columns]
        met[cols + [c for c in met.columns if c not in cols]].to_csv(
            out_dir / "metrics.csv", index=False, encoding="utf-8")
        frames.append(met)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
    ap.add_argument("--fold-kind", choices=list(FOLD_KINDS), default="group",
                    help="彙整哪一種折分的 cell(兩種共用同一棵子樹,必須分開彙整);"
                         "非 group 的輸出落在 <case_mode>/folds_<kind>/")
    ap.add_argument("--out", type=Path, default=None,
                    help="輸出 csv(預設 <cohort>/<case_mode>/[folds_<kind>/]all_metrics.csv)")
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    base = META_ANALYSIS_DIR / cohort_path(*cohort) / args.case_mode
    paths = _cells(base, "metrics.csv", args.fold_kind)
    if not paths:
        logger.warning(f"在 {base} 下找不到 fold_kind={args.fold_kind} 的 <meta_clf>/metrics.csv;"
                       f"請先跑 scripts/meta/run.py。")
        return
    # 輸出根:group 維持原位(既有下游讀這裡),其他折分各自一層,不互相覆蓋
    out_base = base if args.fold_kind == "group" else base / f"folds_{args.fold_kind}"

    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["seed"] = _seed_of(p)          # 路徑為準(舊 metrics 無 seed 欄亦可)
        d["fold_kind"] = args.fold_kind  # 舊 cell 無此欄,由路徑補
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    sort_keys = [c for c in _SORT_KEYS if c in df.columns]
    df = df.sort_values(sort_keys, na_position="last").reset_index(drop=True)

    out_base.mkdir(parents=True, exist_ok=True)
    # all_metrics.csv = seed_0 單一 run 基準(下游 c_curve/confusion/bar 不開 --reps 讀此檔)
    out = args.out or (out_base / "all_metrics.csv")
    df0 = df[df["seed"] == 0].reset_index(drop=True)
    df0.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"collected {len(paths)} metrics.csv -> seed_0 {len(df0)} rows x {df0.shape[1]} cols -> {out}")

    # all_metrics_reps.csv = 跨 seed mean/std/CI(repeated-CV;bar --reps 讀此檔)
    reps = _reps_summary(df)
    reps_out = out_base / "all_metrics_reps.csv"
    reps = reps.sort_values([c for c in _SORT_KEYS if c in reps.columns],
                            na_position="last").reset_index(drop=True)
    reps.to_csv(reps_out, index=False, encoding="utf-8")
    n_seed = df["seed"].nunique()
    logger.info(f"cross-seed ({n_seed} seeds): {len(reps)} cells -> {reps_out}")

    # all_metrics_score_avg.csv = 主要結果(先平均分數,再算一次指標)
    sa = _score_average(base, cohort, args.fold_kind)
    if len(sa):
        sa_out = out_base / "all_metrics_score_avg.csv"
        sa = sa.sort_values([c for c in _SORT_KEYS if c in sa.columns],
                            na_position="last").reset_index(drop=True)
        sa.to_csv(sa_out, index=False, encoding="utf-8")
        logger.info(f"score-average: {len(sa)} rows -> {sa_out}")


if __name__ == "__main__":
    main()
