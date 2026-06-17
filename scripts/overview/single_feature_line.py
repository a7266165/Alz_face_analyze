"""Overview 判別力折線圖:core 的單一輸入特徵各自「單獨」能多會分 AD,外加合併的 core3(3 種 stacker)。

前 3 根 = core3 的單一輸入特徵,各走 univariate logistic OOF(= meta lr stacker,單欄;logistic 自動學
方向、閾值也是 fit 出來的,故 balacc/MCC 公平,AUC 因 rank-based 等同該特徵原始判別力):
  - arcface/original      = embedding_LR_score(原圖 embedding 的 forward logistic OOF)
  - arcface/differences   = asymmetry_LR_score(左右差異圖 embedding 的 forward logistic OOF)
  - predict_age_error     = age_error(實齡 − MiVOLO 預測齡)
第 4–6 根 = core3(= embedding + asymmetry + age_error,即 core4 去掉 real_age 年齡 confound),分別走
三種 meta stacker LR / XGB / TabPFN(同 2070 表、core3 無 NaN,數值等同 meta 的 core3/<variant>/<stacker>);
與前 3 根模型不同,呈現「合併後的 core3」在三個 stacker 下的表現(TabPFN 為 headline)。
(real_age 本身已不在 core3,且年齡是 confound,故折線圖不再放獨立的 age。)

每 bar 的 OOF 交 src.common.evaluate(GroupKFold-by-base_id 無 leakage),取 3 metric(balacc/auc/mcc)
× 3 contrast(ad_vs_hc / ad_vs_nad / ad_vs_acs)→ 3×3 折線圖,每格疊 all 與 1by1 兩條線(x=6 個特徵/
模型),一眼看每個模型配對前後的落差。母體 = full cohort(core3 諸欄皆無 NaN;complete_case=False)。
輸出採 embedding _summary 風格分層,每 eval_unit 一張(內含 all + 1by1 兩線):
  workspace/overview/lineplot/<visit>/<cdr_mmse>/<eval_unit>/all_vs_1by1.png。

用法:
    python scripts/overview/single_feature_line.py
    python scripts/overview/single_feature_line.py --lr-C 0.001 --variant differences
"""
import argparse
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.evaluate import evaluate
from src.config import (
    cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
)
from src.meta import META_FEATURE_SETS, oof_from_table, session_feature_table

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# (欄清單, 標籤, 顏色, stacker) —— 順序 = x 軸順序。
# 前 3 根 = core3 的單一輸入特徵(univariate logistic);後 3 根 = core3 合併,走 LR/XGB/TabPFN。
# real_age 不放(已不在 core3、且是年齡 confound)。
_CORE3 = META_FEATURE_SETS["core3"]   # 單一來源:embedding_LR_score, asymmetry_LR_score, age_error
BARS = [
    (["embedding_LR_score"], "arcface/original", "#4C72B0", "lr"),
    (["asymmetry_LR_score"], "arcface/differences", "#55A868", "lr"),
    (["age_error"], "predict_age_error", "#C44E52", "lr"),
    (_CORE3, "core3 (LR)", "#B6A6CA", "lr"),
    (_CORE3, "core3 (XGB)", "#8C6BB1", "xgb"),
    (_CORE3, "core3 (TabPFN)", "#5E3C99", "tabpfn_v3"),
]
CONTRASTS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
METRICS = ["balacc", "auc", "mcc"]
METRIC_LABEL = {"balacc": "Balanced Acc", "auc": "AUC", "mcc": "MCC"}
CHANCE = {"balacc": 0.5, "auc": 0.5, "mcc": 0.0}
YLIM = {"balacc": (0.4, 1.0), "auc": (0.4, 1.0), "mcc": (-0.1, 0.8)}


def _evaluate_bar(table, cols, clf, cohort, tmpdir, eval_unit):
    """指定欄(單欄=univariate)→ clf 的 fold-aligned OOF → evaluate(指定 eval_unit);回 metrics DataFrame。"""
    oof = oof_from_table(table, cols, meta_clf=clf)
    oof_path = Path(tmpdir) / f"{'_'.join(cols)}__{clf}_oof.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8")
    return evaluate(oof_path, cohort, direction="forward",
                    eval_units=[eval_unit], write=False)


def _pick(m, domain, contrast, *, matched_unit, matching_priority):
    """取某 (domain, contrast) 的 balacc/auc/mcc/n;all=整 cohort,1by1=指定 matched_unit + priority。"""
    if domain == "all":
        r = m[(m["domain"] == "all") & (m["contrast"] == contrast)]
    else:
        r = m[(m["domain"] == "1by1") & (m["contrast"] == contrast)
              & (m["matched_unit"] == matched_unit)
              & (m["matching_priority"] == matching_priority)]
    if not len(r):
        return {mt: float("nan") for mt in METRICS + ["n"]}
    return {mt: float(r.iloc[0][mt]) for mt in METRICS + ["n"]}


def _plot(metrics_by_bar, args, out_png, *, title):
    """每格(metric × contrast)疊 all 與 1by1 兩條折線(x=7 模型);看每個模型配對前後的落差。"""
    labels = [lab for _, lab, _, _ in BARS]
    x = list(range(len(BARS)))
    styles = {
        "all":  dict(color="#1f77b4", marker="o", ms=6, lw=1.5, label="all (full cohort)"),
        "1by1": dict(color="#d62728", marker="s", ms=6, lw=1.5,
                     label=f"1by1 ({args.matched_unit}, {args.matching_priority})"),
    }
    fig, axes = plt.subplots(len(METRICS), len(CONTRASTS), figsize=(13, 11),
                             sharex=True, sharey="row")
    for i, mt in enumerate(METRICS):
        for j, contrast in enumerate(CONTRASTS):
            ax = axes[i][j]
            for domain, st in styles.items():
                vals = [_pick(metrics_by_bar[lab], domain, contrast,
                              matched_unit=args.matched_unit,
                              matching_priority=args.matching_priority)[mt]
                        for _, lab, _, _ in BARS]
                ax.plot(x, vals, zorder=2, **st)
            ax.axhline(CHANCE[mt], color="k", ls=":", lw=0.8)
            ax.set_xlim(-0.5, len(BARS) - 0.5)
            ax.set_ylim(*YLIM[mt])
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels if i == len(METRICS) - 1 else [],
                               rotation=30, ha="right", fontsize=9)
            if i == 0:
                ax.set_title(contrast, fontsize=12)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[mt], fontsize=12)
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default="p_first")
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default="p_cdrall")
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default="hc_all")
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                    default="hc_cdrall_or_mmseall")
    ap.add_argument("--emb", default="arcface")
    ap.add_argument("--bg-mode", choices=["background", "no_background"], default="background")
    ap.add_argument("--photo-mode", choices=["mean", "all"], default="all")
    ap.add_argument("--reducer", default="no_drop")
    ap.add_argument("--variant", default="differences",
                    help="asymmetry 特徵用哪個 variant(arcface/differences 那根 bar 的來源)")
    ap.add_argument("--lr-C", type=float, default=0.001,
                    help="embedding/asymmetry base logistic 的 C(定位落地 OOF;預設 headline 0.001)")
    ap.add_argument("--eval-unit", choices=["eval_by_subject", "eval_by_visit"],
                    default="eval_by_subject",
                    help="評估粒度:eval_by_subject(同 subject 各 visit 分數平均)/ eval_by_visit(每 visit 一樣本)")
    ap.add_argument("--matched-unit", choices=["subject", "visit"], default="visit",
                    help="[1by1 圖] 年齡 1:1 配對的粒度")
    ap.add_argument("--matching-priority",
                    choices=["no_priority", "priority_acs", "priority_nad"],
                    default="priority_acs", help="[1by1 圖] 配對優先序(對齊 overview ACS_first 慣例)")
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    table = session_feature_table(
        cohort, variant=args.variant, emb=args.emb, bg_mode=args.bg_mode,
        photo_mode=args.photo_mode, reducer=args.reducer, base_clf="logistic",
        lr_C=args.lr_C, seed=0, complete_case=False)
    logger.info(f"session table: {len(table)} sessions (full cohort);"
                f" bars={[lab for _, lab, _, _ in BARS]}")

    metrics_by_bar = {}
    with tempfile.TemporaryDirectory() as tmp:
        for cols, label, _, clf in BARS:
            m = _evaluate_bar(table, cols, clf, cohort, tmp, args.eval_unit)
            metrics_by_bar[label] = m
            hc = _pick(m, "all", "ad_vs_hc",
                       matched_unit=args.matched_unit, matching_priority=args.matching_priority)
            logger.info(f"  [{label:22s}] ad_vs_hc all: "
                        f"auc={hc['auc']:.3f} balacc={hc['balacc']:.3f} "
                        f"mcc={hc['mcc']:.3f} n={int(hc['n'])}")

    # embedding _summary 風格分層:lineplot/<cohort>/<eval_unit>/all_vs_1by1.png(all 與 1by1 疊一張)
    out_dir = (PROJECT_ROOT / "workspace" / "overview" / "lineplot"
               / cohort_path(*cohort) / args.eval_unit)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "all_vs_1by1.png"
    _plot(metrics_by_bar, args, out_png,
          title=f"{'/'.join(cohort_path(*cohort).parts)} — single-feature (univariate logistic) vs "
                f"core3 (LR/XGB/TabPFN) discrimination ({args.eval_unit}) — "
                f"all vs 1by1 ({args.matched_unit}, {args.matching_priority}) — "
                f"{args.emb}/{args.bg_mode}/{args.photo_mode}, asym={args.variant}, C={args.lr_C:g}")
    logger.info(f"wrote {out_png}")


if __name__ == "__main__":
    main()
