"""Overview 折線圖:core3 vs core4 在三個 meta stacker(LR / XGB / TabPFN)下、配對前後的判別力。

每格(metric × contrast)4 條線、x 軸只有 3 個 stacker:
  - core3 all  / core3 1by1   (core3 = embedding + asymmetry + age_error,去掉 real_age)
  - core4 all  / core4 1by1   (core4 = core3 + real_age 年齡)
all = 整 cohort;1by1 = 年齡 1:1 配對後子集(matched_unit / matching_priority)。比較 core3 與 core4
在配對前後的落差,凸顯 core4 多出的 real_age 在配對後是否還有貢獻(年齡 confound)。

每條線的 OOF 由 oof_from_table(meta stacker fold-aligned,即時算)→ src.common.evaluate
(GroupKFold-by-base_id 無 leakage),取 3 metric(balacc/auc/mcc)× 3 contrast(ad_vs_hc /
ad_vs_nad / ad_vs_acs)→ 3×3 grid。母體 = full cohort(complete_case=False;core3/4 諸欄皆無 NaN)。
輸出採 embedding _summary 風格分層,每 eval_unit 一張:
  workspace/overview/lineplot/<visit>/<cdr_mmse>/<eval_unit>/all_vs_1by1.png。

用法:
    python scripts/overview/core_compare_line.py --eval-unit eval_by_visit
    python scripts/overview/core_compare_line.py --lr-C 0.001 --variant differences
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

# x 軸 = 3 個 meta stacker(順序固定);每格 4 條線 = {core3, core4} × {all, 1by1}。
STACKERS = [("lr", "LR"), ("xgb", "XGB"), ("tabpfn_v3", "TabPFN")]
FEATURE_SETS = [
    ("core3", META_FEATURE_SETS["core3"]),   # embedding + asymmetry + age_error
    ("core4", META_FEATURE_SETS["core4"]),   # core3 + real_age
]
CONTRASTS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
METRICS = ["balacc", "auc", "mcc"]
METRIC_LABEL = {"balacc": "Balanced Acc", "auc": "AUC", "mcc": "MCC"}
CHANCE = {"balacc": 0.5, "auc": 0.5, "mcc": 0.0}
YLIM = {"balacc": (0.4, 1.0), "auc": (0.4, 1.0), "mcc": (-0.1, 0.8)}
# (feature_set, domain) → 線型:core3 藍 / core4 紅;all 實線 / 1by1 虛線+空心點。
LINE_STYLES = [
    ("core3", "all",  dict(color="#1f77b4", ls="-",  marker="o", ms=6, lw=1.6, label="core3 all")),
    ("core3", "1by1", dict(color="#1f77b4", ls="--", marker="o", ms=6, lw=1.6,
                           mfc="white", label="core3 1by1")),
    ("core4", "all",  dict(color="#d62728", ls="-",  marker="s", ms=6, lw=1.6, label="core4 all")),
    ("core4", "1by1", dict(color="#d62728", ls="--", marker="s", ms=6, lw=1.6,
                           mfc="white", label="core4 1by1")),
]


def _evaluate(table, cols, clf, cohort, tmpdir, eval_unit):
    """指定欄 → clf 的 fold-aligned OOF → evaluate(指定 eval_unit);回 metrics DataFrame。"""
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


def _plot(metrics, args, out_png, *, title):
    """每格(metric × contrast)4 條線({core3,core4} × {all,1by1}),x = 3 個 stacker。"""
    xlabels = [lab for _, lab in STACKERS]
    x = list(range(len(STACKERS)))
    fig, axes = plt.subplots(len(METRICS), len(CONTRASTS), figsize=(13, 11),
                             sharex=True, sharey="row")
    for i, mt in enumerate(METRICS):
        for j, contrast in enumerate(CONTRASTS):
            ax = axes[i][j]
            for fs, domain, st in LINE_STYLES:
                vals = [_pick(metrics[(fs, clf)], domain, contrast,
                              matched_unit=args.matched_unit,
                              matching_priority=args.matching_priority)[mt]
                        for clf, _ in STACKERS]
                ax.plot(x, vals, zorder=2, **st)
            ax.axhline(CHANCE[mt], color="k", ls=":", lw=0.8)
            ax.set_xlim(-0.3, len(STACKERS) - 0.7)
            ax.set_ylim(*YLIM[mt])
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels if i == len(METRICS) - 1 else [], fontsize=11)
            if i == 0:
                ax.set_title(contrast, fontsize=12)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[mt], fontsize=12)
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=12)
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
                    help="asymmetry 特徵用哪個 variant(core3/core4 的 asymmetry_LR_score 來源)")
    ap.add_argument("--lr-C", type=float, default=0.001,
                    help="embedding/asymmetry base logistic 的 C(定位落地 OOF;預設 headline 0.001)")
    ap.add_argument("--eval-unit", choices=["eval_by_subject", "eval_by_visit"],
                    default="eval_by_subject",
                    help="評估粒度:eval_by_subject(同 subject 各 visit 分數平均)/ eval_by_visit(每 visit 一樣本)")
    ap.add_argument("--matched-unit", choices=["subject", "visit"], default="visit",
                    help="[1by1 線] 年齡 1:1 配對的粒度")
    ap.add_argument("--matching-priority",
                    choices=["no_priority", "priority_acs", "priority_nad"],
                    default="priority_acs", help="[1by1 線] 配對優先序(對齊 overview ACS_first 慣例)")
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    table = session_feature_table(
        cohort, variant=args.variant, emb=args.emb, bg_mode=args.bg_mode,
        photo_mode=args.photo_mode, reducer=args.reducer, base_clf="logistic",
        lr_C=args.lr_C, seed=0, complete_case=False)
    logger.info(f"session table: {len(table)} sessions (full cohort); "
                f"stackers={[l for _, l in STACKERS]} feature_sets={[f for f, _ in FEATURE_SETS]}")

    metrics = {}
    with tempfile.TemporaryDirectory() as tmp:
        for fs_name, cols in FEATURE_SETS:
            for clf, clf_label in STACKERS:
                m = _evaluate(table, cols, clf, cohort, tmp, args.eval_unit)
                metrics[(fs_name, clf)] = m
                hc = _pick(m, "all", "ad_vs_hc",
                           matched_unit=args.matched_unit, matching_priority=args.matching_priority)
                hc1 = _pick(m, "1by1", "ad_vs_hc",
                            matched_unit=args.matched_unit, matching_priority=args.matching_priority)
                logger.info(f"  [{fs_name}/{clf_label:6s}] ad_vs_hc auc: all={hc['auc']:.3f} "
                            f"1by1={hc1['auc']:.3f}  n(all)={int(hc['n'])}")

    # embedding _summary 風格分層:lineplot/<cohort>/<eval_unit>/all_vs_1by1.png(4 線一張)
    out_dir = (PROJECT_ROOT / "workspace" / "overview" / "lineplot"
               / cohort_path(*cohort) / args.eval_unit)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "all_vs_1by1.png"
    _plot(metrics, args, out_png,
          title=f"{'/'.join(cohort_path(*cohort).parts)} — core3 vs core4 across stackers "
                f"({args.eval_unit}) — all vs 1by1 ({args.matched_unit}, {args.matching_priority}) — "
                f"{args.emb}/{args.bg_mode}/{args.photo_mode}, asym={args.variant}, C={args.lr_C:g}")
    logger.info(f"wrote {out_png}")


if __name__ == "__main__":
    main()
