"""Overview 單特徵判別力 barplot:core4 的 4 個輸入特徵各自「單獨」能多會分 AD。

4 個特徵(= meta core4 的組成)各走一條 univariate logistic OOF(= meta 的 lr stacker,單欄輸入;
logistic 自動學方向、閾值也是 fit 出來的,故 balacc/MCC 公平,AUC 因 rank-based 等同該特徵原始判別力):
  - arcface/original      = embedding_LR_score(原圖 embedding 的 forward logistic OOF)
  - arcface/differences   = asymmetry_LR_score(左右差異圖 embedding 的 forward logistic OOF)
  - age                   = real_age(實齡)
  - predict_age_error     = age_error(實齡 − MiVOLO 預測齡)

每特徵的 OOF 交 src.common.evaluate(eval_by_subject、GroupKFold-by-base_id 無 leakage),取 domain=all
(整 cohort)的 3 metric(balacc/auc/mcc)× 3 contrast(ad_vs_hc / ad_vs_nad / ad_vs_acs)→ 一張 3×3 barplot。
母體 = full 2070(這 4 欄皆無 NaN;complete_case=False)。輸出採 embedding _summary 風格分層:
  workspace/overview/barplot/<visit>/<cdr_mmse>/<eval_unit>/<domain>.png(domain = all / <matched_unit>_1by1)。

用法:
    python scripts/overview/single_feature_bar.py
    python scripts/overview/single_feature_bar.py --lr-C 0.001 --variant differences
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
from src.meta import oof_from_table, session_feature_table

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# (session 表欄名, x 軸標籤, bar 顏色) —— 順序 = x 軸順序
FEATURES = [
    ("embedding_LR_score", "arcface/original", "#4C72B0"),
    ("asymmetry_LR_score", "arcface/differences", "#55A868"),
    ("real_age", "age", "#DD8452"),
    ("age_error", "predict_age_error", "#C44E52"),
]
CONTRASTS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
METRICS = ["balacc", "auc", "mcc"]
METRIC_LABEL = {"balacc": "Balanced Acc", "auc": "AUC", "mcc": "MCC"}
CHANCE = {"balacc": 0.5, "auc": 0.5, "mcc": 0.0}
YLIM = {"balacc": (0.4, 1.0), "auc": (0.4, 1.0), "mcc": (-0.1, 0.8)}


def _evaluate_feature(table, col, cohort, tmpdir, eval_unit):
    """單欄 → univariate logistic OOF → evaluate(指定 eval_unit);回完整 metrics DataFrame(all + 1by1)。"""
    oof = oof_from_table(table, [col], meta_clf="lr")
    oof_path = Path(tmpdir) / f"{col}_oof.csv"
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


def _plot(metrics_by_feature, domain, args, out_png, *, title):
    """metrics_by_feature[col]=evaluate df → 某 domain 的 3 metric(列)× 3 contrast(欄),每格 4 根 bar。"""
    labels = [lab for _, lab, _ in FEATURES]
    colors = [c for _, _, c in FEATURES]
    x = list(range(len(FEATURES)))
    fig, axes = plt.subplots(len(METRICS), len(CONTRASTS), figsize=(13, 11),
                             sharex=True, sharey="row")
    for i, mt in enumerate(METRICS):
        for j, contrast in enumerate(CONTRASTS):
            ax = axes[i][j]
            vals = [_pick(metrics_by_feature[col], domain, contrast,
                          matched_unit=args.matched_unit,
                          matching_priority=args.matching_priority)[mt]
                    for col, _, _ in FEATURES]
            ax.bar(x, vals, color=colors, width=0.8)
            ax.axhline(CHANCE[mt], color="k", ls=":", lw=0.8)
            for xi, v in zip(x, vals):
                if v == v:
                    ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set_ylim(*YLIM[mt])
            ax.grid(axis="y", alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels if i == len(METRICS) - 1 else [],
                               rotation=30, ha="right", fontsize=9)
            if i == 0:
                ax.set_title(contrast, fontsize=12)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[mt], fontsize=12)
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
                f" features={[c for c, _, _ in FEATURES]}")

    metrics_by_feature = {}
    with tempfile.TemporaryDirectory() as tmp:
        for col, label, _ in FEATURES:
            m = _evaluate_feature(table, col, cohort, tmp, args.eval_unit)
            metrics_by_feature[col] = m
            hc = _pick(m, "all", "ad_vs_hc",
                       matched_unit=args.matched_unit, matching_priority=args.matching_priority)
            logger.info(f"  [{label:22s}] ad_vs_hc all: "
                        f"auc={hc['auc']:.3f} balacc={hc['balacc']:.3f} "
                        f"mcc={hc['mcc']:.3f} n={int(hc['n'])}")

    # embedding _summary 風格分層:barplot/<cohort>/<eval_unit>/<domain>.png
    out_dir = (PROJECT_ROOT / "workspace" / "overview" / "barplot"
               / cohort_path(*cohort) / args.eval_unit)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 兩版:all(整 cohort)+ 1by1(年齡配對,visit/ACS 預設,仿 meta bar 的 all / visit_1by1)
    domains = [("all", "all", "all (full cohort)"),
               ("1by1", f"{args.matched_unit}_1by1",
                f"1by1 ({args.matched_unit}, {args.matching_priority})")]
    for domain, dpart, dtag in domains:
        out_png = out_dir / f"{dpart}.png"
        _plot(metrics_by_feature, domain, args, out_png,
              title=f"single-feature discrimination (core4 inputs, univariate logistic OOF, "
                    f"{args.eval_unit}) @ {dtag} — {args.emb}/{args.bg_mode}/{args.photo_mode}, "
                    f"asym={args.variant}, C={args.lr_C:g}")
        logger.info(f"wrote {out_png}")


if __name__ == "__main__":
    main()
