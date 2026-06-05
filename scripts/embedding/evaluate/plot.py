"""
scripts/embedding/evaluate/plot.py
Embedding 下游 evaluation 的 **best-first 第一層總覽圖** —— 吃 aggregate 的 all_metrics.csv。

對 forward·1by1 的每個 (matching_priority, matched_unit, eval_unit) 切片畫一張
1×3 圖:欄 = metric(balacc / AUC / MCC),每欄 x 軸 = 3 個 contrast(AD/HC、AD/NAD、
AD/ACS)。每個 contrast 一個 box + 灰色散點 = 對『其餘所有軸』
(emb×variant×model×clf_param×bg×photo)取的分布;★ = 該 (contrast, metric) 的冠軍
config(標注值 + emb/model/variant)。

forward·1by1 的 4 個 (matched_unit × eval_unit) 組合全部畫(各一張)。

用法:
    conda run -n Alz_face_main_analysis python scripts/embedding/evaluate/plot.py \\
        --p-visit p_first --p-score p_cdrall --hc-visit hc_all --hc-score hc_cdrall_or_mmseall \\
        --matching-priority priority_acs
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_CLASSIFICATION_REFACTOR_DIR, cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_plot")

METRICS = [("balacc", "balacc"), ("auc", "AUC"), ("mcc", "MCC")]
CONTRASTS = [("ad_vs_hc", "AD/HC"), ("ad_vs_nad", "AD/NAD"), ("ad_vs_acs", "AD/ACS")]
CHANCE = {"balacc": 0.5, "auc": 0.5, "mcc": 0.0}
YLIM = {"balacc": (0.4, 0.8), "auc": (0.4, 1.0), "mcc": (-0.2, 0.8)}
VARIANT_SHORT = {
    "original": "orig", "differences": "diff", "absolute_differences": "|diff|",
    "relative_differences": "rel", "absolute_relative_differences": "|rel|",
}
# forward·1by1 的 4 個 (matched_unit, eval_unit) 組合（全保留）
COMBOS = [("subject", "eval_by_subject"), ("subject", "eval_by_visit"),
          ("visit", "eval_by_subject"), ("visit", "eval_by_visit")]


def make_figure(df, matched_unit, eval_unit, priority, out_path):
    """一個切片 → 一張 1×3(metric)圖;每欄 3 個 contrast 的 box+strip+冠軍★。"""
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (mcol, mlabel) in zip(axes, METRICS):
        arrays = [df[df["contrast"] == con][mcol].dropna().values for con, _ in CONTRASTS]
        box = ax.boxplot(arrays, positions=range(3), widths=0.55,
                         showfliers=False, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor("#cfe8ff")
            patch.set_alpha(0.7)
        ax.axhline(CHANCE[mcol], color="k", ls=":", lw=0.8, zorder=1)
        xticklabels = []
        for i, (con, clabel) in enumerate(CONTRASTS):
            sub = df[df["contrast"] == con].dropna(subset=[mcol])
            n_med = int(sub["n"].median()) if len(sub) else 0
            xticklabels.append(f"{clabel}\nn={n_med}")
            if not len(sub):
                continue
            y = sub[mcol].values
            ax.scatter(rng.normal(i, 0.06, len(y)), y, s=4,
                       color="gray", alpha=0.22, zorder=2)
            champ = sub.loc[sub[mcol].idxmax()]
            cv = champ[mcol]
            cfg = f"{champ['emb']}/{champ['model']}/{VARIANT_SHORT.get(champ['variant'], champ['variant'])}"
            ax.scatter([i], [cv], marker="*", s=280, color="crimson",
                       edgecolor="black", linewidth=0.5, zorder=5)
            ax.annotate(f"{cv:.3f}\n{cfg}", (i, cv), textcoords="offset points",
                        xytext=(9, -4), fontsize=7, color="crimson", fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(xticklabels, fontsize=9)
        ax.set_title(mlabel, fontsize=13, fontweight="bold")
        ax.set_ylim(*YLIM[mcol])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("metric value\n(每點 = 一種其餘軸組合)", fontsize=10)
    fig.suptitle(
        f"Best-first 第1層 | forward 1by1 | {priority} | {matched_unit}-match | {eval_unit}\n"
        "contrast x metric:對『其餘所有軸 (emb x variant x model x clf_param x bg x photo)』取分布"
        "   (*) = 冠軍 config", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])
    ap.add_argument("--matching-priority",
                    choices=["no_priority", "priority_acs", "priority_nad"],
                    default="priority_acs",
                    help="只畫此配對優先序的結果(預設 priority_acs)")
    ap.add_argument("--output-root", type=Path, default=None)
    args = ap.parse_args()

    root = args.output_root or EMBEDDING_CLASSIFICATION_REFACTOR_DIR
    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    csv = root / cohort_path(*cohort) / "all_metrics.csv"
    if not csv.exists():
        ap.error(f"找不到 all_metrics.csv: {csv}（先跑 evaluate/aggregate.py）")
    df = pd.read_csv(csv)
    fwd = df[(df["direction"] == "forward") & (df["domain"] == "1by1")
             & (df["matching_priority"] == args.matching_priority)].copy()
    logger.info(f"all_metrics rows={len(df)}  forward·1by1·{args.matching_priority} rows={len(fwd)}")

    out_dir = root / cohort_path(*cohort) / "_summary" / "best_first"
    for mu, eu in COMBOS:
        sub = fwd[(fwd["matched_unit"] == mu) & (fwd["eval_unit"] == eu)]
        if not len(sub):
            logger.info(f"[skip] {mu}-match × {eu}: no rows (evaluate skips this combo)")
            continue
        out = out_dir / f"best_first_{args.matching_priority}_{mu}match_{eu}.png"
        make_figure(sub, mu, eu, args.matching_priority, out)
        logger.info(f"[ok]   {mu}-match × {eu}: {len(sub)} rows -> {out.name}")
    logger.info(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
