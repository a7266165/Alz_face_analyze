"""
scripts/embedding/evaluate/plot.py
Embedding 下游 evaluation 的 **第一層總覽圖(best-first)** —— 吃 aggregate 的 all_metrics.csv。

對 forward·1by1 的每個 (matching_priority, matched_unit, eval_unit) 切片畫一張
3×4 圖:列 = metric(balacc / AUC / MCC),欄 = embedding(arcface / dlib /
topofr / vggface)。每格 x 軸 = 3 個 contrast(AD/HC、AD/NAD、AD/ACS),分布 =
對該 embedding 底下『其餘所有軸』(variant×model×clf_param×bg×photo)取的
**config 分布**,以 box+strip 或 violin 呈現;★ = 該 (embedding, contrast, metric)
的冠軍 config(標注值 + model/variant)。x 軸標 config 數與受試者 n。

兩種圖各輸出到子夾 _summary/best_first/{box, violin}/;forward·1by1 的 4 個
(matched_unit × eval_unit) 組合全部畫(各一張)。

用法:
    conda run -n Alz_face_main_analysis python scripts/embedding/evaluate/plot.py \\
        --p-visit p_first --p-score p_cdrall --hc-visit hc_all --hc-score hc_cdrall_or_mmseall \\
        --matching-priority priority_acs --styles box violin
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
from scripts.embedding.classification.sweep import EMBEDDINGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_plot")

METRICS = [("balacc", "balacc"), ("auc", "AUC"), ("mcc", "MCC")]
# contrast -> (label, violin color);box 風格固定淺藍,violin 用此三色(沿用 emo_au palette)
CONTRASTS = [("ad_vs_hc", "AD/HC", "#C44E52"),
             ("ad_vs_nad", "AD/NAD", "#4C72B0"),
             ("ad_vs_acs", "AD/ACS", "#55A868")]
CHANCE = {"balacc": 0.5, "auc": 0.5, "mcc": 0.0}
YLIM = {"balacc": (0.4, 0.8), "auc": (0.4, 1.0), "mcc": (-0.2, 0.8)}
VARIANT_SHORT = {
    "original": "orig", "differences": "diff", "absolute_differences": "|diff|",
    "relative_differences": "rel", "absolute_relative_differences": "|rel|",
}
# forward·1by1 的 4 個 (matched_unit, eval_unit) 組合（全保留）
COMBOS = [("subject", "eval_by_subject"), ("subject", "eval_by_visit"),
          ("visit", "eval_by_subject"), ("visit", "eval_by_visit")]


def make_figure(df, matched_unit, eval_unit, priority, out_path, style):
    """一個切片 → 一張 3(metric)×4(embedding)圖;每格 3 個 contrast 的 config 分布 + 冠軍★。

    Args:
        style: box | violin
    """
    rng = np.random.RandomState(0)
    embs = [e for e in EMBEDDINGS if (df["emb"] == e).any()]
    nrow, ncol = len(METRICS), len(embs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow),
                             squeeze=False)
    for r, (mcol, mlabel) in enumerate(METRICS):
        for c, emb in enumerate(embs):
            ax = axes[r][c]
            de = df[df["emb"] == emb]
            ax.axhline(CHANCE[mcol], color="k", ls=":", lw=0.8, zorder=1)
            xticklabels = []
            for i, (con, clabel, color) in enumerate(CONTRASTS):
                sub = de[de["contrast"] == con].dropna(subset=[mcol])
                n_cfg = len(sub)
                n_subj = int(sub["n"].median()) if n_cfg else 0
                xticklabels.append(f"{clabel}\n{n_cfg}cfg\nn={n_subj}")
                if not n_cfg:
                    continue
                y = sub[mcol].values
                if style == "box":
                    box = ax.boxplot([y], positions=[i], widths=0.55,
                                     showfliers=False, patch_artist=True)
                    box["boxes"][0].set_facecolor("#cfe8ff")
                    box["boxes"][0].set_alpha(0.7)
                    ax.scatter(rng.normal(i, 0.06, n_cfg), y, s=4,
                               color="gray", alpha=0.22, zorder=2)
                elif np.ptp(y) > 0:  # violin
                    pc = ax.violinplot([y], positions=[i], widths=0.8, showmedians=True)
                    pc["bodies"][0].set_facecolor(color)
                    pc["bodies"][0].set_alpha(0.6)
                    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                        if key in pc:
                            pc[key].set_color("gray")
                            pc[key].set_linewidth(0.8)
                champ = sub.loc[sub[mcol].idxmax()]
                cv = champ[mcol]
                cfg = f"{champ['model']}/{VARIANT_SHORT.get(champ['variant'], champ['variant'])}"
                ax.scatter([i], [cv], marker="*", s=200, color="crimson",
                           edgecolor="black", linewidth=0.5, zorder=5)
                ax.annotate(f"{cv:.3f}\n{cfg}", (i, cv), textcoords="offset points",
                            xytext=(7, -4), fontsize=6, color="crimson", fontweight="bold")
            ax.set_xticks(range(3))
            ax.set_xticklabels(xticklabels if r == nrow - 1 else [], fontsize=7)
            if r == 0:
                ax.set_title(emb, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=12, fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.set_ylim(*YLIM[mcol])
            ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
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
    ap.add_argument("--styles", nargs="+", choices=["box", "violin"],
                    default=["box", "violin"],
                    help="要產生的圖種(預設兩種都產,各輸出到 best_first/<style>/)")
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

    base = root / cohort_path(*cohort) / "_summary" / "best_first"
    for style in args.styles:
        out_dir = base / style
        for mu, eu in COMBOS:
            sub = fwd[(fwd["matched_unit"] == mu) & (fwd["eval_unit"] == eu)]
            if not len(sub):
                logger.info(f"[skip] {style} {mu}-match × {eu}: no rows")
                continue
            out = out_dir / f"{args.matching_priority}_{mu}match_{eu}.png"
            make_figure(sub, mu, eu, args.matching_priority, out, style)
            logger.info(f"[ok]   {style} {mu}-match × {eu}: {len(sub)} rows -> {out.relative_to(base)}")
    logger.info(f"done -> {base}")


if __name__ == "__main__":
    main()
