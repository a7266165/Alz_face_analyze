"""繪製 evaluation 總覽圖
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    EMBEDDING_CLASSIFICATION_DIR, EMBEDDING_CLASSIFICATION_REFACTOR_DIR, cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.embedding.classification import oof_dir
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


def make_figure(df, out_path, style):
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


# ── ArcFace LogReg on asymmetry：metrics-vs-C 圖（每 variant 一張 3×2）─────────────
# 讀 classification/evaluate sweep 對 6 個 C 值產出的 metrics.csv：
#   欄＝full(domain=all) / 1-by-1 matched(visit·priority_acs)，列＝Balanced ACC/AUC/MCC，
#   每格 x=LogReg C(log)、3 條線=AD vs {HC,NAD,ACS}。固定 eval_by_subject。
LR_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]                 # = classification/run.LR_C_GRID
LR_EVAL_UNIT = "eval_by_subject"
LR_EMB, LR_MODEL, LR_REDUCER, LR_PHOTO, LR_DIRECTION = "arcface", "logistic", "no_drop", "all", "forward"
LR_VARIANTS = [
    ("differences", "diff", r"diff $=L_i-R_i$"),
    ("relative_differences", "rel_diff", r"rel-diff $=(L_i-R_i)/\sqrt{L_i^2+R_i^2}$"),
]
LR_CONTRASTS = [("ad_vs_hc", "AD vs HC", "#4C72B0"),
                ("ad_vs_nad", "AD vs NAD", "#55A868"),
                ("ad_vs_acs", "AD vs ACS", "#C44E52")]
LR_METRICS = [("balacc", "Balanced ACC"), ("auc", "AUC"), ("mcc", "MCC")]
LR_SLICES = [("full (unmatched)", {"domain": "all"}),
             ("1-by-1 matched\n(visit · priority_acs)",
              {"domain": "1by1", "matched_unit": "visit", "matching_priority": "priority_acs"})]


def lr_load_metrics(cohort, bg_mode, variant, root):
    """讀 6 個 C 值 cell 的 metrics.csv，併成帶 C 欄的 DataFrame（缺檔略過）。"""
    frames = []
    for c in LR_C_GRID:
        d = oof_dir(cohort, bg_mode, LR_EMB, variant, LR_PHOTO, LR_REDUCER, LR_MODEL,
                    LR_DIRECTION, lr_C=c, root=root)
        f = d / "metrics.csv"
        if not f.exists():
            logger.warning(f"missing metrics.csv: {f}")
            continue
        dfm = pd.read_csv(f)
        dfm["C"] = c
        frames.append(dfm)
    return pd.concat(frames, ignore_index=True) if frames else None


def lr_series(df, filt, contrast, metric):
    """沿 LR_C_GRID 取某 (slice 過濾, contrast, metric) 的值序列；缺值補 nan。"""
    sub = df[(df["eval_unit"] == LR_EVAL_UNIT) & (df["contrast"] == contrast)]
    for k, v in filt.items():
        sub = sub[sub[k] == v]
    by_c = dict(zip(sub["C"], sub[metric]))
    return [by_c.get(c, np.nan) for c in LR_C_GRID]


def build_lr_fig(df, variant_disp, out_path):
    """3×2：列＝Balanced ACC/AUC/MCC、欄＝full/matched；每格 3 條三族群線 vs C。

    此圖沿用 matplotlib 預設字型/負號（非本模組為 summary 圖設的 JhengHei），故包在 rc_context 內。
    """
    with matplotlib.rc_context({
            "font.sans-serif": matplotlib.rcParamsDefault["font.sans-serif"],
            "axes.unicode_minus": matplotlib.rcParamsDefault["axes.unicode_minus"]}):
        fig, axes = plt.subplots(len(LR_METRICS), len(LR_SLICES),
                                 figsize=(11, 11), sharex=True, sharey="row")
        for ri, (mkey, mlabel) in enumerate(LR_METRICS):
            for ci, (slabel, filt) in enumerate(LR_SLICES):
                ax = axes[ri][ci]
                for ckey, clabel, color in LR_CONTRASTS:
                    ax.plot(LR_C_GRID, lr_series(df, filt, ckey, mkey),
                            marker="o", ms=5, lw=1.8, color=color, label=clabel)
                ax.set_xscale("log")
                ax.axhline(0.5 if mkey in ("balacc", "auc") else 0.0,
                           ls="--", lw=1, color="0.6")
                ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
                if ri == 0:
                    ax.set_title(slabel, fontsize=12, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(mlabel, fontsize=12, fontweight="bold")
                if ri == len(LR_METRICS) - 1:
                    ax.set_xlabel("LogReg  C", fontsize=11)
        axes[0][1].legend(loc="best", fontsize=9, framealpha=0.9)
        fig.suptitle(f"ArcFace · LogReg on {variant_disp}\n"
                     "metrics vs C (forward OOF, eval_by_subject)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    logger.info(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # cohort token 預設留空，依 --kind 在 parse 後決定（summary=分類 sweep 預設；
    # lr_metrics=asymmetry 慣用組合），明確指定仍可覆寫。
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=None)
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=None)
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=None)
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=None)
    ap.add_argument("--kind", choices=["summary", "lr_metrics"], default="summary",
                    help="summary=各 cell 指標總覽(讀 all_metrics.csv)；"
                         "lr_metrics=ArcFace LogReg on asymmetry 的 metrics-vs-C 圖")
    ap.add_argument("--bg-mode", choices=["background", "no_background"], default="background",
                    help="僅 --kind lr_metrics 用")
    ap.add_argument("--matching-priority",
                    choices=["no_priority", "priority_acs", "priority_nad"],
                    default="priority_acs",
                    help="只畫此配對優先序的結果(預設 priority_acs)")
    ap.add_argument("--styles", nargs="+", choices=["box", "violin"],
                    default=["box", "violin"],
                    help="要產生的圖種(預設兩種都產,各輸出到 _summary/<style>/)")
    ap.add_argument("--output-root", type=Path, default=None)
    args = ap.parse_args()

    _coh_def = (("p_first", "p_cdrall", "hc_all", "hc_cdrall_or_mmseall")
                if args.kind == "lr_metrics" else DEFAULT_COHORT_TOKENS)
    cohort = (args.p_visit or _coh_def[0], args.p_score or _coh_def[1],
              args.hc_visit or _coh_def[2], args.hc_score or _coh_def[3])
    if args.kind == "lr_metrics":
        out_base = EMBEDDING_CLASSIFICATION_DIR / cohort_path(*cohort) / args.bg_mode / LR_EMB
        for variant, safe, disp in LR_VARIANTS:
            df = lr_load_metrics(cohort, args.bg_mode, variant,
                                 EMBEDDING_CLASSIFICATION_REFACTOR_DIR)
            if df is None:
                logger.warning(f"no metrics for variant={variant}, skip")
                continue
            build_lr_fig(df, disp, out_base / f"lr_metrics_{safe}.png")
        return

    root = args.output_root or EMBEDDING_CLASSIFICATION_REFACTOR_DIR
    csv = root / cohort_path(*cohort) / "all_metrics.csv"
    if not csv.exists():
        ap.error(f"找不到 all_metrics.csv: {csv}（先跑 evaluate/aggregate.py）")
    df = pd.read_csv(csv)
    fwd = df[(df["direction"] == "forward") & (df["domain"] == "1by1")
             & (df["matching_priority"] == args.matching_priority)].copy()
    logger.info(f"all_metrics rows={len(df)}  forward·1by1·{args.matching_priority} rows={len(fwd)}")

    base = root / cohort_path(*cohort) / "_summary"
    for style in args.styles:
        out_dir = base / style
        for mu, eu in COMBOS:
            sub = fwd[(fwd["matched_unit"] == mu) & (fwd["eval_unit"] == eu)]
            if not len(sub):
                logger.info(f"[skip] {style} {mu}-match × {eu}: no rows")
                continue
            out = out_dir / f"{args.matching_priority}_{mu}match_{eu}.png"
            make_figure(sub, out, style)
            logger.info(f"[ok]   {style} {mu}-match × {eu}: {len(sub)} rows -> {out.relative_to(base)}")
    logger.info(f"done -> {base}")


if __name__ == "__main__":
    main()
