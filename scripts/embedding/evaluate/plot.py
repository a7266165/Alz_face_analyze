"""繪製 evaluation 圖。

--kind summary    建 _summary/<eval_unit>/ 結構：box、violin（讀 all_metrics.csv，跨 embedding）
                  + confusion_matrix、combo_metrics（讀 arcface seed_0 per-cell）。
--kind lr_metrics ArcFace LogReg on asymmetry 的 metrics-vs-C 圖。
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


# ── _summary 結構：confusion_matrix（per variant×C）+ combo_metrics（LR/XGB per variant）─────
# 讀 arcface 的 per-cell seed_0 forward metrics.csv（與 box/violin 讀 all_metrics.csv 不同來源）。
SUMMARY_EMB = "arcface"
SUMMARY_VARIANTS = ["differences", "relative_differences", "absolute_differences",
                    "absolute_relative_differences", "original"]
SUMMARY_CGRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SUMMARY_METRICS = [("balacc", "Balanced ACC"), ("auc", "AUC"), ("mcc", "MCC")]
CM_CONTRASTS = [("ad_vs_hc", "AD", "HC"), ("ad_vs_nad", "AD", "NAD"), ("ad_vs_acs", "AD", "ACS")]
SUMMARY_LINE_COLOR = {"ad_vs_hc": "#4C72B0", "ad_vs_nad": "#55A868", "ad_vs_acs": "#C44E52"}


def _sum_cell(base, variant, model, seg):
    """seed_0 forward 的 metrics.csv 路徑；seg = C_<c>（logistic）或 xgb config 名。"""
    return base / variant / "all" / "no_drop" / model / "seed_0" / seg / "fwd" / "metrics.csv"


def _sum_read(path, eval_unit):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["eval_unit"] == eval_unit]


def _sum_pick(df, filt, contrast):
    sub = df[df["contrast"] == contrast]
    for k, v in filt.items():
        sub = sub[sub[k] == v]
    return sub.iloc[0] if len(sub) else None


def _sum_slices(priority):
    return [("full", {"domain": "all"}),
            ("1by1 matched", {"domain": "1by1", "matched_unit": "visit",
                              "matching_priority": priority})]


def _xgb_segs(base, variant):
    """xgb seed_0 的 config 子目錄名，依 (ne, md, lr) 排序。回 [(short_label, seg_name)]。"""
    d = base / variant / "all" / "no_drop" / "xgb" / "seed_0"
    if not d.exists():
        return []
    segs = sorted((p.name for p in d.iterdir() if p.is_dir()),
                  key=lambda n: (int(n.split("_")[1]), int(n.split("_")[3]), float(n.split("_")[5])))
    return [("/".join(s.split("_")[i] for i in (1, 3, 5)), s) for s in segs]


def build_cm_fig(base, variant, c, eval_unit, priority, out_path):
    """2 列(full / 1by1-matched) × 3 欄(AD vs HC/NAD/ACS) 的 logistic 混淆矩陣（該 C、eval_unit）。"""
    df = _sum_read(_sum_cell(base, variant, "logistic", f"C_{c}"), eval_unit)
    if df is None:
        return
    slices = _sum_slices(priority)
    fig, axes = plt.subplots(len(slices), len(CM_CONTRASTS), figsize=(11, 7.2))
    for ri, (slabel, sfilt) in enumerate(slices):
        for ci, (ckey, pos, neg) in enumerate(CM_CONTRASTS):
            ax = axes[ri][ci]
            r = _sum_pick(df, sfilt, ckey)
            if r is None:
                ax.text(0.5, 0.5, "NA", ha="center", va="center"); ax.axis("off"); continue
            cm = np.array([[int(r.tp), int(r.fn)], [int(r.fp), int(r.tn)]], float)
            rp = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(rp, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{int(cm[i, j])}\n({rp[i, j]*100:.0f}%)", ha="center",
                            va="center", fontsize=11, color="white" if rp[i, j] > 0.55 else "black")
            ax.set_xticks([0, 1]); ax.set_xticklabels([f"pred {pos}", f"pred {neg}"], fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels([f"act {pos}", f"act {neg}"], fontsize=8)
            ax.set_title(f"{pos} vs {neg} - {slabel}\nn={int(r.n)}  balacc={r.balacc:.3f}  "
                         f"AUC={r.auc:.3f}  MCC={r.mcc:.3f}", fontsize=9)
    fig.suptitle(f"ArcFace - LogReg (C={c}, seed_0, {eval_unit}) - {variant} confusion matrices\n"
                 f"rows: full / 1by1 matched(visit-{priority})   cols: AD vs HC / NAD / ACS",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)


def build_combo_fig(variant, model, series, xlabel, eval_unit, priority, out_path):
    """3 列(balacc/AUC/MCC) × 2 欄(full/matched)，3 對比線 vs 超參數。series=[(xlabel, metrics_df)]。"""
    slices = _sum_slices(priority)
    xs = list(range(len(series)))
    fig, axes = plt.subplots(len(SUMMARY_METRICS), len(slices), figsize=(13, 11),
                             sharex=True, sharey="row")
    for ri, (mkey, mlabel) in enumerate(SUMMARY_METRICS):
        for cidx, (slabel, sfilt) in enumerate(slices):
            ax = axes[ri][cidx]
            for ckey, pos, neg in CM_CONTRASTS:
                ys = []
                for _x, df in series:
                    r = _sum_pick(df, sfilt, ckey) if df is not None else None
                    ys.append(np.nan if r is None else float(r[mkey]))
                ax.plot(xs, ys, marker="o", ms=4, lw=1.6,
                        color=SUMMARY_LINE_COLOR[ckey], label=f"{pos} vs {neg}")
            ax.axhline(0.5 if mkey in ("balacc", "auc") else 0.0, ls="--", lw=1, color="0.6")
            ax.grid(True, ls=":", lw=0.5, alpha=0.5)
            if ri == 0:
                ax.set_title(slabel, fontsize=12, fontweight="bold")
            if cidx == 0:
                ax.set_ylabel(mlabel, fontsize=12, fontweight="bold")
            if ri == len(SUMMARY_METRICS) - 1:
                ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in series], rotation=90, fontsize=6)
                ax.set_xlabel(xlabel, fontsize=10)
    axes[0][1].legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle(f"ArcFace - {model} - {variant} (seed_0, {eval_unit}) metrics vs {xlabel}\n"
                 f"cols: full / 1by1 matched(visit-{priority})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)


def build_summary_grids(cohort, bg_mode, priority, sum_base):
    """為 eval_by_subject / eval_by_visit 各產 confusion_matrix + combo_metrics（arcface、seed_0）。"""
    base = EMBEDDING_CLASSIFICATION_DIR / cohort_path(*cohort) / bg_mode / SUMMARY_EMB
    for eu in ("eval_by_subject", "eval_by_visit"):
        for variant in SUMMARY_VARIANTS:
            for c in SUMMARY_CGRID:
                build_cm_fig(base, variant, c, eu, priority,
                             sum_base / eu / "confusion_matrix" / f"C_{c}" / f"{variant}.png")
            lr_series = [(str(c), _sum_read(_sum_cell(base, variant, "logistic", f"C_{c}"), eu))
                         for c in SUMMARY_CGRID]
            build_combo_fig(variant, "LogReg", lr_series, "C", eu, priority,
                            sum_base / eu / "combo_metrics" / "LR" / f"{variant}.png")
            xsegs = _xgb_segs(base, variant)
            if xsegs:
                xseries = [(lbl, _sum_read(_sum_cell(base, variant, "xgb", seg), eu)) for lbl, seg in xsegs]
                build_combo_fig(variant, "XGBoost", xseries, "XGB grid (ne/md/lr)", eu, priority,
                                sum_base / eu / "combo_metrics" / "XGB" / f"{variant}.png")
        logger.info(f"[summary grids] {eu}: confusion_matrix + combo_metrics done")


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
                    help="summary=建 _summary/<eval_unit>/{box,violin,confusion_matrix,combo_metrics}；"
                         "lr_metrics=ArcFace LogReg on asymmetry 的 metrics-vs-C 圖")
    ap.add_argument("--bg-mode", choices=["background", "no_background"], default="background",
                    help="confusion_matrix/combo_metrics（summary）與 lr_metrics 用的 bg")
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
    # box/violin：eval_unit 改為資料夾層（_summary/<eu>/<style>/），與 confusion_matrix/combo_metrics 一致
    for style in args.styles:
        for mu, eu in COMBOS:
            sub = fwd[(fwd["matched_unit"] == mu) & (fwd["eval_unit"] == eu)]
            if not len(sub):
                logger.info(f"[skip] {style} {mu}-match × {eu}: no rows")
                continue
            out = base / eu / style / f"{args.matching_priority}_{mu}match.png"
            make_figure(sub, out, style)
            logger.info(f"[ok]   {style} {mu}-match × {eu}: {len(sub)} rows -> {out.relative_to(base)}")
    # confusion_matrix + combo_metrics（arcface、seed_0；兩個 eval_unit）
    build_summary_grids(cohort, args.bg_mode, args.matching_priority, base)
    logger.info(f"done -> {base}")


if __name__ == "__main__":
    main()
