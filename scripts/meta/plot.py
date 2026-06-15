"""畫 meta combo 的彙整圖,純讀 cohort 層 all_metrics.csv(run.py + aggregate.py 落地)。

--kind c_curve  : 不同 base-logistic C 下的指標折線,3 metric × 3 contrast。
--kind confusion: 單一 C 的 confusion matrix。
--kind bar      : 固定 9 個 combo 的比較柱狀圖(3 metric × 3 contrast,每格 9 根 bar);影像 combo 取
                  固定 C(--c,預設 0.001)+ --variant;all / 1by1 各一張。輸出 _summary/barplot/…。

--feature-set <name> : 單一 combo。
    c_curve   每格 all vs 1by1 兩線;confusion 2(domain)× 3(contrast)。
--feature-set all    : 8 個 combo 疊一張(影像 combo 固定 --variant;認知 combo 因無 C 畫水平線)。
    all / 1by1 各出一張(共 2 張)。c_curve 每格 8 條 combo 線(3 metric × 3 contrast);
    confusion 8 combo(列)× 3 contrast(欄)。

只有含影像 OOF 的 combo(core4*)有 C 軸;純認知 combo(mmse/casi/mmse_casi)無曲線 / 無 C。

輸出(仿 embedding 分層 _summary):
  c_curve  → _summary/c_curve/<meta_clf>/<variant>/{allcombos_all.png, allcombos_visit_1by1.png}
  confusion→ _summary/confusion_matrix/<meta_clf>/<variant>/C_<c>/{allcombos_all.png, allcombos_visit_1by1.png}
             (confusion 預設每個 C 各一張、各落在自己的 C_<c>/ 資料夾)
  單一 combo:檔名 <feature_set>_<matched_unit>.png(c_curve 另含 .csv)。
用法:
    python scripts/meta/plot.py --kind c_curve   --feature-set all --variant all --matched-unit visit
    python scripts/meta/plot.py --kind confusion --feature-set all --variant all --matched-unit visit  # 每個 C 一張
    python scripts/meta/plot.py --kind confusion --feature-set all --variant all --c 0.001             # 只畫某個 C
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    META_ANALYSIS_DIR, cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
)
from src.meta import ASYM_VARIANTS, META_CLASSIFIERS, META_FEATURE_SETS, feature_set_needs_oof

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CONTRASTS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
METRICS = ["balacc", "auc", "mcc"]
METRIC_LABEL = {"balacc": "Balanced Acc", "auc": "AUC", "mcc": "MCC"}
COMBOS = list(META_FEATURE_SETS)  # 8 combo 的標準順序(認知在前、影像在後)
_IMG = {fs: feature_set_needs_oof(META_FEATURE_SETS[fs]) for fs in COMBOS}
_UNIT_SHORT = {"subject": "subj", "visit": "visit"}
_PRIORITY_SHORT = {"no_priority": "none", "priority_acs": "ACS", "priority_nad": "NAD"}
# domain 選擇:(key, 圖中標籤, matched_unit, matching_priority)。matched_unit/priority 由 args 帶入。
_DOMAINS = ["all", "1by1"]

# --kind bar:固定比較的 9 個 combo(順序 = x 軸;含影像者吃 variant + 單一 C,認知者無)。
BAR_COMBOS = [
    ("core4", "core4"),
    ("core4_bmi", "core4+BMI"),
    ("mmse", "MMSE"),
    ("core4_mmse", "core4+MMSE"),
    ("core4_bmi_mmse", "core4+BMI+MMSE"),
    ("mmse_casi", "MMSE+CASI"),
    ("casi", "CASI"),
    ("core4_casi", "core4+CASI"),
    ("core4_bmi_casi", "core4+BMI+CASI"),
]
CHANCE = {"balacc": 0.5, "auc": 0.5, "mcc": 0.0}
YLIM = {"balacc": (0.4, 1.0), "auc": (0.4, 1.0), "mcc": (-0.1, 0.8)}


def _parse_c(label):
    """clf_param 'C_0.001' → 0.001;非字串(認知 combo 的 NaN)→ nan。"""
    return float(str(label).split("C_", 1)[1]) if isinstance(label, str) else float("nan")


def _domain_tag(domain, args):
    return ("all (full cohort)" if domain == "all"
            else f"1by1 ({_UNIT_SHORT[args.matched_unit]}, {_PRIORITY_SHORT[args.matching_priority]})")


def _pick_domain(df, domain, args):
    """取某 domain 的列(all = 整 cohort;1by1 = 指定 matched_unit + matching_priority)。"""
    if domain == "all":
        return df[df["domain"] == "all"]
    return df[(df["domain"] == "1by1") & (df["matched_unit"] == args.matched_unit)
              & (df["matching_priority"] == args.matching_priority)]


def _summary_dir(base, kind, meta_clf, variant):
    """仿 embedding 的分層 _summary:_summary/<kind>/<meta_clf>/<variant>/(建好回傳)。"""
    d = base / "_summary" / kind / meta_clf / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_am(args, *, reps=False):
    """讀 cohort 層 all_metrics.csv(reps=True 改讀 all_metrics_reps.csv),回 (base_dir, DataFrame)。"""
    base = META_ANALYSIS_DIR / cohort_path(args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    am_path = base / ("all_metrics_reps.csv" if reps else "all_metrics.csv")
    if not am_path.exists():
        raise FileNotFoundError(
            f"找不到 {am_path}\n  請先跑 scripts/meta/run.py + scripts/meta/aggregate.py。")
    return base, pd.read_csv(am_path)


def _slice_one(am, args):
    """單一 combo:依 feature_set + variant + emb/bg/photo/reducer 篩(影像 combo 用)。"""
    sub = am[(am["feature_set"] == args.feature_set) & (am["variant"] == args.variant)
             & (am["emb"] == args.emb) & (am["bg"] == args.bg_mode)
             & (am["photo"] == args.photo_mode) & (am["reducer"] == args.reducer)
             & (am["meta_clf"] == args.meta_clf)].copy()
    if not len(sub):
        raise SystemExit(
            f"all_metrics 篩不到列:feature_set={args.feature_set} variant={args.variant} "
            f"meta_clf={args.meta_clf} emb={args.emb}/{args.bg_mode}/{args.photo_mode}/{args.reducer}。\n"
            f"  確認該 combo 為影像 combo 且已用對應 --asym-variant / --meta-clf 跑過 run.py。")
    return sub


def _slice_all(am, args, variant):
    """all 模式:認知 combo(variant 為空)+ 影像 combo(該 variant),同一 emb/bg/photo/reducer。"""
    env = ((am["emb"] == args.emb) & (am["bg"] == args.bg_mode)
           & (am["photo"] == args.photo_mode) & (am["reducer"] == args.reducer)
           & (am["meta_clf"] == args.meta_clf))
    cog_sets = [fs for fs in COMBOS if not _IMG[fs]]
    img_sets = [fs for fs in COMBOS if _IMG[fs]]
    cog = am[env & am["variant"].isna() & am["feature_set"].isin(cog_sets)]
    img = am[env & (am["variant"] == variant) & am["feature_set"].isin(img_sets)]
    sub = pd.concat([cog, img], ignore_index=True)
    if not len(sub):
        raise SystemExit(
            f"all_metrics 篩不到列:variant={variant} "
            f"emb={args.emb}/{args.bg_mode}/{args.photo_mode}/{args.reducer}。請先跑 run.py + aggregate.py。")
    sub = sub.copy()
    sub["C"] = sub["clf_param"].map(_parse_c)
    return sub


# ---------------------------------------------------------------------------
# c_curve
# ---------------------------------------------------------------------------

def _tidy(df, *, matched_unit, matching_priority):
    """單一 combo 子表 → tidy[C, contrast, domain, metric, value, n]。"""
    df = df.copy()
    df["C"] = df["clf_param"].map(_parse_c)
    rows = []
    for contrast in CONTRASTS:
        picks = {
            "all": df[(df["contrast"] == contrast) & (df["domain"] == "all")],
            "1by1": df[(df["contrast"] == contrast) & (df["domain"] == "1by1")
                       & (df["matched_unit"] == matched_unit)
                       & (df["matching_priority"] == matching_priority)],
        }
        for domain, d in picks.items():
            for _, r in d.iterrows():
                for m in METRICS:
                    rows.append({"C": float(r["C"]), "contrast": contrast,
                                 "domain": domain, "metric": m,
                                 "value": float(r[m]), "n": int(r["n"])})
    return pd.DataFrame(rows)


def _plot_c_curve(tidy, out_png, *, title, matched_label):
    """單一 combo:3 metric × 3 contrast,每格 all vs 1by1 兩線,x=C(log)。"""
    fig, axes = plt.subplots(len(METRICS), len(CONTRASTS),
                             figsize=(13, 11), sharex=True, sharey=True)
    styles = {"all": dict(marker="o", color="#1f77b4", label="all (full cohort)"),
              "1by1": dict(marker="s", color="#d62728", label=matched_label)}
    cs = sorted(tidy["C"].unique())
    for i, m in enumerate(METRICS):
        for j, contrast in enumerate(CONTRASTS):
            ax = axes[i][j]
            for domain, st in styles.items():
                d = tidy[(tidy["contrast"] == contrast) & (tidy["metric"] == m)
                         & (tidy["domain"] == domain)].sort_values("C")
                if len(d):
                    ax.plot(d["C"], d["value"], **st)
            ax.set_xscale("log"); ax.set_xticks(cs)
            ax.set_xticklabels([f"{c:g}" for c in cs]); ax.minorticks_off()
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(contrast, fontsize=12)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[m], fontsize=12)
    axes[0][0].legend(loc="best", fontsize=9)
    fig.supxlabel("C (logistic, log scale)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_c_curve(args, base, am):
    sub = _slice_one(am, args)
    tidy = _tidy(sub, matched_unit=args.matched_unit, matching_priority=args.matching_priority)
    mtag = f"{_UNIT_SHORT[args.matched_unit]}, {_PRIORITY_SHORT[args.matching_priority]}"
    out_dir = _summary_dir(base, "c_curve", args.meta_clf, args.variant)
    stem = f"{args.feature_set}_{args.matched_unit}"
    tidy.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8")
    _plot_c_curve(tidy, out_dir / f"{stem}.png",
                  title=f"meta {args.feature_set} ({args.variant}/{args.meta_clf}) — metric vs C: all vs 1by1({mtag})",
                  matched_label=f"1by1 ({mtag})")
    logger.info(f"wrote {out_dir / stem}.csv + .png ({len(tidy)} rows)")


def _run_c_curve_all(args, base, sub, variant):
    """all 模式:每 domain 一張,8 combo 疊在 3 metric × 3 contrast(影像=線、認知=水平線)。"""
    cs = sorted(c for c in sub["C"].dropna().unique())
    out_dir = _summary_dir(base, "c_curve", args.meta_clf, variant)
    for domain in _DOMAINS:
        dom = _pick_domain(sub, domain, args)
        fig, axes = plt.subplots(len(METRICS), len(CONTRASTS),
                                 figsize=(14, 12), sharex=True, sharey=True)
        handles, labels = [], []
        for ci, fs in enumerate(COMBOS):
            color = plt.cm.tab10(ci % 10)
            dfs = dom[dom["feature_set"] == fs]
            if dfs.empty:
                continue
            for i, m in enumerate(METRICS):
                for j, contrast in enumerate(CONTRASTS):
                    d = dfs[dfs["contrast"] == contrast]
                    if d.empty:
                        continue
                    ax = axes[i][j]
                    if _IMG[fs]:
                        d = d.sort_values("C")
                        ln, = ax.plot(d["C"], d[m], marker="o", ms=4, color=color, label=fs)
                    else:
                        ln = ax.axhline(float(d.iloc[0][m]), color=color, ls="--", lw=1.4, label=fs)
                    if i == 0 and j == 0:
                        handles.append(ln); labels.append(fs)
        for i, m in enumerate(METRICS):
            for j, contrast in enumerate(CONTRASTS):
                ax = axes[i][j]
                ax.set_xscale("log"); ax.set_xticks(cs)
                ax.set_xticklabels([f"{c:g}" for c in cs]); ax.minorticks_off()
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.set_title(contrast, fontsize=12)
                if j == 0:
                    ax.set_ylabel(METRIC_LABEL[m], fontsize=12)
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
        fig.supxlabel("C (logistic, log scale)")
        fig.suptitle(f"meta all combos (imaging={variant}, {args.meta_clf}) — metric vs C @ {_domain_tag(domain, args)}",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0.07, 1, 0.98])
        dpart = "all" if domain == "all" else f"{args.matched_unit}_1by1"
        stem = f"allcombos_{dpart}"
        fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"wrote {out_dir / stem}.png")


# ---------------------------------------------------------------------------
# confusion
# ---------------------------------------------------------------------------

def _cell(df, contrast, domain, matched_unit, matching_priority):
    """從子表挑某 (contrast, domain) 那列,回 tn/fp/fn/tp + sens/spec/auc/n(無則 None)。"""
    if domain == "all":
        r = df[(df["contrast"] == contrast) & (df["domain"] == "all")]
    else:
        r = df[(df["contrast"] == contrast) & (df["domain"] == "1by1")
               & (df["matched_unit"] == matched_unit)
               & (df["matching_priority"] == matching_priority)]
    if not len(r):
        return None
    r = r.iloc[0]
    return dict(tn=int(r["tn"]), fp=int(r["fp"]), fn=int(r["fn"]), tp=int(r["tp"]),
                sens=float(r["sens"]), spec=float(r["spec"]),
                auc=float(r["auc"]), n=int(r["n"]))


def _draw(ax, c, title):
    """畫一格 2×2 confusion matrix(列=True[HC,AD]、欄=Pred[HC,AD]),只標 count。"""
    M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=10,
                color="white" if v > M.max() * 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HC", "AD"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["HC", "AD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)


def _confusion_params(sub, args):
    """要畫的 C(每個自成一層資料夾):--c all → slice 內所有 imaging clf_param(升冪);
    否則 → 指定的那個。純認知 slice(無 clf_param)回 [None](不分 C 層)。"""
    avail = sorted(sub.loc[sub["clf_param"].notna(), "clf_param"].unique(), key=_parse_c)
    if not avail:
        return [None]
    if args.c == "all":
        return avail
    sel = [p for p in avail if abs(_parse_c(p) - float(args.c)) < 1e-12]
    if not sel:
        raise SystemExit(f"找不到 C={args.c} 的 cell(可用 C:{[_parse_c(p) for p in avail]})。")
    return sel


def _run_confusion(args, base, am):
    """單一 combo:每個 C 一張(2 domain × 3 contrast),C 自成一層資料夾。"""
    sub = _slice_one(am, args)
    mtag = f"{_UNIT_SHORT[args.matched_unit]}, {_PRIORITY_SHORT[args.matching_priority]}"
    domains = [("all", "all (full cohort)"), ("1by1", f"1by1 ({mtag})")]
    out_root = _summary_dir(base, "confusion_matrix", args.meta_clf, args.variant)
    for param in _confusion_params(sub, args):
        sub_c = sub[sub["clf_param"] == param] if param else sub
        out_dir = (out_root / param) if param else out_root
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(domains), len(CONTRASTS), figsize=(12, 8))
        for i, (domain, dlabel) in enumerate(domains):
            for j, contrast in enumerate(CONTRASTS):
                ax = axes[i][j]
                c = _cell(sub_c, contrast, domain, args.matched_unit, args.matching_priority)
                if c is None:
                    ax.set_axis_off()
                    ax.set_title(f"{contrast} | {dlabel}\n(無資料)", fontsize=9)
                    continue
                _draw(ax, c, f"{contrast} | {dlabel}\n"
                             f"n={c['n']}  sens={c['sens']:.2f}  spec={c['spec']:.2f}  auc={c['auc']:.2f}")
        fig.suptitle(f"meta {args.feature_set} ({args.variant}/{args.meta_clf}) confusion @ {param or 'n/a'}, "
                     f"thr=0.5 (eval_by_subject, match={mtag})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = out_dir / f"{args.feature_set}_{args.matched_unit}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"wrote {out_png}")


def _run_confusion_all(args, base, sub, variant):
    """all 模式:每 (C, domain) 一張,8 combo(列)× 3 contrast(欄)。C 自成一層資料夾。"""
    out_root = _summary_dir(base, "confusion_matrix", args.meta_clf, variant)
    for param in _confusion_params(sub, args):
        out_dir = (out_root / param) if param else out_root
        out_dir.mkdir(parents=True, exist_ok=True)
        for domain in _DOMAINS:
            fig, axes = plt.subplots(len(COMBOS), len(CONTRASTS),
                                     figsize=(10.5, 2.6 * len(COMBOS)))
            for i, fs in enumerate(COMBOS):
                dfs = sub[sub["feature_set"] == fs]
                if _IMG[fs]:
                    dfs = dfs[dfs["clf_param"] == param]
                for j, contrast in enumerate(CONTRASTS):
                    ax = axes[i][j]
                    c = _cell(dfs, contrast, domain, args.matched_unit, args.matching_priority)
                    if c is None:
                        ax.set_axis_off()
                        ax.set_title(f"{fs} | {contrast}\n(無資料)", fontsize=8)
                        continue
                    _draw(ax, c, f"{fs} | {contrast}\n"
                                 f"n={c['n']} sens={c['sens']:.2f} spec={c['spec']:.2f} auc={c['auc']:.2f}")
            fig.suptitle(f"meta all combos (imaging={variant}, {args.meta_clf}) confusion @ "
                         f"{_domain_tag(domain, args)}, imaging {param or 'n/a'}, thr=0.5", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.99])
            dpart = "all" if domain == "all" else f"{args.matched_unit}_1by1"
            fig.savefig(out_dir / f"allcombos_{dpart}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"wrote {out_dir / ('allcombos_' + dpart)}.png")


# ---------------------------------------------------------------------------
# bar(9-combo 比較)
# ---------------------------------------------------------------------------

def _bar_value(r, m, reps):
    """從某 (combo, contrast, domain) 那列取 (value, lower_err, upper_err)。
    reps=False → (metric, 0, 0);reps=True → (mean, mean−ci_low, ci_high−mean)。無資料 → (nan,0,0)。"""
    if not len(r):
        return float("nan"), 0.0, 0.0
    row = r.iloc[0]
    if not reps:
        return float(row[m]), 0.0, 0.0
    mean = float(row[f"{m}_mean"])
    return mean, mean - float(row[f"{m}_ci_low"]), float(row[f"{m}_ci_high"]) - mean


def _run_bar(args, base, am, variant, *, reps=False):
    """9 combo 比較柱狀圖:每 domain 一張,3 metric(列)× 3 contrast(欄),每格 9 根 bar。

    影像 combo 取固定 C(--c,預設 0.001)+ 指定 variant;認知 combo 無 C/variant(灰色 bar)。
    reps=True:bar=跨 seed mean、誤差棒=95% CI(讀 all_metrics_reps.csv)。
    """
    c_val = 0.001 if args.c == "all" else float(args.c)
    per_combo = {}
    for fs, _ in BAR_COMBOS:
        d = am[(am["feature_set"] == fs) & (am["emb"] == args.emb) & (am["bg"] == args.bg_mode)
               & (am["photo"] == args.photo_mode) & (am["reducer"] == args.reducer)
               & (am["meta_clf"] == args.meta_clf)]
        if _IMG[fs]:
            d = d[(d["variant"] == variant)
                  & d["clf_param"].map(lambda x: isinstance(x, str) and abs(_parse_c(x) - c_val) < 1e-12)]
        per_combo[fs] = d
    labels = [lab for _, lab in BAR_COMBOS]
    colors = ["#4C72B0" if _IMG[fs] else "#7f7f7f" for fs, _ in BAR_COMBOS]
    x = list(range(len(BAR_COMBOS)))
    n_rep = int(am["n_rep"].max()) if reps and "n_rep" in am.columns else 1

    out_dir = _summary_dir(base, "barplot", args.meta_clf, variant) / f"C_{c_val:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for domain in _DOMAINS:
        fig, axes = plt.subplots(len(METRICS), len(CONTRASTS), figsize=(16, 13),
                                 sharex=True, sharey="row")
        for i, m in enumerate(METRICS):
            for j, contrast in enumerate(CONTRASTS):
                ax = axes[i][j]
                vals, los, his = [], [], []
                for fs, _ in BAR_COMBOS:
                    dom = _pick_domain(per_combo[fs], domain, args)
                    v, lo, hi = _bar_value(dom[dom["contrast"] == contrast], m, reps)
                    vals.append(v); los.append(lo); his.append(hi)
                yerr = np.array([los, his]) if reps else None
                ax.bar(x, vals, yerr=yerr, color=colors, width=0.8,
                       error_kw=dict(elinewidth=0.8, capsize=2) if reps else {})
                ax.axhline(CHANCE[m], color="k", ls=":", lw=0.8)
                for xi, v, hi in zip(x, vals, his):
                    if v == v:
                        ax.text(xi, v + (hi if reps else 0), f"{v:.2f}", ha="center",
                                va="bottom", fontsize=6, rotation=90)
                ax.set_ylim(*YLIM[m])
                ax.grid(axis="y", alpha=0.3)
                ax.set_xticks(x)
                ax.set_xticklabels(labels if i == len(METRICS) - 1 else [],
                                   rotation=80, ha="right", fontsize=7)
                if i == 0:
                    ax.set_title(contrast, fontsize=12)
                if j == 0:
                    ax.set_ylabel(METRIC_LABEL[m], fontsize=12)
        mtag = ("all (full cohort)" if domain == "all"
                else f"1by1 ({_UNIT_SHORT[args.matched_unit]}, {_PRIORITY_SHORT[args.matching_priority]})")
        stat = f"mean ± 95% CI over {n_rep} reps" if reps else "thr=0.5"
        fig.suptitle(f"meta 9-combo compare (imaging={variant}, {args.meta_clf}) @ {mtag}, "
                     f"imaging C_{c_val:g}, {stat}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        dpart = "all" if domain == "all" else f"{args.matched_unit}_1by1"
        suffix = "_reps" if reps else ""
        fig.savefig(out_dir / f"9combos_{dpart}{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"wrote {out_dir / ('9combos_' + dpart + suffix)}.png")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["c_curve", "confusion", "bar"], default="c_curve")
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default="p_first")
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default="p_cdrall")
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default="hc_all")
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                    default="hc_cdrall_or_mmseall")
    ap.add_argument("--feature-set", choices=list(META_FEATURE_SETS) + ["all"], default="all",
                    help="單一 combo;或 all=8 combo 疊一張(all / 1by1 各一張)")
    ap.add_argument("--variant", choices=list(ASYM_VARIANTS) + ["all"],
                    default="relative_differences",
                    help="影像 combo 用哪個 asymmetry variant;all=4 種各出一套(僅 --feature-set all)")
    ap.add_argument("--meta-clf", choices=list(META_CLASSIFIERS), default="tabpfn_v3",
                    help="畫哪個 meta stacker 的結果(tabpfn_v3 / xgb)")
    ap.add_argument("--emb", default="arcface")
    ap.add_argument("--bg-mode", default="background")
    ap.add_argument("--photo-mode", default="all")
    ap.add_argument("--reducer", default="no_drop")
    ap.add_argument("--matched-unit", choices=["subject", "visit"], default="visit")
    ap.add_argument("--matching-priority",
                    choices=["no_priority", "priority_acs", "priority_nad"],
                    default="priority_acs")
    ap.add_argument("--c", default="all",
                    help="[confusion] 畫哪些 C:all=每個 C 一張(各落在 C_<c>/ 資料夾);或指定單一數值如 0.001")
    ap.add_argument("--reps", action="store_true",
                    help="[bar] 讀 all_metrics_reps.csv,bar=mean、誤差棒=跨 seed 95%% CI(repeated-CV)")
    args = ap.parse_args()

    base, am = _load_am(args, reps=args.reps and args.kind == "bar")
    variants = list(ASYM_VARIANTS) if args.variant == "all" else [args.variant]

    if args.kind == "bar":
        for v in variants:
            _run_bar(args, base, am, v, reps=args.reps)
        return

    combined = args.feature_set == "all"
    if args.variant == "all" and not combined:
        ap.error("--variant all 只用於 --feature-set all 或 --kind bar;單一 combo 請指定單一 variant。")
    if not combined:
        (_run_c_curve if args.kind == "c_curve" else _run_confusion)(args, base, am)
        return
    for v in variants:
        sub = _slice_all(am, args, v)
        if args.kind == "c_curve":
            _run_c_curve_all(args, base, sub, v)
        else:
            _run_confusion_all(args, base, sub, v)


if __name__ == "__main__":
    main()
