"""
scripts/emo_au/plot_group_compare.py

emo_au 特徵的「族群統計與比較」violin 網格(類比 scripts/age/error/violin.py，但
emo_au 是高維特徵，故用網格而非單圖)。

每張圖:列 = 特徵、欄 = 模型，每個有資料的格子畫 g 把 violin(AD=P / NAD / ACS)，
格上標 Kruskal–Wallis q(該圖內 BH-FDR 校正)+ 各族群 n；每列共用 y 軸。模型沒有該
特徵 → 該格留白。三張圖(各再出 full + 1:1 配齡兩版):
  F1_emotions  情緒 7 列 × 至多 9 模型
  F2_au        AU(強度 + 偵測) × {openface, libreface, pyfeat}
  F3_extra     valence/arousal/gaze/contempt × {emonet, openface}

每人一個值 = 該特徵欄「跨幀 nanmean」(eval unit = subject)。AU 欄名去前導零歸併
(AU01→AU1)，使同一 AU 在不同模型落在同一列；_det(偵測)維持獨立列。

Usage:
  conda run -n Alz_face_main_analysis python scripts/emo_au/plot_group_compare.py
  conda run -n Alz_face_main_analysis python scripts/emo_au/plot_group_compare.py \
      --features-dir <dir> --variants full --p-visit p_first ...
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    EMO_AU_FEATURES_DIR, EMO_AU_ANALYSIS_DIR, cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.common.cohort import cohort_list
from src.common.matching import match_by_age
from src.emo_au.extractor.au_config import HARMONIZED_EMOTIONS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# (group code in cohort, display label, violin color)
GROUPS = [("P", "AD", "#C44E52"), ("NAD", "NAD", "#4C72B0"), ("ACS", "ACS", "#55A868")]
MODEL_ORDER = ["openface", "libreface", "pyfeat", "dan", "hsemotion", "vit",
               "poster_pp", "fer", "emonet"]
EXTRA_ROWS = ["contempt", "valence", "arousal", "gaze_pitch", "gaze_yaw"]
_AU_RE = re.compile(r"AU0*(\d+)(_det)?$")


# ── tiny stats:BH-FDR 就地內聯（單一教科書公式，不值得為它引入共用統計模組）─────────
def bh_fdr(pvals):
    """Benjamini–Hochberg FDR:把 p 值陣列轉成單調遞增的 q 值（NaN 原樣保留）。"""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    res = np.empty_like(q)
    res[order] = q
    out[ok] = res
    return out


def _stars(q):
    if np.isnan(q):
        return ""
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""


def _blank(ax):
    """空格:藏掉框線與刻度，但保留 ax(這樣列標/欄名仍可掛上去)。"""
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


# ── feature naming ────────────────────────────────────────────────────────────
def _norm(col: str) -> str:
    """AU01→AU1（去前導零）；_det 與其他欄原樣。同一 AU 跨模型併同列。"""
    m = _AU_RE.fullmatch(col)
    return f"AU{int(m.group(1))}{m.group(2) or ''}" if m else col


def _au_sort_key(r):
    m = _AU_RE.fullmatch(r)
    return (int(m.group(1)), 1 if m.group(2) else 0)


# ── data loading ──────────────────────────────────────────────────────────────
def _load_schema(features_dir: Path) -> dict:
    f = features_dir / "_schema.json"
    if not f.exists():
        raise FileNotFoundError(f"找不到 schema: {f}（先跑 extract.py）")
    return json.loads(f.read_text(encoding="utf-8"))


def _load_model_means(features_dir: Path, model: str, ids) -> pd.DataFrame:
    """每位受試者一列、欄 = 該模型實際欄名、值 = 該欄跨幀 nanmean。讀不到的受試者略過。"""
    mdir = features_dir / model
    rows = {}
    for sid in ids:
        p = mdir / f"{sid}.npz"
        if not p.exists():
            continue
        try:
            z = np.load(p, allow_pickle=False)
            data, cols = z["data"], [str(c) for c in z["columns"]]
        except Exception:
            continue
        if data.ndim != 2 or data.shape[1] != len(cols):
            continue
        with np.errstate(all="ignore"):
            means = np.nanmean(data, axis=0)
        rows[sid] = dict(zip(cols, means))
    return pd.DataFrame.from_dict(rows, orient="index")


# ── drawing ───────────────────────────────────────────────────────────────────
def _cell_groups(series: pd.Series, id_group: dict, keep_ids: set):
    """series(index=ID,值=該特徵 per-subject mean) → 各族群的值陣列(依 GROUPS 序)。"""
    out = []
    for code, _, _ in GROUPS:
        vals = [v for sid, v in series.items()
                if id_group.get(sid) == code and sid in keep_ids and np.isfinite(v)]
        out.append(np.asarray(vals, dtype=float))
    return out


def _kw_p(arrays):
    """≥2 個非空族群且整體有變異 → Kruskal–Wallis p；否則 NaN。"""
    nonempty = [a for a in arrays if len(a) >= 1]
    if len(nonempty) < 2 or sum(len(a) for a in nonempty) < 3:
        return np.nan
    if np.allclose(np.concatenate(nonempty), np.concatenate(nonempty)[0]):
        return np.nan
    try:
        return float(stats.kruskal(*nonempty).pvalue)
    except ValueError:
        return np.nan


def build_figure(fig_name, subtitle, rows, means_by_model, normmap, id_group,
                 keep_ids, out_path):
    models = [m for m in MODEL_ORDER
              if m in means_by_model and any(r in normmap[m] for r in rows)]
    rows = [r for r in rows if any(r in normmap[m] for m in models)]
    if not models or not rows:
        logger.warning(f"{fig_name}: 無資料可畫，略過")
        return

    # pass 1: 收集每格 KW p + 該格的族群陣列(供 pass 2 直接畫)，再做圖內 BH-FDR。
    cell, pvals = {}, []
    for r in rows:
        for m in models:
            actual = normmap[m].get(r)
            if actual is None or actual not in means_by_model[m].columns:
                continue
            arrs = _cell_groups(means_by_model[m][actual], id_group, keep_ids)
            p = _kw_p(arrs)
            cell[(r, m)] = (arrs, p)
            pvals.append(((r, m), p))
    if pvals:
        qs = bh_fdr([p for _, p in pvals])
        qmap = {k: q for (k, _), q in zip(pvals, qs)}
    else:
        qmap = {}

    nrow, ncol = len(rows), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol + 1.2, 2.1 * nrow + 1.0),
                             squeeze=False)
    for i, r in enumerate(rows):
        # 該列共用 y 軸範圍
        allv = np.concatenate([cell[(r, m)][0][k]
                               for m in models if (r, m) in cell
                               for k in range(len(GROUPS)) if len(cell[(r, m)][0][k])]
                              ) if any((r, m) in cell for m in models) else np.array([])
        if allv.size:
            lo, hi = float(np.min(allv)), float(np.max(allv))
            pad = (hi - lo) * 0.12 or 0.05
            ylim = (lo - pad, hi + pad)
        else:
            ylim = None
        for j, m in enumerate(models):
            ax = axes[i][j]
            # 列標 / 欄名獨立於該格有無資料（否則首欄/首列為空時會掉標籤）
            if i == 0:
                ax.annotate(m, xy=(0.5, 1.28), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(r, fontsize=8, rotation=0, ha="right", va="center")
            if (r, m) not in cell:
                _blank(ax)
                continue
            arrs, _ = cell[(r, m)]
            q = qmap.get((r, m), np.nan)
            for k, (code, label, color) in enumerate(GROUPS):
                a = arrs[k]
                pos = k + 1
                if len(a) >= 2 and np.ptp(a) > 0:
                    pc = ax.violinplot([a], positions=[pos], showmedians=True,
                                       widths=0.8)
                    for body in pc["bodies"]:
                        body.set_facecolor(color); body.set_alpha(0.6)
                    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                        if key in pc:
                            pc[key].set_color("gray"); pc[key].set_linewidth(0.8)
                elif len(a) >= 1:
                    ax.scatter([pos], [np.mean(a)], s=12, color=color)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels([f"{lab}\n{len(arrs[k])}"
                                for k, (_, lab, _) in enumerate(GROUPS)], fontsize=6)
            if ylim:
                ax.set_ylim(*ylim)
            ax.tick_params(axis="y", labelsize=6)
            qtxt = "n/a" if np.isnan(q) else f"q={q:.2g}{_stars(q)}"
            ax.set_title(qtxt, fontsize=7,
                         color=("black" if (not np.isnan(q) and q < .05) else "gray"))

    fig.suptitle(subtitle, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out_path}")


def build_emotion_splits(subtitle_prefix, emotions, means_by_model, normmap,
                         id_group, keep_ids, out_dir):
    """把 F1 情緒圖按列拆成每情緒一張(1 列 × 模型),輸出 out_dir/<emotion>.png。
    q 值沿用 F1 全體(7 情緒 × 模型)的 BH-FDR，與合併圖一致；每張內模型共用 y。"""
    models = [m for m in MODEL_ORDER
              if m in means_by_model and any(e in normmap[m] for e in emotions)]
    emotions = [e for e in emotions if any(e in normmap[m] for m in models)]
    if not models or not emotions:
        return
    # pass 1: 收集每格族群陣列 + KW p，FDR 範圍 = 全部情緒格(等同 F1)
    cell, pvals = {}, []
    for e in emotions:
        for m in models:
            actual = normmap[m].get(e)
            if actual is None or actual not in means_by_model[m].columns:
                continue
            arrs = _cell_groups(means_by_model[m][actual], id_group, keep_ids)
            cell[(e, m)] = arrs
            pvals.append(((e, m), _kw_p(arrs)))
    qmap = ({k: q for (k, _), q in zip(pvals, bh_fdr([p for _, p in pvals]))}
            if pvals else {})

    out_dir.mkdir(parents=True, exist_ok=True)
    ncol = len(models)
    for e in emotions:
        allv = np.concatenate([cell[(e, m)][k] for m in models if (e, m) in cell
                               for k in range(len(GROUPS)) if len(cell[(e, m)][k])]
                              ) if any((e, m) in cell for m in models) else np.array([])
        if allv.size:
            lo, hi = float(np.min(allv)), float(np.max(allv))
            pad = (hi - lo) * 0.12 or 0.05
            ylim = (lo - pad, hi + pad)
        else:
            ylim = None
        fig, axes = plt.subplots(1, ncol, figsize=(2.0 * ncol + 1.2, 3.2), squeeze=False)
        for j, m in enumerate(models):
            ax = axes[0][j]
            ax.annotate(m, xy=(0.5, 1.12), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            if (e, m) not in cell:
                _blank(ax)
                continue
            arrs = cell[(e, m)]
            q = qmap.get((e, m), np.nan)
            for k, (code, label, color) in enumerate(GROUPS):
                a = arrs[k]
                pos = k + 1
                if len(a) >= 2 and np.ptp(a) > 0:
                    pc = ax.violinplot([a], positions=[pos], showmedians=True, widths=0.8)
                    for body in pc["bodies"]:
                        body.set_facecolor(color); body.set_alpha(0.6)
                    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                        if key in pc:
                            pc[key].set_color("gray"); pc[key].set_linewidth(0.8)
                elif len(a) >= 1:
                    ax.scatter([pos], [np.mean(a)], s=12, color=color)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels([f"{lab}\n{len(arrs[k])}"
                                for k, (_, lab, _) in enumerate(GROUPS)], fontsize=6)
            if ylim:
                ax.set_ylim(*ylim)
            ax.tick_params(axis="y", labelsize=6)
            qtxt = "n/a" if np.isnan(q) else f"q={q:.2g}{_stars(q)}"
            ax.set_title(qtxt, fontsize=7,
                         color=("black" if (not np.isnan(q) and q < .05) else "gray"))
        fig.suptitle(f"{e} — {subtitle_prefix}", fontsize=11, y=1.02)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(str(out_dir / f"{e}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"saved {out_dir / f'{e}.png'}")


def _figures(normmap):
    """依跨模型欄位聯集(normalized)切成三家族的(fig_name, 說明, 列)。"""
    union = set().union(*[set(nm) for nm in normmap.values()]) if normmap else set()
    emotions = [e for e in HARMONIZED_EMOTIONS if e in union]
    au_rows = sorted([r for r in union if _AU_RE.fullmatch(r)], key=_au_sort_key)
    extra = [x for x in EXTRA_ROWS if x in union]
    return [
        ("F1_emotions", "emotion (per-subject mean)", emotions),
        ("F2_au", "AU intensity / detection (per-subject mean)", au_rows),
        ("F3_extra", "valence / arousal / gaze / contempt (per-subject mean)", extra),
    ]


def _roster_from_config():
    """各模型欄位(canonical 序)——純由 au_config 重建，不需 extract 結果、不載模型，
    供 --skeleton 版面預覽。libreface 欄表內嵌(其 maps 在會載 torch 的模組內)。"""
    from src.emo_au.extractor.au_config import (
        canonical_order, OPENFACE_AU_INDEX, OPENFACE_EMOTION_INDEX,
        OPENFACE_GAZE_COLUMNS, PYFEAT_AU_MAP, PYFEAT_EMOTION_MAP,
        POSTER_PP_EMOTION_INDEX)
    E = list(HARMONIZED_EMOTIONS)
    libreface = (["AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU17",
                  "AU20", "AU25", "AU26"]
                 + ["AU1_det", "AU2_det", "AU4_det", "AU6_det", "AU7_det", "AU10_det",
                    "AU12_det", "AU14_det", "AU15_det", "AU17_det", "AU23_det", "AU24_det"]
                 + E)
    raw = {
        "openface": (list(OPENFACE_AU_INDEX.values())
                     + list(OPENFACE_EMOTION_INDEX.values())
                     + list(OPENFACE_GAZE_COLUMNS)),
        "libreface": libreface,
        "pyfeat": list(PYFEAT_AU_MAP.keys()) + list(PYFEAT_EMOTION_MAP.keys()),
        "dan": E, "hsemotion": E, "vit": E,
        "poster_pp": list(POSTER_PP_EMOTION_INDEX.values()), "fer": E,
        "emonet": E + ["valence", "arousal"],
    }
    return {m: canonical_order(c) for m, c in raw.items()}


def build_skeleton(fig_name, subtitle, rows, normmap, out_path):
    """空白版面預覽:每個有資料的格畫淡色三槽(AD/NAD/ACS 該放 violin 的位置)，
    模型沒有的特徵留白。不需任何特徵檔。"""
    models = [m for m in MODEL_ORDER if m in normmap and any(r in normmap[m] for r in rows)]
    rows = [r for r in rows if any(r in normmap[m] for m in models)]
    if not models or not rows:
        return
    fig, axes = plt.subplots(len(rows), len(models),
                             figsize=(2.0 * len(models) + 1.2, 2.1 * len(rows) + 1.0),
                             squeeze=False)
    for i, r in enumerate(rows):
        for j, m in enumerate(models):
            ax = axes[i][j]
            if i == 0:
                ax.annotate(m, xy=(0.5, 1.28), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(r, fontsize=8, rotation=0, ha="right", va="center")
            if r not in normmap[m]:
                _blank(ax)
                continue
            ax.bar([1, 2, 3], [1, 1, 1], color=[g[2] for g in GROUPS],
                   alpha=0.18, width=0.7)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels([g[1] for g in GROUPS], fontsize=6)
            ax.set_yticks([]); ax.set_ylim(0, 1.25); ax.set_xlim(0.4, 3.6)
    fig.suptitle(subtitle, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])
    ap.add_argument("--features-dir", type=Path, default=EMO_AU_FEATURES_DIR)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="覆寫輸出目錄；留空依 cohort 自動決定")
    ap.add_argument("--variants", nargs="+", default=["full", "1by1matched"],
                    choices=["full", "1by1matched"])
    ap.add_argument("--caliper", type=float, default=1.0)
    ap.add_argument("--skeleton", action="store_true",
                    help="只畫空白版面預覽(不需 extract 結果/cohort)")
    args = ap.parse_args()

    if args.skeleton:
        normmap = {m: {_norm(c): c for c in cols}
                   for m, cols in _roster_from_config().items()}
        out = args.output_dir or (
            EMO_AU_ANALYSIS_DIR / "_layout_preview")
        logger.info(f"版面預覽(空白) → {out}")
        for fig_name, fam, rows in _figures(normmap):
            build_skeleton(
                fig_name,
                f"emo_au {fig_name} — LAYOUT PREVIEW (blank, no data)\n"
                f"{len(rows)} feature rows x models; each cell = AD/NAD/ACS slots "
                f"(faded placeholders), blank where model lacks the feature | {fam}",
                rows, normmap, out / f"{fig_name}.png")
        return

    tokens = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    out_base = args.output_dir or (
        EMO_AU_ANALYSIS_DIR / cohort_path(*tokens) / "group_compare")
    logger.info(f"cohort = {tokens}")
    logger.info(f"features-dir = {args.features_dir}")
    logger.info(f"output-dir   = {out_base}")

    cohort = cohort_list(*tokens)
    id_group = dict(zip(cohort["ID"], cohort["Group"]))
    ids = list(cohort["ID"])
    logger.info(f"cohort: {len(ids)} 人 {cohort['Group'].value_counts().to_dict()}")

    schema = _load_schema(args.features_dir)
    methods = schema.get("methods", {})
    normmap = {m: {_norm(c): c for c in methods[m]["columns"]} for m in methods}
    logger.info(f"schema methods: {list(methods)} | bg={schema.get('bg_variant')}")

    means_by_model = {}
    for m in MODEL_ORDER:
        if m not in methods:
            continue
        df = _load_model_means(args.features_dir, m, ids)
        if not df.empty:
            means_by_model[m] = df
            logger.info(f"  {m}: {df.shape[0]} 人有特徵")
    if not means_by_model:
        logger.error("沒有任何模型的特徵檔，先跑 extract.py")
        return

    figures = _figures(normmap)  # 列(特徵)集合依跨模型欄位聯集切三家族

    for variant in args.variants:
        if variant == "full":
            keep_ids = set(ids)
        else:
            p_ids, hc_ids = match_by_age(*tokens, priority=["ACS"], caliper=args.caliper)
            keep_ids = set(p_ids) | set(hc_ids)
        n_by = pd.Series({c: sum(1 for s in keep_ids if id_group.get(s) == c)
                          for c, _, _ in GROUPS})
        logger.info(f"--- variant={variant} | n={n_by.to_dict()} ---")
        for fig_name, fam_label, rows in figures:
            sub = (f"emo_au group comparison — {fig_name} ({variant})\n"
                   f"cohort={tokens}  AD={n_by['P']} NAD={n_by['NAD']} ACS={n_by['ACS']}"
                   f"  | {fam_label}; KW q (BH-FDR within figure)")
            build_figure(fig_name, sub, rows, means_by_model, normmap, id_group,
                         keep_ids, out_base / variant / f"{fig_name}.png")

        # F1 情緒圖另外按情緒拆成每張一情緒 → <variant>/emotions/<emotion>.png
        emo_rows = next((r for fn, _, r in figures if fn == "F1_emotions"), [])
        if emo_rows:
            build_emotion_splits(
                f"emotion (per-subject mean), {variant} | "
                f"AD={n_by['P']} NAD={n_by['NAD']} ACS={n_by['ACS']} | "
                f"KW q (BH-FDR within emotions)",
                emo_rows, means_by_model, normmap, id_group, keep_ids,
                out_base / variant / "emotions")


if __name__ == "__main__":
    main()
