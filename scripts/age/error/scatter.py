"""年齡預測散點圖，分 ACS / NAD / P 三組（兩面板：HC = NAD+ACS vs Patient）
—— 完整 cohort 與 AD-vs-HC 1:1 年齡配對子集。

輸出於 <AGE_ANALYSIS_DIR>/<cohort>/scatter/{full,1by1matched}/：
  predicted_ages_scatter.png  —— 真實 vs 預測年齡
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    AGE_ANALYSIS_DIR,
    cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.age.utils import build_cohort_with_age_error
from src.common.matching import match_by_age

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── 面板小工具 ────────────────────────────────────────────────────────────

def _draw_panel(ax, df, title, colors, labels,
                x_col="real_age", y_col="predicted_age",
                x_label="Real Age", y_label="Predicted Age (MiVOLO)",
                reverse_line=False):
    """散點 + y=x + OLS 迴歸線。橘線一律 y_col 對 x_col 回歸（換軸時自動重定義）。

    reverse_line=True 時再疊一條紫虛線＝反方向 OLS（x_col 對 y_col），代數轉到本座標
    畫成 y=…。換軸面板用它顯示「舊迴歸關係座標轉換後」那條線（＝ pred 對 real）。
    r（對稱）、MAE=|real−predicted|（對稱）不受換軸影響；只有座標與迴歸線改變。
    """
    for grp, color in colors.items():
        sub = df[df["group"] == grp]
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub[y_col],
                   c=color, label=labels.get(grp, grp),
                   alpha=0.6, s=30, edgecolors="white", linewidth=0.3)

    age_min, age_max = 25, 110  # 固定軸範圍，供跨 cohort/版本比較
    xs = np.array([age_min, age_max])
    ax.plot([age_min, age_max], [age_min, age_max],
            "k--", alpha=0.5, linewidth=1, label="y = x")

    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ss_xx = float(np.sum((x - x.mean()) ** 2))
    if ss_xx > 0:
        a = float(np.sum((x - x.mean()) * (y - y.mean()))) / ss_xx
        b = float(y.mean() - a * x.mean())
        ax.plot(xs, a * xs + b, color="#FF9800", linewidth=2, alpha=0.8,
                label=f"y = {a:.2f}x + {b:.2f}")

    # 反方向 OLS（x 對 y，最小化水平殘差），轉到本座標畫成 y=(1/a')x - b'/a'
    if reverse_line:
        ss_yy = float(np.sum((y - y.mean()) ** 2))
        if ss_yy > 0:
            a_r = float(np.sum((x - x.mean()) * (y - y.mean()))) / ss_yy
            b_r = float(x.mean() - a_r * y.mean())
            if abs(a_r) > 1e-9:
                inv_a, inv_b = 1.0 / a_r, -b_r / a_r
                ax.plot(xs, inv_a * xs + inv_b, color="#7B1FA2", linewidth=2,
                        alpha=0.9, linestyle=(0, (6, 4)),
                        label=f"y = {inv_a:.2f}x {inv_b:+.2f}  (old fit, axes swapped)")

    n = len(df)
    r = df[x_col].corr(df[y_col])
    mae = df["error"].abs().mean()
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f"{title}\n(n={n}, r={r:.3f}, MAE={mae:.1f})", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(age_min, age_max)
    ax.set_ylim(age_min, age_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

# ── 散點圖 ────────────────────────────────────────────────────────────

def plot_main_scatter(df, scatter_dir, note=""):
    df_hc = df[df["group"].isin(["ACS", "NAD"])]
    df_p = df[df["group"] == "P"]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
    _draw_panel(ax_l, df_hc, f"Healthy Controls (NAD + ACS){note}",
                {"NAD": "#2196F3", "ACS": "#4CAF50"},
                {"NAD": "NAD", "ACS": "ACS"})
    _draw_panel(ax_r, df_p, f"Patients (P){note}",
                {"P": "#F44336"}, {"P": "Patient"})
    plt.tight_layout()
    out = scatter_dir / "predicted_ages_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out}")

# ── 分組面板散點（兩種座標方向各出一份） ─────────────────────────────────

GROUP_COLORS = {"P": "#F44336", "NAD": "#2196F3", "ACS": "#4CAF50"}
GROUP_LABELS = {"P": "Patient", "NAD": "SCD", "ACS": "ACS"}   # NAD 顯示為 SCD
GROUP_TITLES = {"P": "Patients (P)", "NAD": "SCD", "ACS": "ACS"}
# 兩種座標方向：realx=傳統(X 真實)；predx=換軸(X 預測)。迴歸線一律 y 對 x 回歸。
_ORIENT = {
    "realx": dict(x_col="real_age", y_col="predicted_age",
                  x_label="Real Age", y_label="Predicted Age (MiVOLO)"),
    "predx": dict(x_col="predicted_age", y_col="real_age",
                  x_label="Predicted Age (MiVOLO)", y_label="Real Age"),
}


def _group_panel(ax, df, grp, orient, note=""):
    _draw_panel(ax, df[df["group"] == grp], f"{GROUP_TITLES[grp]}{note}",
                {grp: GROUP_COLORS[grp]}, {grp: GROUP_LABELS[grp]},
                reverse_line=(orient == "predx"), **_ORIENT[orient])


def _all_panel(ax, df, orient, note=""):
    _draw_panel(ax, df, f"All (P + SCD + ACS){note}",
                GROUP_COLORS, GROUP_LABELS,
                reverse_line=(orient == "predx"), **_ORIENT[orient])


def _save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out_path}")


def plot_grid4(df, scatter_dir, orient, note=""):
    """2×2：左上 All、右上 P、左下 SCD、右下 ACS。orient∈{realx,predx}。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    _all_panel(axes[0, 0], df, orient, note)
    _group_panel(axes[0, 1], df, "P", orient, note)
    _group_panel(axes[1, 0], df, "NAD", orient, note)
    _group_panel(axes[1, 1], df, "ACS", orient, note)
    fig.tight_layout()
    _save(fig, scatter_dir / f"grid4_scatter_{orient}.png")


def plot_pairs(df, scatter_dir, orient, note=""):
    """三對六格：P vs SCD、P vs ACS、SCD vs ACS（每列一對）。orient∈{realx,predx}。"""
    pairs = [("P", "NAD"), ("P", "ACS"), ("NAD", "ACS")]
    fig, axes = plt.subplots(3, 2, figsize=(16, 24))
    for row, (left, right) in enumerate(pairs):
        _group_panel(axes[row, 0], df, left, orient, note)
        _group_panel(axes[row, 1], df, right, orient, note)
    fig.tight_layout()
    _save(fig, scatter_dir / f"pairs_scatter_{orient}.png")

# ── 主流程 ─────────────────────────────────────────────────────────────────────

_COLS = ["ID", "real_age", "predicted_age", "group", "error"]


def _prep(df):
    return df.rename(columns={"age_error": "error"})[_COLS]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="覆寫輸出目錄；留空依 cohort 自動決定")
    ap.add_argument("--match-level", choices=["subject", "visit"], default="subject",
                    help="AD-vs-HC 配對粒度；visit 時 matched 輸出至 1by1matched_visit/（不重產 full）")
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    output_dir = args.output_dir or (
        AGE_ANALYSIS_DIR / cohort_path(*cohort) / "scatter")
    level = args.match_level
    matched_name = "1by1matched" if level == "subject" else "1by1matched_visit"
    logger.info(f"cohort = {cohort}")
    logger.info(f"output-dir  = {output_dir}")
    logger.info(f"match-level = {level}")

    full = build_cohort_with_age_error(*cohort)
    p_ids, hc_ids = match_by_age(*cohort, priority=["ACS"], level=level)  # ACS 優先：稀少的 ACS 對照先配對
    matched = full[full["ID"].isin(set(p_ids) | set(hc_ids))].reset_index(drop=True)
    logger.info(f"full={len(full)} ({full['group'].value_counts().to_dict()}), "
                f"{matched_name}={len(matched)} "
                f"({matched['group'].value_counts().to_dict()})")

    panels = []
    if level == "subject":
        panels.append(("full", _prep(full), ""))
    panels.append((matched_name, _prep(matched), "\n(age-matched 1:1)"))
    for name, d, note in panels:
        plot_main_scatter(d, output_dir / name, note=note)
        for orient in ("realx", "predx"):  # realx=傳統, predx=換軸，各出一份
            plot_grid4(d, output_dir / name, orient, note=note)
            plot_pairs(d, output_dir / name, orient, note=note)


if __name__ == "__main__":
    main()
