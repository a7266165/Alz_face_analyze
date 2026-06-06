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

def _draw_panel(ax, df, title, colors, labels):
    for grp, color in colors.items():
        sub = df[df["group"] == grp]
        if sub.empty:
            continue
        ax.scatter(sub["real_age"], sub["predicted_age"],
                   c=color, label=labels.get(grp, grp),
                   alpha=0.6, s=30, edgecolors="white", linewidth=0.3)

    age_min, age_max = 25, 110  # 固定軸範圍，供跨 cohort/版本比較
    ax.plot([age_min, age_max], [age_min, age_max],
            "k--", alpha=0.5, linewidth=1, label="y = x")

    x = df["real_age"].to_numpy(float)
    y = df["predicted_age"].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ss_xx = float(np.sum((x - x.mean()) ** 2))
    if ss_xx > 0:
        a = float(np.sum((x - x.mean()) * (y - y.mean()))) / ss_xx
        b = float(y.mean() - a * x.mean())
        xs = np.array([age_min, age_max])
        ax.plot(xs, a * xs + b, color="#FF9800", linewidth=2, alpha=0.8,
                label=f"y = {a:.2f}x + {b:.2f}")

    n = len(df)
    r = df["real_age"].corr(df["predicted_age"])
    mae = df["error"].abs().mean()
    ax.set_xlabel("Real Age", fontsize=12)
    ax.set_ylabel("Predicted Age (MiVOLO)", fontsize=12)
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
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    output_dir = args.output_dir or (
        AGE_ANALYSIS_DIR / cohort_path(*cohort) / "scatter")
    logger.info(f"cohort = {cohort}")
    logger.info(f"output-dir  = {output_dir}")

    full = build_cohort_with_age_error(*cohort)
    p_ids, hc_ids = match_by_age(*cohort, priority=["ACS"])  # ACS 優先：稀少的 ACS 對照先配對
    matched = full[full["ID"].isin(set(p_ids) | set(hc_ids))].reset_index(drop=True)
    logger.info(f"full={len(full)} ({full['group'].value_counts().to_dict()}), "
                f"1by1matched={len(matched)} "
                f"({matched['group'].value_counts().to_dict()})")

    plot_main_scatter(_prep(full), output_dir / "full")
    plot_main_scatter(_prep(matched), output_dir / "1by1matched",
                      note="\n(age-matched 1:1)")


if __name__ == "__main__":
    main()
