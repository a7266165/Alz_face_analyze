"""逐真實年齡的預測殘差（real − predicted）折線圖，分 ACS / NAD / P 三組
—— 完整 cohort 與 AD-vs-HC 1:1 年齡配對子集。

輸出於 <AGE_ANALYSIS_DIR>/<cohort>/lines/{full,1by1matched}/：
  no_sliding_window/lines_internal.png       —— 逐整數年齡的殘差（mean ± std）
  sliding_window_10/lines_internal_sw10.png  —— 10 年滑動視窗的殘差
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

COLORS = {
    "ACS": "#4CAF50",
    "NAD": "#2196F3",
    "P":   "#F44336",
}
GROUPS = ["ACS", "NAD", "P"]

# ── 繪圖函式 ───────────────────────────────────────────────────────────

def plot_combined(df_all, output_path, groups, title, ylabel, y_col):
    fig, ax = plt.subplots(figsize=(14, 5))
    for grp in groups:
        color = COLORS.get(grp)
        if color is None:
            continue
        sub = df_all[df_all["group"] == grp]
        if sub.empty:
            continue
        st = sub.groupby("age_int")[y_col].agg(["mean", "std", "count"])
        st = st[st["count"] >= 3].sort_index()
        if st.empty:
            continue
        ages, means, stds = st.index.values, st["mean"].values, st["std"].fillna(0).values
        ax.plot(ages, means, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{grp} (n={len(sub)})")
        ax.fill_between(ages, means - stds, means + stds, color=color, alpha=0.15)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("True Age (y)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(50, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {output_path}")


def plot_sliding_window(df_all, output_path, groups, title, ylabel, y_col,
                        window=10, step=1, min_count=5, xlim=(50, 100)):
    fig, ax = plt.subplots(figsize=(14, 5))
    lo_x, hi_x = xlim
    starts = np.arange(lo_x, hi_x - window + 1 + step, step)

    for grp in groups:
        color = COLORS.get(grp)
        if color is None:
            continue
        sub = df_all[df_all["group"] == grp]
        if sub.empty:
            continue
        real, vals = sub["real_age"].values, sub[y_col].values
        xs, ms, ss = [], [], []
        for s in starts:
            mask = (real >= s) & (real < s + window)
            if mask.sum() < min_count:
                continue
            xs.append(s + window / 2.0)
            ms.append(vals[mask].mean())
            ss.append(vals[mask].std(ddof=0))
        if not xs:
            continue
        xs, ms, ss = np.array(xs), np.array(ms), np.array(ss)
        ax.plot(xs, ms, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{grp} (n={len(sub)})")
        ax.fill_between(xs, ms - ss, ms + ss, color=color, alpha=0.15)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel(f"True Age (y) — {window}-y sliding window center", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(lo_x, hi_x)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {output_path}")

# ── 主流程 ─────────────────────────────────────────────────────────────────────

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
        AGE_ANALYSIS_DIR / cohort_path(*cohort) / "lines")

    logger.info(f"cohort = {cohort}")
    logger.info(f"output-dir  = {output_dir}")

    full = build_cohort_with_age_error(*cohort)
    full["age_int"] = full["real_age"].astype(int)
    p_ids, hc_ids = match_by_age(*cohort, priority=["ACS"])  # ACS 優先：稀少的 ACS 對照先配對
    matched = full[full["ID"].isin(set(p_ids) | set(hc_ids))].reset_index(drop=True)
    logger.info(f"full={len(full)} ({full['group'].value_counts().to_dict()}), "
                f"1by1matched={len(matched)} "
                f"({matched['group'].value_counts().to_dict()})")

    ylabel = "Prediction Residual (real − predicted)"
    title_state = "Residual = real − predicted"

    for sub_name, sub_df in [("full", full), ("1by1matched", matched)]:
        base = output_dir / sub_name
        suffix = "" if sub_name == "full" else " — age-matched 1:1"
        plot_combined(
            sub_df, base / "no_sliding_window" / "lines_internal.png", groups=GROUPS,
            title=f"{title_state} by True Age — ACS / NAD / P (mean ± std){suffix}",
            ylabel=ylabel, y_col="age_error")
        plot_sliding_window(
            sub_df, base / "sliding_window_10" / "lines_internal_sw10.png", groups=GROUPS,
            title=f"{title_state} by True Age — ACS / NAD / P (10-y sliding window){suffix}",
            ylabel=ylabel, y_col="age_error")


if __name__ == "__main__":
    main()
