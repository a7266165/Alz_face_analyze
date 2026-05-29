"""
scripts/age/scatter.py
Age prediction scatter plots (internal ACS / NAD / P).

Outputs (to scatter/ directory):
  predicted_ages_scatter.png   — HC (NAD+ACS) vs P

Usage:
  conda run -n Alz_face_age python scripts/age/scatter.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    DEMOGRAPHICS_DIR,
    AGE_SCATTER_DIR,
    PREDICTED_AGES_FILE,
)
from src.age.utils import load_predicted_ages

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── data loading ─────────────────────────────────────────────────────────────

def load_demographics(demo_dir: Path) -> pd.DataFrame:
    # 唯一讀取點：cohort.load_demographics() 已組好 ID(完整鍵) 並解析 Age。
    from src.common.cohort import load_demographics as _load_demo
    df = _load_demo()
    df["group"] = df["Group"]
    return df[["ID", "Age", "group"]]


def match_ages(predicted_ages: dict, demo: pd.DataFrame) -> pd.DataFrame:
    records = []
    for sid, pred in predicted_ages.items():
        row = demo[demo["ID"] == sid]
        if row.empty:
            continue
        real = row["Age"].values[0]
        if pd.isna(real):
            continue
        records.append({"ID": sid, "real_age": real, "predicted_age": pred,
                        "group": row["group"].values[0],
                        "error": real - pred})
    return pd.DataFrame(records)

# ── panel helpers ────────────────────────────────────────────────────────────

def _draw_panel(ax, df, title, colors, labels):
    for grp, color in colors.items():
        sub = df[df["group"] == grp]
        if sub.empty:
            continue
        ax.scatter(sub["real_age"], sub["predicted_age"],
                   c=color, label=labels.get(grp, grp),
                   alpha=0.6, s=30, edgecolors="white", linewidth=0.3)

    age_min = min(df["real_age"].min(), df["predicted_age"].min()) - 5
    age_max = max(df["real_age"].max(), df["predicted_age"].max()) + 5
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

# ── scatter plots ────────────────────────────────────────────────────────────

def plot_main_scatter(df_matched, scatter_dir):
    df_hc = df_matched[df_matched["group"].isin(["ACS", "NAD"])]
    df_p = df_matched[df_matched["group"] == "P"]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
    _draw_panel(ax_l, df_hc, "Healthy Controls (NAD + ACS)",
                {"NAD": "#2196F3", "ACS": "#4CAF50"},
                {"NAD": "NAD", "ACS": "ACS"})
    _draw_panel(ax_r, df_p, "Patients (P)",
                {"P": "#F44336"}, {"P": "Patient"})
    plt.tight_layout()
    out = scatter_dir / "predicted_ages_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scatter-dir", type=Path, default=AGE_SCATTER_DIR)
    args = ap.parse_args()

    args.scatter_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predicted_ages(PREDICTED_AGES_FILE)
    demo = load_demographics(DEMOGRAPHICS_DIR)
    df_matched = match_ages(preds, demo)
    logger.info(f"matched={len(df_matched)}")

    plot_main_scatter(df_matched, args.scatter_dir)


if __name__ == "__main__":
    main()
