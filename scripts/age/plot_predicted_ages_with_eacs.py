"""
在原 predicted_ages_scatter.png 的基礎上，把 E-ACS 的 UTKFace 子集
加入左邊 Healthy Controls panel。

兩欄 layout（同原圖）：
  左：Healthy Controls (NAD + ACS + E-ACS UTKFace)
  右：Patients (P)

樣式 match 原圖：scatter + y=x 虛線 + 橘色迴歸線 + n/r/MAE。

輸出：workspace/age/age_prediction/predicted_ages_scatter_with_utkface.png
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # type: ignore

from src.config import (
    DEMOGRAPHICS_DIR,
    AGE_PREDICTION_DIR,
    PREDICTED_AGES_FILE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HC_COLORS = {
    "NAD":     "#2196F3",   # blue
    "ACS":     "#4CAF50",   # green
    "UTKFace": "#d62728",   # red (E-ACS UTKFace subset)
}
P_COLOR = "#F44336"


def load_preds():
    with open(PREDICTED_AGES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_internal_with_group():
    rows = []
    for csv in ["ACS.csv", "NAD.csv", "P.csv"]:
        df = pd.read_csv(DEMOGRAPHICS_DIR / csv, encoding="utf-8-sig")
        df["group"] = csv.replace(".csv", "")
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        rows.append(df[["ID", "Age", "group"]])
    return pd.concat(rows, ignore_index=True)


def load_eacs_utkface():
    df = pd.read_csv(DEMOGRAPHICS_DIR / "EACS.csv", encoding="utf-8-sig")
    df = df[df["Source"] == "UTKFace"].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["group"] = "UTKFace"  # 以 group 欄位走同一條 color 邏輯
    return df[["ID", "Age", "group"]]


def attach_pred(df, preds):
    df = df.copy()
    df["predicted_age"] = df["ID"].map(preds)
    df = df.dropna(subset=["Age", "predicted_age"]).reset_index(drop=True)
    df = df.rename(columns={"Age": "real_age"})
    df["error"] = df["real_age"] - df["predicted_age"]
    return df


def _draw_panel(ax, df, title, colors, labels):
    for cat, color in colors.items():
        sub = df[df["group"] == cat]
        if sub.empty:
            continue
        ax.scatter(sub["real_age"], sub["predicted_age"],
                   c=color, label=labels.get(cat, cat),
                   alpha=0.6, s=30, edgecolors="white", linewidth=0.3)

    age_min = min(df["real_age"].min(), df["predicted_age"].min()) - 5
    age_max = max(df["real_age"].max(), df["predicted_age"].max()) + 5

    ax.plot([age_min, age_max], [age_min, age_max],
            "k--", alpha=0.5, linewidth=1, label="y = x")

    x = df["real_age"].to_numpy(float)
    y = df["predicted_age"].to_numpy(float)
    ss_xx = float(np.sum((x - x.mean()) ** 2))
    if ss_xx > 0:
        a = float(np.sum((x - x.mean()) * (y - y.mean()))) / ss_xx
        b = float(y.mean() - a * x.mean())
        xs = np.array([age_min, age_max])
        ax.plot(xs, a * xs + b, color="#FF9800", linewidth=2, alpha=0.85,
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


def main():
    preds = load_preds()
    logger.info(f"loaded {len(preds)} predicted ages")

    internal = attach_pred(load_internal_with_group(), preds)
    utkface = attach_pred(load_eacs_utkface(), preds)

    hc = internal[internal["group"].isin(["NAD", "ACS"])]
    p = internal[internal["group"] == "P"]
    hc_plus = pd.concat([hc, utkface], ignore_index=True)

    logger.info(f"HC(NAD+ACS+UTKFace): {len(hc_plus)}  "
                f"r={hc_plus['real_age'].corr(hc_plus['predicted_age']):.3f}")
    logger.info(f"  NAD: {int((hc_plus['group']=='NAD').sum())}  "
                f"ACS: {int((hc_plus['group']=='ACS').sum())}  "
                f"UTKFace: {int((hc_plus['group']=='UTKFace').sum())}")
    logger.info(f"P: {len(p)}  r={p['real_age'].corr(p['predicted_age']):.3f}")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))

    _draw_panel(
        ax_l, hc_plus,
        title="Healthy Controls (NAD + ACS + E-ACS UTKFace)",
        colors=HC_COLORS,
        labels={"NAD": "NAD", "ACS": "ACS", "UTKFace": "UTKFace (E-ACS)"},
    )

    _draw_panel(
        ax_r, p, title="Patients (P)",
        colors={"P": P_COLOR}, labels={"P": "Patient"},
    )

    out = AGE_PREDICTION_DIR / "predicted_ages_scatter_with_utkface.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out}")


if __name__ == "__main__":
    main()
