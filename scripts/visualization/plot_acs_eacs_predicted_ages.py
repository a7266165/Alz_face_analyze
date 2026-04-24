"""
ACS (internal) vs E-ACS (external) — predicted age scatter.

並排兩 panel：
  左：ACS (data/demographics/ACS.csv)，多 visit
  右：E-ACS (data/demographics/EACS.csv)，外部公開資料集的 7 個 source
       可用 --color-by-source 依來源上色

輸出：workspace/age/age_prediction/predicted_ages_scatter_acs_vs_eacs.png
"""

import argparse
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

from src.config import DEMOGRAPHICS_DIR, AGE_PREDICTION_DIR, PREDICTED_AGES_FILE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EACS_SOURCE_COLORS = {
    "IMDB":        "#1f77b4",
    "MegaAge":     "#ff7f0e",
    "FairFace":    "#2ca02c",
    "UTKFace":     "#d62728",
    "SZU-EmoDage": "#9467bd",
    "AFAD":        "#8c564b",
    "DiverseAsian": "#e377c2",
}


def load_preds(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_acs():
    df = pd.read_csv(DEMOGRAPHICS_DIR / "ACS.csv", encoding="utf-8-sig")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df[["ID", "Age"]].rename(columns={"Age": "real_age"})


def load_eacs():
    path = DEMOGRAPHICS_DIR / "EACS.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    keep = ["ID", "Age"] + (["Source"] if "Source" in df.columns else [])
    df = df[keep].rename(columns={"Age": "real_age"})
    if "Source" not in df.columns:
        df["Source"] = "unknown"
    return df


def attach_pred(df, preds):
    df = df.copy()
    df["predicted_age"] = df["ID"].map(preds)
    df = df.dropna(subset=["real_age", "predicted_age"]).reset_index(drop=True)
    df["error"] = df["real_age"] - df["predicted_age"]
    return df


def _draw_panel(ax, df, title, color_by="single", single_color="#4CAF50"):
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return

    if color_by == "source":
        for src, color in EACS_SOURCE_COLORS.items():
            sub = df[df["Source"] == src]
            if sub.empty:
                continue
            ax.scatter(sub["real_age"], sub["predicted_age"],
                       c=color, label=f"{src} (n={len(sub)})",
                       alpha=0.55, s=22, edgecolors="white", linewidth=0.25)
        rest = df[~df["Source"].isin(EACS_SOURCE_COLORS)]
        if not rest.empty:
            ax.scatter(rest["real_age"], rest["predicted_age"],
                       c="lightgray", label=f"other (n={len(rest)})",
                       alpha=0.45, s=18, edgecolors="white", linewidth=0.2)
    else:
        ax.scatter(df["real_age"], df["predicted_age"],
                   c=single_color, alpha=0.6, s=30,
                   edgecolors="white", linewidth=0.3,
                   label=f"n={len(df)}")

    age_min = min(df["real_age"].min(), df["predicted_age"].min()) - 5
    age_max = max(df["real_age"].max(), df["predicted_age"].max()) + 5

    # y = x
    ax.plot([age_min, age_max], [age_min, age_max],
            "k--", alpha=0.5, linewidth=1, label="y = x")

    # Regression line
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
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(age_min, age_max)
    ax.set_ylim(age_min, age_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="per_source",
                    choices=["combined", "per_source"],
                    help="combined: ACS + EACS 兩 panel；per_source: 每 dataset 一 panel")
    ap.add_argument("--color-by-source", action="store_true",
                    help="combined 模式下 E-ACS panel 依 Source 分色")
    ap.add_argument("--min-pred", type=float, default=None,
                    help="過濾掉 predicted_age < 此閾值的點（face detection 失敗會產 <40 的離群值）")
    ap.add_argument("--output", type=Path, default=None,
                    help="輸出 PNG 路徑；留空用預設")
    args = ap.parse_args()

    preds = load_preds(PREDICTED_AGES_FILE)
    logger.info(f"loaded {len(preds)} predicted ages")

    acs = attach_pred(load_acs(), preds)
    eacs = attach_pred(load_eacs(), preds)
    if args.min_pred is not None:
        n_before_a, n_before_e = len(acs), len(eacs)
        acs = acs[acs["predicted_age"] >= args.min_pred].reset_index(drop=True)
        eacs = eacs[eacs["predicted_age"] >= args.min_pred].reset_index(drop=True)
        logger.info(f"  min_pred={args.min_pred} filter: "
                    f"ACS {n_before_a}→{len(acs)}, EACS {n_before_e}→{len(eacs)}")
    logger.info(f"ACS matched: {len(acs)} rows  (r={acs['real_age'].corr(acs['predicted_age']):.3f})")
    logger.info(f"EACS matched: {len(eacs)} rows (r={eacs['real_age'].corr(eacs['predicted_age']):.3f})")

    if args.mode == "combined":
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
        _draw_panel(ax_l, acs, title="ACS (internal)",
                    color_by="single", single_color="#4CAF50")
        _draw_panel(ax_r, eacs,
                    title="E-ACS (external public datasets)",
                    color_by="source" if args.color_by_source else "single",
                    single_color="#2196F3")
        default_name = "predicted_ages_scatter_acs_vs_eacs.png"
    else:
        # per_source: ACS + 7 EACS sources, 2×4 grid
        sources_in_order = [
            ("ACS (internal)", acs, "#4CAF50"),
            ("E-ACS | IMDB", eacs[eacs["Source"] == "IMDB"],
             EACS_SOURCE_COLORS["IMDB"]),
            ("E-ACS | MegaAge", eacs[eacs["Source"] == "MegaAge"],
             EACS_SOURCE_COLORS["MegaAge"]),
            ("E-ACS | FairFace", eacs[eacs["Source"] == "FairFace"],
             EACS_SOURCE_COLORS["FairFace"]),
            ("E-ACS | UTKFace", eacs[eacs["Source"] == "UTKFace"],
             EACS_SOURCE_COLORS["UTKFace"]),
            ("E-ACS | SZU-EmoDage", eacs[eacs["Source"] == "SZU-EmoDage"],
             EACS_SOURCE_COLORS["SZU-EmoDage"]),
            ("E-ACS | AFAD", eacs[eacs["Source"] == "AFAD"],
             EACS_SOURCE_COLORS["AFAD"]),
            ("E-ACS | DiverseAsian", eacs[eacs["Source"] == "DiverseAsian"],
             EACS_SOURCE_COLORS["DiverseAsian"]),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        for ax, (title, sub, color) in zip(axes.flat, sources_in_order):
            _draw_panel(ax, sub, title=title, color_by="single",
                        single_color=color)
        default_name = "predicted_ages_scatter_per_source.png"

    # append min-pred tag to filename
    if args.min_pred is not None:
        stem = Path(default_name).stem
        default_name = f"{stem}_minpred{int(args.min_pred)}.png"

    out = args.output if args.output is not None else (
        AGE_PREDICTION_DIR / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out}")


if __name__ == "__main__":
    main()
