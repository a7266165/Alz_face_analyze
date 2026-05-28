"""
scripts/age/scatter.py
Age prediction scatter plots.

Outputs (to scatter/ directory):
  predicted_ages_scatter.png             — HC (NAD+ACS) vs P
  predicted_ages_scatter_with_utkface.png — HC+UTKFace vs P
  predicted_ages_scatter_acs_vs_eacs.png  — ACS vs EACS (combined or per-source)

Usage:
  conda run -n Alz_face_age python scripts/age/scatter.py
  conda run -n Alz_face_age python scripts/age/scatter.py --eacs-mode combined
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
from src.age.calibrator import load_predicted_ages

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
    "AgeDB":       "#17becf",
    "APPA-REAL":   "#bcbd22",
}

# ── data loading ─────────────────────────────────────────────────────────────

def load_demographics(demo_dir: Path) -> pd.DataFrame:
    keep_cols = ["ID", "Age", "group"]
    dfs = []
    for csv_file in ["ACS.csv", "NAD.csv", "P.csv"]:
        df = pd.read_csv(demo_dir / csv_file, encoding="utf-8-sig")
        df["group"] = csv_file.replace(".csv", "")
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        dfs.append(df[["ID", "Age", "group"]])
    return pd.concat(dfs, ignore_index=True)


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


def load_eacs(demo_dir: Path) -> pd.DataFrame:
    path = demo_dir / "EACS.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if "Source" not in df.columns:
        df["Source"] = "unknown"
    return df


def attach_pred(df: pd.DataFrame, preds: dict) -> pd.DataFrame:
    df = df.copy()
    df["predicted_age"] = df["ID"].map(preds)
    df = df.dropna(subset=["Age", "predicted_age"]).reset_index(drop=True)
    df = df.rename(columns={"Age": "real_age"})
    df["error"] = df["real_age"] - df["predicted_age"]
    return df

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


def _draw_panel_single(ax, df, title, color_by="single", single_color="#4CAF50"):
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
    else:
        ax.scatter(df["real_age"], df["predicted_age"],
                   c=single_color, alpha=0.6, s=30,
                   edgecolors="white", linewidth=0.3,
                   label=f"n={len(df)}")

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
    ax.legend(fontsize=9, loc="upper left")
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


def plot_scatter_with_utkface(preds, scatter_dir):
    internal = load_demographics(DEMOGRAPHICS_DIR)
    internal = attach_pred(internal, preds)
    eacs = load_eacs(DEMOGRAPHICS_DIR)
    eacs = eacs[eacs["Source"] == "UTKFace"].copy()
    eacs["group"] = "UTKFace"
    eacs = attach_pred(eacs[["ID", "Age", "group"]], preds)

    hc = internal[internal["group"].isin(["NAD", "ACS"])]
    p = internal[internal["group"] == "P"]
    hc_plus = pd.concat([hc, eacs], ignore_index=True)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
    _draw_panel(ax_l, hc_plus, "Healthy Controls (NAD + ACS + UTKFace)",
                {"NAD": "#2196F3", "ACS": "#4CAF50", "UTKFace": "#d62728"},
                {"NAD": "NAD", "ACS": "ACS", "UTKFace": "UTKFace (E-ACS)"})
    _draw_panel(ax_r, p, "Patients (P)",
                {"P": "#F44336"}, {"P": "Patient"})
    plt.tight_layout()
    out = scatter_dir / "predicted_ages_scatter_with_utkface.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out}")


def plot_acs_vs_eacs(preds, scatter_dir, mode="per_source",
                     color_by_source=False, min_pred=None):
    acs_df = pd.read_csv(DEMOGRAPHICS_DIR / "ACS.csv", encoding="utf-8-sig")
    acs_df["Age"] = pd.to_numeric(acs_df["Age"], errors="coerce")
    acs_df = acs_df[["ID", "Age"]].rename(columns={"Age": "real_age"}).copy()
    acs_df["predicted_age"] = acs_df["ID"].map(preds)
    acs_df = acs_df.dropna(subset=["real_age", "predicted_age"]).reset_index(drop=True)
    acs_df["error"] = acs_df["real_age"] - acs_df["predicted_age"]

    eacs_df = load_eacs(DEMOGRAPHICS_DIR)
    eacs_df = eacs_df[["ID", "Age", "Source"]].rename(columns={"Age": "real_age"}).copy()
    eacs_df["predicted_age"] = eacs_df["ID"].map(preds)
    eacs_df = eacs_df.dropna(subset=["real_age", "predicted_age"]).reset_index(drop=True)
    eacs_df["error"] = eacs_df["real_age"] - eacs_df["predicted_age"]

    if min_pred is not None:
        acs_df = acs_df[acs_df["predicted_age"] >= min_pred].reset_index(drop=True)
        eacs_df = eacs_df[eacs_df["predicted_age"] >= min_pred].reset_index(drop=True)

    if mode == "combined":
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
        _draw_panel_single(ax_l, acs_df, "ACS (internal)", single_color="#4CAF50")
        _draw_panel_single(ax_r, eacs_df, "E-ACS (external)",
                           color_by="source" if color_by_source else "single",
                           single_color="#2196F3")
        fname = "predicted_ages_scatter_acs_vs_eacs.png"
    else:
        sources = [("ACS (internal)", acs_df, "#4CAF50")] + [
            (f"E-ACS | {src}", eacs_df[eacs_df["Source"] == src], EACS_SOURCE_COLORS[src])
            for src in EACS_SOURCE_COLORS
        ]
        fig, axes = plt.subplots(2, 5, figsize=(27, 11))
        for ax, (title, sub, color) in zip(axes.flat, sources):
            _draw_panel_single(ax, sub, title=title, single_color=color)
        fname = "predicted_ages_scatter_per_source.png"

    if min_pred is not None:
        fname = f"{Path(fname).stem}_minpred{int(min_pred)}.png"

    out = scatter_dir / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scatter-dir", type=Path, default=AGE_SCATTER_DIR)
    ap.add_argument("--eacs-mode", default="per_source",
                    choices=["combined", "per_source"])
    ap.add_argument("--min-pred", type=float, default=None)
    args = ap.parse_args()

    args.scatter_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predicted_ages(PREDICTED_AGES_FILE)
    demo = load_demographics(DEMOGRAPHICS_DIR)
    df_matched = match_ages(preds, demo)
    logger.info(f"matched={len(df_matched)}")

    plot_main_scatter(df_matched, args.scatter_dir)
    plot_scatter_with_utkface(preds, args.scatter_dir)
    plot_acs_vs_eacs(preds, args.scatter_dir, mode=args.eacs_mode,
                     min_pred=args.min_pred)


if __name__ == "__main__":
    main()
