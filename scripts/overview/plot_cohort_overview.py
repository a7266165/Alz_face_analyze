"""
Cohort age ladder overview across 3 cross-sectional + longitudinal designs.

Reads:
  workspace/overview/<cohort>/cross_naive/cohort.csv
  workspace/overview/<cohort>/cross_matched/mmse_high_vs_low/matched_features.csv
  workspace/overview/<cohort>/longi_naive/mmse_high_vs_low/matched_features.csv

Produces:
  workspace/overview/<cohort>/cohort_overview.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.config import cohort_path  # noqa: E402

COHORT = cohort_path("p_first_cdr05_hc_first_cdrall_or_mmseall")
OVERVIEW_COHORT = ROOT / "workspace" / "overview" / COHORT
CROSS_NAIVE_CSV   = OVERVIEW_COHORT / "cross_naive" / "cohort.csv"
CROSS_MATCHED_CSV = OVERVIEW_COHORT / "cross_matched" / "mmse_high_vs_low" / "matched_features.csv"
LONGI_NAIVE_CSV   = OVERVIEW_COHORT / "longi_naive" / "mmse_high_vs_low" / "matched_features.csv"
OUT_PNG = OVERVIEW_COHORT / "cohort_overview.png"

COLOR_POS = "#C44E52"  # red  = AD / MMSE-high / last visit
COLOR_NEG = "#4C72B0"  # blue = HC / MMSE-low  / baseline


def _fmt_mean_sd(vals):
    vals = pd.Series(vals).dropna().astype(float)
    if len(vals) == 0:
        return "—"
    return f"{vals.mean():.1f} ± {vals.std():.1f}"


def _fmt_range(vals):
    vals = pd.Series(vals).dropna().astype(float)
    if len(vals) == 0:
        return "—"
    return f"[{vals.min():.1f}–{vals.max():.1f}]"


def plot_overlay_hist(ax, series_dict, colors, bins, xlim, title, xlabel="Age (years)"):
    for (label, vals), color in zip(series_dict.items(), colors):
        ax.hist(pd.Series(vals).dropna(), bins=bins, range=xlim,
                alpha=0.55, color=color, label=label, edgecolor="white", linewidth=0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_table(ax, cell_text, row_labels, col_labels, footnote=None):
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels,
                   colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    if footnote:
        ax.text(0.5, -0.05, footnote, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, style="italic", color="dimgray")


def panel_cross_naive(ax_hist, ax_table, df, xlim, bins):
    ad = df[df["label"] == 1]
    hc = df[df["label"] == 0]
    plot_overlay_hist(
        ax_hist,
        {f"AD (n={len(ad)})": ad["Age"], f"HC (n={len(hc)})": hc["Age"]},
        [COLOR_POS, COLOR_NEG], bins=bins, xlim=xlim,
        title="cross_naive — AD vs HC (no age control)",
    )
    cell_text = [
        [str(len(ad)), _fmt_mean_sd(ad["Age"]), _fmt_range(ad["Age"]), _fmt_mean_sd(ad["MMSE"])],
        [str(len(hc)), _fmt_mean_sd(hc["Age"]), _fmt_range(hc["Age"]), _fmt_mean_sd(hc["MMSE"])],
    ]
    render_table(ax_table, cell_text,
                 row_labels=["AD", "HC"],
                 col_labels=["N", "Age μ±σ", "Age range", "MMSE μ±σ"])


def panel_cross_matched(ax_hist, ax_table, df, xlim, bins):
    hi = df[df["mmse_group"] == "high"]
    lo = df[df["mmse_group"] == "low"]
    plot_overlay_hist(
        ax_hist,
        {f"MMSE-high (n={len(hi)})": hi["Age"], f"MMSE-low (n={len(lo)})": lo["Age"]},
        [COLOR_POS, COLOR_NEG], bins=bins, xlim=xlim,
        title="cross_matched — MMSE matched (AD only)",
    )
    n_pairs = df["pair_id"].nunique() if "pair_id" in df.columns else len(df) // 2
    cell_text = [
        [str(len(hi)), _fmt_mean_sd(hi["Age"]), _fmt_range(hi["Age"]), _fmt_mean_sd(hi["MMSE"])],
        [str(len(lo)), _fmt_mean_sd(lo["Age"]), _fmt_range(lo["Age"]), _fmt_mean_sd(lo["MMSE"])],
    ]
    render_table(ax_table, cell_text,
                 row_labels=["MMSE-high", "MMSE-low"],
                 col_labels=["N", "Age μ±σ", "Age range", "MMSE μ±σ"],
                 footnote=f"{n_pairs} matched pairs (1:1 NN, caliper = 2.0 yr)")


def panel_longi_naive(ax_hist, ax_table, df, xlim, bins):
    plot_overlay_hist(
        ax_hist,
        {f"Baseline (n={len(df)})": df["first_age"],
         f"Last visit (n={len(df)})": df["last_age"]},
        [COLOR_NEG, COLOR_POS], bins=bins, xlim=xlim,
        title="longi_naive — Longitudinal (AD, matched)",
    )

    hi = df[df["mmse_group"] == "high"]
    lo = df[df["mmse_group"] == "low"]

    cell_text = [
        [str(len(df)),
         _fmt_mean_sd(df["first_age"]),
         _fmt_mean_sd(df["last_age"]),
         _fmt_mean_sd(df["first_MMSE"])],
        [str(len(hi)),
         _fmt_mean_sd(hi["first_age"]),
         _fmt_mean_sd(hi["last_age"]),
         _fmt_mean_sd(hi["first_MMSE"])],
        [str(len(lo)),
         _fmt_mean_sd(lo["first_age"]),
         _fmt_mean_sd(lo["last_age"]),
         _fmt_mean_sd(lo["first_MMSE"])],
    ]
    follow_fmt = _fmt_mean_sd(df["follow_up_years"])
    n_visits_fmt = _fmt_mean_sd(df["n_visits"])
    footnote = f"Follow-up: {follow_fmt} yr · visits per patient: {n_visits_fmt}"
    render_table(ax_table, cell_text,
                 row_labels=["All", "MMSE-high", "MMSE-low"],
                 col_labels=["N", "Baseline Age", "Last-visit Age", "Baseline MMSE"],
                 footnote=footnote)


def main():
    cn = pd.read_csv(CROSS_NAIVE_CSV)
    cm = pd.read_csv(CROSS_MATCHED_CSV)
    ln = pd.read_csv(LONGI_NAIVE_CSV)

    print(f"cross_naive:   {len(cn)} rows  (AD={int((cn['label']==1).sum())}, "
          f"HC={int((cn['label']==0).sum())})")
    print(f"cross_matched: {len(cm)} rows  (high={int((cm['mmse_group']=='high').sum())}, "
          f"low={int((cm['mmse_group']=='low').sum())})")
    print(f"longi_naive:   {len(ln)} patients  (high={int((ln['mmse_group']=='high').sum())}, "
          f"low={int((ln['mmse_group']=='low').sum())})")

    all_ages = pd.concat([
        cn["Age"], cm["Age"], ln["first_age"], ln["last_age"],
    ]).dropna()
    xlim = (float(np.floor(all_ages.min() / 5) * 5),
            float(np.ceil(all_ages.max() / 5) * 5))
    bins = 24

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 8.5),
        gridspec_kw={"height_ratios": [3, 1.2]},
    )
    panel_cross_naive(axes[0, 0],   axes[1, 0], cn, xlim, bins)
    panel_cross_matched(axes[0, 1], axes[1, 1], cm, xlim, bins)
    panel_longi_naive(axes[0, 2],   axes[1, 2], ln, xlim, bins)

    fig.suptitle("Cohort age ladder — cross-sec + longitudinal designs",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
