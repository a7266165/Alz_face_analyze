"""
Cohort overview for the 4-arm age ladder.

Reads:
  workspace/age_ladder/arm_a_ad_vs_hc/cohort.csv
  workspace/age_ladder/mmse_hilo_standalone/matched_features.csv
  workspace/age_ladder/arm_c_longitudinal_matched/matched_features_longitudinal.csv

Produces:
  workspace/age_ladder/cohort_overview.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "workspace" / "age_ladder"
ARM_A_CSV = LADDER / "arm_a_ad_vs_hc" / "cohort.csv"
ARM_B_CSV = LADDER / "mmse_hilo_standalone" / "matched_features.csv"
ARM_C_CSV = LADDER / "arm_c_longitudinal_matched" / "matched_features_longitudinal.csv"
OUT_PNG = LADDER / "cohort_overview.png"

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


def panel_arm_a(ax_hist, ax_table, df, xlim, bins):
    ad = df[df["label"] == 1]
    hc = df[df["label"] == 0]
    plot_overlay_hist(
        ax_hist,
        {f"AD (n={len(ad)})": ad["Age"], f"HC (n={len(hc)})": hc["Age"]},
        [COLOR_POS, COLOR_NEG], bins=bins, xlim=xlim,
        title="Arm A — AD vs HC (no age control)",
    )
    cell_text = [
        [str(len(ad)), _fmt_mean_sd(ad["Age"]), _fmt_range(ad["Age"]), _fmt_mean_sd(ad["MMSE"])],
        [str(len(hc)), _fmt_mean_sd(hc["Age"]), _fmt_range(hc["Age"]), _fmt_mean_sd(hc["MMSE"])],
    ]
    render_table(ax_table, cell_text,
                 row_labels=["AD", "HC"],
                 col_labels=["N", "Age μ±σ", "Age range", "MMSE μ±σ"])


def panel_arm_b(ax_hist, ax_table, df, xlim, bins):
    hi = df[df["mmse_group"] == "high"]
    lo = df[df["mmse_group"] == "low"]
    plot_overlay_hist(
        ax_hist,
        {f"MMSE-high (n={len(hi)})": hi["Age"], f"MMSE-low (n={len(lo)})": lo["Age"]},
        [COLOR_POS, COLOR_NEG], bins=bins, xlim=xlim,
        title="Arm B — MMSE matched (AD only)",
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


def panel_arm_c(ax_hist, ax_table, df, xlim, bins):
    plot_overlay_hist(
        ax_hist,
        {f"Baseline (n={len(df)})": df["first_age"],
         f"Last visit (n={len(df)})": df["last_age"]},
        [COLOR_NEG, COLOR_POS], bins=bins, xlim=xlim,
        title="Arm C — Longitudinal (AD, matched)",
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
    arm_a = pd.read_csv(ARM_A_CSV)
    arm_b = pd.read_csv(ARM_B_CSV)
    arm_c = pd.read_csv(ARM_C_CSV)

    print(f"Arm A: {len(arm_a)} rows  (AD={int((arm_a['label']==1).sum())}, "
          f"HC={int((arm_a['label']==0).sum())})")
    print(f"Arm B: {len(arm_b)} rows  (high={int((arm_b['mmse_group']=='high').sum())}, "
          f"low={int((arm_b['mmse_group']=='low').sum())})")
    print(f"Arm C: {len(arm_c)} patients  (high={int((arm_c['mmse_group']=='high').sum())}, "
          f"low={int((arm_c['mmse_group']=='low').sum())})")

    all_ages = pd.concat([
        arm_a["Age"], arm_b["Age"], arm_c["first_age"], arm_c["last_age"],
    ]).dropna()
    xlim = (float(np.floor(all_ages.min() / 5) * 5),
            float(np.ceil(all_ages.max() / 5) * 5))
    bins = 24

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 8.5),
        gridspec_kw={"height_ratios": [3, 1.2]},
    )
    panel_arm_a(axes[0, 0], axes[1, 0], arm_a, xlim, bins)
    panel_arm_b(axes[0, 1], axes[1, 1], arm_b, xlim, bins)
    panel_arm_c(axes[0, 2], axes[1, 2], arm_c, xlim, bins)

    fig.suptitle("Cohort overview — 4-arm age ladder", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
