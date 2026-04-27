"""
Per-arm confusion-matrix grid: 3 (groups) × 4 (model × feat-set) = 12 cells per arm.

X-axis = model × feat-set:
  - XGBoost ┌ 2-feat
            └ 3-feat
  - TabPFN  ┌ 2-feat
            └ 3-feat
Y-axis = comparison group (HC, NAD, ACS).

Each cell: independent 2×2 confusion matrix at threshold=0.5 from that
combo's OOF predictions, with AUC/BalAcc/MCC printed below.

Reads:  workspace/arms_analysis/per_arm/<arm>/ad_vs_<cmp>/age/classifier_<set>_<model>_oof.csv
        workspace/arms_analysis/per_arm/<arm>/ad_vs_<cmp>/age/classifier_summary.csv

Output: workspace/arms_analysis/per_arm/<arm>/age_classifier_grid.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
ARMS_DIR = ROOT / "workspace" / "arms_analysis" / "per_arm"
ARMS = ["A", "B", "C", "D"]
COMPARISONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
COMP_LABEL = {"ad_vs_hc": "AD vs HC",
              "ad_vs_nad": "AD vs NAD",
              "ad_vs_acs": "AD vs ACS"}
MODELS = [("xgb", "XGBoost"), ("tabpfn", "TabPFN")]
FEAT_SETS = [("2feat", "2-feat"), ("3feat", "3-feat")]
# Flat column order: XGB-2f, XGB-3f, TabPFN-2f, TabPFN-3f
COLS = [(model, mlbl, fs, flbl)
        for (model, mlbl) in MODELS
        for (fs, flbl) in FEAT_SETS]


def load_oof(arm, cmp, model, feat_set):
    p = (ARMS_DIR / f"arm_{arm.lower()}" / cmp / "age" /
         f"classifier_{feat_set}_{model}_oof.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_summary(arm, cmp):
    p = (ARMS_DIR / f"arm_{arm.lower()}" / cmp / "age" /
         "classifier_summary.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)


def metric_row(summary, model, feat_set):
    row = summary[(summary["model"] == model) &
                  (summary["feature_set"] == feat_set)]
    if row.empty:
        return None
    r = row.iloc[0]
    return dict(auc=r["auc"], balacc=r["balacc"], mcc=r["mcc"], n=int(r["n"]))


def draw_cm(ax, cm, neg_label, pos_label):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            color = "white" if v > cm.sum() * 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([neg_label, pos_label], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([neg_label, pos_label], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)


def render_arm(arm):
    n_rows = len(COMPARISONS)
    n_cols = len(COLS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols + 1.0, 3.5 * n_rows + 1.5),
        gridspec_kw=dict(hspace=0.85, wspace=0.45),
    )

    for ri, cmp in enumerate(COMPARISONS):
        summary = load_summary(arm, cmp)
        neg_label = cmp.replace("ad_vs_", "").upper()
        for ci, (model, model_lbl, fset, flbl) in enumerate(COLS):
            ax = axes[ri, ci]
            oof = load_oof(arm, cmp, model, fset)
            if oof is None or summary is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            y = oof["label"].astype(int).values
            p = oof["prob"].values
            valid = ~np.isnan(p)
            y, p = y[valid], p[valid]
            yhat = (p >= 0.5).astype(int)
            cm = confusion_matrix(y, yhat, labels=[0, 1])
            draw_cm(ax, cm, neg_label, "AD")

            m = metric_row(summary, model, fset)
            ax.set_title(f"{COMP_LABEL[cmp]}", fontsize=10,
                         fontweight="bold", pad=4)
            if m:
                metric_text = (
                    f"AUC    {m['auc']:.3f}\n"
                    f"BalAcc {m['balacc']:.3f}\n"
                    f"MCC    {m['mcc']:.3f}\n"
                    f"n      {m['n']}"
                )
                ax.text(0.5, -0.42, metric_text,
                        ha="center", va="top",
                        transform=ax.transAxes,
                        fontsize=8, family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5",
                                  ec="gray", lw=0.5))

    # Two-tier column header: model on top, feature-set below
    for ci, (model, model_lbl, fset, flbl) in enumerate(COLS):
        # feat-set sub-label (above each column)
        bbox = axes[0, ci].get_position()
        x_centre = (bbox.x0 + bbox.x1) / 2
        fig.text(x_centre, bbox.y1 + 0.025, flbl,
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#444")

    # Top-level model header spanning each pair of feat-set columns
    n_per_model = len(FEAT_SETS)
    for mi, (model, model_lbl) in enumerate(MODELS):
        left_bbox = axes[0, mi * n_per_model].get_position()
        right_bbox = axes[0, mi * n_per_model + n_per_model - 1].get_position()
        x_centre = (left_bbox.x0 + right_bbox.x1) / 2
        fig.text(x_centre, left_bbox.y1 + 0.06, model_lbl,
                 ha="center", va="bottom",
                 fontsize=14, fontweight="bold")
        # Group separator line under model header
        fig.add_artist(plt.Line2D(
            [left_bbox.x0, right_bbox.x1],
            [left_bbox.y1 + 0.055, left_bbox.y1 + 0.055],
            color="black", linewidth=1.0, transform=fig.transFigure,
        ))

    fig.suptitle(
        f"Arm {arm} — confusion matrices @ threshold=0.5  (10-fold OOF)",
        fontsize=13, fontweight="bold", y=1.04,
    )

    out_path = ARMS_DIR / f"arm_{arm.lower()}" / "age_classifier_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    for arm in ARMS:
        render_arm(arm)


if __name__ == "__main__":
    main()
