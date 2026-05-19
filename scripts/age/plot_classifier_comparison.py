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

Reads:
  workspace/age/analysis/classification/<cohort>/<partition>/<full|matched>/<feat_set>/<model>_oof.csv
  workspace/age/analysis/classification/<cohort>/<partition>/summary_<full|matched>.csv
  + longitudinal/analysis/age/classification/... for longitudinal_naive/_matched.

Output: stdout (`age_classifier_grid.png` is no longer written; summary.csv
        tables carry the metrics directly).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.config import cohort_name  # noqa: E402

COHORT = cohort_name("default")  # V2.2: p_first_cdr05_hc_first_cdrall_or_mmseall
AGE_CLF_ROOT = ROOT / "workspace" / "age" / "analysis" / "classification" / COHORT
LONGI_AGE_CLF_ROOT = (ROOT / "workspace" / "longitudinal_analysis" / "age"
                      / "analysis" / "classification" / COHORT)
# design → (timeframe, bucket subdir name)
DESIGN_TO_TIMEFRAME_BUCKET = {
    "cross_naive":          ("cross", "full"),
    "cross_matched":        ("cross", "matched"),
    "longitudinal_naive":   ("longi", "full"),
    "longitudinal_matched": ("longi", "matched"),
}
DESIGNS = list(DESIGN_TO_TIMEFRAME_BUCKET.keys())
COMPARISONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
COMP_LABEL = {"ad_vs_hc": "AD vs HC",
              "ad_vs_nad": "AD vs NAD",
              "ad_vs_acs": "AD vs ACS"}
MODELS = [("xgb", "XGBoost"), ("tabpfn", "TabPFN")]
FEAT_SETS = [("2feat", "2-feat"), ("3feat", "3-feat")]
COLS = [(model, mlbl, fs, flbl)
        for (model, mlbl) in MODELS
        for (fs, flbl) in FEAT_SETS]


def _design_root(design):
    timeframe, _ = DESIGN_TO_TIMEFRAME_BUCKET[design]
    return LONGI_AGE_CLF_ROOT if timeframe == "longi" else AGE_CLF_ROOT


def load_oof(design, cmp, model, feat_set):
    _, bucket = DESIGN_TO_TIMEFRAME_BUCKET[design]
    p = _design_root(design) / cmp / bucket / feat_set / f"{model}_oof.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_summary(design, cmp):
    _, bucket = DESIGN_TO_TIMEFRAME_BUCKET[design]
    p = _design_root(design) / cmp / f"summary_{bucket}.csv"
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


def render_design(design):
    n_rows = len(COMPARISONS)
    n_cols = len(COLS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols + 1.0, 3.5 * n_rows + 1.5),
        gridspec_kw=dict(hspace=0.85, wspace=0.45),
    )

    for ri, cmp in enumerate(COMPARISONS):
        summary = load_summary(design, cmp)
        neg_label = cmp.replace("ad_vs_", "").upper()
        for ci, (model, model_lbl, fset, flbl) in enumerate(COLS):
            ax = axes[ri, ci]
            oof = load_oof(design, cmp, model, fset)
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
        f"{design} — confusion matrices @ threshold=0.5  (10-fold OOF)",
        fontsize=13, fontweight="bold", y=1.04,
    )

    # Output goes to overview/<cohort>/<timeframe>_<design_full>/.
    overview_cohort = ROOT / "workspace" / "overview" / COHORT
    timeframe, bucket = DESIGN_TO_TIMEFRAME_BUCKET[design]
    bucket_full = {"full": "naive", "matched": "matched"}[bucket]
    out_path = (overview_cohort / f"{timeframe}_{bucket_full}"
                / f"classifier_grid_{design}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    for design in DESIGNS:
        render_design(design)


if __name__ == "__main__":
    main()
