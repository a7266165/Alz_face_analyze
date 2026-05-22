"""
Overview grid-search plot: 3 embeddings × 3 metrics in one figure.

For each (classifier, partition_group), produces ONE figure:
    Rows: AUC / BalAcc / MCC
    Cols: arcface / dlib / topofr
    Lines: partitions (ad_vs_hc, ad_vs_nad, ad_vs_acs for hc_family)

Reduces 18 images → 4 per root (2 classifiers × 2 partition groups).

Usage:
    conda run -n Alz_face_main_analysis python scripts/embedding/plot_grid_search_overview.py \
        --root workspace/embedding/analysis/classification/match_acs_first/matched/original/p_first_cdrall_hc_all_cdrall_or_mmseall/no_drop
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS = [("auc", "AUC"), ("balacc", "BalAcc"), ("mcc", "MCC")]
EMBEDDINGS = ["arcface", "dlib", "topofr"]
EMB_DISPLAY = {"arcface": "ArcFace", "dlib": "dlib", "topofr": "TopoFR"}

HC_FAMILY = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
HILO = ["mmse_hilo", "casi_hilo"]

PART_COLORS = {
    "ad_vs_hc":  "#4C72B0",
    "ad_vs_nad": "#DD8452",
    "ad_vs_acs": "#55A868",
    "mmse_hilo": "#C44E52",
    "casi_hilo": "#8172B3",
}
PART_LABELS = {
    "ad_vs_hc":  "AD vs HC",
    "ad_vs_nad": "AD vs NAD",
    "ad_vs_acs": "AD vs ACS",
    "mmse_hilo": "MMSE hi/lo",
    "casi_hilo": "CASI hi/lo",
}


def _resolve_root(arg_root):
    p = Path(arg_root)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), PROJECT_ROOT):
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    return (PROJECT_ROOT / p).resolve()


def _combo_sort(df, classifier):
    if classifier == "logistic":
        return df.sort_values("lr_C").reset_index(drop=True)
    if classifier == "xgb":
        return df.sort_values(
            ["xgb_learning_rate", "xgb_max_depth", "xgb_n_estimators"]
        ).reset_index(drop=True)
    return df.reset_index(drop=True)


def _x_for_classifier(df, classifier):
    if classifier == "logistic":
        x = np.log10(df["lr_C"].astype(float).to_numpy())
        labels = [f"{c:g}" for c in df["lr_C"]]
        return x, labels, "log₁₀(C)"
    if classifier == "xgb":
        x = np.arange(len(df))
        labels = [
            f"n_tree{int(r['xgb_n_estimators'])}_max_depth{int(r['xgb_max_depth'])}"
            f"_lr{r['xgb_learning_rate']:g}"
            for _, r in df.iterrows()
        ]
        return x, labels, "combo"
    return np.arange(len(df)), [""] * len(df), "idx"


def _plot_line(ax, x, y, label, color, ci_lo=None, ci_hi=None):
    valid = np.isfinite(y)
    if not valid.any():
        return
    ax.plot(x, y, marker="o", color=color, linewidth=1.3, markersize=3,
            label=label)
    if ci_lo is not None and ci_hi is not None:
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.12, color=color)
    best = int(np.nanargmax(y))
    ax.scatter(x[best], y[best], s=120, marker="*", color="gold",
               edgecolors=color, linewidths=1, zorder=10)
    ax.annotate(f"{y[best]:.3f}", (x[best], y[best]),
                textcoords="offset points", xytext=(4, 6),
                fontsize=7, color=color, fontweight="bold")


STRATEGY_SCOPES = {
    "fwd": [("forward_full", "full"), ("forward_matched", "matched")],
    "rev": [("reverse_matched_oof", "matched_oof"), ("reverse_unmatched", "unmatched")],
}


def plot_overview(all_df, classifier, partitions, group_label, strategy,
                  out_path):
    """3 rows (metrics) × 6 cols (3 embeddings × 2 scopes).

    *strategy*: ``"fwd"`` or ``"rev"`` — selects which scopes to show.
    Black lines separate the 3 embedding blocks.
    """
    scopes = STRATEGY_SCOPES[strategy]
    n_scopes = len(scopes)
    n_rows = len(METRICS)
    n_emb = len(EMBEDDINGS)
    n_cols = n_emb * n_scopes
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.2 * n_rows),
                             sharey="row")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    scope_keys = [s for s, _ in scopes]
    sub = all_df[(all_df["classifier"] == classifier)
                 & (all_df["partition"].isin(partitions))
                 & (all_df["scope"].isin(scope_keys))].copy()
    if sub.empty:
        plt.close(fig)
        return False

    ref_scope_key = scope_keys[-1]
    for ei, emb in enumerate(EMBEDDINGS):
        emb_df = sub[sub["embedding"] == emb]
        if emb_df.empty:
            continue

        ref_scope = emb_df[emb_df["scope"] == ref_scope_key]
        ref = _combo_sort(
            ref_scope[ref_scope["partition"] == partitions[0]].copy(),
            classifier)
        if ref.empty:
            ref = _combo_sort(ref_scope.drop_duplicates(
                subset=[c for c in ref_scope.columns
                        if c.startswith(("lr_", "xgb_"))]),
                classifier)
        x, tick_labels, xlab = _x_for_classifier(ref, classifier)

        for si, (scope_key, scope_label) in enumerate(scopes):
            col = ei * n_scopes + si
            scope_df = emb_df[emb_df["scope"] == scope_key]

            for ri, (metric, mlabel) in enumerate(METRICS):
                ax = axes[ri, col]
                for part in partitions:
                    pdf = scope_df[scope_df["partition"] == part]
                    if pdf.empty:
                        continue
                    pdf = _combo_sort(pdf.copy(), classifier)
                    y = pdf[metric].to_numpy(dtype=float)
                    ci_lo = (pdf["auc_ci_low"].to_numpy(dtype=float)
                             if metric == "auc" else None)
                    ci_hi = (pdf["auc_ci_high"].to_numpy(dtype=float)
                             if metric == "auc" else None)
                    _plot_line(ax, x[:len(y)], y,
                               PART_LABELS.get(part, part),
                               PART_COLORS.get(part, "#888"),
                               ci_lo, ci_hi)
                ax.grid(True, alpha=0.25)
                ax.tick_params(axis="both", labelsize=7, labelleft=True)
                if col == 0:
                    ax.set_ylabel(mlabel, fontsize=10, fontweight="bold")
                if ri == 0:
                    ax.set_title(f"{EMB_DISPLAY.get(emb, emb)}\n{scope_label}",
                                 fontsize=10, fontweight="bold")
                if ri == n_rows - 1:
                    if classifier == "xgb":
                        step = max(1, len(x) // 6)
                        idx = list(range(0, len(x), step))
                        ax.set_xticks(np.array(idx))
                        ax.set_xticklabels(
                            [tick_labels[i] for i in idx],
                            rotation=55, ha="right", fontsize=6)
                    else:
                        ax.tick_params(axis="x", labelsize=8)
                    ax.set_xlabel(xlab, fontsize=8)

    for ei in range(1, n_emb):
        x_frac = ei / n_emb
        fig.add_artist(plt.Line2D(
            [x_frac, x_frac], [0, 0.95], transform=fig.transFigure,
            color="black", linewidth=1.5, zorder=100))

    axes[0, -1].legend(loc="lower left", fontsize=7,
                       framealpha=0.8)
    strat_label = "Forward" if strategy == "fwd" else "Reverse"
    clf_label = "Logistic Regression" if classifier == "logistic" else "XGBoost"
    fig.suptitle(f"{group_label}  —  {clf_label}  ({strat_label})",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", required=True,
                   help="Dir containing _summary/all_metrics_with_cm.csv")
    args = p.parse_args()
    root = _resolve_root(args.root)
    csv = root / "_summary" / "all_metrics_with_cm.csv"
    if not csv.exists():
        print(f"ERROR: {csv} not found")
        return
    df = pd.read_csv(csv)
    summary_dir = root / "_summary"
    n = 0
    for clf in ["logistic", "xgb"]:
        for partitions, tag, label in [
            (HC_FAMILY, "hc_family", "HC family"),
            (HILO, "hilo", "MMSE / CASI hi-lo"),
        ]:
            for strategy in ["fwd", "rev"]:
                out = summary_dir / f"overview_{tag}_{clf}_{strategy}.png"
                if plot_overview(df, clf, partitions, label, strategy, out):
                    print(f"  wrote {out.relative_to(root)}")
                    n += 1
    print(f"Done: {n} overview plots written.")


if __name__ == "__main__":
    main()
