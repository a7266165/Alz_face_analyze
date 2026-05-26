"""
Plot grid_search results from `<OUTPUT_DIR>/_summary/all_metrics_with_cm.csv`.

For each (embedding, classifier) combo present in the CSV, renders a 3×2
panel summary:

    Cols:    full union eval | matched eval
    Rows:    AUC / BalAcc / MCC
    Full col:    one line per panel       (group-agnostic)
    Matched col: up to 3 lines per panel  (HC family: ad_vs_hc / ad_vs_nad / ad_vs_acs)

X-axis:
    LR  → log10(C)
    XGB → combo idx 0..N-1, ordered by (lr, max_depth, n_estimators), with
          compound tick labels `ne_X_md_Y_lr_Z`

Best per-line combo is highlighted with a ★ marker.

Output:
    <OUTPUT_DIR>/_summary/grid_plot_hc_family_<embedding>_<classifier>.png
    <OUTPUT_DIR>/_summary/grid_plot_<partition>_<embedding>_<classifier>.png
        (for mmse_hilo / casi_hilo: matched row 1 line, no group split)

Usage:
    conda run -n Alz_face_main_analysis python scripts/embedding/plot_grid_search.py \\
        --root workspace/embedding/analysis/classification/original_background/p_first_cdr05_hc_all_cdrall_or_mmseall/no_drop
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

HC_FAMILY = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
HC_GROUP_COLORS = {
    "ad_vs_hc":  "#4C72B0",   # blue
    "ad_vs_nad": "#DD8452",   # orange
    "ad_vs_acs": "#55A868",   # green
}
HC_GROUP_LABELS = {
    "ad_vs_hc":  "HC (NAD+ACS)",
    "ad_vs_nad": "NAD only",
    "ad_vs_acs": "ACS only",
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
    """Sort a df of grid rows by a canonical hyperparam order.
    LR: by lr_C asc.
    XGB: by (lr, max_depth, n_estimators) asc.
    """
    if classifier == "logistic":
        return df.sort_values("lr_C").reset_index(drop=True)
    if classifier == "xgb":
        return df.sort_values(
            ["xgb_learning_rate", "xgb_max_depth", "xgb_n_estimators"]
        ).reset_index(drop=True)
    return df.reset_index(drop=True)


def _x_for_classifier(df, classifier):
    """Return (x_values, tick_labels, x_label) for the sorted df."""
    if classifier == "logistic":
        x = np.log10(df["lr_C"].astype(float).to_numpy())
        labels = [f"{c:g}" for c in df["lr_C"]]
        return x, labels, "log10(C)"
    if classifier == "xgb":
        x = np.arange(len(df))
        labels = [
            f"n_tree{int(r['xgb_n_estimators'])}_max_depth{int(r['xgb_max_depth'])}"
            f"_lr{r['xgb_learning_rate']:g}"
            for _, r in df.iterrows()
        ]
        return x, labels, "combo (sorted by lr → max_depth → n_estimators)"
    return np.arange(len(df)), [""] * len(df), "idx"


def _plot_line(ax, x, df, metric, label, color, with_ci=True):
    """Plot one metric line on `ax`; star-mark the best combo."""
    y = df[metric].to_numpy(dtype=float)
    if not np.isfinite(y).any():
        return
    n = len(y)
    x = x[:n]
    if with_ci and metric == "auc":
        lo = df["auc_ci_low"].to_numpy(dtype=float)
        hi = df["auc_ci_high"].to_numpy(dtype=float)
        if len(lo) == n and len(hi) == n:
            ax.fill_between(x, lo, hi, alpha=0.15, color=color)
    ax.plot(x, y, marker="o", color=color, linewidth=1.4, markersize=4,
             label=label)
    best_idx = int(np.nanargmax(y))
    best_val = y[best_idx]
    ax.scatter(x[best_idx], best_val, s=160, marker="*",
                color="gold", edgecolors=color, linewidths=1.2,
                zorder=10)
    if label == "NAD only":
        oy, va_ = -6, "top"
    else:
        oy, va_ = 6, "bottom"
    ax.annotate(f"{best_val:.3f}",
                xy=(x[best_idx], best_val),
                xytext=(0, oy), textcoords="offset points",
                ha="center", va=va_, fontsize=13, fontweight="bold",
                color=color, zorder=11,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.8))


def _draw_3x2(rows_full, rows_matched_by_part, classifier, out_path,
              title_prefix):
    """Render 3×2 panel grid: rows = AUC/BalAcc/MCC, cols = full/matched.

    rows_full: DataFrame for forward_full scope, sorted by combo (1 line).
    rows_matched_by_part: dict[partition] → DataFrame of forward_matched rows,
        sorted by combo. Multiple partitions ⇒ multiple lines per matched panel.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharex="col", sharey="row")
    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=20)

    canonical = (rows_full if rows_full is not None and not rows_full.empty
                 else next(iter(rows_matched_by_part.values())))
    x, tick_labels, xlab = _x_for_classifier(canonical, classifier)
    n_combos = len(canonical)

    for row, (metric, label) in enumerate(METRICS):
        # --- Col 0: full union ---
        ax_full = axes[row, 0]
        if rows_full is not None and not rows_full.empty:
            _plot_line(ax_full, x, rows_full, metric, "full union",
                       "#444444", with_ci=True)
        ax_full.set_ylabel(label, fontsize=11, fontweight="bold")
        ax_full.grid(True, alpha=0.3)

        # --- Col 1: matched (1 or 3 lines) ---
        ax_m = axes[row, 1]
        for partition, df_part in rows_matched_by_part.items():
            if df_part is None or df_part.empty:
                continue
            color = HC_GROUP_COLORS.get(partition, "#888888")
            lab = HC_GROUP_LABELS.get(partition, partition)
            _plot_line(ax_m, x, df_part, metric, lab, color, with_ci=True)
        ax_m.grid(True, alpha=0.3)

        # X-axis only on bottom row
        if row == len(METRICS) - 1:
            ax_full.set_xlabel(xlab, fontsize=22)
            ax_m.set_xlabel(xlab, fontsize=22)
            if classifier == "xgb":
                step = max(1, n_combos // 9)
                tick_idx = list(range(0, n_combos, step))
                for ax in (ax_full, ax_m):
                    ax.set_xticks(np.array(tick_idx))
                    ax.set_xticklabels([tick_labels[i] for i in tick_idx],
                                        rotation=60, ha="right", fontsize=14)

    # Column headers (full / matched)
    axes[0, 0].set_title("full union eval", fontsize=24, fontweight="bold",
                          pad=10)
    axes[0, 1].set_title("matched eval", fontsize=24, fontweight="bold",
                          pad=10)

    # Legends — first row each col, lower-left for uniform placement
    axes[0, 0].legend(loc="lower left", fontsize=14)
    axes[0, 1].legend(loc="lower left", fontsize=14)

    fig.suptitle(title_prefix, fontsize=24, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_for(df_emb_clf, embedding, classifier, summary_dir):
    """Render plots for one (embedding, classifier) slice. Returns list of
    output paths written."""
    # Drop rows where the relevant hyperparam column is NaN (e.g. legacy
    # cells written before the lr_C / xgb_params fields existed).
    if classifier == "logistic":
        df_emb_clf = df_emb_clf[df_emb_clf["lr_C"].notna()]
    elif classifier == "xgb":
        df_emb_clf = df_emb_clf[
            df_emb_clf["xgb_n_estimators"].notna()
            & df_emb_clf["xgb_max_depth"].notna()
            & df_emb_clf["xgb_learning_rate"].notna()
        ]
    if df_emb_clf.empty:
        return []

    out_paths = []

    # HC family — combine 3 partitions into one plot.
    hc_full = df_emb_clf[
        (df_emb_clf["partition"].isin(HC_FAMILY))
        & (df_emb_clf["scope"] == "forward_full")
    ]
    hc_matched_by_part = {}
    for part in HC_FAMILY:
        sub = df_emb_clf[
            (df_emb_clf["partition"] == part)
            & (df_emb_clf["scope"] == "forward_matched")
        ]
        if not sub.empty:
            hc_matched_by_part[part] = _combo_sort(sub, classifier)

    if hc_matched_by_part:
        # All 3 partitions share full-union metrics, take the first one.
        rep_part = next(iter(hc_matched_by_part))
        full_one = df_emb_clf[
            (df_emb_clf["partition"] == rep_part)
            & (df_emb_clf["scope"] == "forward_full")
        ]
        full_sorted = _combo_sort(full_one, classifier) if not full_one.empty else None
        out = summary_dir / f"grid_plot_hc_family_{embedding}_{classifier}.png"
        _draw_3x2(full_sorted, hc_matched_by_part, classifier, out,
                  f"HC family — {embedding} / {classifier} forward "
                  f"(n_combos={len(full_sorted) if full_sorted is not None else '?'})")
        out_paths.append(out)

    # Other partitions (mmse_hilo / casi_hilo) — 1 partition each, 1 matched line.
    for part in df_emb_clf["partition"].unique():
        if part in HC_FAMILY:
            continue
        sub_full = df_emb_clf[
            (df_emb_clf["partition"] == part)
            & (df_emb_clf["scope"] == "forward_full")
        ]
        sub_matched = df_emb_clf[
            (df_emb_clf["partition"] == part)
            & (df_emb_clf["scope"] == "forward_matched")
        ]
        if sub_matched.empty:
            continue
        full_sorted = _combo_sort(sub_full, classifier) if not sub_full.empty else None
        matched_sorted = _combo_sort(sub_matched, classifier)
        out = summary_dir / f"grid_plot_{part}_{embedding}_{classifier}.png"
        _draw_3x2(full_sorted, {part: matched_sorted}, classifier, out,
                  f"{part} — {embedding} / {classifier} forward "
                  f"(n_combos={len(matched_sorted)})")
        out_paths.append(out)

    return out_paths


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True,
                        help="OUTPUT_DIR containing _summary/all_metrics_with_cm.csv")
    args = parser.parse_args()

    root = _resolve_root(args.root)
    summary_dir = root / "_summary"
    csv = summary_dir / "all_metrics_with_cm.csv"
    if not csv.exists():
        raise SystemExit(
            f"missing: {csv}\n"
            f"run scripts/embedding/build_sweep_metrics.py first to "
            f"refresh the aggregate."
        )

    df = pd.read_csv(csv)
    if df.empty:
        raise SystemExit(f"{csv} is empty")

    match_levels = (sorted(df["match_level"].dropna().unique())
                    if "match_level" in df.columns else [None])
    eval_units = (sorted(df["eval_unit"].dropna().unique())
                  if "eval_unit" in df.columns else [None])

    n_done = 0
    for ml in match_levels:
        for eu in eval_units:
            slice_df = df.copy()
            if ml is not None:
                slice_df = slice_df[slice_df["match_level"] == ml]
            if eu is not None:
                slice_df = slice_df[slice_df["eval_unit"] == eu]
            if slice_df.empty:
                continue

            if ml is not None and eu is not None:
                out_dir = summary_dir / ml / eu
            else:
                out_dir = summary_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            for embedding in sorted(slice_df["embedding"].dropna().unique()):
                for classifier in sorted(slice_df["classifier"].dropna().unique()):
                    if classifier not in ("logistic", "xgb"):
                        continue
                    sub = slice_df[(slice_df["embedding"] == embedding)
                                   & (slice_df["classifier"] == classifier)]
                    if sub.empty:
                        continue
                    paths = render_for(sub, embedding, classifier, out_dir)
                    for p in paths:
                        rel = p.relative_to(summary_dir.parent)
                        print(f"  wrote {rel}")
                        n_done += 1

    print(f"Done: {n_done} plots written.")


if __name__ == "__main__":
    main()
