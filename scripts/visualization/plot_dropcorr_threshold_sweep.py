"""
Aggregate dropcorr threshold dirs (no_drop + drop_0.95..0.05) into a
cross-threshold AUC comparison.

Default mode reads `embedding_classification/<reducer>/_summary/all_metrics_with_cm.csv`.
With --variant, reads `embedding_asymmetry_classification/<reducer>/<variant>/_summary/all_metrics_with_cm.csv`.

Output goes to:
    embedding_classification/_dropcorr_summary/                       (default)
    embedding_asymmetry_classification/_dropcorr_summary/<variant>/   (--variant set)

Usage:
    # original
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_dropcorr_threshold_sweep.py
    # asymmetry variant
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_dropcorr_threshold_sweep.py --variant difference
"""
import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARMS_ROOT = PROJECT_ROOT / "workspace" / "arms_analysis"
ASYM_VARIANTS = ["difference", "absolute_difference", "average",
                 "relative_differences", "absolute_relative_differences"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_paths(variant):
    """Return (ROOT, OUT, csv_subpath) for the given variant.
    csv_subpath is the path under each <reducer>/ dir that holds the cell csv.
    """
    if variant is None:
        root = ARMS_ROOT / "p_first_hc_strict" / "embedding_classification"
        out = root / "_dropcorr_summary"
        # csv at <reducer>/_summary/all_metrics_with_cm.csv
        return root, out, Path("_summary") / "all_metrics_with_cm.csv"
    root = ARMS_ROOT / "p_first_hc_strict" / "embedding_asymmetry_classification"
    out = root / "_dropcorr_summary" / variant
    # csv at <reducer>/<variant>/_summary/all_metrics_with_cm.csv
    return root, out, Path(variant) / "_summary" / "all_metrics_with_cm.csv"


def threshold_label(name):
    if name == "no_drop":
        return None
    m = re.match(r"drop_([0-9.]+)$", name)
    return float(m.group(1)) if m else None


def load_all(root, csv_subpath):
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name not in ("no_drop",) and not sub.name.startswith("drop_"):
            continue
        csv = sub / csv_subpath
        if not csv.exists():
            logger.warning(f"missing {csv}")
            continue
        df = pd.read_csv(csv)
        df["drop_root"] = sub.name
        df["drop_threshold"] = threshold_label(sub.name)
        rows.append(df)
        logger.info(f"  {sub.name}: {len(df)} rows")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", default=None, choices=ASYM_VARIANTS,
                        help="Asymmetry variant; if omitted, uses "
                             "embedding_classification (original).")
    args = parser.parse_args()

    root, out, csv_subpath = resolve_paths(args.variant)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"ROOT: {root}")
    logger.info(f"OUT : {out}")

    df = load_all(root, csv_subpath)
    if df.empty:
        logger.warning("no data found; nothing to write")
        return
    out_csv = out / "all_thresholds_metrics.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv} ({len(df)} rows)")

    # Wide AUC pivot — one cell+scope per row, one threshold per column.
    df["cell"] = (df["partition"] + "/" + df["embedding"] + "/"
                  + df["classifier"] + "/" + df["strategy"] + "/" + df["scope"])
    pivot = df.pivot_table(
        index=["partition", "embedding", "classifier", "strategy", "scope"],
        columns="drop_root", values="auc", aggfunc="first",
    )

    col_order = ["no_drop"] + [f"drop_{t}" for t in
                               (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
                                0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3,
                                0.25, 0.2, 0.15, 0.1, 0.05)]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    pivot_csv = out / "auc_by_threshold_pivot.csv"
    pivot.to_csv(pivot_csv)
    logger.info(f"Wrote {pivot_csv}")

    # Line plots — one per partition, AUC vs threshold, per (embedding, classifier, scope).
    plot_scopes = ["forward_matched", "reverse_ensemble_matched_oof",
                   "reverse_ensemble_full"]
    line_df = df[df["scope"].isin(plot_scopes)].copy()
    # Numeric x: threshold (lower → more aggressive drop). no_drop → 1.0 placeholder.
    line_df["x"] = line_df["drop_threshold"].fillna(1.0)

    partitions = sorted(line_df["partition"].unique())
    for part in partitions:
        sub = line_df[line_df["partition"] == part]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, scope in zip(axes, plot_scopes):
            scope_df = sub[sub["scope"] == scope]
            for (emb, clf), grp in scope_df.groupby(["embedding", "classifier"]):
                grp = grp.sort_values("x")
                ax.plot(grp["x"], grp["auc"], marker="o",
                        label=f"{emb}/{clf}", linewidth=1.5)
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xlabel("drop threshold (1.0 = no_drop)")
            ax.set_title(scope)
            ax.set_xlim(1.02, 0.0)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("AUC")
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                        fontsize=8)
        fig.suptitle(f"{part} — AUC vs drop_correlated threshold")
        fig.tight_layout()
        png = out / f"auc_by_threshold_{part}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {png}")


if __name__ == "__main__":
    main()
