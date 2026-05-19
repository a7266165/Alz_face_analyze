"""
Aggregate drop_feats reducer dirs (pearson_r_X.X) plus no_drop reference into
a cross-threshold AUC comparison.

Reads (`<variant>` defaults to `original`, switch with `--variant`):
    embedding/analysis/classification/<variant>/<cohort>/no_drop/_summary/all_metrics_with_cm.csv
    embedding/analysis/classification/<variant>/<cohort>/drop_feats/pearson_r_X.X/_summary/all_metrics_with_cm.csv

Output:
    embedding/analysis/classification/<variant>/<cohort>/drop_feats/_summary/

Usage:
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_dropcorr_threshold_sweep.py
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_dropcorr_threshold_sweep.py --variant difference --cohort-mode p_all_hc_all
"""
import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT))
ASYM_VARIANTS = ["difference", "absolute_difference", "average",
                 "relative_differences", "absolute_relative_differences"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_paths(variant, cohort_mode="default"):
    """Return (class_root, out, reducer_dirs)."""
    from src.config import EMBEDDING_CLASSIFICATION_DIR, cohort_name
    cohort_dir = cohort_name(cohort_mode)
    v = variant if variant is not None else "original"
    class_root = EMBEDDING_CLASSIFICATION_DIR / v / cohort_dir
    out = class_root / "drop_feats" / "_summary"
    reducer_dirs = []
    if class_root.is_dir():
        # NEW layout: class_root/<reducer>/<partition>/{fwd,rev}/<emb>/<clf>/
        seen = set()
        for marker_name in ("fwd", "rev"):
            for marker in class_root.rglob(marker_name):
                if marker.is_dir():
                    seen.add(marker.parent.parent)
        for reducer in sorted(seen):
            rel_parts = reducer.relative_to(class_root).parts
            if any(p.startswith("_") for p in rel_parts):
                continue
            # drop_feats plot covers no_drop reference + drop_feats reducers.
            if rel_parts[0] not in ("no_drop", "drop_feats"):
                continue
            reducer_dirs.append(reducer)
    return class_root, out, reducer_dirs


def threshold_label(name):
    if name == "no_drop":
        return None
    m = re.match(r"pearson_r_([0-9.]+)$", name)
    return float(m.group(1)) if m else None


def load_all(reducer_dirs):
    rows = []
    for sub in reducer_dirs:
        csv = sub / "_summary" / "all_metrics_with_cm.csv"
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
                        help="Asymmetry variant; if omitted, uses original.")
    from src.config import VALID_COHORT_CHOICES
    parser.add_argument("--cohort-mode", default="default",
                        choices=VALID_COHORT_CHOICES)
    args = parser.parse_args()

    class_root, out, reducer_dirs = resolve_paths(args.variant, args.cohort_mode)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"ROOT: {class_root}")
    logger.info(f"OUT : {out}")
    logger.info(f"reducers: {[r.name for r in reducer_dirs]}")

    df = load_all(reducer_dirs)
    if df.empty:
        logger.warning("no data found; nothing to write")
        return
    out_csv = out / "all_thresholds_metrics.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv} ({len(df)} rows)")

    df["cell"] = (df["partition"] + "/" + df["embedding"] + "/"
                  + df["classifier"] + "/" + df["strategy"] + "/" + df["scope"])
    pivot = df.pivot_table(
        index=["partition", "embedding", "classifier", "strategy", "scope"],
        columns="drop_root", values="auc", aggfunc="first",
    )

    col_order = ["no_drop"] + [f"pearson_r_{t}" for t in
                               (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
                                0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3,
                                0.25, 0.2, 0.15, 0.1, 0.05)]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    pivot_csv = out / "auc_by_threshold_pivot.csv"
    pivot.to_csv(pivot_csv)
    logger.info(f"Wrote {pivot_csv}")

    plot_scopes = ["forward_matched", "reverse_matched_oof", "reverse_unmatched"]
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
