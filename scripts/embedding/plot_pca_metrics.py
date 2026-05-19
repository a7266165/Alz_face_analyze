"""
Per-partition PCA-sweep metric plot: AUC + Balanced Accuracy + MCC vs PCA
n_components, with the cumulative eigenvalue ratio panel below.

Two directions share the same plot scaffold:
  --direction fwd  → scopes {forward_full, forward_matched}
                     output: pca/_summary/fwd/forward_metrics_{full|matched}_by_pca_<partition>.png
  --direction rev  → scopes {reverse_matched_oof, reverse_unmatched}
                     output: pca/_summary/rev/reverse_{matched_oof|unmatched}_by_pca_<partition>.png
                     (also shades AUC CI when auc_ci_low/auc_ci_high present)

Reverse classifiers are trained on the matched 1:1 cohort via 10-fold
GroupKFold and evaluated on:
  - matched_oof: subject-level OOF on the matched cohort (validation domain)
  - unmatched:   ensemble of fold-models on unmatched subjects (held-out)

Reads the long-form metrics + cumulative_eigenvalue_ratio.csv produced by
plot_pca_components_sweep.py
(embedding/analysis/classification/<variant>/<cohort>/pca/_summary/...).

Usage:
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_pca_metrics.py --direction fwd
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_pca_metrics.py --direction rev --variant difference
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


def resolve_paths(variant, cohort_mode="p_first_cdr05_hc_first_cdrall_or_mmseall"):
    from src.config import EMBEDDING_CLASSIFICATION_DIR, cohort_name
    cohort_dir = cohort_name(cohort_mode)
    v = variant if variant is not None else "original"
    return EMBEDDING_CLASSIFICATION_DIR / v / cohort_dir / "pca" / "_summary"


INPUT_DIM = {"arcface": 512, "topofr": 512, "dlib": 128}
EMB_CLF_COLOR = {
    ("arcface", "logistic"): "#5da3d9",
    ("arcface", "xgb"):      "#08306b",
    ("topofr",  "logistic"): "#ffb061",
    ("topofr",  "xgb"):      "#8c3a04",
    ("dlib",    "logistic"): "#6dc06d",
    ("dlib",    "xgb"):      "#0d4d10",
}
EMB_COLOR = {"arcface": "#1f77b4", "topofr": "#ff7f0e", "dlib": "#2ca02c"}
EMB_CLF_LINESTYLE = {"logistic": "-", "xgb": "--"}

METRICS = [("auc", "AUC", 0.5), ("balacc", "Balanced accuracy", 0.5),
           ("mcc", "MCC", 0.0)]

# scope value in CSV → (file-name tag, panel label, output-image prefix)
# file_tag is chosen so that alphabetical FS sort within each direction
# folder yields the intended order:
#   fwd: full (f) < matched (m)        → full first, matched second
#   rev: oof  (o) < unmatched (u)      → oof first, unmatched second
# This avoids needing a numeric scope_idx prefix in the filename.
SCOPES_BY_DIRECTION = {
    "fwd": [
        ("forward_full",    "full",     "forward_full",        "forward_metrics"),
        ("forward_matched", "matched",  "forward_matched",     "forward_metrics"),
    ],
    "rev": [
        ("reverse_matched_oof", "oof",       "reverse_matched_oof", "reverse"),
        ("reverse_unmatched",   "unmatched", "reverse_unmatched",   "reverse"),
    ],
}

# Display label for each partition (used in suptitle).
# Insertion order also defines the output-file ordering: filenames get
# `{idx}_<part>_<scope>.png` so alphabetical FS display matches this order.
PARTITION_LABELS = {
    "ad_vs_hc":  "AD vs HC",
    "ad_vs_nad": "AD vs NAD",
    "ad_vs_acs": "AD vs ACS",
    "mmse_hilo": "MMSE Hi-Lo",
    "casi_hilo": "CASI Hi-Lo",
}
PARTITION_ORDER = list(PARTITION_LABELS.keys())

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_pca_label(name):
    """Parse n_components_<int> reducer-dir name into integer x; everything
    else (no_drop, var_ratio_*, drop_feats/*) returns None."""
    m = re.match(r"n_components_([0-9]+)", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _draw_partition_scope(df, eig_df, part, scope_val, scope_label,
                          shade_ci):
    sub = df[(df["partition"] == part) & (df["scope"] == scope_val)]
    if not len(sub):
        return None
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True,
                              sharey="row")
    EMBEDDINGS = ["arcface", "topofr", "dlib"]
    for row_idx, (metric_col, metric_label, chance) in enumerate(METRICS):
        for col_idx, emb in enumerate(EMBEDDINGS):
            ax = axes[row_idx, col_idx]
            emb_sub = sub[sub["embedding"] == emb]
            for (clf, lr_C), grp in emb_sub.groupby(
                ["classifier", "lr_C"], dropna=False
            ):
                grp = grp.sort_values("pca_x")
                label = (f"{clf}/C={lr_C:g}"
                         if clf == "logistic" and pd.notna(lr_C) else clf)
                color = EMB_CLF_COLOR.get((emb, clf), EMB_COLOR.get(emb))
                linestyle = EMB_CLF_LINESTYLE.get(clf, "-")
                if clf == "logistic" and pd.notna(lr_C) and float(lr_C) != 1.0:
                    linestyle = ":"
                ax.plot(grp["pca_x"], grp[metric_col], marker="o",
                        label=label, linewidth=1.5, color=color,
                        linestyle=linestyle)
                if (shade_ci
                        and metric_col == "auc"
                        and "auc_ci_low" in grp.columns
                        and "auc_ci_high" in grp.columns
                        and grp[["auc_ci_low", "auc_ci_high"]].notna().all().all()):
                    ax.fill_between(grp["pca_x"],
                                    grp["auc_ci_low"],
                                    grp["auc_ci_high"],
                                    color=color, alpha=0.15, linewidth=0)
            ax.axhline(chance, color="grey", linestyle=":", linewidth=0.8)
            ax.grid(alpha=0.3, which="both")
            ax.tick_params(labelsize=15)
            if row_idx == 0:
                ax.set_title(f"{emb} (input dim={INPUT_DIM[emb]})", fontsize=18)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=15, labelpad=15)
            ax.legend(loc="lower right", fontsize=14)

    for col_idx, emb in enumerate(EMBEDDINGS):
        ax = axes[3, col_idx]
        sub_e = eig_df[eig_df["embedding"] == emb]
        if len(sub_e):
            ax.plot(sub_e["n_components"],
                    sub_e["cumulative_variance_ratio"],
                    color=EMB_COLOR.get(emb), linewidth=1.8)
        ax.axhline(0.95, color="grey", linestyle=":", linewidth=0.6)
        ax.axhline(0.99, color="grey", linestyle=":", linewidth=0.6)
        ax.text(1.05, 0.95, "0.95", fontsize=7, color="grey", va="center")
        ax.text(1.05, 0.99, "0.99", fontsize=7, color="grey", va="center")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=15)
        if col_idx == 0:
            ax.set_ylabel("Cumulative\neigenvalue / total", fontsize=15, labelpad=15)

    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlim(1, 512)

    part_label = PARTITION_LABELS.get(part, part)
    fig.suptitle(f"{part_label} — {scope_label}", fontsize=45)
    fig.supxlabel("PCA n_components", fontsize=30)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.88, bottom=0.10,
                        hspace=0.30, wspace=0.06)
    return fig


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--direction", choices=["fwd", "rev"], required=True,
                        help="fwd: forward_{full,matched}; "
                             "rev: reverse_{matched_oof,unmatched}.")
    parser.add_argument("--variant", default=None,
                        choices=ASYM_VARIANTS + ["original", "original_background"],
                        help="Variant under classification/.  Default None == 'original'.")
    from src.config import VALID_COHORT_CHOICES
    parser.add_argument("--cohort-mode", default="p_first_cdr05_hc_first_cdrall_or_mmseall",
                        choices=VALID_COHORT_CHOICES)
    args = parser.parse_args()

    out = resolve_paths(args.variant, args.cohort_mode)
    logger.info(f"OUT: {out}")
    long_csv = out / "all_pca_metrics.csv"
    eig_csv = out / "cumulative_eigenvalue_ratio.csv"
    if not long_csv.exists() or not eig_csv.exists():
        raise SystemExit(
            f"Missing {long_csv} or {eig_csv}. Run "
            "plot_pca_components_sweep.py first."
        )
    df = pd.read_csv(long_csv)
    eig_df = pd.read_csv(eig_csv)

    if "pca_x" not in df.columns:
        df["pca_x"] = df["pca_root"].map(_parse_pca_label)

    scopes = SCOPES_BY_DIRECTION[args.direction]
    target_scopes = [s[0] for s in scopes]
    df = df[df["pca_x"].notna() & df["scope"].isin(target_scopes)].copy()

    shade_ci = (args.direction == "rev")
    direction_dir = out / args.direction
    direction_dir.mkdir(parents=True, exist_ok=True)

    # Custom partition ordering: PARTITION_ORDER first, then any unknown
    # partitions tail-appended in alphabetical order so they remain reachable.
    found = set(df["partition"].unique())
    ordered = [p for p in PARTITION_ORDER if p in found]
    unknown = sorted(p for p in found if p not in PARTITION_ORDER)
    partitions = ordered + unknown
    # Filename pattern: {file_tag}_{part_idx}_{part}.png
    #   file_tag (alphabetical) gives scope ordering — see SCOPES_BY_DIRECTION
    #   part_idx 1-5 (from PARTITION_ORDER) orders within each scope group as
    #     hc, nad, acs, mmse, casi
    for part_idx, part in enumerate(partitions, start=1):
        for scope_val, file_tag, scope_label, prefix in scopes:
            fig = _draw_partition_scope(df, eig_df, part, scope_val,
                                        scope_label, shade_ci=shade_ci)
            if fig is None:
                continue
            png = direction_dir / f"{file_tag}_{part_idx}_{part}.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
            logger.info(f"Wrote {png}")


if __name__ == "__main__":
    main()
