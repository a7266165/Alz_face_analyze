"""
Per-partition reverse PCA plot: AUC + Balanced Accuracy + MCC vs PCA
n_components, with the cumulative eigenvalue ratio panel below.

Mirror of plot_pca_forward_metrics.py for the reverse strategy. Reverse
classifiers are trained on the matched 1:1 cohort and evaluated either on
matched-OOF (age-controlled) or on the full cohort (ensemble inference).

Reads the long-form metrics + cumulative_eigenvalue_ratio.csv produced by
plot_pca_components_sweep.py (default mode -> embedding_classification;
--variant <v> -> embedding_asymmetry_classification/<v>).

Output (per partition x scope):
    embedding_classification/_pca_summary/rev/
        reverse_ens_full_by_pca_<partition>.png
        reverse_ens_matched_oof_by_pca_<partition>.png
    embedding_asymmetry_classification/_pca_summary/<variant>/rev/
        ...

Usage:
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_pca_reverse_metrics.py
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_pca_reverse_metrics.py --variant average
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARMS_ROOT = PROJECT_ROOT / "workspace" / "arms_analysis"
ASYM_VARIANTS = ["difference", "absolute_difference", "average",
                 "relative_differences", "absolute_relative_differences"]


def resolve_paths(variant, cohort_mode="default"):
    cohort_dir = "p_first_hc_all" if cohort_mode == "p_first_hc_all" else "p_first_hc_strict"
    if variant is None:
        return ARMS_ROOT / cohort_dir / "embedding_classification" / "_pca_summary"
    return (ARMS_ROOT / cohort_dir / "embedding_asymmetry_classification"
            / "_pca_summary" / variant)

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

# (scope value in CSV, file-name tag, panel label)
REV_SCOPES = [
    ("reverse_ensemble_full",        "full",        "reverse_ensemble_full"),
    ("reverse_ensemble_matched_oof", "matched_oof", "reverse_ensemble_matched_oof"),
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_pca_label(name):
    if name.startswith("no_drop") or not name.startswith("pca_"):
        return None
    import re
    m = re.match(r"pca_([0-9.]+)", name)
    if not m:
        return None
    try:
        x = float(m.group(1))
    except ValueError:
        return None
    return x if x >= 1 else None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", default=None, choices=ASYM_VARIANTS)
    parser.add_argument("--cohort-mode", default="default",
                        choices=["default", "p_first_hc_all"])
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

    target_scopes = [s[0] for s in REV_SCOPES]
    df = df[df["pca_x"].notna() & df["scope"].isin(target_scopes)].copy()

    EMBEDDINGS = ["arcface", "topofr", "dlib"]
    partitions = sorted(df["partition"].unique())
    for part in partitions:
      for scope_val, scope_tag, scope_label in REV_SCOPES:
        sub = df[(df["partition"] == part) & (df["scope"] == scope_val)]
        if not len(sub):
            continue
        fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True,
                                   sharey="row")
        for row_idx, (metric_col, metric_label, chance) in enumerate(METRICS):
            for col_idx, emb in enumerate(EMBEDDINGS):
                ax = axes[row_idx, col_idx]
                emb_sub = sub[sub["embedding"] == emb]
                for clf, grp in emb_sub.groupby("classifier"):
                    grp = grp.sort_values("pca_x")
                    color = EMB_CLF_COLOR.get((emb, clf), EMB_COLOR.get(emb))
                    ax.plot(grp["pca_x"], grp[metric_col], marker="o",
                            label=clf, linewidth=1.5, color=color,
                            linestyle=EMB_CLF_LINESTYLE.get(clf, "-"))
                    if (metric_col == "auc"
                            and "auc_ci_low" in grp.columns
                            and "auc_ci_high" in grp.columns
                            and grp[["auc_ci_low", "auc_ci_high"]].notna().all().all()):
                        ax.fill_between(grp["pca_x"],
                                         grp["auc_ci_low"],
                                         grp["auc_ci_high"],
                                         color=color, alpha=0.15, linewidth=0)
                ax.axhline(chance, color="grey", linestyle=":", linewidth=0.8)
                ax.grid(alpha=0.3, which="both")
                if row_idx == 0:
                    ax.set_title(f"{emb} (input dim={INPUT_DIM[emb]})")
                if col_idx == 0:
                    ax.set_ylabel(metric_label)
                if row_idx == 0 and col_idx == len(EMBEDDINGS) - 1:
                    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                              fontsize=8)

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
            ax.set_xlabel("PCA n_components")
            if col_idx == 0:
                ax.set_ylabel("Cumulative\neigenvalue / total")

        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlim(1, 512)

        fig.suptitle(f"{part} — {scope_label} metrics + cumulative "
                     f"eigenvalue ratio vs PCA n_components",
                     fontsize=13)
        fig.subplots_adjust(left=0.07, right=0.92, top=0.94, bottom=0.05,
                              hspace=0.18, wspace=0.06)
        rev_dir = out / "rev"
        rev_dir.mkdir(parents=True, exist_ok=True)
        png = rev_dir / f"reverse_ens_{scope_tag}_by_pca_{part}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        logger.info(f"Wrote {png}")


if __name__ == "__main__":
    main()
