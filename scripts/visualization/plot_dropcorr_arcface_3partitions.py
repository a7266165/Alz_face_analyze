"""
ArcFace forward-strategy logistic — MCC / AUC / BalAcc vs DropCorrelatedFeatures
threshold, for the 3 disease partitions (ad_vs_hc / ad_vs_nad / ad_vs_acs),
shown for both full cohort and matched (selected) subset.

Output: embedding_classification/_dropcorr_summary/
        arcface_3partitions_metrics_by_threshold.png

Usage:
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_dropcorr_arcface_3partitions.py
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "workspace" / "arms_analysis" / "p_first_hc_strict" / "embedding_classification"
OUT = ROOT / "_dropcorr_summary"

PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
COLORS = {"ad_vs_hc": "#1f77b4", "ad_vs_nad": "#d62728", "ad_vs_acs": "#2ca02c"}
METRICS = [("mcc", "MCC"), ("auc", "AUC"), ("balacc", "Balanced accuracy")]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    df = pd.read_csv(OUT / "all_thresholds_metrics.csv")
    df = df[(df["embedding"] == "arcface")
            & (df["classifier"] == "logistic")
            & (df["strategy"] == "forward")
            & (df["partition"].isin(PARTITIONS))
            & (df["scope"].isin(["forward_full", "forward_matched"]))].copy()
    df["x"] = df["drop_threshold"].fillna(1.0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    scope_label = {"forward_full": "Full cohort", "forward_matched": "Selected (matched)"}

    for row, scope in enumerate(["forward_full", "forward_matched"]):
        for col, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row, col]
            sub = df[df["scope"] == scope]
            for part in PARTITIONS:
                grp = sub[sub["partition"] == part].sort_values("x")
                ax.plot(grp["x"], grp[metric], marker="o",
                        color=COLORS[part], linewidth=1.8, label=part)
            chance = 0.0 if metric == "mcc" else 0.5
            ax.axhline(chance, color="grey", linestyle=":", linewidth=0.8)
            ax.axvline(1.0, color="grey", linestyle=":", linewidth=0.6)
            ax.set_xlim(1.05, 0.0)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(metric_label)
            if col == 0:
                ax.set_ylabel(scope_label[scope])
            if row == 1:
                ax.set_xlabel("DropCorrelatedFeatures threshold (1.0 = no drop)")

    axes[0, -1].legend(loc="lower left", fontsize=9)
    fig.suptitle("ArcFace + Logistic — metrics vs drop_correlated threshold (3 partitions)",
                 fontsize=13)
    fig.tight_layout()
    png = OUT / "arcface_3partitions_metrics_by_threshold.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {png}")


if __name__ == "__main__":
    main()
