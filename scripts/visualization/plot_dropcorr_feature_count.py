"""
Plot the number of features retained by DropCorrelatedFeatures at each
threshold, per embedding model.

Default mode reads `drop_X.X/fwd/<partition>/<emb>/logistic/forward_matched_metrics.json`.
With --variant, reads `drop_X.X/<variant>/fwd/<partition>/<emb>/logistic/...`.

Outputs:
    embedding_classification/_dropcorr_summary/                   (default)
    embedding_asymmetry_classification/_dropcorr_summary/<variant>/  (--variant)
        feature_count_by_threshold.csv
        feature_count_by_threshold.png

Usage:
    # original
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_dropcorr_feature_count.py
    # asymmetry variant
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_dropcorr_feature_count.py --variant difference
"""
import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARMS_ROOT = PROJECT_ROOT / "workspace" / "arms_analysis"
ASYM_VARIANTS = ["difference", "absolute_difference", "average",
                 "relative_differences", "absolute_relative_differences"]

EMBEDDINGS = ["arcface", "topofr", "dlib"]
PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs", "mmse_hilo", "casi_hilo"]
COLORS = {"arcface": "#1f77b4", "topofr": "#ff7f0e", "dlib": "#2ca02c"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_paths(variant):
    if variant is None:
        root = ARMS_ROOT / "p_first_hc_strict" / "embedding_classification"
        out = root / "_dropcorr_summary"
        # cell json at <reducer>/fwd/<part>/<emb>/<clf>/forward_matched_metrics.json
        return root, out, lambda reducer, part, emb, clf: (
            reducer / "fwd" / part / emb / clf / "forward_matched_metrics.json"
        )
    root = ARMS_ROOT / "p_first_hc_strict" / "embedding_asymmetry_classification"
    out = root / "_dropcorr_summary" / variant
    return root, out, lambda reducer, part, emb, clf: (
        reducer / variant / "fwd" / part / emb / clf / "forward_matched_metrics.json"
    )


def threshold_value(name):
    if name == "no_drop":
        return None
    m = re.match(r"drop_([0-9.]+)$", name)
    return float(m.group(1)) if m else None


def collect(root, cell_json_for):
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name != "no_drop" and not sub.name.startswith("drop_"):
            continue
        thr = threshold_value(sub.name)
        for part in PARTITIONS:
            for emb in EMBEDDINGS:
                p = cell_json_for(sub, part, emb, "logistic")
                if not p.exists():
                    continue
                payload = json.loads(p.read_text(encoding="utf-8"))
                info = payload.get("drop_corr_info", {})
                kept = info.get("n_features_kept_per_fold")
                n_input = info.get("n_features_input")
                if kept is None or n_input is None:
                    # no_drop: no drop_corr_info → use input dim from a known drop dir
                    n_input = {"arcface": 512, "topofr": 512, "dlib": 128}[emb]
                    kept = [n_input] * 10
                for fold, n in enumerate(kept):
                    rows.append({
                        "drop_root": sub.name,
                        "drop_threshold": thr,
                        "partition": part,
                        "embedding": emb,
                        "fold": fold,
                        "n_kept": n,
                        "n_input": n_input,
                    })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", default=None, choices=ASYM_VARIANTS)
    args = parser.parse_args()

    root, out, cell_json_for = resolve_paths(args.variant)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"ROOT: {root}")
    logger.info(f"OUT : {out}")

    df = collect(root, cell_json_for)
    if df.empty:
        logger.warning("no data found; nothing to write")
        return
    csv_path = out / "feature_count_by_threshold.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path} ({len(df)} rows)")

    # Aggregate: mean ± std across (partition × fold) per (embedding, threshold).
    df["x"] = df["drop_threshold"].fillna(1.0)
    agg = (df.groupby(["embedding", "x"])
             .agg(n_kept_mean=("n_kept", "mean"),
                  n_kept_std=("n_kept", "std"),
                  n_kept_min=("n_kept", "min"),
                  n_kept_max=("n_kept", "max"),
                  n_input=("n_input", "first"))
             .reset_index())

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for emb in EMBEDDINGS:
        sub = agg[agg["embedding"] == emb].sort_values("x")
        n_in = int(sub["n_input"].iloc[0])
        color = COLORS[emb]
        ax.plot(sub["x"], sub["n_kept_mean"], marker="o", color=color,
                linewidth=2, label=f"{emb} (input dim={n_in})")
        ax.fill_between(sub["x"],
                        sub["n_kept_mean"] - sub["n_kept_std"],
                        sub["n_kept_mean"] + sub["n_kept_std"],
                        color=color, alpha=0.15)
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.text(1.005, ax.get_ylim()[1] * 0.95, "no_drop", color="grey",
            fontsize=8, va="top")
    ax.set_xlim(1.05, 0.0)
    ax.set_xlabel("DropCorrelatedFeatures threshold (1.0 = no drop)")
    ax.set_ylabel("Features retained (mean across 5 partitions × 10 folds)")
    ax.set_title("Feature count vs drop_correlated threshold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = out / "feature_count_by_threshold.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {png}")

    logger.info("=== summary table ===")
    pivot = agg.pivot(index="x", columns="embedding", values="n_kept_mean")
    pivot = pivot.sort_index(ascending=False)
    print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
