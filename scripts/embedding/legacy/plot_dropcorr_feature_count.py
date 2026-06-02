"""
Plot the number of features retained by DropCorrelatedFeatures at each
threshold, per embedding model.

Reads `embedding/analysis/classification/<variant>/<cohort>/drop_feats/
pearson_r_X.X/<partition>/fwd/<emb>/logistic/forward_matched_metrics.json`
(`<variant>` defaults to `original`, switch with `--variant`).

Output:
    embedding/analysis/classification/<variant>/<cohort>/drop_feats/_summary/
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
import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (DEFAULT_COHORT_TOKENS, P_VISIT_TOKENS, P_SCORE_TOKENS,
                        HC_VISIT_TOKENS, HC_SCORE_TOKENS)
ASYM_VARIANTS = ["difference", "absolute_difference", "average",
                 "relative_differences", "absolute_relative_differences"]

EMBEDDINGS = ["arcface", "topofr", "dlib"]
PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs", "mmse_hilo", "casi_hilo"]
COLORS = {"arcface": "#1f77b4", "topofr": "#ff7f0e", "dlib": "#2ca02c"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_paths(variant, embedding, bg_mode="no_background",
                   cohort=DEFAULT_COHORT_TOKENS,
                   photo_mode="mean", match_strategy="no_priority"):
    from src.config import EMBEDDING_CLASSIFICATION_DIR, cohort_dirs
    visit_dir, cdr_mmse_dir = cohort_dirs(*cohort)
    v = variant if variant is not None else "original"
    class_root = (EMBEDDING_CLASSIFICATION_DIR / visit_dir / cdr_mmse_dir
                  / bg_mode / embedding / v / photo_mode)
    out = class_root / "drop_feats" / "_summary"
    reducer_dirs = []
    if class_root.is_dir():
        seen = set()
        for marker_name in ("fwd", "rev"):
            for marker in class_root.rglob(marker_name):
                if marker.is_dir():
                    seen.add(marker.parent.parent)
        for reducer in sorted(seen):
            rel_parts = reducer.relative_to(class_root).parts
            if any(p.startswith("_") for p in rel_parts):
                continue
            if rel_parts[0] not in ("no_drop", "drop_feats"):
                continue
            reducer_dirs.append(reducer)

    def cell_json_for(reducer, part, clf):
        clf_sub = clf
        if clf == "logistic":
            clf_sub = clf + "/C_1"
        return (reducer / clf_sub / "fwd" / "1by1matched"
                / "subject_match" / "eval_by_subject"
                / match_strategy / part / "forward_matched_metrics.json")

    return class_root, out, reducer_dirs, cell_json_for


def threshold_value(name):
    if name == "no_drop":
        return None
    m = re.match(r"pearson_r_([0-9.]+)$", name)
    return float(m.group(1)) if m else None


def collect(reducer_dirs, cell_json_for):
    rows = []
    for sub in reducer_dirs:
        thr = threshold_value(sub.name)
        for part in PARTITIONS:
            p = cell_json_for(sub, part, "logistic")
            if not p.exists():
                continue
            payload = json.loads(p.read_text(encoding="utf-8"))
            emb = payload.get("embedding", "unknown")
            info = payload.get("drop_corr_info", {})
            kept = info.get("n_features_kept_per_fold")
            n_input = info.get("n_features_input")
            if kept is None or n_input is None:
                n_input = {"arcface": 512, "topofr": 512, "dlib": 128,
                           "vggface": 4096}.get(emb, 512)
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
    parser.add_argument("--embedding", default="arcface",
                        choices=["arcface", "topofr", "dlib", "vggface"])
    parser.add_argument("--bg-mode", default="no_background",
                        choices=["background", "no_background"])
    parser.add_argument("--p-visit",  choices=list(P_VISIT_TOKENS),  default=DEFAULT_COHORT_TOKENS[0])
    parser.add_argument("--p-score",  choices=list(P_SCORE_TOKENS),  default=DEFAULT_COHORT_TOKENS[1])
    parser.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    parser.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])
    args = parser.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    class_root, out, reducer_dirs, cell_json_for = resolve_paths(
        args.variant, args.embedding, args.bg_mode, cohort
    )
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"ROOT: {class_root}")
    logger.info(f"OUT : {out}")

    df = collect(reducer_dirs, cell_json_for)
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
