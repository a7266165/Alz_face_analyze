"""
Aggregate PCA reducer dirs (n_components_1..400, plus var_ratio_0.95/0.99 and
no_drop reference) into a cross-component AUC comparison.

Reads (`<variant>` defaults to `original`, switch with `--variant`):
    embedding/analysis/classification/<variant>/<cohort>/pca/n_components_<n>/_summary/all_metrics_with_cm.csv
    embedding/analysis/classification/<variant>/<cohort>/pca/var_ratio_<r>/_summary/all_metrics_with_cm.csv
    embedding/analysis/classification/<variant>/<cohort>/no_drop/_summary/all_metrics_with_cm.csv

For non-`original` variants, cumulative eigenvalues are computed from that
variant's feature matrix (workspace/embedding/features/<emb>/<variant>/).

Output:
    embedding/analysis/classification/<variant>/<cohort>/pca/_summary/

Usage:
    # original
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_pca_components_sweep.py
    # asymmetry variant
    conda run -n Alz_face_main_analysis python scripts/visualization/plot_pca_components_sweep.py --variant difference
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
ASYM_VARIANTS = ["difference", "absolute_difference", "average",
                 "relative_differences", "absolute_relative_differences"]

EMBEDDING_FEAT_DIR = (PROJECT_ROOT / "workspace" / "embedding" / "features")
EMBEDDING_ABTEST_FEAT_DIR = (PROJECT_ROOT / "workspace" / "embedding_ABtest" / "features")
INPUT_DIM = {"arcface": 512, "topofr": 512, "dlib": 128}
EMB_COLOR = {"arcface": "#1f77b4", "topofr": "#ff7f0e", "dlib": "#2ca02c"}
# Per (embedding, classifier): same hue per embedding but lighter for LR /
# darker for XGB so the two classifiers are distinguishable at a glance.
EMB_CLF_COLOR = {
    ("arcface", "logistic"): "#5da3d9",  # light blue
    ("arcface", "xgb"):      "#08306b",  # dark blue
    ("topofr",  "logistic"): "#ffb061",  # light orange
    ("topofr",  "xgb"):      "#8c3a04",  # dark orange
    ("dlib",    "logistic"): "#6dc06d",  # light green
    ("dlib",    "xgb"):      "#0d4d10",  # dark green
}
EMB_CLF_LINESTYLE = {"logistic": "-", "xgb": "--"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_paths(variant, cohort_mode="default", abtest=False):
    """Return (class_root, out, reducer_dirs, cell_json_for, feature_subdir, feat_root).

    class_root             → <classification_root>/<variant>/<cohort>
    feat_root              → workspace/embedding[_ABtest]/features
    reducer_dirs           → list of reducer dirs to walk
                              (no_drop + pca/n_components_X + pca/var_ratio_X)
    cell_json_for(reducer, part, emb, clf) → cell metrics json
    feature_subdir         → which sub-dir of <feat_root>/<emb>/ to use for
                              cumulative eigenvalue ('original' for default,
                              variant name for asymmetry / original_background).
    """
    from src.config import (
        EMBEDDING_CLASSIFICATION_DIR,
        EMBEDDING_ABTEST_CLASSIFICATION_DIR,
        cohort_name,
    )
    cohort_dir = cohort_name(cohort_mode)
    feature_subdir = variant if variant is not None else "original"
    classification_root = (
        EMBEDDING_ABTEST_CLASSIFICATION_DIR if abtest else EMBEDDING_CLASSIFICATION_DIR
    )
    class_root = classification_root / feature_subdir / cohort_dir
    out = class_root / "pca" / "_summary"
    feat_root = EMBEDDING_ABTEST_FEAT_DIR if abtest else EMBEDDING_FEAT_DIR

    def _reducer_dirs():
        if not class_root.is_dir():
            return []
        # NEW layout: class_root/<reducer>/<partition>/{fwd,rev}/<emb>/<clf>/
        seen = set()
        for marker_name in ("fwd", "rev"):
            for marker in class_root.rglob(marker_name):
                if marker.is_dir():
                    seen.add(marker.parent.parent)
        out = []
        for reducer in sorted(seen):
            rel_parts = reducer.relative_to(class_root).parts
            if any(p.startswith("_") for p in rel_parts):
                continue
            # PCA components plot only cares about no_drop + pca/ subtree.
            if rel_parts[0] not in ("no_drop", "pca"):
                continue
            out.append(reducer)
        return out

    def cell_json_for(reducer, part, emb, clf):
        # NEW layout: reducer/<partition>/<fwd|rev>/<emb>/<clf>/
        base = reducer / part / "fwd" / emb / clf
        if clf == "logistic":
            base = base / "C_1"
        return base / "forward_matched_metrics.json"

    return class_root, out, _reducer_dirs(), cell_json_for, feature_subdir, feat_root


def _parse_pca_label(name):
    """Reducer dir name → numeric x-axis position. Returns the integer
    n_components for `n_components_<int>`, None for everything else
    (variance ratios, no_drop, drop_feats).

    Strips visit/photo mode suffix (e.g. `n_components_100__visit_all` -> 100).
    """
    m = re.match(r"n_components_([0-9]+)", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _load_visit_all_cohort_ids():
    """Build the union of visit=all cohort IDs across the 5 partitions
    (ad_vs_hc / ad_vs_nad / ad_vs_acs / mmse_hilo / casi_hilo).

    Each partition's visit-all cohort = subjects passing per-visit eligibility
    (Global_CDR>=0.5 for AD; strict HC criteria for HC; metric+age present for
    hi-lo). Union ≈ 3478 unique visit IDs (cf. ad_vs_hc visit=all = 3478,
    hilo cohorts overlap with the AD subset).

    Returns: set of visit-level IDs (e.g. {"P1-2", "P1-3", "ACS5-1", ...}).
    """
    DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
    frames = []
    for grp in ["P", "NAD", "ACS", "EACS"]:
        path = DEMOGRAPHICS_DIR / f"{grp}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "ID" not in df.columns:
            for col in df.columns:
                if col in ("ACS", "NAD"):
                    df = df.rename(columns={col: "ID"})
                    break
        df["group"] = grp if grp != "EACS" else "ACS"
        frames.append(df)
    demo = pd.concat(frames, ignore_index=True)
    demo["Age"] = pd.to_numeric(demo.get("Age"), errors="coerce")
    demo["Global_CDR"] = pd.to_numeric(demo.get("Global_CDR"), errors="coerce")
    demo["MMSE"] = pd.to_numeric(demo.get("MMSE"), errors="coerce")
    demo["CASI"] = pd.to_numeric(demo.get("CASI"), errors="coerce")
    demo["base_id"] = demo["ID"].astype(str).str.extract(r"^(.+)-\d+$")[0]
    demo["visit"] = demo["ID"].astype(str).str.extract(r"-(\d+)$").astype(float)

    # AD-side per-visit eligibility (P, CDR>=0.5, Age present)
    ad_mask = (demo["group"] == "P") & (demo["Global_CDR"] >= 0.5) \
              & demo["Age"].notna()

    # HC-side per-visit eligibility (NAD/ACS strict)
    has_cog = demo["Global_CDR"].notna() | demo["MMSE"].notna()
    ok_strict = has_cog & (
        (demo["Global_CDR"] == 0)
        | (demo["Global_CDR"].isna() & (demo["MMSE"] >= 26))
    )
    hc_mask = demo["group"].isin(["NAD", "ACS"]) & ok_strict \
              & demo["Age"].notna()

    # Hi-lo (mmse / casi): AD-only with Global_CDR>=0.5 + metric+Age present
    hilo_mmse_mask = ad_mask & demo["MMSE"].notna()
    hilo_casi_mask = ad_mask & demo["CASI"].notna()

    union = (ad_mask | hc_mask | hilo_mmse_mask | hilo_casi_mask)
    ids = set(demo.loc[union, "ID"].astype(str))
    logger.info(f"visit=all union cohort: {len(ids)} unique IDs "
                f"(ad={ad_mask.sum()}, hc={hc_mask.sum()}, "
                f"mmse={hilo_mmse_mask.sum()}, casi={hilo_casi_mask.sum()})")
    return ids


def compute_cumulative_eigenvalue_ratio(feature_subdir, cohort_mode="all_npy", feat_root=None):
    """For each embedding model, fit PCA on the feature pool determined by
    cohort_mode and return the cumulative explained-variance ratio.

    cohort_mode:
      'all_npy'        : every .npy in the dir, mean-pooled per .npy (subject
                         OR visit, depending on .npy granularity). Default —
                         current 4317-row behavior for visit-level .npy files.
      'visit_all'      : restrict to .npy whose basename matches the union of
                         visit-all cohort IDs (≈ 3478 visits). Use this when
                         analyzing visit=all sweeps so the eigenvalue panel
                         matches the data the classifier actually saw.
    """
    from sklearn.decomposition import PCA
    keep_ids = None
    if cohort_mode == "visit_all":
        keep_ids = _load_visit_all_cohort_ids()
    if feat_root is None:
        feat_root = EMBEDDING_FEAT_DIR
    rows = []
    for emb in ("arcface", "topofr", "dlib"):
        feat_dir = feat_root / emb / feature_subdir
        if not feat_dir.exists():
            logger.warning(f"missing {feat_dir}; skipping {emb}")
            continue
        vecs = []
        for npy in sorted(feat_dir.glob("*.npy")):
            if keep_ids is not None and npy.stem not in keep_ids:
                continue
            a = np.load(npy, allow_pickle=True)
            if a.dtype == object:
                a = list(a.item().values())[0]
            v = a.mean(axis=0) if a.ndim == 2 else a
            vecs.append(v)
        if not vecs:
            continue
        X = np.stack(vecs).astype(float)
        pca = PCA().fit(X)
        cum = np.cumsum(pca.explained_variance_ratio_)
        for k, val in enumerate(cum, start=1):
            rows.append({"embedding": emb, "n_components": k,
                         "cumulative_variance_ratio": float(val)})
        logger.info(f"  {emb}: {X.shape} → cumulative ratio "
                    f"computed for {len(cum)} components")
    return pd.DataFrame(rows)


def collect(reducer_dirs):
    rows = []
    for sub in reducer_dirs:
        csv = sub / "_summary" / "all_metrics_with_cm.csv"
        if not csv.exists():
            logger.warning(f"missing {csv}")
            continue
        df = pd.read_csv(csv)
        df["pca_root"] = sub.name
        df["pca_x"] = _parse_pca_label(sub.name)
        df["pca_label"] = sub.name
        rows.append(df)
        logger.info(f"  {sub.name}: {len(df)} rows")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def collect_feature_counts(reducer_dirs, cell_json_for):
    rows = []
    for sub in reducer_dirs:
        if not sub.name.startswith("n_components_") \
                and not sub.name.startswith("var_ratio_"):
            continue
        for embedding in ["arcface", "topofr", "dlib"]:
            for partition in ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs",
                               "mmse_hilo", "casi_hilo"]:
                p = cell_json_for(sub, partition, embedding, "logistic")
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                info = d.get("drop_corr_info", {})
                kept = info.get("n_features_kept_per_fold")
                n_in = info.get("n_features_input")
                if kept is None or n_in is None:
                    continue
                for fold, n in enumerate(kept):
                    rows.append({
                        "pca_root": sub.name,
                        "pca_x": _parse_pca_label(sub.name),
                        "embedding": embedding,
                        "partition": partition,
                        "fold": fold,
                        "n_kept": int(n),
                        "n_input": int(n_in),
                    })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", default=None,
                        choices=ASYM_VARIANTS + ["original", "original_background"],
                        help="Variant under classification/.  Default None == 'original'.")
    parser.add_argument("--cohort-mode", default="default",
                        choices=["default", "p_first_hc_all", "p_all_hc_all"],
                        help="Output cohort routing. 'default'=p_first_hc_strict; "
                             "'p_first_hc_all'=p_first_hc_all/; "
                             "'p_all_hc_all'=p_all_hc_all/.")
    parser.add_argument("--eigen-source", default="all_npy",
                        choices=["all_npy", "visit_all"],
                        help="Eigenvalue panel data source. 'all_npy' (default): "
                             "fit PCA on every .npy in the feature dir "
                             "(no cohort filter). 'visit_all': filter to the "
                             "union of visit=all cohort IDs so the panel "
                             "matches what a --visit-mode all sweep saw.")
    parser.add_argument("--embedding-abtest", action="store_true",
                        help="Read from embedding_ABtest/ tree (features + "
                             "analysis/classification) instead of production embedding/.")
    args = parser.parse_args()

    class_root, out, reducer_dirs, cell_json_for, feature_subdir, feat_root = resolve_paths(
        args.variant, args.cohort_mode, abtest=args.embedding_abtest
    )
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"ROOT: {class_root}")
    logger.info(f"OUT : {out}")
    logger.info(f"feature_subdir for eigenvalue PCA: {feature_subdir}")
    logger.info(f"cohort_mode: {args.cohort_mode}  eigen_source: {args.eigen_source}")
    logger.info(f"reducer_dirs: {[r.name for r in reducer_dirs]}")

    df = collect(reducer_dirs)
    if df.empty:
        logger.warning("no data found; nothing to write")
        return
    out_csv = out / "all_pca_metrics.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv} ({len(df)} rows)")

    # Cumulative eigenvalue ratio per embedding (saved as csv for downstream
    # forward/reverse plot scripts to read).
    eig_df = compute_cumulative_eigenvalue_ratio(feature_subdir,
                                                   cohort_mode=args.eigen_source,
                                                   feat_root=feat_root)
    if len(eig_df):
        eig_csv_name = ("cumulative_eigenvalue_ratio.csv"
                        if args.eigen_source == "all_npy"
                        else f"cumulative_eigenvalue_ratio_{args.eigen_source}.csv")
        eig_df.to_csv(out / eig_csv_name, index=False)
        logger.info(f"Wrote {out / eig_csv_name}")

    # Feature count plot — effective n_components retained per (embedding, x).
    fc = collect_feature_counts(reducer_dirs, cell_json_for)
    if len(fc):
        fc_csv = out / "feature_count_by_pca.csv"
        fc.to_csv(fc_csv, index=False)
        agg = (fc.groupby(["embedding", "pca_x"])
                  .agg(n_kept_mean=("n_kept", "mean"),
                       n_kept_std=("n_kept", "std"),
                       n_input=("n_input", "first")).reset_index())
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for emb in ["arcface", "topofr", "dlib"]:
            sub = agg[agg["embedding"] == emb].sort_values("pca_x")
            n_in = int(sub["n_input"].iloc[0]) if len(sub) else 0
            ax.plot(sub["pca_x"], sub["n_kept_mean"], marker="o",
                    label=f"{emb} (input dim={n_in})", linewidth=2)
            ax.fill_between(sub["pca_x"],
                            sub["n_kept_mean"] - sub["n_kept_std"],
                            sub["n_kept_mean"] + sub["n_kept_std"],
                            alpha=0.15)
        # diagonal y=x reference (PCA setting = effective components for ints)
        x = np.array([1, 2, 5, 10, 20, 50, 100, 200, 400])
        ax.plot(x, x, color="grey", linestyle=":", linewidth=0.8,
                label="y = PCA setting (int)")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("PCA n_components setting")
        ax.set_ylabel("Effective components retained "
                       "(mean across 5 partitions × 10 folds)")
        ax.set_title("PCA — effective component count vs setting")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fc_png = out / "feature_count_by_pca.png"
        fig.savefig(fc_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {fc_png}")


if __name__ == "__main__":
    main()
