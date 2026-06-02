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

from src.config import (  # noqa: E402
    DEFAULT_COHORT_TOKENS,
    HC_SCORE_TOKENS,
    HC_VISIT_TOKENS,
    P_SCORE_TOKENS,
    P_VISIT_TOKENS,
    cohort_dirs,
)
from src.common.cohort import (  # noqa: E402
    hc_filter,
    p_filter,
)
ASYM_VARIANTS = ["difference", "absolute_difference", "average",
                 "relative_differences", "absolute_relative_differences"]

EMBEDDING_FEAT_DIR = (PROJECT_ROOT / "workspace" / "embedding" / "features")
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


def resolve_paths(variant, embedding, bg_mode="no_background",
                   cohort=DEFAULT_COHORT_TOKENS,
                   photo_mode="mean", match_strategy="no_priority"):
    """Return (class_root, out, reducer_dirs, cell_json_for, feature_subdir, feat_root).

    class_root             → .../<visit>/<cdr_mmse>/<bg_mode>/<emb>/<variant>/<photo>
    cell_json_for(reducer, part, clf) → cell metrics json
    """
    from src.config import EMBEDDING_CLASSIFICATION_DIR
    visit_dir, cdr_mmse_dir = cohort_dirs(*cohort)
    feature_subdir = variant if variant is not None else "original"
    class_root = (EMBEDDING_CLASSIFICATION_DIR / visit_dir / cdr_mmse_dir
                  / bg_mode / embedding / feature_subdir / photo_mode)
    out = class_root / "pca" / "_summary"
    feat_root = EMBEDDING_FEAT_DIR

    def _reducer_dirs():
        if not class_root.is_dir():
            return []
        seen = set()
        for marker_name in ("fwd", "rev"):
            for marker in class_root.rglob(marker_name):
                if marker.is_dir():
                    seen.add(marker.parent.parent)
        result = []
        for reducer in sorted(seen):
            rel_parts = reducer.relative_to(class_root).parts
            if any(p.startswith("_") for p in rel_parts):
                continue
            if rel_parts[0] not in ("no_drop", "pca"):
                continue
            result.append(reducer)
        return result

    def cell_json_for(reducer, part, clf):
        clf_sub = clf
        if clf == "logistic":
            clf_sub = clf + "/C_1"
        return (reducer / clf_sub / "fwd" / "1by1matched"
                / "subject_match" / "eval_by_subject"
                / match_strategy / part / "forward_matched_metrics.json")

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


def _load_visit_all_cohort_ids(cohort=DEFAULT_COHORT_TOKENS):
    """Build the union of visit=all cohort IDs across the 5 partitions
    (ad_vs_hc / ad_vs_nad / ad_vs_acs / mmse_hilo / casi_hilo).

    Each partition's visit-all cohort = subjects passing per-visit eligibility:
      - AD-side: P group + ``p_filter(p_score)``
      - HC-side: NAD/ACS group + ``hc_filter(hc_score)``
      - hi-lo: AD-side + metric+age present

    Returns: set of visit-level IDs (e.g. {"P1-2", "P1-3", "ACS5-1", ...}).
    """
    DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
    frames = []
    # 內部 P/NAD/ACS 走唯一讀取點 load_demographics()（已組好完整 ID "P1-2"）。
    from src.common.cohort import load_demographics
    internal = load_demographics()
    internal["group"] = internal["Group"]
    frames.append(internal)
    # 外部 EACS（完整 ID）仍標為 ACS。
    eacs_path = DEMOGRAPHICS_DIR / "EACS.csv"
    if eacs_path.exists():
        df_e = pd.read_csv(eacs_path)
        df_e["group"] = "ACS"
        frames.append(df_e)
    demo = pd.concat(frames, ignore_index=True)
    demo["Age"] = pd.to_numeric(demo.get("Age"), errors="coerce")
    demo["Global_CDR"] = pd.to_numeric(demo.get("Global_CDR"), errors="coerce")
    demo["MMSE"] = pd.to_numeric(demo.get("MMSE"), errors="coerce")
    demo["CASI"] = pd.to_numeric(demo.get("CASI"), errors="coerce")
    demo["base_id"] = demo["ID"].astype(str).str.extract(r"^(.+)-\d+$")[0]
    demo["visit"] = demo["ID"].astype(str).str.extract(r"-(\d+)$").astype(float)

    # AD-side per-visit eligibility (P group + cohort p_score + Age present)
    ad_pool = demo[(demo["group"] == "P") & demo["Age"].notna()].copy()
    ad_pool = p_filter(ad_pool, cohort[1])
    ad_ids = set(ad_pool["ID"].astype(str))

    # HC-side per-visit eligibility (NAD/ACS + cohort hc_score + Age present)
    hc_pool = demo[demo["group"].isin(["NAD", "ACS"]) & demo["Age"].notna()].copy()
    hc_pool = hc_filter(hc_pool, cohort[3])
    hc_ids = set(hc_pool["ID"].astype(str))

    # Hi-lo (mmse / casi): AD-side + metric present
    mmse_pool = ad_pool[ad_pool["MMSE"].notna()]
    casi_pool = ad_pool[ad_pool["CASI"].notna()]
    mmse_ids = set(mmse_pool["ID"].astype(str))
    casi_ids = set(casi_pool["ID"].astype(str))

    ids = ad_ids | hc_ids | mmse_ids | casi_ids
    logger.info(f"visit=all union cohort: {len(ids)} unique IDs "
                f"(ad={len(ad_ids)}, hc={len(hc_ids)}, "
                f"mmse={len(mmse_ids)}, casi={len(casi_ids)})  "
                f"cohort={'_'.join(cohort)}")
    return ids


def compute_cumulative_eigenvalue_ratio(feature_subdir, cohort_mode="all_npy",
                                         feat_root=None,
                                         cohort=DEFAULT_COHORT_TOKENS):
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

    ``cohort`` (4-token tuple) is forwarded to ``_load_visit_all_cohort_ids``
    to drive the P CDR + HC filters.
    """
    from sklearn.decomposition import PCA
    keep_ids = None
    if cohort_mode == "visit_all":
        keep_ids = _load_visit_all_cohort_ids(cohort=cohort)
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
            a = np.load(npy)
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
        for partition in ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs",
                           "mmse_hilo", "casi_hilo"]:
            p = cell_json_for(sub, partition, "logistic")
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            emb = d.get("embedding", "unknown")
            info = d.get("drop_corr_info", {})
            kept = info.get("n_features_kept_per_fold")
            n_in = info.get("n_features_input")
            if kept is None or n_in is None:
                continue
            for fold, n in enumerate(kept):
                rows.append({
                    "pca_root": sub.name,
                    "pca_x": _parse_pca_label(sub.name),
                    "embedding": emb,
                    "partition": partition,
                    "fold": fold,
                    "n_kept": int(n),
                    "n_input": int(n_in),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", default=None,
                        choices=ASYM_VARIANTS + ["original"],
                        help="Variant under classification/.  Default None == 'original'.")
    parser.add_argument("--embedding", default="arcface",
                        choices=["arcface", "topofr", "dlib", "vggface"])
    parser.add_argument("--bg-mode", default="no_background",
                        choices=["background", "no_background"])
    parser.add_argument("--p-visit", choices=list(P_VISIT_TOKENS),
                        default=DEFAULT_COHORT_TOKENS[0],
                        help="P visit token (p_first / p_all).")
    parser.add_argument("--p-score", choices=list(P_SCORE_TOKENS),
                        default=DEFAULT_COHORT_TOKENS[1],
                        help="P CDR-score token (p_cdrall / p_cdr05 / p_cdr1 / p_cdr2).")
    parser.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS),
                        default=DEFAULT_COHORT_TOKENS[2],
                        help="HC visit token (hc_first / hc_all).")
    parser.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                        default=DEFAULT_COHORT_TOKENS[3],
                        help="HC cognitive-filter token "
                             "(hc_cdrall_or_mmseall / hc_cdr0_or_mmse26).")
    parser.add_argument("--eigen-source", default="all_npy",
                        choices=["all_npy", "visit_all"],
                        help="Eigenvalue panel data source. 'all_npy' (default): "
                             "fit PCA on every .npy in the feature dir "
                             "(no cohort filter). 'visit_all': filter to the "
                             "union of visit=all cohort IDs so the panel "
                             "matches what a --visit-mode all sweep saw.")
    args = parser.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    class_root, out, reducer_dirs, cell_json_for, feature_subdir, feat_root = resolve_paths(
        args.variant, args.embedding, args.bg_mode, cohort
    )
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"ROOT: {class_root}")
    logger.info(f"OUT : {out}")
    logger.info(f"feature_subdir for eigenvalue PCA: {feature_subdir}")
    logger.info(f"cohort: {cohort}  eigen_source: {args.eigen_source}")
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
                                                   feat_root=feat_root,
                                                   cohort=cohort)
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
        ax.tick_params(labelsize=15)
        ax.set_xlabel("PCA n_components setting", fontsize=15)
        ax.set_ylabel("Effective components retained "
                       "(mean across 5 partitions × 10 folds)", fontsize=15)
        ax.set_title("PCA — effective component count vs setting", fontsize=18)
        ax.legend(loc="lower right", fontsize=18)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fc_png = out / "feature_count_by_pca.png"
        fig.savefig(fc_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {fc_png}")


if __name__ == "__main__":
    main()
