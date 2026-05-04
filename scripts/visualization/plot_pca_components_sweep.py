"""
Aggregate PCA component-count dirs (pca_1..400, plus pca_0.95/0.99 and
no_drop reference) into a cross-component AUC comparison.

Default mode reads `embedding_classification/pca_<n>/_summary/all_metrics_with_cm.csv`.
With --variant, reads `embedding_asymmetry_classification/pca_<n>/<variant>/_summary/all_metrics_with_cm.csv`
and computes cumulative eigenvalues from that variant's feature matrix.

Output:
    embedding_classification/_pca_summary/                       (default)
    embedding_asymmetry_classification/_pca_summary/<variant>/   (--variant set)

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
ARMS_ROOT = PROJECT_ROOT / "workspace" / "arms_analysis"
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


def resolve_paths(variant, cohort_mode="default"):
    """Return (root, out, cell_csv_for, cell_json_for, feature_subdir).

    cell_csv_for(reducer)  → Path to <reducer>'s _summary/all_metrics_with_cm.csv
    cell_json_for(reducer, part, emb, clf) → cell metrics json
    feature_subdir         → which sub-dir of workspace/embedding/features/<emb>/
                              to use for cumulative eigenvalue ('original' for
                              default, variant name for asymmetry).
    """
    cohort_dir = "p_first_hc_all" if cohort_mode == "p_first_hc_all" else "p_first_hc_strict"
    if variant is None:
        root = ARMS_ROOT / cohort_dir / "embedding_classification"
        out = root / "_pca_summary"
        return (
            root, out,
            lambda r: r / "_summary" / "all_metrics_with_cm.csv",
            lambda r, part, emb, clf: (
                r / "fwd" / part / emb / clf / "forward_matched_metrics.json"
            ),
            "original",
        )
    root = ARMS_ROOT / cohort_dir / "embedding_asymmetry_classification"
    out = root / "_pca_summary" / variant
    return (
        root, out,
        lambda r: r / variant / "_summary" / "all_metrics_with_cm.csv",
        lambda r, part, emb, clf: (
            r / variant / "fwd" / part / emb / clf
            / "forward_matched_metrics.json"
        ),
        variant,
    )


def _parse_pca_label(name):
    """Folder name → numeric x-axis position. Returns None for non-integer PCA
    settings (variance ratios), `no_drop`, and `drop_*`; those rows are
    dropped from the integer-axis line plot.

    Strips visit/photo mode suffix (e.g. `pca_100__visit_all` -> 100).
    """
    if name.startswith("no_drop"):
        return None
    m = re.match(r"pca_([0-9.]+)", name)
    if not m:
        return None
    val = m.group(1)
    try:
        x = float(val)
    except ValueError:
        return None
    return x if x >= 1 else None  # drop variance ratios (0 < x < 1)


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


def compute_cumulative_eigenvalue_ratio(feature_subdir, cohort_mode="all_npy"):
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
    rows = []
    for emb in ("arcface", "topofr", "dlib"):
        feat_dir = EMBEDDING_FEAT_DIR / emb / feature_subdir
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


def collect(root, cell_csv_for):
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name != "no_drop" and not sub.name.startswith("pca_"):
            continue
        csv = cell_csv_for(sub)
        if not csv.exists():
            logger.warning(f"missing {csv}")
            continue
        df = pd.read_csv(csv)
        df["pca_root"] = sub.name
        df["pca_x"] = _parse_pca_label(sub.name)
        df["pca_label"] = (sub.name.replace("pca_", "") if sub.name != "no_drop"
                            else "no_drop")
        rows.append(df)
        logger.info(f"  {sub.name}: {len(df)} rows")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def collect_feature_counts(root, cell_json_for):
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith("pca_"):
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
    parser.add_argument("--variant", default=None, choices=ASYM_VARIANTS)
    parser.add_argument("--cohort-mode", default="default",
                        choices=["default", "p_first_hc_all"],
                        help="Output cohort routing. 'default'=p_first_hc_strict; "
                             "'p_first_hc_all'=p_first_hc_all/.")
    parser.add_argument("--eigen-source", default="all_npy",
                        choices=["all_npy", "visit_all"],
                        help="Eigenvalue panel data source. 'all_npy' (default): "
                             "fit PCA on every .npy in the feature dir "
                             "(no cohort filter). 'visit_all': filter to the "
                             "union of visit=all cohort IDs so the panel "
                             "matches what a --visit-mode all sweep saw.")
    args = parser.parse_args()

    root, out, cell_csv_for, cell_json_for, feature_subdir = resolve_paths(
        args.variant, args.cohort_mode
    )
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"ROOT: {root}")
    logger.info(f"OUT : {out}")
    logger.info(f"feature_subdir for eigenvalue PCA: {feature_subdir}")
    logger.info(f"cohort_mode: {args.cohort_mode}  eigen_source: {args.eigen_source}")

    df = collect(root, cell_csv_for)
    if df.empty:
        logger.warning("no data found; nothing to write")
        return
    out_csv = out / "all_pca_metrics.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv} ({len(df)} rows)")

    pivot = df.pivot_table(
        index=["partition", "embedding", "classifier", "strategy", "scope"],
        columns="pca_root", values="auc", aggfunc="first",
    )
    col_order = [f"pca_{v}" for v in
                  (1, 2, 5, 10, 20, 50, 100, 200, 400, 0.95, 0.99)] + ["no_drop"]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    pivot.to_csv(out / "auc_by_pca_pivot.csv")
    logger.info(f"Wrote {out / 'auc_by_pca_pivot.csv'}")

    # Cumulative eigenvalue ratio per embedding (shared across the 5 figures).
    eig_df = compute_cumulative_eigenvalue_ratio(feature_subdir,
                                                   cohort_mode=args.eigen_source)
    if len(eig_df):
        eig_csv_name = ("cumulative_eigenvalue_ratio.csv"
                        if args.eigen_source == "all_npy"
                        else f"cumulative_eigenvalue_ratio_{args.eigen_source}.csv")
        eig_df.to_csv(out / eig_csv_name, index=False)
        logger.info(f"Wrote {out / eig_csv_name}")

    # Per-partition stacked plot — split into fwd/ and rev/ subdirs.
    #   fwd/auc_by_pca_<part>.png : forward_matched (1 panel) + eigenvalue
    #   rev/auc_by_pca_<part>.png : reverse_matched_oof + reverse_full_ensemble
    #                                (2 panels) + eigenvalue
    fwd_scopes = ["forward_matched"]
    rev_scopes = ["reverse_ensemble_matched_oof", "reverse_ensemble_full"]
    all_scopes = fwd_scopes + rev_scopes
    line_df = df[df["scope"].isin(all_scopes) & df["pca_x"].notna()].copy()

    from matplotlib.gridspec import GridSpec
    fwd_dir = out / "fwd"
    rev_dir = out / "rev"
    fwd_dir.mkdir(exist_ok=True)
    rev_dir.mkdir(exist_ok=True)

    def _draw(part, scopes, out_path):
        sub = line_df[(line_df["partition"] == part)
                       & line_df["scope"].isin(scopes)]
        n_top = len(scopes)
        fig = plt.figure(figsize=(6 * n_top, 8))
        gs = GridSpec(2, n_top, height_ratios=[2.2, 1], hspace=0.35,
                       figure=fig)
        top_axes = [fig.add_subplot(gs[0, i]) for i in range(n_top)]
        bot_ax = fig.add_subplot(gs[1, :])
        for ax_idx, (ax, scope) in enumerate(zip(top_axes, scopes)):
            scope_df = sub[sub["scope"] == scope]
            for (emb, clf), grp in scope_df.groupby(["embedding", "classifier"]):
                grp = grp.sort_values("pca_x")
                ax.plot(grp["pca_x"], grp["auc"], marker="o",
                        label=f"{emb}/{clf}", linewidth=1.5,
                        color=EMB_CLF_COLOR.get((emb, clf),
                                                 EMB_COLOR.get(emb)),
                        linestyle=EMB_CLF_LINESTYLE.get(clf, "-"))
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlim(1, 512)
            ax.set_title(scope)
            ax.grid(alpha=0.3, which="both")
            if ax_idx == 0:
                ax.set_ylabel("AUC")
        top_axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                             fontsize=8)
        for emb in ("arcface", "topofr", "dlib"):
            sub_eig = eig_df[eig_df["embedding"] == emb]
            if not len(sub_eig):
                continue
            bot_ax.plot(sub_eig["n_components"],
                        sub_eig["cumulative_variance_ratio"],
                        label=f"{emb} (input dim={INPUT_DIM[emb]})",
                        color=EMB_COLOR.get(emb), linewidth=1.8)
        bot_ax.axhline(0.95, color="grey", linestyle=":", linewidth=0.6)
        bot_ax.axhline(0.99, color="grey", linestyle=":", linewidth=0.6)
        bot_ax.text(1.05, 0.95, "0.95", fontsize=7, color="grey", va="center")
        bot_ax.text(1.05, 0.99, "0.99", fontsize=7, color="grey", va="center")
        bot_ax.set_xscale("log")
        bot_ax.set_xlim(1, 512)
        bot_ax.set_ylim(0, 1.02)
        bot_ax.set_xlabel("PCA n_components")
        bot_ax.set_ylabel("Cumulative eigenvalue / total")
        bot_ax.grid(alpha=0.3, which="both")
        bot_ax.legend(loc="lower right", fontsize=8)
        fig.suptitle(f"{part} — AUC vs PCA n_components (top) ·  "
                     f"cumulative eigenvalue ratio (bottom)",
                     fontsize=13)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {out_path}")

    partitions = sorted(line_df["partition"].unique())
    for part in partitions:
        _draw(part, fwd_scopes, fwd_dir / f"auc_by_pca_{part}.png")
        _draw(part, rev_scopes, rev_dir / f"auc_by_pca_{part}.png")

    # Feature count plot — effective n_components retained per (embedding, x).
    fc = collect_feature_counts(root, cell_json_for)
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
