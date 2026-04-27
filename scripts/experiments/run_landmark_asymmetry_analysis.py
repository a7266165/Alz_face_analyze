"""
Landmark 不對稱性分析

三個任務:
  1. visualize — 視覺化 65 組 landmark pairs
  2. cross-sectional — CDR 嚴重度三類別分類 (CDR 0.5 vs 1 vs 2+)
  3. longitudinal — 首末 visit landmark 變化量 vs 認知退化

Usage:
  python run_landmark_asymmetry_analysis.py --task all
  python run_landmark_asymmetry_analysis.py --task visualize
  python run_landmark_asymmetry_analysis.py --task cross-sectional
  python run_landmark_asymmetry_analysis.py --task longitudinal
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import directly to avoid __init__.py chain that pulls in unresolvable deps
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "regional_landmark",
    PROJECT_ROOT / "src" / "extractor" / "features" / "asymmetry" / "regional_landmark.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ALL_PAIRS = _mod.ALL_PAIRS
EYE_PAIRS = _mod.EYE_PAIRS
NOSE_PAIRS = _mod.NOSE_PAIRS
MOUTH_PAIRS = _mod.MOUTH_PAIRS
FACE_OVAL_PAIRS = _mod.FACE_OVAL_PAIRS
compute_pair_features = _mod.compute_pair_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Paths ===
LANDMARK_FEATURES_CSV = PROJECT_ROOT / "workspace" / "asymmetry" / "landmark_features.csv"
LANDMARKS_DIR = PROJECT_ROOT / "workspace" / "asymmetry" / "landmarks"
DEMOGRAPHICS_CSV = PROJECT_ROOT / "data" / "demographics" / "P.csv"
PREDICTED_AGES_FILE = PROJECT_ROOT / "workspace" / "age" / "age_prediction" / "predicted_ages.json"
PATIENT_DELTAS_CSV = PROJECT_ROOT / "workspace" / "longitudinal" / "patient_deltas.csv"
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "asymmetry" / "analysis"

N_FOLDS = 5
RANDOM_SEED = 42
CDR_LABELS = {0: "CDR 0.5", 1: "CDR 1", 2: "CDR 2+"}

REGION_COLORS = {
    "eye": "#2196F3",
    "nose": "#4CAF50",
    "mouth": "#F44336",
    "face_oval": "#FF9800",
}

# Feature column subsets
REGION_FEATURE_SUBSETS = {
    "eye": ([f"eye_x_pair_{i}" for i in range(1, 17)]
            + [f"eye_y_pair_{i}" for i in range(1, 17)]
            + ["eye_area_diff"]),
    "nose": ([f"nose_x_pair_{i}" for i in range(1, 15)]
             + [f"nose_y_pair_{i}" for i in range(1, 15)]
             + ["nose_area_diff"]),
    "mouth": ([f"mouth_x_pair_{i}" for i in range(1, 19)]
              + [f"mouth_y_pair_{i}" for i in range(1, 19)]
              + ["mouth_area_diff"]),
    "face_oval": ([f"face_oval_x_pair_{i}" for i in range(1, 18)]
                  + [f"face_oval_y_pair_{i}" for i in range(1, 18)]
                  + ["face_oval_area_diff"]),
    "areas_only": ["eye_area_diff", "nose_area_diff", "mouth_area_diff", "face_oval_area_diff"],
}


# ============================================================
#  Shared: XGBoost Multi-class CV
# ============================================================

def run_multiclass_cv(X, y, base_ids, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """Run multi-class XGBoost with GroupKFold CV."""
    gkf = GroupKFold(n_splits=n_folds)
    n_classes = len(np.unique(y))

    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=base_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.append(y_prob)

        fold_metrics.append({
            "balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
        })

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.vstack(all_y_prob)

    bal_acc = balanced_accuracy_score(all_y_true, all_y_pred)
    mcc = matthews_corrcoef(all_y_true, all_y_pred)
    cm = confusion_matrix(all_y_true, all_y_pred)

    try:
        macro_auc = roc_auc_score(all_y_true, all_y_prob, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = None

    report = classification_report(
        all_y_true, all_y_pred,
        target_names=[CDR_LABELS[i] for i in range(n_classes)],
        output_dict=True,
    )

    return {
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "macro_auc": float(macro_auc) if macro_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "fold_metrics": fold_metrics,
        "n_samples": len(all_y_true),
    }


# ============================================================
#  Task 1: Visualize Pairs
# ============================================================

def visualize_pairs():
    """視覺化 65 組 landmark pairs，依區域上色。"""
    logger.info("Task 1: Visualizing landmark pairs...")

    # Find a P-group subject with landmarks
    npy_files = sorted(LANDMARKS_DIR.glob("P*.npy"))
    if not npy_files:
        logger.error("No P-group .npy files found")
        return

    # Pick one with reasonable number of images
    chosen = npy_files[len(npy_files) // 2]
    arr = np.load(chosen)  # (n_images, 468, 2)
    landmarks = arr[0]  # Use first image
    logger.info(f"Using subject: {chosen.stem}, landmarks shape: {arr.shape}")

    fig, ax = plt.subplots(figsize=(8, 10))

    # All 468 landmarks as gray dots
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=2, c="gray", alpha=0.3, zorder=1)

    # Draw pairs by region
    region_names = {"eye": f"Eye ({len(EYE_PAIRS)} pairs)",
                    "nose": f"Nose ({len(NOSE_PAIRS)} pairs)",
                    "mouth": f"Mouth ({len(MOUTH_PAIRS)} pairs)",
                    "face_oval": f"Face Oval ({len(FACE_OVAL_PAIRS)} pairs)"}

    for region, pairs in ALL_PAIRS.items():
        color = REGION_COLORS[region]
        for i, (r_idx, l_idx) in enumerate(pairs):
            r_pt = landmarks[r_idx]
            l_pt = landmarks[l_idx]
            label = region_names[region] if i == 0 else None
            ax.plot([r_pt[0], l_pt[0]], [r_pt[1], l_pt[1]],
                    c=color, linewidth=1.0, alpha=0.7, zorder=2)
            ax.scatter([r_pt[0], l_pt[0]], [r_pt[1], l_pt[1]],
                       s=15, c=color, zorder=3, label=label)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("65 Landmark Pairs for Facial Asymmetry Analysis", fontsize=13)
    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=10)

    out_path = OUTPUT_DIR / "landmark_pairs_verification.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ============================================================
#  Task 2: Cross-sectional Severity Classification
# ============================================================

def prepare_cross_sectional_data(visit_selection):
    """Prepare landmark features for severity classification."""
    # Load features
    feat_df = pd.read_csv(LANDMARK_FEATURES_CSV)
    feat_df = feat_df[feat_df["subject_id"].str.startswith("P")].copy()

    # Load demographics
    demo = pd.read_csv(DEMOGRAPHICS_CSV)
    demo["Global_CDR"] = pd.to_numeric(demo["Global_CDR"], errors="coerce")

    # Load predicted ages
    with open(PREDICTED_AGES_FILE) as f:
        pred_ages = json.load(f)

    # Join
    merged = feat_df.merge(demo[["ID", "Global_CDR", "Age"]], left_on="subject_id", right_on="ID", how="inner")
    merged = merged.drop(columns=["ID"])

    # Filter: CDR >= 0.5, predicted_age >= 65
    merged = merged[merged["Global_CDR"] >= 0.5]
    merged = merged[merged["subject_id"].apply(lambda x: pred_ages.get(x, 65) >= 65)]

    # Severity labels
    def severity_label(cdr):
        if cdr == 0.5:
            return 0
        elif cdr == 1.0:
            return 1
        return 2

    merged["label"] = merged["Global_CDR"].apply(severity_label)
    merged["base_id"] = merged["subject_id"].str.extract(r"^([A-Z]+\d+)")
    merged["visit"] = merged["subject_id"].str.extract(r"-(\d+)$").astype(int)

    # Single visit per patient
    ascending = visit_selection == "first"
    merged = (
        merged.sort_values("visit", ascending=ascending)
        .groupby("base_id")
        .first()
        .reset_index()
    )

    logger.info(f"Cross-sectional data ({visit_selection}): {len(merged)} patients, "
                f"CDR distribution: {merged['label'].value_counts().sort_index().to_dict()}")
    return merged


def run_cross_sectional():
    """橫斷面嚴重度分類。"""
    logger.info("Task 2: Cross-sectional severity classification...")
    results = {}

    # All feature columns (exclude metadata)
    exclude_cols = {"subject_id", "ID", "Global_CDR", "Age", "label", "base_id", "visit"}

    for visit_sel in ["first", "latest"]:
        merged = prepare_cross_sectional_data(visit_sel)

        # Get all feature columns
        all_feature_cols = [c for c in merged.columns if c not in exclude_cols]

        # Define subsets
        subsets = {"all": all_feature_cols}
        for region, cols in REGION_FEATURE_SUBSETS.items():
            subsets[region] = [c for c in cols if c in merged.columns]

        for subset_name, feature_cols in subsets.items():
            key = f"{visit_sel}_{subset_name}"
            logger.info(f"  {key}: {len(feature_cols)} features")

            X = merged[feature_cols].values.astype(float)
            y = merged["label"].values
            base_ids = merged["base_id"].values

            # Drop NaN rows
            valid = ~np.isnan(X).any(axis=1)
            X, y, base_ids = X[valid], y[valid], base_ids[valid]

            if len(X) < 50:
                logger.warning(f"    Too few samples: {len(X)}")
                continue

            result = run_multiclass_cv(X, y, base_ids)
            result["n_features"] = X.shape[1]
            result["n_persons"] = len(set(base_ids))
            results[key] = result

            logger.info(f"    BalAcc={result['balanced_accuracy']:.3f}, "
                        f"MCC={result['mcc']:.3f}, AUC={result['macro_auc']}")

    # Save detailed results
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    detail_path = OUTPUT_DIR / "cross_sectional_results.json"
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    # Save summary CSV
    rows = []
    for key, result in results.items():
        visit_sel, subset_name = key.split("_", 1)
        rows.append({
            "visit_selection": visit_sel,
            "module": "landmark_asymmetry",
            "config": subset_name,
            "n_persons": result.get("n_persons"),
            "n_features": result.get("n_features"),
            "balanced_accuracy": result["balanced_accuracy"],
            "mcc": result["mcc"],
            "macro_auc": result.get("macro_auc"),
            "f1_CDR05": result["classification_report"].get("CDR 0.5", {}).get("f1-score"),
            "f1_CDR1": result["classification_report"].get("CDR 1", {}).get("f1-score"),
            "f1_CDR2p": result["classification_report"].get("CDR 2+", {}).get("f1-score"),
        })

    summary_df = pd.DataFrame(rows).sort_values(
        ["visit_selection", "balanced_accuracy"], ascending=[True, False]
    )
    summary_path = OUTPUT_DIR / "cross_sectional_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved: {summary_path}")

    # Print results
    print("\n" + "=" * 80)
    print("CROSS-SECTIONAL SEVERITY CLASSIFICATION RESULTS")
    print("=" * 80)
    for visit_sel in ["first", "latest"]:
        print(f"\n--- Visit: {visit_sel} ---")
        sub = summary_df[summary_df["visit_selection"] == visit_sel]
        for _, row in sub.iterrows():
            print(f"  {row['config']:>12} | n={row['n_persons']:>4}, feat={row['n_features']:>3} | "
                  f"BalAcc={row['balanced_accuracy']:.3f} | MCC={row['mcc']:.3f} | "
                  f"AUC={row['macro_auc']:.3f}")

    # Compare with existing severity classification results
    existing_summary = PROJECT_ROOT / "workspace" / "severity_classification" / "summary.csv"
    if existing_summary.exists():
        print("\n--- Comparison with existing best results ---")
        existing = pd.read_csv(existing_summary)
        for visit_sel in ["first", "latest"]:
            best_existing = existing[existing["visit_selection"] == visit_sel].iloc[0]
            best_landmark = summary_df[summary_df["visit_selection"] == visit_sel].iloc[0]
            print(f"  {visit_sel}: Existing best={best_existing['module']}/"
                  f"{best_existing['config']} BalAcc={best_existing['balanced_accuracy']:.3f} | "
                  f"Landmark best={best_landmark['config']} BalAcc={best_landmark['balanced_accuracy']:.3f}")

    return results


# ============================================================
#  Task 3: Longitudinal Analysis
# ============================================================

def run_longitudinal():
    """縱向分析：landmark 變化量 vs 認知退化。"""
    logger.info("Task 3: Longitudinal analysis...")

    # Load patient deltas
    deltas = pd.read_csv(PATIENT_DELTAS_CSV)
    logger.info(f"Patient deltas: {len(deltas)} patients")

    # Compute landmark feature deltas for each patient
    landmark_delta_rows = []
    missing_count = 0

    for _, row in deltas.iterrows():
        first_npy = LANDMARKS_DIR / f"{row['first_visit_id']}.npy"
        last_npy = LANDMARKS_DIR / f"{row['last_visit_id']}.npy"

        if not first_npy.exists() or not last_npy.exists():
            missing_count += 1
            continue

        # Load and mean-pool across images
        first_arr = np.load(first_npy)  # (n_images, 468, 2)
        last_arr = np.load(last_npy)
        first_lm = first_arr.mean(axis=0)  # (468, 2)
        last_lm = last_arr.mean(axis=0)

        # Compute features for each visit
        first_feats = compute_pair_features(first_lm)
        last_feats = compute_pair_features(last_lm)

        # Delta
        delta_feats = {k: last_feats[k] - first_feats[k] for k in first_feats}
        delta_feats["base_id"] = row["base_id"]

        # Scalar summaries
        delta_vec = np.array([delta_feats[k] for k in first_feats])
        delta_feats["landmark_l2_norm"] = float(np.linalg.norm(delta_vec))
        delta_feats["landmark_mean_abs"] = float(np.mean(np.abs(delta_vec)))

        # Per-region L2 norms
        for region, pairs in ALL_PAIRS.items():
            n_pairs = len(pairs)
            region_cols = ([f"{region}_x_pair_{i}" for i in range(1, n_pairs + 1)]
                           + [f"{region}_y_pair_{i}" for i in range(1, n_pairs + 1)]
                           + [f"{region}_area_diff"])
            region_vec = np.array([delta_feats[c] for c in region_cols if c in delta_feats])
            delta_feats[f"{region}_l2"] = float(np.linalg.norm(region_vec))

        landmark_delta_rows.append(delta_feats)

    logger.info(f"Landmark deltas computed: {len(landmark_delta_rows)} patients "
                f"({missing_count} missing .npy)")

    landmark_deltas = pd.DataFrame(landmark_delta_rows)

    # Save landmark deltas
    delta_out = OUTPUT_DIR / "longitudinal_landmark_deltas.csv"
    landmark_deltas.to_csv(delta_out, index=False)
    logger.info(f"Saved: {delta_out}")

    # Merge with cognitive deltas
    merged = landmark_deltas.merge(
        deltas[["base_id", "delta_MMSE", "delta_CASI", "delta_CDR_SB",
                "emb_cosine_dist", "follow_up_days", "n_visits"]],
        on="base_id",
        how="inner",
    )
    logger.info(f"Merged with cognitive deltas: {len(merged)} patients")

    # Correlation analysis
    scalar_metrics = ["landmark_l2_norm", "landmark_mean_abs",
                      "eye_l2", "nose_l2", "mouth_l2", "face_oval_l2"]
    cognitive_measures = ["delta_CDR_SB", "delta_MMSE", "delta_CASI"]

    corr_rows = []
    for metric in scalar_metrics:
        for cog in cognitive_measures:
            valid = merged[[metric, cog]].dropna()
            if len(valid) < 10:
                continue
            x, y = valid[metric].values, valid[cog].values
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            corr_rows.append({
                "metric": metric,
                "cognitive_measure": cog,
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
                "n": len(valid),
            })

    # Also compute embedding cosine dist correlations for comparison
    for cog in cognitive_measures:
        valid = merged[["emb_cosine_dist", cog]].dropna()
        if len(valid) < 10:
            continue
        x, y = valid["emb_cosine_dist"].values, valid[cog].values
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        corr_rows.append({
            "metric": "emb_cosine_dist",
            "cognitive_measure": cog,
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "n": len(valid),
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_path = OUTPUT_DIR / "longitudinal_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Saved: {corr_path}")

    # Print results
    print("\n" + "=" * 80)
    print("LONGITUDINAL CORRELATION RESULTS")
    print("=" * 80)
    for cog in cognitive_measures:
        print(f"\n--- vs {cog} ---")
        sub = corr_df[corr_df["cognitive_measure"] == cog].sort_values(
            "spearman_r", ascending=False, key=abs
        )
        for _, row in sub.iterrows():
            sig = "***" if row["spearman_p"] < 0.001 else "**" if row["spearman_p"] < 0.01 else "*" if row["spearman_p"] < 0.05 else ""
            print(f"  {row['metric']:>20} | n={row['n']:>4} | "
                  f"Pearson r={row['pearson_r']:+.3f} (p={row['pearson_p']:.4f}) | "
                  f"Spearman r={row['spearman_r']:+.3f} (p={row['spearman_p']:.4f}) {sig}")

    return corr_df


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Landmark asymmetry analysis")
    parser.add_argument("--task", default="all",
                        choices=["all", "visualize", "cross-sectional", "longitudinal"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.task in ("all", "visualize"):
        visualize_pairs()

    if args.task in ("all", "cross-sectional"):
        run_cross_sectional()

    if args.task in ("all", "longitudinal"):
        run_longitudinal()


if __name__ == "__main__":
    main()
