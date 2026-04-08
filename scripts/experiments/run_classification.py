"""
AU 特徵分類分析（統一版）

支援分組方案:
  - cdr2: CDR>=2 vs Normal (CDR=0 或 MMSE>=26)
  - nostrat: All Patient vs Normal

支援年齡控制:
  - original: 未控制年齡
  - age: 年齡作為共變量加入模型
  - psm: PSM 年齡配對 (+/-2歲)

Usage:
    python run_classification.py
    python run_classification.py --strat cdr2 --age-methods original psm
    python run_classification.py --tools openface --strat nostrat
"""

import re
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.extractor.features.emotion.extractor.au_config import AU_AGGREGATED_DIR, AU_ANALYSIS_DIR
from src.meta_analysis.classifier.xgboost import XGBoostAnalyzer
from src.meta_analysis.loader.base import Dataset
from src.meta_analysis.evaluation.shap_explainer import AUSHAPExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
OUTPUT_BASE = AU_ANALYSIS_DIR / "classification"
N_FOLDS = 5


# ═══════════════════════════════════════════════════════════
# 分組邏輯
# ═══════════════════════════════════════════════════════════

def load_demographics() -> pd.DataFrame:
    frames = []
    for group in ["P", "NAD", "ACS"]:
        df = pd.read_csv(DEMOGRAPHICS_DIR / f"{group}.csv")
        df["group"] = group
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def assign_label_cdr2(row):
    """CDR>=2 = AD, CDR=0 or MMSE>=26 = Normal"""
    g, cdr, mmse = row["group"], row["Global_CDR"], row["MMSE"]
    if g == "P":
        return "AD" if pd.notna(cdr) and cdr >= 2 else None
    if pd.notna(cdr) and cdr == 0:
        return "Normal"
    if pd.isna(cdr) and pd.notna(mmse) and mmse >= 26:
        return "Normal"
    return None


def assign_label_nostrat(row):
    """All Patient = AD, CDR=0 or MMSE>=26 = Normal"""
    g, cdr, mmse = row["group"], row["Global_CDR"], row["MMSE"]
    if g == "P":
        return "AD"
    if pd.notna(cdr) and cdr == 0:
        return "Normal"
    if pd.isna(cdr) and pd.notna(mmse) and mmse >= 26:
        return "Normal"
    return None


STRATIFICATIONS = {
    "cdr2": ("CDR>=2 vs Normal", assign_label_cdr2),
    "nostrat": ("All Patient vs Normal", assign_label_nostrat),
}


# ═══════════════════════════════════════════════════════════
# PSM
# ═══════════════════════════════════════════════════════════

def psm_match(df: pd.DataFrame, caliper: float = 2.0) -> pd.DataFrame:
    ad_df = df[df["label"] == 1].copy().reset_index(drop=True)
    nm_df = df[df["label"] == 0].copy().reset_index(drop=True)

    ad_ages = ad_df["age"].values.reshape(-1, 1)
    nm_ages = nm_df["age"].values.reshape(-1, 1)

    tree = KDTree(ad_ages)
    k = min(20, len(ad_ages))
    dists, indices = tree.query(nm_ages, k=k)

    order = np.argsort(dists[:, 0])
    matched_ad, matched_nm = set(), set()

    for nm_i in order:
        for j in range(k):
            if dists[nm_i, j] > caliper:
                break
            if indices[nm_i, j] not in matched_ad:
                matched_ad.add(indices[nm_i, j])
                matched_nm.add(nm_i)
                break

    matched = pd.concat([
        ad_df.iloc[list(matched_ad)],
        nm_df.iloc[list(matched_nm)],
    ], ignore_index=True)

    logger.info(
        f"  PSM: AD={len(matched_ad)}, Normal={len(matched_nm)}, "
        f"Age: AD={matched[matched['label']==1]['age'].mean():.1f}, "
        f"Normal={matched[matched['label']==0]['age'].mean():.1f}"
    )
    return matched


# ═══════════════════════════════════════════════════════════
# 資料載入
# ═══════════════════════════════════════════════════════════

def load_features_with_demo(tool, feature_set, demo, label_fn):
    csv_path = AU_AGGREGATED_DIR / f"{tool}_{feature_set}.csv"
    if not csv_path.exists():
        logger.warning(f"找不到: {csv_path}")
        return None, None

    feat_df = pd.read_csv(csv_path)
    feature_columns = [c for c in feat_df.columns if c != "subject_id"]

    label_map, age_map = {}, {}
    for _, row in demo.iterrows():
        lbl = label_fn(row)
        if lbl is not None:
            label_map[row["ID"]] = 1 if lbl == "AD" else 0
            age_map[row["ID"]] = row["Age"]

    records = []
    for _, row in feat_df.iterrows():
        sid = row["subject_id"]
        if sid not in label_map or pd.isna(age_map.get(sid)):
            continue
        match = re.match(r"^(.+?)-\d+$", sid)
        record = {
            "subject_id": sid,
            "base_id": match.group(1) if match else sid,
            "label": label_map[sid],
            "age": age_map[sid],
        }
        for c in feature_columns:
            record[c] = row[c]
        records.append(record)

    return pd.DataFrame(records) if records else None, feature_columns


def build_dataset(df, feature_columns, extra_columns, tool, feature_set, suffix):
    all_cols = feature_columns + extra_columns
    X = df[all_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0)

    n_ad = np.sum(y == 1)
    n_nm = np.sum(y == 0)
    logger.info(f"  {tool}_{feature_set}_{suffix}: {len(df)} subjects, "
                f"{X.shape[1]} features, AD={n_ad}, Normal={n_nm}")

    return Dataset(
        X=X, y=y,
        subject_ids=df["subject_id"].tolist(),
        base_ids=df["base_id"].tolist(),
        metadata={
            "tool": tool, "feature_set": feature_set,
            "model": f"au_{tool}",
            "feature_type": f"{feature_set}_{suffix}",
            "n_features": X.shape[1],
            "feature_names": all_cols,
        },
        data_format="averaged",
    )


# ═══════════════════════════════════════════════════════════
# 分類 + SHAP
# ═══════════════════════════════════════════════════════════

def run_experiment(datasets, output_dir, label):
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    pred_prob_dir = output_dir / "pred_probability"
    shap_dir = output_dir / "shap"

    analyzer = XGBoostAnalyzer(
        models_dir=models_dir, reports_dir=reports_dir,
        pred_prob_dir=pred_prob_dir, n_folds=N_FOLDS, n_drop_features=0,
    )
    results = analyzer.analyze(datasets)

    # 彙總
    logger.info(f"\n{'=' * 60}")
    logger.info(label)
    logger.info(f"{'=' * 60}")

    for dk, nfr in results.items():
        for nf, r in nfr.items():
            m = r["test_metrics"]
            logger.info(
                f"  {dk} (n={nf}): Acc={m['accuracy']:.3f}, "
                f"AUC={m.get('auc', 'N/A')}, MCC={m['mcc']:.3f}, "
                f"Sens={m['sensitivity']:.3f}, Spec={m['specificity']:.3f}"
            )

    # SHAP
    shap_dir.mkdir(parents=True, exist_ok=True)
    explainer = AUSHAPExplainer()

    dataset_map = {
        f"au_{ds.metadata['tool']}_{ds.metadata['feature_type']}_{dk.split('_')[-1]}": ds
        for ds in datasets
        for dk in results.keys()
    }
    # Rebuild properly
    dataset_map = {}
    for ds in datasets:
        for dk in results.keys():
            dataset_map[dk] = ds
            break

    for dk, nfr in results.items():
        ds = dataset_map.get(dk)
        if ds is None:
            continue
        fn = ds.metadata.get("feature_names", [])
        for nf, r in nfr.items():
            fm = r.get("fold_models", [])
            if not fm:
                continue
            si = r.get("selected_indices", list(range(nf)))
            Xs = ds.X[:, si]
            sn = [fn[i] for i in si]
            try:
                sr = explainer.explain_all_folds(
                    fold_models=fm,
                    fold_data=[(Xs, ds.y)] * len(fm),
                    feature_names=sn,
                )
                explainer.save_results(sr, shap_dir / f"{dk}_shap.json")
                logger.info(f"  SHAP Top 5 ({dk}):")
                for item in sr["ranking"][:5]:
                    logger.info(f"    #{item['rank']} {item['feature']}: {item['mean_abs_shap']:.4f}")
            except Exception as e:
                logger.error(f"  SHAP failed {dk}: {e}")

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AU 特徵分類分析")
    parser.add_argument(
        "--strat", nargs="+", default=["cdr2", "nostrat"],
        choices=["cdr2", "nostrat"],
        help="分組方案",
    )
    parser.add_argument(
        "--age-methods", nargs="+", default=["original", "age", "psm"],
        choices=["original", "age", "psm"],
        help="年齡控制方法",
    )
    parser.add_argument(
        "--tools", nargs="+", default=["openface", "libreface"],
        choices=["openface", "libreface", "pyfeat"],
    )
    parser.add_argument(
        "--feature-sets", nargs="+", default=["harmonized", "extended"],
    )
    args = parser.parse_args()

    demo = load_demographics()

    for strat_key in args.strat:
        strat_name, label_fn = STRATIFICATIONS[strat_key]

        for tool in args.tools:
            for fs in args.feature_sets:
                merged, feat_cols = load_features_with_demo(tool, fs, demo, label_fn)
                if merged is None:
                    continue

                for age_method in args.age_methods:
                    if age_method == "psm":
                        data = psm_match(merged.copy(), caliper=2.0)
                        extra_cols = []
                        suffix = "psm"
                    elif age_method == "age":
                        data = merged.copy()
                        extra_cols = ["age"]
                        suffix = "age"
                    else:
                        data = merged.copy()
                        extra_cols = []
                        suffix = "original"

                    config_key = f"{strat_key}_{suffix}"
                    out_dir = OUTPUT_BASE / config_key

                    ds = build_dataset(data, feat_cols, extra_cols, tool, fs, suffix)

                    label = f"{strat_name} | {tool} {fs} | {age_method}"
                    logger.info(f"\n{'#' * 60}")
                    logger.info(f"# {label}")
                    logger.info(f"{'#' * 60}")

                    run_experiment([ds], out_dir, label)

    logger.info("\n全部完成！")


if __name__ == "__main__":
    main()
