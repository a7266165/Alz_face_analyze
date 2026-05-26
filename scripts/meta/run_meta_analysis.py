"""
Meta Analysis 執行腳本 (v2)

依 embedding pipeline 分層格式輸出：
  meta/analysis/{visit}/{cdr}/{bg}/{emb_model}/{asymmetry_variant}/{photo}/{reducer}/
    meta_{meta_clf}/fwd/{eval_chain}/

對每個 asymmetry variant 各跑一次 (original 固定 + asymmetry 可選)。

使用方式:
    conda run -n Alz_face_main_analysis python scripts/meta/run_meta_analysis.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT

from src.config import (
    cohort_name,
    cohort_spec_from_name,
)
from src.meta import MetaConfig, MetaPipeline

# ========== 路徑設定 ==========

WORKSPACE_DIR = PROJECT_ROOT / "workspace"
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"

# ========== 分析設定 ==========

COHORT_MODE = "p_first_cdrall_hc_all_cdrall_or_mmseall"
BG_MODE = "background"
PHOTO_MODE = "mean"
REDUCER = "no_drop"
MATCH_STRATEGY = "match_acs_first"
BASE_CLASSIFIER = "logistic"
BASE_CLASSIFIER_PARAM = "C_1"

EMB_MODELS = ["arcface"]
META_CLASSIFIERS = ["tabpfn", "logistic", "xgboost"]
NORMALIZE_OPTIONS = [None, "minmax"]
ASYMMETRY_VARIANTS = [
    "difference",
    "absolute_difference",
    "relative_differences",
    "absolute_relative_differences",
]

# Age predictions (cohort-specific)
spec = cohort_spec_from_name(cohort_name(COHORT_MODE))
PREDICTED_AGES = (
    WORKSPACE_DIR / "age" / "predictions"
    / spec.visit_dir / spec.cdr_mmse_dir
    / "correction" / "calibration" / "predicted_ages_calibrated.json"
)

# Emotion (None for now)
EMOTION_METHOD = None

# Meta output root (mirrors embedding structure)
META_ROOT = WORKSPACE_DIR / "meta" / "analysis"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def build_output_dir(emb_model, asymmetry_variant, meta_clf, normalize=None):
    """Build embedding-style output directory for meta analysis."""
    norm_tag = normalize if normalize else "raw"
    return (
        META_ROOT / spec.visit_dir / spec.cdr_mmse_dir
        / BG_MODE / emb_model / asymmetry_variant
        / PHOTO_MODE / REDUCER / norm_tag / meta_clf / "fwd"
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Meta Analysis (v2) — multi-variant")
    logger.info("=" * 60)
    logger.info(f"Cohort: {COHORT_MODE}")
    logger.info(f"Embedding models: {EMB_MODELS}")
    logger.info(f"Asymmetry variants: {ASYMMETRY_VARIANTS}")
    logger.info(f"Meta classifiers: {META_CLASSIFIERS}")
    logger.info(f"Normalize options: {NORMALIZE_OPTIONS}")

    if not DEMOGRAPHICS_DIR.exists():
        logger.error(f"找不到人口學資料目錄: {DEMOGRAPHICS_DIR}")
        sys.exit(1)
    if not PREDICTED_AGES.exists():
        logger.error(f"找不到預測年齡檔案: {PREDICTED_AGES}")
        sys.exit(1)

    all_summaries = []

    for norm_opt in NORMALIZE_OPTIONS:
        norm_tag = norm_opt if norm_opt else "raw"
        for asym_variant in ASYMMETRY_VARIANTS:
            for emb_model in EMB_MODELS:
                for meta_clf in META_CLASSIFIERS:
                    output_dir = build_output_dir(
                        emb_model, asym_variant, meta_clf, norm_opt,
                    )

                    if (output_dir / "summary.csv").exists():
                        logger.info(f"  SKIP (已存在): {output_dir}")
                        summary_df = pd.read_csv(output_dir / "summary.csv")
                        if not summary_df.empty:
                            summary_df["asymmetry_variant"] = asym_variant
                            summary_df["normalize"] = norm_tag
                            all_summaries.append(summary_df)
                        continue

                    logger.info(f"\n{'='*60}")
                    logger.info(f"  {emb_model} / {asym_variant} / {meta_clf} / {norm_tag}")
                    logger.info(f"  → {output_dir}")
                    logger.info(f"{'='*60}")

                    config = MetaConfig(
                        cohort_mode=COHORT_MODE,
                        bg_mode=BG_MODE,
                        photo_mode=PHOTO_MODE,
                        reducer=REDUCER,
                        base_classifier=BASE_CLASSIFIER,
                        base_classifier_param=BASE_CLASSIFIER_PARAM,
                        match_strategy=MATCH_STRATEGY,
                        models=[emb_model],
                        meta_classifiers=[meta_clf],
                        emotion_method=EMOTION_METHOD,
                        demographics_dir=DEMOGRAPHICS_DIR,
                        predicted_ages_file=PREDICTED_AGES,
                        normalize=norm_opt,
                    )

                    pipeline = MetaPipeline(
                        output_dir=output_dir,
                        config=config,
                        asymmetry_variant=asym_variant,
                    )

                    try:
                        summary_df = pipeline.run()
                        if not summary_df.empty:
                            summary_df["asymmetry_variant"] = asym_variant
                            summary_df["normalize"] = norm_tag
                            all_summaries.append(summary_df)
                    except Exception as e:
                        logger.error(f"失敗: {e}")

    # Combined summary
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined = combined.sort_values("mcc", ascending=False)
        summary_path = META_ROOT / spec.visit_dir / spec.cdr_mmse_dir / "summary_all.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
        logger.info(f"\n合併結果已儲存: {summary_path}")

        logger.info("\n" + "=" * 60)
        logger.info("最佳結果 (依 MCC 排序):")
        logger.info("=" * 60)
        for _, row in combined.head(10).iterrows():
            auc_str = f"{row['auc']:.4f}" if row['auc'] is not None else "N/A"
            logger.info(
                f"  {row['emb_model']} / {row['asymmetry_variant']} + {row['meta_classifier']}: "
                f"MCC={row['mcc']:.4f}, Acc={row['accuracy']:.4f}, AUC={auc_str}"
            )

    logger.info("\n分析完成！")


if __name__ == "__main__":
    main()
