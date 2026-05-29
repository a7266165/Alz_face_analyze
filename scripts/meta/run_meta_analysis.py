"""
Meta Analysis 執行腳本 (v3)

多維度 sweep:
  bg_mode × extra_feat × emb_model × (asym_variant, scoring_method) × normalize × meta_clf

輸出路徑:
  meta/analysis/{visit}/{cdr}/{bg}/{extra_feat}/{emb}/{asym_variant}/{scoring}/
    {photo}/{reducer}/fwd/{normalize}/{meta_clf}/

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
    META_ANALYSIS_DIR,
    PREDICTED_AGES_FILE,
    cohort_name,
    cohort_spec_from_name,
)
from src.meta import MetaConfig, MetaPipeline
from src.meta.evaluation.matched_eval import build_matching_cache

# ========== 路徑設定 ==========

WORKSPACE_DIR = PROJECT_ROOT / "workspace"
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"

# ========== 分析設定 ==========

COHORT_MODE = "p_first_cdrall_hc_all_cdrall_or_mmseall"
PHOTO_MODE = "mean"
REDUCER = "no_drop"
BASE_CLASSIFIER = "logistic"
BASE_CLASSIFIER_PARAM = "C_1"

BG_MODES = ["background", "no_background"]
EMB_MODELS = ["arcface", "dlib", "topofr", "vggface"]
META_CLASSIFIERS = ["logistic", "xgboost", "tabpfn"]
NORMALIZE_OPTIONS = [None, "minmax"]
EXTRA_FEATURES_OPTIONS = [[], ["bmi"]]

# (asymmetry_variant, scoring_method)
ASYMMETRY_CONFIGS = [
    ("none", "none"),
    ("difference", "l2_norm"),
    ("difference", "centroid_dist"),
    ("difference", "lda_projection"),
    ("absolute_difference", "l2_norm"),
    ("absolute_difference", "centroid_dist"),
    ("absolute_difference", "lda_projection"),
    ("relative_differences", "l2_norm"),
    ("relative_differences", "centroid_dist"),
    ("relative_differences", "lda_projection"),
    ("absolute_relative_differences", "l2_norm"),
    ("absolute_relative_differences", "centroid_dist"),
    ("absolute_relative_differences", "lda_projection"),
]

# Age predictions (raw MiVOLO predictions; cohort-agnostic)
spec = cohort_spec_from_name(cohort_name(COHORT_MODE))
PREDICTED_AGES = PREDICTED_AGES_FILE

EMOTION_METHOD = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def build_output_dir(bg_mode, extra_feat_tag, emb_model,
                     asym_variant, scoring_method,
                     normalize, meta_clf):
    norm_tag = normalize if normalize else "raw"
    return (
        META_ANALYSIS_DIR / spec.visit_dir / spec.cdr_mmse_dir
        / bg_mode / extra_feat_tag / emb_model
        / asym_variant / scoring_method
        / PHOTO_MODE / REDUCER / "fwd"
        / norm_tag / meta_clf
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Meta Analysis (v3) — multi-dimensional sweep")
    logger.info("=" * 60)
    logger.info(f"Cohort: {COHORT_MODE}")
    logger.info(f"BG modes: {BG_MODES}")
    logger.info(f"Embedding models: {EMB_MODELS}")
    logger.info(f"Asymmetry configs: {len(ASYMMETRY_CONFIGS)}")
    logger.info(f"Meta classifiers: {META_CLASSIFIERS}")
    logger.info(f"Normalize: {NORMALIZE_OPTIONS}")
    logger.info(f"Extra features: {EXTRA_FEATURES_OPTIONS}")

    if not DEMOGRAPHICS_DIR.exists():
        logger.error(f"找不到人口學資料目錄: {DEMOGRAPHICS_DIR}")
        sys.exit(1)
    if not PREDICTED_AGES.exists():
        logger.error(f"找不到預測年齡檔案: {PREDICTED_AGES}")
        sys.exit(1)

    logger.info("建構全局 matching cache...")
    matching_cache = build_matching_cache(
        cohort_mode=COHORT_MODE,
    )

    all_summaries = []

    for bg_mode in BG_MODES:
        for extra_feats in EXTRA_FEATURES_OPTIONS:
            extra_feat_tag = "with_bmi" if "bmi" in extra_feats else "no_bmi"
            for emb_model in EMB_MODELS:
                for asym_variant, scoring_method in ASYMMETRY_CONFIGS:
                    for norm_opt in NORMALIZE_OPTIONS:
                        for meta_clf in META_CLASSIFIERS:
                            output_dir = build_output_dir(
                                bg_mode, extra_feat_tag, emb_model,
                                asym_variant, scoring_method,
                                norm_opt, meta_clf,
                            )
                            norm_tag = norm_opt if norm_opt else "raw"

                            if (output_dir / "summary.csv").exists():
                                logger.info(f"  SKIP: {output_dir}")
                                summary_df = pd.read_csv(output_dir / "summary.csv")
                                if not summary_df.empty:
                                    summary_df["asymmetry_variant"] = asym_variant
                                    summary_df["scoring_method"] = scoring_method
                                    summary_df["normalize"] = norm_tag
                                    summary_df["bg_mode"] = bg_mode
                                    summary_df["extra_features"] = extra_feat_tag
                                    all_summaries.append(summary_df)
                                continue

                            label = (f"{bg_mode}/{extra_feat_tag}/{emb_model}/"
                                     f"{asym_variant}/{scoring_method}/"
                                     f"{norm_tag}/{meta_clf}")
                            logger.info(f"\n{'='*60}")
                            logger.info(f"  {label}")
                            logger.info(f"  → {output_dir}")
                            logger.info(f"{'='*60}")

                            config = MetaConfig(
                                cohort_mode=COHORT_MODE,
                                bg_mode=bg_mode,
                                photo_mode=PHOTO_MODE,
                                reducer=REDUCER,
                                base_classifier=BASE_CLASSIFIER,
                                base_classifier_param=BASE_CLASSIFIER_PARAM,
                                match_strategy="priority_acs",
                                scoring_method=scoring_method,
                                extra_features=extra_feats,
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
                                matching_cache=matching_cache,
                            )

                            try:
                                summary_df = pipeline.run()
                                if not summary_df.empty:
                                    summary_df["asymmetry_variant"] = asym_variant
                                    summary_df["scoring_method"] = scoring_method
                                    summary_df["normalize"] = norm_tag
                                    summary_df["bg_mode"] = bg_mode
                                    summary_df["extra_features"] = extra_feat_tag
                                    all_summaries.append(summary_df)
                            except Exception as e:
                                logger.error(f"失敗: {e}")

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined = combined.sort_values("mcc", ascending=False)
        summary_path = META_ANALYSIS_DIR / spec.visit_dir / spec.cdr_mmse_dir / "summary_all.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
        logger.info(f"\n合併結果已儲存: {summary_path}")

        logger.info("\n" + "=" * 60)
        logger.info("最佳結果 (依 MCC 排序):")
        logger.info("=" * 60)
        for _, row in combined.head(10).iterrows():
            auc_str = f"{row['auc']:.4f}" if row['auc'] is not None else "N/A"
            logger.info(
                f"  {row['bg_mode']}/{row.get('extra_features','')}/{row['emb_model']}/"
                f"{row.get('asymmetry_variant','')}/{row.get('scoring_method','')}"
                f" + {row['meta_classifier']}: "
                f"MCC={row['mcc']:.4f}, Acc={row['accuracy']:.4f}, AUC={auc_str}"
            )

    logger.info("\n分析完成！")


if __name__ == "__main__":
    main()
