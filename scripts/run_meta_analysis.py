"""
Meta Analysis 執行腳本

整合 14 個特徵，訓練 TabPFN meta-model。

特徵：
  1. age_error (real - predicted)
  2. real_age
  3. lr_score_original (原始臉 LR 分數)
  4. lr_score_asymmetry (不對稱性 LR 分數)
  5-12. 8 種表情
  13. Valence
  14. Arousal

使用方式:
    poetry run python scripts/run_meta_analysis.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# 加入專案根目錄到 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.meta_analysis import MetaConfig, MetaPipeline

# ========== 設定 ==========

# 路徑設定
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
PREDICTED_AGES_FILE = WORKSPACE_DIR / "predicted_ages.json"

# LR 預測分數目錄 (根據你的實際目錄名稱修改)
PREDICTIONS_DIR = (
    WORKSPACE_DIR
    / "analysis_20260226_100333_logistic_balancing_False_allvisits_True"
    / "pred_probability"
)

# Emotion 分數檔案
EMOTION_SCORES_FILE = WORKSPACE_DIR / "emotion_score_EmoNet.csv"

# 輸出目錄
OUTPUT_DIR = WORKSPACE_DIR / f"tabpfn_meta_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 分析設定
CONFIG = MetaConfig(
    # 資料設定
    cdr_threshold=0,

    # 訓練設定（折數由 base model 預測檔自動決定）
    random_seed=42,

    # 輸出設定
    save_models=True,
    save_predictions=True,
    save_reports=True,

    # 分析範圍
    # models=["arcface", "topofr", "dlib".],
    models=["arcface"],
    asymmetry_method="absolute_relative_differences",
    n_features_list=None,  # None = 自動發現全部 n_features

    # 資料路徑
    demographics_dir=DEMOGRAPHICS_DIR,
    predicted_ages_file=PREDICTED_AGES_FILE,
)


def setup_logging():
    """設定日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def main():
    """主程式"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("TabPFN Meta Analysis (14 特徵)")
    logger.info("=" * 60)

    # 檢查路徑
    if not PREDICTIONS_DIR.exists():
        logger.error(f"找不到預測目錄: {PREDICTIONS_DIR}")
        sys.exit(1)

    if not EMOTION_SCORES_FILE.exists():
        logger.error(f"找不到 emotion 檔案: {EMOTION_SCORES_FILE}")
        sys.exit(1)

    if not DEMOGRAPHICS_DIR.exists():
        logger.error(f"找不到人口學資料目錄: {DEMOGRAPHICS_DIR}")
        sys.exit(1)

    if not PREDICTED_AGES_FILE.exists():
        logger.error(f"找不到預測年齡檔案: {PREDICTED_AGES_FILE}")
        sys.exit(1)

    logger.info(f"預測目錄: {PREDICTIONS_DIR}")
    logger.info(f"Emotion 檔案: {EMOTION_SCORES_FILE}")
    logger.info(f"人口學目錄: {DEMOGRAPHICS_DIR}")
    logger.info(f"預測年齡檔案: {PREDICTED_AGES_FILE}")
    logger.info(f"輸出目錄: {OUTPUT_DIR}")

    # 執行分析
    pipeline = MetaPipeline(
        predictions_dir=PREDICTIONS_DIR,
        emotion_scores_file=EMOTION_SCORES_FILE,
        output_dir=OUTPUT_DIR,
        config=CONFIG,
    )

    summary_df = pipeline.run()

    # 顯示最佳結果
    if not summary_df.empty:
        logger.info("\n" + "=" * 60)
        logger.info("最佳結果 (依 MCC 排序前 10):")
        logger.info("=" * 60)

        top_10 = summary_df.head(10)
        for _, row in top_10.iterrows():
            auc_str = f"{row['auc']:.4f}" if row['auc'] is not None else "N/A"
            logger.info(
                f"  {row['model']}, n_features={row['n_features']:3d}: "
                f"MCC={row['mcc']:.4f}, Acc={row['accuracy']:.4f}, "
                f"AUC={auc_str}"
            )

    logger.info("\n分析完成！")
    logger.info(f"結果已儲存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
