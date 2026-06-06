"""
src 模組

匯出全專案共用配置
"""

from .config import (
    # 路徑常數
    PROJECT_ROOT,
    EXTERNAL_DIR,
    RAW_IMAGES_DIR,
    DATA_DIR,
    DEMOGRAPHICS_DIR,
    HOSPITAL_A_CSV,
    WORKSPACE_DIR,
    # Embedding
    EMBEDDING_DIR,
    EMBEDDING_FEATURES_DIR,
    EMBEDDING_ANALYSIS_DIR,
    EMBEDDING_FEATURE_STAT_DIR,
    EMBEDDING_CLASSIFICATION_DIR,
    # Preprocess
    PREPROCESSING_DIR,
    preprocess_dir,
    # Age
    AGE_DIR,
    AGE_PREDICTIONS_DIR,
    AGE_BENCHMARK_DIR,
    PREDICTED_AGES_FILE,
    AGE_SCATTER_DIR,
    AGE_STAT_DIR,
    AGE_LINES_DIR,
    AGE_HISTOGRAM_DIR,
    AGE_VIOLIN_DIR,
    AGE_ANALYSIS_DIR,
    # Emo_au
    EMO_AU_DIR,
    EMO_AU_FEATURES_DIR,
    EMO_AU_FEATURES_SCHEMA_FILE,
    EMO_AU_ANALYSIS_DIR,
    EMO_AU_FEATURE_STAT_DIR,
    EMO_AU_CLASSIFICATION_DIR,
    # Asymmetry
    ASYMMETRY_DIR,
    ASYMMETRY_FEATURES_DIR,
    ASYMMETRY_LANDMARKS_DIR,
    ASYMMETRY_PAIR_FEATURES_FILE,
    ASYMMETRY_ANALYSIS_DIR,
    ASYMMETRY_FEATURE_STAT_DIR,
    ASYMMETRY_CLASSIFICATION_DIR,
    # Rotation
    ROTATION_DIR,
    ROTATION_FIG_DIR,
    ROTATION_FEATURES_DIR,
    # Cross-modality
    OVERVIEW_DIR,
    # Cohort tokens + helpers
    P_VISIT_TOKENS,
    P_SCORE_TOKENS,
    HC_VISIT_TOKENS,
    HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
    validate_cohort_tokens,
    cohort_dirs,
    cohort_path,
    embedding_classification_path,
    get_raw_images_subdir,
    # 處理參數
    PreprocessConfig,
    MirrorConfig,
)

__all__ = [
    # 路徑常數
    "PROJECT_ROOT",
    "EXTERNAL_DIR",
    "RAW_IMAGES_DIR",
    "DATA_DIR",
    "DEMOGRAPHICS_DIR",
    "HOSPITAL_A_CSV",
    "WORKSPACE_DIR",
    # Embedding
    "EMBEDDING_DIR",
    "EMBEDDING_FEATURES_DIR",
    "EMBEDDING_ANALYSIS_DIR",
    "EMBEDDING_FEATURE_STAT_DIR",
    "EMBEDDING_CLASSIFICATION_DIR",
    # Preprocess
    "PREPROCESSING_DIR",
    "preprocess_dir",
    # Age
    "AGE_DIR",
    "AGE_PREDICTIONS_DIR",
    "AGE_ANALYSIS_DIR",
    "AGE_BENCHMARK_DIR",
    "PREDICTED_AGES_FILE",
    "AGE_SCATTER_DIR",
    "AGE_STAT_DIR",
    "AGE_LINES_DIR",
    "AGE_HISTOGRAM_DIR",
    "AGE_VIOLIN_DIR",
    # Emo_au
    "EMO_AU_DIR",
    "EMO_AU_FEATURES_DIR",
    "EMO_AU_FEATURES_SCHEMA_FILE",
    "EMO_AU_ANALYSIS_DIR",
    "EMO_AU_FEATURE_STAT_DIR",
    "EMO_AU_CLASSIFICATION_DIR",
    # Asymmetry
    "ASYMMETRY_DIR",
    "ASYMMETRY_FEATURES_DIR",
    "ASYMMETRY_LANDMARKS_DIR",
    "ASYMMETRY_PAIR_FEATURES_FILE",
    "ASYMMETRY_ANALYSIS_DIR",
    "ASYMMETRY_FEATURE_STAT_DIR",
    "ASYMMETRY_CLASSIFICATION_DIR",
    # Rotation
    "ROTATION_DIR",
    "ROTATION_FIG_DIR",
    "ROTATION_FEATURES_DIR",
    # Cross-modality
    "OVERVIEW_DIR",
    # Cohort tokens + helpers
    "P_VISIT_TOKENS",
    "P_SCORE_TOKENS",
    "HC_VISIT_TOKENS",
    "HC_SCORE_TOKENS",
    "DEFAULT_COHORT_TOKENS",
    "validate_cohort_tokens",
    "cohort_dirs",
    "cohort_path",
    "embedding_classification_path",
    "get_raw_images_subdir",
    # 處理參數
    "PreprocessConfig",
    "MirrorConfig",
]
