"""
Meta Analysis 設定
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MetaConfig:

    # --- embedding pipeline 路徑參數 ---
    # cohort 4-token(順序同 cohort_list);取代舊的 cohort_mode 字串。
    p_visit: str = "p_first"
    p_score: str = "p_cdr05"
    hc_visit: str = "hc_first"
    hc_score: str = "hc_cdrall_or_mmseall"
    bg_mode: str = "no_background"
    photo_mode: str = "mean"
    reducer: str = "no_drop"
    base_classifier: str = "logistic"
    base_classifier_param: str = "C_1"
    direction: str = "fwd"
    eval_method: str = "1by1matched"
    match_level: str = "subject_match"
    eval_unit: str = "eval_by_subject"
    match_strategy: str = "no_priority"
    hc_source_mode: str = "ACS"
    partition: str = "ad_vs_hc"

    # --- asymmetry scoring ---
    scoring_method: str = "none"  # "none", "l2_norm", "centroid_dist", "lda_projection"

    # --- extra features ---
    extra_features: List[str] = field(default_factory=list)  # e.g. ["bmi"]

    # --- emotion 設定 ---
    emotion_method: Optional[str] = None
    emotion_features_dir: Optional[Path] = None
    emotion_schema_file: Optional[Path] = None
    emonet_csv: Optional[Path] = None

    # --- meta classifier ---
    meta_classifiers: List[str] = field(
        default_factory=lambda: ["tabpfn"],
    )

    # --- 前處理 ---
    normalize: Optional[str] = None  # None, "minmax", "standard"

    # --- 訓練 ---
    random_seed: int = 42

    # --- 分析範圍 ---
    models: List[str] = field(
        default_factory=lambda: ["arcface", "topofr"],
    )

    # --- 資料路徑 ---
    demographics_dir: Optional[Path] = None
    predicted_ages_file: Optional[Path] = None

    def __post_init__(self):
        valid_models = {"arcface", "topofr", "dlib", "vggface"}
        for m in self.models:
            if m not in valid_models:
                raise ValueError(f"無效的模型: {m}，必須是 {valid_models}")

        valid_clf = {"tabpfn", "logistic", "xgboost"}
        for c in self.meta_classifiers:
            if c not in valid_clf:
                raise ValueError(
                    f"無效的 meta classifier: {c}，必須是 {valid_clf}"
                )

        valid_scoring = {"none", "l2_norm", "centroid_dist", "lda_projection"}
        if self.scoring_method not in valid_scoring:
            raise ValueError(
                f"無效的 scoring_method: {self.scoring_method}，必須是 {valid_scoring}"
            )
