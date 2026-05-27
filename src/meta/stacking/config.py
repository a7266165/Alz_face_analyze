"""
Meta Analysis 設定
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MetaConfig:
    """
    Meta 分析設定

    Attributes:
        cohort_mode: cohort 模式（用於建構 OOF 路徑）
        bg_mode: 背景模式
        photo_mode: 照片聚合模式
        reducer: 降維方式（固定 no_drop）
        base_classifier: base model 分類器名稱
        base_classifier_param: base model 分類器參數子目錄
        direction: forward / reverse
        eval_method: 評估匹配方法
        match_level: 匹配層級
        eval_unit: 評估單位
        match_strategy: 匹配策略（統一指定）
        partition: 資料分割
        emotion_method: emotion 來源 ("emonet" 或 schema 裡的 tool 名)
        emotion_features_dir: .npy 特徵目錄
        emotion_schema_file: _schema.json 路徑
        emonet_csv: EmoNet CSV 路徑
        meta_classifiers: meta-level classifier 清單
        random_seed: 隨機種子
        models: 要分析的 embedding 模型列表
        demographics_dir: 人口學資料目錄
        predicted_ages_file: 預測年齡 JSON 路徑
    """

    # --- embedding pipeline 路徑參數 ---
    cohort_mode: str = "p_first_cdr05_hc_first_cdrall_or_mmseall"
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

    # --- emotion 設定 ---
    emotion_method: str = "emonet"
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

    # --- 輸出 ---
    save_models: bool = True
    save_predictions: bool = True
    save_reports: bool = True

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
