"""
全專案共用配置

路徑常數、專案級設定、處理參數
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple


# =============================================================================
# 路徑常數
# =============================================================================

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent

# 專案內資料目錄
DATA_DIR = PROJECT_ROOT / "data"
DEMOGRAPHICS_DIR = DATA_DIR / "demographics"

# 原始影像目錄（外部資料，從 data/path.txt 讀取）
_RAW_PATH_FILE = DATA_DIR / "path.txt"
if not _RAW_PATH_FILE.exists():
    raise FileNotFoundError(
        f"找不到原始影像路徑設定檔: {_RAW_PATH_FILE}\n"
        f"請建立此檔案並寫入原始影像目錄路徑"
    )
RAW_IMAGES_DIR = Path(_RAW_PATH_FILE.read_text(encoding="utf-8").strip())

# 外部依賴目錄
EXTERNAL_DIR = PROJECT_ROOT / "external"
EXTERNAL_PUBLIC_FACE_DIR = EXTERNAL_DIR / "public_face_datasets"
EXTERNAL_DATASETS_DIR = EXTERNAL_PUBLIC_FACE_DIR / "datasets"
EXTERNAL_FILTERED_DIR = EXTERNAL_PUBLIC_FACE_DIR / "filtered"

# 工作區根
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# -----------------------------------------------------------------------------
# Preprocess（ABtest branch 預設）
# -----------------------------------------------------------------------------
PREPROCESSING_DIR = WORKSPACE_DIR / "preprocess_ABtest"
SELECTED_DIR = PREPROCESSING_DIR / "selected"
ALIGNED_DIR = PREPROCESSING_DIR / "aligned"
ALIGNED_BACKGROUND_DIR = PREPROCESSING_DIR / "aligned_background"
MIRRORS_DIR = PREPROCESSING_DIR / "mirrors"

# -----------------------------------------------------------------------------
# Embedding (ABtest branch — extraction target)
# -----------------------------------------------------------------------------
EMBEDDING_ABTEST_DIR = WORKSPACE_DIR / "embedding_ABtest"
FEATURES_DIR = EMBEDDING_ABTEST_DIR / "features"
STATISTICS_DIR = EMBEDDING_ABTEST_DIR / "statistics"
EMBEDDING_ABTEST_ANALYSIS_DIR = EMBEDDING_ABTEST_DIR / "analysis"
EMBEDDING_ABTEST_CLASSIFICATION_DIR = EMBEDDING_ABTEST_ANALYSIS_DIR / "classification"

# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------
EMBEDDING_DIR = WORKSPACE_DIR / "embedding"
EMBEDDING_FEATURES_DIR = EMBEDDING_DIR / "features"
EMBEDDING_ANALYSIS_DIR = EMBEDDING_DIR / "analysis"
EMBEDDING_FEATURE_STAT_DIR = EMBEDDING_ANALYSIS_DIR / "feature_stat"
EMBEDDING_CLASSIFICATION_DIR = EMBEDDING_ANALYSIS_DIR / "classification"
# 6 個 variant 為 EMBEDDING_CLASSIFICATION_DIR 之 L1 sub-dir：
# original / difference / absolute_difference / average / relative_differences / absolute_relative_differences

# -----------------------------------------------------------------------------
# Age
# -----------------------------------------------------------------------------
AGE_DIR = WORKSPACE_DIR / "age"
AGE_PREDICTIONS_DIR = AGE_DIR / "predictions"
# 預設指向 V2.2 default canonical (p_first_cdr05_hc_first_cdrall_or_mmseall)；
# 其他 cohort 用 AGE_PREDICTIONS_DIR / cohort_name(...) 動態組合。
AGE_PREDICTION_DIR = AGE_PREDICTIONS_DIR / "p_first_cdr05_hc_first_cdrall_or_mmseall"
AGE_BENCHMARK_DIR = AGE_PREDICTIONS_DIR / "benchmark"

CORRECTIONS_DIR = AGE_PREDICTION_DIR / "corrections"
CALIBRATION_DIR = CORRECTIONS_DIR / "calibration"
BOOTSTRAP_DIR = CORRECTIONS_DIR / "bootstrap_correction"
MEAN_CORRECTION_DIR = CORRECTIONS_DIR / "mean_correction"
PREDICTED_AGES_FILE = AGE_PREDICTION_DIR / "predicted_ages.json"
PREDICTED_AGES_CALIBRATED_FILE = AGE_PREDICTION_DIR / "predicted_ages_calibrated.json"

# Age analysis 子樹
AGE_ANALYSIS_DIR = AGE_DIR / "analysis"
AGE_PRED_ERROR_STAT_DIR = AGE_ANALYSIS_DIR / "pred_error_stat"
AGE_CLASSIFICATION_DIR = AGE_ANALYSIS_DIR / "classification"
AGE_WINDOW_CLASSIFIER_DIR = AGE_CLASSIFICATION_DIR / "window_classifier"

# -----------------------------------------------------------------------------
# Longitudinal — features (raw deltas) + analysis (per-modality)
# -----------------------------------------------------------------------------
LONGITUDINAL_DIR = WORKSPACE_DIR / "longitudinal"
LONGITUDINAL_FEATURES_DIR = LONGITUDINAL_DIR / "features"
LONGITUDINAL_ANALYSIS_DIR = LONGITUDINAL_DIR / "analysis"
LONGI_AGE_DIR = LONGITUDINAL_ANALYSIS_DIR / "age"
LONGI_AGE_CLASSIFICATION_DIR = LONGI_AGE_DIR / "classification"
LONGI_AGE_PRED_ERROR_STAT_DIR = LONGI_AGE_DIR / "pred_error_stat"

# -----------------------------------------------------------------------------
# Emo_au
# -----------------------------------------------------------------------------
EMO_AU_DIR = WORKSPACE_DIR / "emo_au"
EMO_AU_FEATURES_DIR = EMO_AU_DIR / "features"
EMO_AU_FEATURES_SCHEMA_FILE = EMO_AU_FEATURES_DIR / "_schema.json"
EMO_AU_ANALYSIS_DIR = EMO_AU_DIR / "analysis"
EMO_AU_FEATURE_STAT_DIR = EMO_AU_ANALYSIS_DIR / "feature_stat"
EMO_AU_CLASSIFICATION_DIR = EMO_AU_ANALYSIS_DIR / "classification"

# -----------------------------------------------------------------------------
# Asymmetry (landmark)
# -----------------------------------------------------------------------------
ASYMMETRY_DIR = WORKSPACE_DIR / "asymmetry"
ASYMMETRY_FEATURES_DIR = ASYMMETRY_DIR / "features"
ASYMMETRY_LANDMARKS_DIR = ASYMMETRY_FEATURES_DIR / "landmarks"
ASYMMETRY_PAIR_FEATURES_FILE = ASYMMETRY_FEATURES_DIR / "pair_features.csv"
ASYMMETRY_ANALYSIS_DIR = ASYMMETRY_DIR / "analysis"
ASYMMETRY_FEATURE_STAT_DIR = ASYMMETRY_ANALYSIS_DIR / "feature_stat"
ASYMMETRY_CLASSIFICATION_DIR = ASYMMETRY_ANALYSIS_DIR / "classification"

# -----------------------------------------------------------------------------
# Overview — 跨 modality cohort metadata + matching artifacts + per-design summaries
# + cross-modality stat grid（per-cohort × per-hc_source）
# -----------------------------------------------------------------------------
OVERVIEW_DIR = WORKSPACE_DIR / "overview"

# -----------------------------------------------------------------------------
# Cohort spec (V2.2 — explicit 5-axis canonical naming)
#
# Canonical:
#   p_<p_visit>_<p_cdr>_hc_<hc_visit>_<hc_cdr>_or_<hc_mmse>
#
# Axes:
#   p_visit:  first | all
#   p_cdr:    cdr05 (Global_CDR>=0.5) | cdrall (no filter)
#   hc_visit: first | all
#   hc_cdr:   cdr0  (Global_CDR==0)   | cdrall (no filter)
#   hc_mmse:  mmse26 (MMSE>=26)       | mmseall (no filter)
#
# HC two tokens are coupled (Legacy OR rule).  Only two valid combos:
#   (cdrall, mmseall) -> HC loose (no filter)
#   (cdr0,   mmse26)  -> HC strict: CDR==0 OR (CDR.isna() AND MMSE>=26)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    """5-axis cohort specification (V2.2).

    Use ``cohort_spec_from_name`` to parse from alias/canonical strings, or
    construct directly.  ``canonical_name`` always returns the explicit form.
    """

    p_visit: Literal["first", "all"]
    p_cdr: Literal["cdr05", "cdrall"]
    hc_visit: Literal["first", "all"]
    hc_cdr: Literal["cdr0", "cdrall"]
    hc_mmse: Literal["mmse26", "mmseall"]

    def __post_init__(self) -> None:
        if (self.hc_cdr, self.hc_mmse) not in _VALID_HC_COMBOS:
            raise ValueError(
                f"Invalid HC filter combo: hc_cdr={self.hc_cdr!r}, "
                f"hc_mmse={self.hc_mmse!r}. Only (cdrall, mmseall) and "
                f"(cdr0, mmse26) are supported (Legacy OR rule)."
            )

    @property
    def canonical_name(self) -> str:
        return (f"p_{self.p_visit}_{self.p_cdr}"
                f"_hc_{self.hc_visit}_{self.hc_cdr}_or_{self.hc_mmse}")

    @property
    def hc_strict(self) -> bool:
        """True iff HC applies the Legacy OR strict rule."""
        return (self.hc_cdr, self.hc_mmse) == ("cdr0", "mmse26")


_VALID_HC_COMBOS = frozenset({("cdrall", "mmseall"), ("cdr0", "mmse26")})


DEFAULT_COHORT_SPEC = CohortSpec(
    p_visit="first", p_cdr="cdr05",
    hc_visit="first", hc_cdr="cdrall", hc_mmse="mmseall",
)


# Aliases (V0/V1 names → V2 canonical).
COHORT_ALIASES = {
    "default":           "p_first_cdr05_hc_first_cdrall_or_mmseall",
    "p_first_hc_first":  "p_first_cdr05_hc_first_cdrall_or_mmseall",
    "p_first_hc_all":    "p_first_cdr05_hc_all_cdrall_or_mmseall",
    "p_all_hc_all":      "p_all_cdr05_hc_all_cdrall_or_mmseall",
}

# 16 valid canonical names + 4 legacy aliases (de-duped).
VALID_COHORT_CHOICES = sorted(set([
    *COHORT_ALIASES.keys(),
    *[
        CohortSpec(pv, pc, hv, hc, hm).canonical_name
        for pv in ("first", "all")
        for pc in ("cdr05", "cdrall")
        for hv in ("first", "all")
        for (hc, hm) in _VALID_HC_COMBOS
    ],
]))


_CANONICAL_RE = re.compile(
    r"^p_(?P<pv>first|all)_(?P<pc>cdr05|cdrall)"
    r"_hc_(?P<hv>first|all)_(?P<hc>cdr0|cdrall)_or_(?P<hm>mmse26|mmseall)$"
)


def cohort_name(cohort_mode: str) -> str:
    """Resolve alias → canonical, or pass through if already canonical.

    Examples:
        ``cohort_name("default")`` →
            ``"p_first_cdr05_hc_first_cdrall_or_mmseall"``
        ``cohort_name("p_first_cdrall_hc_first_cdr0_or_mmse26")`` →
            same (already canonical)
    """
    return COHORT_ALIASES.get(cohort_mode, cohort_mode)


def cohort_spec_from_name(name: str) -> CohortSpec:
    """Parse alias or canonical name to CohortSpec; raise on invalid."""
    canonical = cohort_name(name)
    m = _CANONICAL_RE.match(canonical)
    if m is None:
        raise ValueError(
            f"Cannot parse cohort name {name!r} (resolved to {canonical!r}). "
            f"Expected canonical form "
            f"'p_<first|all>_<cdr05|cdrall>_hc_<first|all>_<cdr0|cdrall>_or_<mmse26|mmseall>'."
        )
    return CohortSpec(
        p_visit=m["pv"], p_cdr=m["pc"],
        hc_visit=m["hv"], hc_cdr=m["hc"], hc_mmse=m["hm"],
    )


def embedding_classification_path(
    variant: str,
    cohort: str,
    reducer: str = "no_drop",
    partition: Optional[str] = None,
    direction: Optional[str] = None,
    emb: Optional[str] = None,
    clf: Optional[str] = None,
) -> Path:
    """
    Compose embedding classification output path.

    Layout: embedding/analysis/classification/<variant>/<cohort>/<reducer>/<partition>/<direction>/<emb>/<clf>/

    Args:
        variant: original | difference | absolute_difference | average |
                 relative_differences | absolute_relative_differences
        cohort: alias or canonical cohort name (will be resolved via cohort_name)
        reducer: no_drop | pca/n_components_X | drop_feats/pearson_r_X.X
        partition, direction, emb, clf: 可選、None 時止於前面那層
    """
    p = EMBEDDING_CLASSIFICATION_DIR / variant / cohort_name(cohort) / reducer
    for seg in (partition, direction, emb, clf):
        if seg is None:
            break
        p = p / seg
    return p


def get_raw_images_subdir(group: str) -> Path:
    """
    取得原始影像子目錄

    Args:
        group: "ACS", "NAD", "P" 或 "health/ACS", "health/NAD", "patient"

    Returns:
        完整路徑
    """
    if group == "EACS":
        return EXTERNAL_FILTERED_DIR
    group_mapping = {
        "ACS": "health/ACS",
        "NAD": "health/NAD",
        "P": "patient",
    }
    subdir = group_mapping.get(group, group)
    return RAW_IMAGES_DIR / subdir


# =============================================================================
# 處理參數 Dataclass
# =============================================================================

@dataclass
class MirrorConfig:
    """鏡射生成配置"""

    mirror_method: str = "flip"  # "midline" (沿臉部中線) 或 "flip" (水平翻轉)
    mirror_size: Tuple[int, int] = (512, 512)  # 輸出鏡射影像大小
    feather_px: int = 2  # 邊緣羽化像素
    margin: float = 0.08  # 畫布邊緣留白比例
    midline_points: Tuple[int, ...] = (10, 168, 4, 2)  # 臉部中軸線特徵點索引


@dataclass
class PreprocessConfig:
    """共用預處理配置"""

    # ========== MediaPipe 特徵點 ==========
    midline_points: Tuple[int, ...] = (10, 168, 4, 2)  # 同 src.common.mediapipe_utils.MIDLINE_POINTS

    # ========== 相片選擇參數 ==========
    n_select: int = 10  # 選擇多少張最正的臉部相片
    detection_confidence: float = 0.5  # MediaPipe 偵測信心度閾值

    # ========== CLAHE 參數 ==========
    apply_clahe: bool = False  # 是否應用 CLAHE
    clahe_clip_limit: float = 2.0  # CLAHE 限制參數
    clahe_tile_size: int = 8  # CLAHE 區塊大小

    # ========== 儲存控制 ==========
    save_intermediate: bool = False  # 是否儲存中間結果
    subject_id: Optional[str] = None  # 受試者 ID（用於建立子目錄）

    # 額外輸出未去背版本到 ALIGNED_BACKGROUND_DIR（不影響既有 aligned/）
    # ABtest branch 預設 True，產出 aligned/ + aligned_background/ 雙變體
    also_save_aligned_background: bool = True

    # ========== 處理流程控制 ==========
    steps: List[str] = field(
        default_factory=lambda: [
            "select",  # 選擇最正面的 n 張
            "align",   # 角度校正
        ]
    )


@dataclass
class APIConfig(PreprocessConfig):
    """API 配置"""

    save_intermediate: bool = False  # API 預設不儲存
    cleanup_on_complete: bool = True  # 完成後清理暫存檔


@dataclass
class AnalyzeConfig(PreprocessConfig):
    """Analyze 配置"""

    save_intermediate: bool = True  # Analyze 預設儲存
    mirror: MirrorConfig = field(default_factory=MirrorConfig)
