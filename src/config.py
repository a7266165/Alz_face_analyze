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
# 單一乾淨人口學表（P/NAD/ACS 合併；scripts/external/build_hospital_A.py 產出）。
# 欄位：Group, ID(受試者數字), Photo_Session, Photo_Date, Birth_Date, Sex,
#       Age, BMI, NPT_Date, NPT_Session, Diff_Days, MMSE, CASI, Global_CDR
HOSPITAL_A_CSV = DEMOGRAPHICS_DIR / "hospital_A.csv"

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
# Preprocess
# -----------------------------------------------------------------------------
PREPROCESSING_DIR = WORKSPACE_DIR / "preprocess"
_PREPROCESS_STAGES = ("selected", "aligned", "mirrors")


def preprocess_dir(stage: str, background: bool = False) -> Path:
    """預處理輸出目錄 = PREPROCESSING_DIR / {no_background|background} / {stage}。

    stage      ∈ {selected, aligned, mirrors}
    background  False（預設）→ no_background（去背版）；True → background（保留背景版）

    取代舊的扁平常數（ALIGNED_DIR / ALIGNED_BACKGROUND_DIR / MIRRORS_DIR …），
    bg/no_bg 由參數決定，不再每個葉子各開一個常數。
    """
    if stage not in _PREPROCESS_STAGES:
        raise ValueError(
            f"stage must be one of {_PREPROCESS_STAGES}, got {stage!r}")
    variant = "background" if background else "no_background"
    return PREPROCESSING_DIR / variant / stage

# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------
EMBEDDING_DIR = WORKSPACE_DIR / "embedding"
EMBEDDING_FEATURES_DIR = EMBEDDING_DIR / "features"
EMBEDDING_ANALYSIS_DIR = EMBEDDING_DIR / "analysis"
EMBEDDING_FEATURE_STAT_DIR = EMBEDDING_ANALYSIS_DIR / "feature_stat"
EMBEDDING_CLASSIFICATION_DIR = EMBEDDING_ANALYSIS_DIR / "classification"

# -----------------------------------------------------------------------------
# Age
# -----------------------------------------------------------------------------
AGE_DIR = WORKSPACE_DIR / "age"
AGE_PREDICTIONS_DIR = AGE_DIR / "predictions"
AGE_BENCHMARK_DIR = AGE_PREDICTIONS_DIR
AGE_ANALYSIS_DIR = AGE_DIR / "analysis"

# 預設指向 DEFAULT_COHORT_MODE（analysis 下）。
_AGE_DEFAULT_ANALYSIS = AGE_ANALYSIS_DIR / "P_first_HC_first" / "P_cdr05_HC_cdrall_mmseall"

PREDICTED_AGES_FILE = AGE_PREDICTIONS_DIR / "1_MiVOLO" / "predicted_ages.json"

# 視覺化子樹（直接在 cohort 下）
AGE_SCATTER_DIR = _AGE_DEFAULT_ANALYSIS / "scatter"
AGE_STAT_DIR = _AGE_DEFAULT_ANALYSIS / "stat"
AGE_LINES_DIR = _AGE_DEFAULT_ANALYSIS / "lines"
AGE_HISTOGRAM_DIR = _AGE_DEFAULT_ANALYSIS / "histogram"
AGE_VIOLIN_DIR = _AGE_DEFAULT_ANALYSIS / "violin"

# -----------------------------------------------------------------------------
# BMI
# -----------------------------------------------------------------------------
BMI_DIR = WORKSPACE_DIR / "bmi"
BMI_MODELS_DIR = BMI_DIR / "models"
BMI_PREDICTIONS_DIR = BMI_DIR / "predictions"
BMI_ANALYSIS_DIR = BMI_DIR / "analysis"

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
# HC two tokens are coupled.  Only two valid combos:
#   (cdrall, mmseall) -> no HC cognitive filter
#   (cdr0,   mmse26)  -> HC cognitive filter: CDR==0 OR (CDR.isna() AND MMSE>=26)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    """5-axis cohort specification.

    Use ``cohort_spec_from_name`` to parse canonical strings, or construct
    directly.  ``canonical_name`` always returns the explicit form.
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
                f"(cdr0, mmse26) are supported."
            )

    @property
    def canonical_name(self) -> str:
        return (f"p_{self.p_visit}_{self.p_cdr}"
                f"_hc_{self.hc_visit}_{self.hc_cdr}_or_{self.hc_mmse}")

    @property
    def visit_dir(self) -> str:
        return f"P_{self.p_visit}_HC_{self.hc_visit}"

    @property
    def cdr_mmse_dir(self) -> str:
        return f"P_{self.p_cdr}_HC_{self.hc_cdr}_{self.hc_mmse}"

    @property
    def hc_strict(self) -> bool:
        """True when HC cognitive filter is active (CDR==0 OR MMSE>=26)."""
        return (self.hc_cdr, self.hc_mmse) == ("cdr0", "mmse26")


_VALID_HC_COMBOS = frozenset({("cdrall", "mmseall"), ("cdr0", "mmse26")})


DEFAULT_COHORT_SPEC = CohortSpec(
    p_visit="first", p_cdr="cdr05",
    hc_visit="first", hc_cdr="cdrall", hc_mmse="mmseall",
)

DEFAULT_COHORT_MODE = DEFAULT_COHORT_SPEC.canonical_name

# 16 valid canonical names (4 p_visit×p_cdr combos × 2 hc_visit × 2 hc_filter).
VALID_COHORT_CHOICES = sorted([
    CohortSpec(pv, pc, hv, hc, hm).canonical_name
    for pv in ("first", "all")
    for pc in ("cdr05", "cdrall")
    for hv in ("first", "all")
    for (hc, hm) in _VALID_HC_COMBOS
])


_CANONICAL_RE = re.compile(
    r"^p_(?P<pv>first|all)_(?P<pc>cdr05|cdrall)"
    r"_hc_(?P<hv>first|all)_(?P<hc>cdr0|cdrall)_or_(?P<hm>mmse26|mmseall)$"
)


def cohort_name(cohort_mode: str) -> str:
    """Validate and return the canonical cohort name (passthrough)."""
    if cohort_mode not in VALID_COHORT_CHOICES:
        raise ValueError(
            f"Unknown cohort_mode {cohort_mode!r}. "
            f"Must be one of the 16 canonical names.")
    return cohort_mode


def cohort_path(cohort_mode: str) -> Path:
    """Return two-level directory path: <visit_dir>/<cdr_mmse_dir>."""
    spec = cohort_spec_from_name(cohort_name(cohort_mode))
    return Path(spec.visit_dir) / spec.cdr_mmse_dir


def cohort_spec_from_name(name: str) -> CohortSpec:
    """Parse canonical name to CohortSpec; raise on invalid."""
    m = _CANONICAL_RE.match(name)
    if m is None:
        raise ValueError(
            f"Cannot parse cohort name {name!r}. "
            f"Expected canonical form "
            f"'p_<first|all>_<cdr05|cdrall>_hc_<first|all>_<cdr0|cdrall>_or_<mmse26|mmseall>'."
        )
    return CohortSpec(
        p_visit=m["pv"], p_cdr=m["pc"],
        hc_visit=m["hv"], hc_cdr=m["hc"], hc_mmse=m["hm"],
    )


def embedding_classification_path(
    cohort: str,
    bg_mode: str,
    emb: str,
    variant: str,
    photo_mode: str = "mean",
    reducer: str = "no_drop",
    clf: Optional[str] = None,
    direction: Optional[str] = None,
    eval_method: Optional[str] = None,
    match_level: Optional[str] = None,
    eval_unit: Optional[str] = None,
    match_strategy: Optional[str] = None,
    partition: Optional[str] = None,
) -> Path:
    """
    Compose embedding classification output path.

    Layout (follows 10-variable pipeline order):
      classification/<visit>/<cdr_mmse>/<bg_mode>/<emb>/<variant>/<photo>/<reducer>/
        <clf>/<direction>/<eval_method>/<match_level>/<eval_unit>/<match_strategy>/<partition>/

    Args:
        cohort: canonical cohort name → split into visit_dir + cdr_mmse_dir
        bg_mode: background | no_background
        emb: arcface | topofr | dlib | vggface
        variant: original | difference | absolute_difference | average |
                 relative_differences | absolute_relative_differences
        photo_mode: mean | all
        reducer: no_drop | pca/n_components_X | drop_feats/pearson_r_X.X
        clf, direction, eval_method, match_level, eval_unit,
        match_strategy, partition: 可選
    """
    spec = cohort_spec_from_name(cohort_name(cohort))
    p = (EMBEDDING_CLASSIFICATION_DIR / spec.visit_dir / spec.cdr_mmse_dir
         / bg_mode / emb / variant / photo_mode / reducer)
    for seg in (clf, direction, eval_method, match_level, eval_unit,
                match_strategy, partition):
        if seg is None:
            break
        p = p / seg
    return p


META_DIR = WORKSPACE_DIR / "meta"
META_ANALYSIS_DIR = META_DIR / "analysis"


def meta_analysis_path(
    cohort: str,
    bg_mode: str,
    emb_model: str,
    asymmetry_variant: str,
    photo_mode: str = "mean",
    reducer: str = "no_drop",
    base_classifier: Optional[str] = None,
    base_classifier_param: Optional[str] = None,
    direction: Optional[str] = None,
    eval_method: Optional[str] = None,
    match_level: Optional[str] = None,
    eval_unit: Optional[str] = None,
    match_strategy: Optional[str] = None,
    partition: Optional[str] = None,
    normalize_tag: Optional[str] = None,
    meta_classifier: Optional[str] = None,
) -> Path:
    """
    Compose meta analysis output path.

    Layout (aligns with embedding up to partition, then meta-specific):
      meta/analysis/<visit>/<cdr_mmse>/<bg_mode>/<emb>/<asym_variant>/<photo>/<reducer>/
        <base_clf>/<base_param>/<direction>/<eval_method>/<match_level>/<eval_unit>/
        <match_strategy>/<partition>/<normalize_tag>/<meta_clf>/
    """
    spec = cohort_spec_from_name(cohort_name(cohort))
    p = (META_ANALYSIS_DIR / spec.visit_dir / spec.cdr_mmse_dir
         / bg_mode / emb_model / asymmetry_variant / photo_mode / reducer)
    for seg in (base_classifier, base_classifier_param, direction,
                eval_method, match_level, eval_unit, match_strategy,
                partition, normalize_tag, meta_classifier):
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
    # 預設 True，產出 aligned/ + aligned_background/ 雙變體
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
