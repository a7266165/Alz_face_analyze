"""
全專案共用配置

路徑常數、專案級設定、處理參數
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


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

# Refactor sandbox (2026-06): the rebuilt downstream writes its OOF / metrics here
# so results never mix with the legacy workspace/ outputs. Mirrors the same relative
# layout under a separate root → trivial A/B diff (same relative path, two roots).
WORKSPACE_REFACTOR_DIR = PROJECT_ROOT / "workspace_refactor_20260601"
EMBEDDING_CLASSIFICATION_REFACTOR_DIR = (
    WORKSPACE_REFACTOR_DIR / "embedding" / "analysis" / "classification")

# -----------------------------------------------------------------------------
# Age
# -----------------------------------------------------------------------------
AGE_DIR = WORKSPACE_DIR / "age"
AGE_PREDICTIONS_DIR = AGE_DIR / "predictions"
AGE_BENCHMARK_DIR = AGE_PREDICTIONS_DIR
AGE_ANALYSIS_DIR = AGE_DIR / "analysis"

# 預設指向 DEFAULT_COHORT_TOKENS（analysis 下）。
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
# Cohort tokens (4-axis — same signature as src.common.cohort.cohort_list)
#
#   p_visit  ∈ {p_first, p_all}
#   p_score  ∈ {p_cdrall, p_cdr05, p_cdr1, p_cdr2}     (Global_CDR >= 0 / .5 / 1 / 2)
#   hc_visit ∈ {hc_first, hc_all}
#   hc_score ∈ {hc_cdrall_or_mmseall, hc_cdr0_or_mmse26}
#
# 一個 cohort = 這 4 個 token 的 tuple,順序同 cohort_list,程式內以 ``*cohort`` 流通。
# 取代舊的 cohort_mode 字串 / 5-axis CohortSpec —— 後者 p_cdr 只有 cdr05/cdrall,表達
# 不出 p_cdr1 / p_cdr2;4-token 是其嚴格超集。輸出路徑沿用舊命名,逐字相容。
# -----------------------------------------------------------------------------

P_VISIT_TOKENS = ("p_first", "p_all")
P_SCORE_TOKENS = ("p_cdrall", "p_cdr05", "p_cdr1", "p_cdr2")
HC_VISIT_TOKENS = ("hc_first", "hc_all")
HC_SCORE_TOKENS = ("hc_cdrall_or_mmseall", "hc_cdr0_or_mmse26")

DEFAULT_COHORT_TOKENS = ("p_first", "p_cdr05", "hc_first", "hc_cdrall_or_mmseall")

# hc_score token → 路徑片段(對齊 legacy cdr_mmse_dir 的 HC 部分:cdr0_mmse26 / cdrall_mmseall)
_HC_SCORE_DIR = {
    "hc_cdrall_or_mmseall": "cdrall_mmseall",
    "hc_cdr0_or_mmse26": "cdr0_mmse26",
}


def validate_cohort_tokens(p_visit, p_score, hc_visit, hc_score) -> None:
    """4 token 各自落在合法字彙;否則 raise。"""
    for tok, vocab in ((p_visit, P_VISIT_TOKENS), (p_score, P_SCORE_TOKENS),
                       (hc_visit, HC_VISIT_TOKENS), (hc_score, HC_SCORE_TOKENS)):
        if tok not in vocab:
            raise ValueError(f"invalid cohort token {tok!r}; expected one of {vocab}")


def cohort_dirs(p_visit, p_score, hc_visit, hc_score) -> Tuple[str, str]:
    """4 token → (visit_dir, cdr_mmse_dir)。逐字相容舊 CohortSpec 命名,並支援新的
    p_cdr1 / p_cdr2(舊 5-axis 表達不出)。"""
    validate_cohort_tokens(p_visit, p_score, hc_visit, hc_score)
    visit_dir = f"P_{p_visit.split('_', 1)[1]}_HC_{hc_visit.split('_', 1)[1]}"
    cdr_mmse_dir = f"P_{p_score.split('_', 1)[1]}_HC_{_HC_SCORE_DIR[hc_score]}"
    return visit_dir, cdr_mmse_dir


def cohort_path(p_visit, p_score, hc_visit, hc_score) -> Path:
    """Two-level cohort 目錄:<visit_dir>/<cdr_mmse_dir>。"""
    visit_dir, cdr_mmse_dir = cohort_dirs(p_visit, p_score, hc_visit, hc_score)
    return Path(visit_dir) / cdr_mmse_dir


def embedding_classification_path(
    p_visit: str,
    p_score: str,
    hc_visit: str,
    hc_score: str,
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
    root: Optional[Path] = None,
) -> Path:
    """
    Compose embedding classification output path.

    Layout (follows 10-variable pipeline order):
      classification/<visit>/<cdr_mmse>/<bg_mode>/<emb>/<variant>/<photo>/<reducer>/
        <clf>/<direction>/<eval_method>/<match_level>/<eval_unit>/<match_strategy>/<partition>/

    Args:
        p_visit, p_score, hc_visit, hc_score: cohort 4-token(見上方 cohort token 區塊)
        bg_mode: background | no_background
        emb: arcface | topofr | dlib | vggface
        variant: original | difference | absolute_difference | average |
                 relative_differences | absolute_relative_differences
        photo_mode: mean | all
        reducer: no_drop | pca/n_components_X | drop_feats/pearson_r_X.X
        clf, direction, eval_method, match_level, eval_unit,
        match_strategy, partition: 可選
    """
    visit_dir, cdr_mmse_dir = cohort_dirs(p_visit, p_score, hc_visit, hc_score)
    base = root if root is not None else EMBEDDING_CLASSIFICATION_DIR
    p = (base / visit_dir / cdr_mmse_dir
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
    p_visit: str,
    p_score: str,
    hc_visit: str,
    hc_score: str,
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
    visit_dir, cdr_mmse_dir = cohort_dirs(p_visit, p_score, hc_visit, hc_score)
    p = (META_ANALYSIS_DIR / visit_dir / cdr_mmse_dir
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
    """預處理各站參數（detect / select / align / mirror 共用）。

    去背/鏡射「要不要做」由 run_preprocess.py 的 toggle 控制，
    不再放在 config（昔日的 steps / also_save_aligned_background 已移除）。
    """

    # MediaPipe 特徵點（同 src.common.mediapipe_utils.MIDLINE_POINTS）
    midline_points: Tuple[int, ...] = (10, 168, 4, 2)

    # 相片選擇 / 偵測
    n_select: int = 10  # 選擇多少張最正的臉部相片
    detection_confidence: float = 0.5  # MediaPipe 偵測信心度閾值

    # 鏡射參數
    mirror: MirrorConfig = field(default_factory=MirrorConfig)
