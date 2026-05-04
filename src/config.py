"""
全專案共用配置

路徑常數、專案級設定、處理參數
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


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

# 公開人臉資料集（ACS 擴充）
EXTERNAL_PUBLIC_FACE_DIR = EXTERNAL_DIR / "public_face_datasets"
EXTERNAL_DATASETS_DIR = EXTERNAL_PUBLIC_FACE_DIR / "datasets"
EXTERNAL_FILTERED_DIR = EXTERNAL_PUBLIC_FACE_DIR / "filtered"

# 工作區路徑（按模組分類）
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# preprocess 模組（ABtest branch：切到 _ABtest namespace 避免污染 production）
PREPROCESSING_DIR = WORKSPACE_DIR / "preprocess_ABtest"
SELECTED_DIR = PREPROCESSING_DIR / "selected"
ALIGNED_DIR = PREPROCESSING_DIR / "aligned"
ALIGNED_BACKGROUND_DIR = PREPROCESSING_DIR / "aligned_background"
MIRRORS_DIR = PREPROCESSING_DIR / "mirrors"

# embedding 模組（ABtest branch：切到 _ABtest namespace）
EMBEDDING_DIR = WORKSPACE_DIR / "embedding_ABtest"
FEATURES_DIR = EMBEDDING_DIR / "features"
STATISTICS_DIR = EMBEDDING_DIR / "statistics"

# age 模組
AGE_PREDICTION_DIR = WORKSPACE_DIR / "age" / "age_prediction"
CORRECTIONS_DIR = AGE_PREDICTION_DIR / "corrections"
CALIBRATION_DIR = CORRECTIONS_DIR / "calibration"
BOOTSTRAP_DIR = CORRECTIONS_DIR / "bootstrap_correction"
MEAN_CORRECTION_DIR = CORRECTIONS_DIR / "mean_correction"
PREDICTED_AGES_FILE = AGE_PREDICTION_DIR / "predicted_ages.json"
PREDICTED_AGES_CALIBRATED_FILE = AGE_PREDICTION_DIR / "predicted_ages_calibrated.json"

# age 模組 — 新 cohort（first-visit P + ALL NAD/ACS, no strict HC filter）
AGE_PREDICTION_DIR_V2 = WORKSPACE_DIR / "age" / "age_prediction_p_first_hc_all"
CORRECTIONS_DIR_V2 = AGE_PREDICTION_DIR_V2 / "corrections"
CALIBRATION_DIR_V2 = CORRECTIONS_DIR_V2 / "calibration"
BOOTSTRAP_DIR_V2 = CORRECTIONS_DIR_V2 / "bootstrap_correction"
MEAN_CORRECTION_DIR_V2 = CORRECTIONS_DIR_V2 / "mean_correction"

# arms_analysis 模組
ARMS_ANALYSIS_DIR = WORKSPACE_DIR / "arms_analysis"

# arms_analysis 模組 — default cohort
#   P : first-visit + Global_CDR>=0.5 + .npy fallback
#   HC: strict (CDR=0 OR (CDR=NaN AND MMSE>=26)) + first-visit per HC subject
ARMS_P_FIRST_HC_STRICT_DIR = ARMS_ANALYSIS_DIR / "p_first_hc_strict"
ARMS_P_FIRST_HC_STRICT_PER_ARM = ARMS_P_FIRST_HC_STRICT_DIR / "per_arm"
ARMS_P_FIRST_HC_STRICT_GRID = ARMS_P_FIRST_HC_STRICT_DIR / "grid"

# arms_analysis 模組 — 新 cohort（first-visit P + ALL NAD/ACS, no strict HC filter）
ARMS_P_FIRST_HC_ALL_DIR = ARMS_ANALYSIS_DIR / "p_first_hc_all"
ARMS_P_FIRST_HC_ALL_PER_ARM = ARMS_P_FIRST_HC_ALL_DIR / "per_arm"
ARMS_P_FIRST_HC_ALL_GRID = ARMS_P_FIRST_HC_ALL_DIR / "grid"

# 別名（向後相容）— ARMS_PER_ARM_DIR / ARMS_GRID_DIR 預設指向 p_first_hc_strict
ARMS_PER_ARM_DIR = ARMS_P_FIRST_HC_STRICT_PER_ARM
ARMS_GRID_DIR = ARMS_P_FIRST_HC_STRICT_GRID


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
