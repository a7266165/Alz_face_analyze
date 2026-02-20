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

# 原始影像目錄（外部資料，不在專案內）
RAW_IMAGES_DIR = Path(r"D:\project\Alz\face\data\datung\raw")

# 專案內資料目錄
DATA_DIR = PROJECT_ROOT / "data"
DEMOGRAPHICS_DIR = DATA_DIR / "demographics"

# 外部依賴目錄
EXTERNAL_DIR = PROJECT_ROOT / "external"

# 工作區路徑
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
FEATURES_DIR = WORKSPACE_DIR / "features"

# 預處理路徑
PREPROCESSING_DIR = WORKSPACE_DIR / "preprocessing"
SELECTED_DIR = PREPROCESSING_DIR / "selected"
ALIGNED_DIR = PREPROCESSING_DIR / "aligned"
MIRRORS_DIR = PREPROCESSING_DIR / "mirrors"

STATISTICS_DIR = WORKSPACE_DIR / "statistics"
PREDICTED_AGES_FILE = WORKSPACE_DIR / "predicted_ages.json"


def get_raw_images_subdir(group: str) -> Path:
    """
    取得原始影像子目錄

    Args:
        group: "ACS", "NAD", "P" 或 "health/ACS", "health/NAD", "patient"

    Returns:
        完整路徑
    """
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
class PreprocessConfig:
    """共用預處理配置"""

    # ========== MediaPipe 特徵點 ==========
    midline_points: Tuple[int, ...] = (10, 168, 4, 2)  # 臉部中軸線特徵點索引

    # ========== 相片選擇參數 ==========
    n_select: int = 10  # 選擇多少張最正的臉部相片
    detection_confidence: float = 0.5  # MediaPipe 偵測信心度閾值

    # ========== 角度校正參數 ==========
    align_face: bool = True  # 是否校正角度

    # ========== 鏡射參數 ==========
    mirror_method: str = "flip"  # "midline" (沿臉部中線) 或 "flip" (水平翻轉)
    mirror_size: Tuple[int, int] = (512, 512)  # 輸出鏡射影像大小
    feather_px: int = 2  # 邊緣羽化像素
    margin: float = 0.08  # 畫布邊緣留白比例

    # ========== CLAHE 參數 ==========
    apply_clahe: bool = False  # 是否應用 CLAHE
    clahe_clip_limit: float = 2.0  # CLAHE 限制參數
    clahe_tile_size: int = 8  # CLAHE 區塊大小

    # ========== 儲存控制 ==========
    save_intermediate: bool = False  # 是否儲存中間結果
    subject_id: Optional[str] = None  # 受試者 ID（用於建立子目錄）

    # ========== 處理流程控制 ==========
    steps: List[str] = field(
        default_factory=lambda: [
            "select",  # 選擇最正面的 n 張
            "align",   # 角度校正
            "mirror",  # 生成左右鏡射
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
