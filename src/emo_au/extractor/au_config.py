"""統一的 AU / 情緒特徵名稱、欄位對映與量綱轉換規則。"""

import re
from typing import Dict, List

from src.config import EXTERNAL_DIR, EMO_AU_FEATURES_DIR

# =============================================================================
# 統一特徵名稱
# =============================================================================

# OpenFace 3.0 輸出的 8 個 AU（DISFA dataset）
# 注意：老師原始表有 12 AU，但 OpenFace 3.0 只支援 8 個
# 缺少 AU5, AU15, AU17, AU20
OPENFACE_AUS: List[str] = [
    "AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26",
]

# harmonized AU 基準集:目前直接採 OpenFace 3.0 的 8 AU（跨工具交集尚未啟用）。
HARMONIZED_AUS: List[str] = OPENFACE_AUS

# 統一情緒名稱（7 種基本情緒，排除 contempt）
HARMONIZED_EMOTIONS: List[str] = [
    "anger", "disgust", "fear", "happiness",
    "sadness", "surprise", "neutral",
]

# =============================================================================
# 統一物理欄序
# =============================================================================
# 各工具輸出的欄位不一致（不同情緒/AU 子集），但落地（npz / schema）一律套用同一套
# 物理順序，讓任一工具的同名欄都落在可比位置；每個工具只填自己有的欄。
# 順序：共有 7 情緒(固定) → contempt(額外情緒) → AU(依編號；同號原始強度在 _det 之前)
# → 其他額外欄(gaze 等，字典序)。純規則、不依賴其他工具，故各工具能在自己的 env 內
# 獨立套用（見 src/emo_au/extractor/base.py 對 output_columns 的說明）。
_AU_RE = re.compile(r"AU0*(\d+)(_det)?")


def _canonical_key(col: str):
    if col in HARMONIZED_EMOTIONS:
        return (0, HARMONIZED_EMOTIONS.index(col), 0, col)
    if col == "contempt":
        return (1, 0, 0, col)
    m = _AU_RE.fullmatch(col)
    if m:
        return (2, int(m.group(1)), 1 if m.group(2) else 0, col)
    return (3, 0, 0, col)


def canonical_order(columns: List[str]) -> List[str]:
    """把任一工具的欄位排成全庫統一的物理順序（規則見上方），不改名、只重排。"""
    return sorted(columns, key=_canonical_key)


# pyfeat / libreface 的「原始名 → harmonized 名」皆為 identity（原生已用 harmonized 名、同序）；
# 用單一映射表達，兩者各保留具名別名供其 extractor import。
IDENTITY_EMOTION_MAP: Dict[str, str] = {e: e for e in HARMONIZED_EMOTIONS}

# =============================================================================
# OpenFace 3.0 欄位對映
# =============================================================================

# MultitaskPredictor 輸出 8 AU（DISFA order）
# au_output tensor index → harmonized AU name
OPENFACE_AU_INDEX: Dict[int, str] = {
    0: "AU1",
    1: "AU2",
    2: "AU4",
    3: "AU6",
    4: "AU9",
    5: "AU12",
    6: "AU25",
    7: "AU26",
}

# 情緒輸出（8 classes，softmax 後取機率）
OPENFACE_EMOTION_INDEX: Dict[int, str] = {
    0: "neutral",
    1: "happiness",
    2: "sadness",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "anger",
    7: "contempt",  # 不在 harmonized 7 情緒中，但仍保存到 raw
}

# Gaze 輸出：2D（yaw, pitch）
OPENFACE_GAZE_COLUMNS: List[str] = ["gaze_yaw", "gaze_pitch"]

# =============================================================================
# Py-Feat 欄位對映
# =============================================================================

# Py-Feat Detector.detect_image() 輸出 20 AU（probability [0,1]）
PYFEAT_AU_MAP: Dict[str, str] = {
    "AU01": "AU1",
    "AU02": "AU2",
    "AU04": "AU4",
    "AU05": "AU5",
    "AU06": "AU6",
    "AU07": "AU7",
    "AU09": "AU9",
    "AU10": "AU10",
    "AU11": "AU11",
    "AU12": "AU12",
    "AU14": "AU14",
    "AU15": "AU15",
    "AU17": "AU17",
    "AU20": "AU20",
    "AU23": "AU23",
    "AU24": "AU24",
    "AU25": "AU25",
    "AU26": "AU26",
    "AU28": "AU28",
    "AU43": "AU43",
}

PYFEAT_EMOTION_MAP: Dict[str, str] = IDENTITY_EMOTION_MAP

# =============================================================================
# LibreFace 欄位對映
# =============================================================================

LIBREFACE_EMOTION_MAP: Dict[str, str] = IDENTITY_EMOTION_MAP

# =============================================================================
# POSTER++ (POSTER V2) 欄位對映
# =============================================================================

# POSTER++ 只輸出 7-class emotion probability，無 AU
# RAF-DB ImageFolder 按資料夾名（1-7）字母排序：
#   index 0 → label 1 (surprise)
#   index 1 → label 2 (fear)
#   index 2 → label 3 (disgust)
#   index 3 → label 4 (happiness)
#   index 4 → label 5 (sadness)
#   index 5 → label 6 (anger)
#   index 6 → label 7 (neutral)
POSTER_PP_EMOTION_INDEX: Dict[int, str] = {
    0: "surprise",
    1: "fear",
    2: "disgust",
    3: "happiness",
    4: "sadness",
    5: "anger",
    6: "neutral",
}

# =============================================================================
# EmoNet 欄位對映
# =============================================================================

# EmoNet 輸出 8-class emotion（或 5-class）
# 8-class order: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
EMONET_EMOTION_INDEX: Dict[int, str] = {
    0: "neutral",
    1: "happiness",
    2: "sadness",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "anger",
    7: "contempt",  # 不在 harmonized 7 情緒中
}

# =============================================================================
# 路徑配置
# =============================================================================

AU_HARMONIZED_DIR = EMO_AU_FEATURES_DIR / "harmonized"
AU_AGGREGATED_DIR = EMO_AU_FEATURES_DIR / "aggregated"

# 權重目錄
WEIGHTS_DIR = EXTERNAL_DIR / "emotion"
OPENFACE_WEIGHTS_DIR = WEIGHTS_DIR / "openface"
LIBREFACE_WEIGHTS_DIR = WEIGHTS_DIR / "libreface"
POSTER_PP_WEIGHTS_DIR = WEIGHTS_DIR / "poster_pp"
DAN_DIR = WEIGHTS_DIR / "DAN"
DAN_WEIGHTS_DIR = WEIGHTS_DIR / "dan_weights"
EMONET_DIR = WEIGHTS_DIR / "emonet"
EMONET_WEIGHTS_DIR = EMONET_DIR / "pretrained"
