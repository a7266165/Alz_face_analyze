"""
年齡模組共用工具

load_predicted_ages:         載入預測年齡 JSON，回傳 {ID: mean_age}（函數本身僅用標準函式庫）。
calculate_age_error:         年齡預測誤差 = 真實 − 預測（逐元素，釘正負號慣例）。
build_cohort_with_age_error: cohort 表外掛年齡預測欄（real_age/predicted_age/age_error/group）的寬表。

依賴（pandas / config / cohort）一律於模組開頭 import；import 本模組會連帶拉進
pandas + cohort，但不涉及 cv2/torch，meta 等輕環境仍可直接 import。
"""

import json
from pathlib import Path
import pandas as pd

from src.config import PREDICTED_AGES_FILE
from src.common.cohort import cohort_list

def load_predicted_ages(path: Path) -> dict:
    """載入預測年齡 JSON，回傳 {ID: mean_age}。

    支援格式：
      {id: {"predicted_ages": [...]}}   ← predict.py 輸出
      {id: [floats]}
      {id: float}

    無有效預測值的 ID 會被略過（不再以 0.0 代入，以免污染下游統計）。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            ages = v.get("predicted_ages") or []
        elif isinstance(v, list):
            ages = v
        else:
            out[k] = float(v)
            continue
        if ages:
            out[k] = sum(ages) / len(ages)
    return out


def calculate_age_error(real_age, predicted_age):
    """年齡預測誤差 = 真實年齡 − 預測年齡（正值 = 預測偏年輕）。逐元素，吃 Series/array/scalar。"""
    return real_age - predicted_age


def build_cohort_with_age_error(p_visit, p_score, hc_visit, hc_score, *,
                                predictions_file=None):
    """指定 cohort 的 cohort 表外掛年齡預測欄，供 error 繪圖/統計腳本共用。

    在 cohort（cohort_list：真實 Age + metadata）上對齊預測值（load_predicted_ages）
    並經 calculate_age_error 算出誤差；無預測值（或 Age 非數值）的受試者已 dropna。
    下游各取所需欄位。

    Returns:
        DataFrame：除 cohort_list 既有欄（ID/Group/Age/MMSE/CASI/Global_CDR …）外，
        另含 group=Group、real_age、predicted_age、age_error。
    """
    preds = load_predicted_ages(predictions_file or PREDICTED_AGES_FILE)
    df = cohort_list(p_visit, p_score, hc_visit, hc_score).copy()
    df["group"] = df["Group"]
    df["real_age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["predicted_age"] = df["ID"].map(preds)
    df["age_error"] = calculate_age_error(df["real_age"], df["predicted_age"])
    return df.dropna(subset=["age_error"]).reset_index(drop=True)
