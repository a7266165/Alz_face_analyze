"""
年齡模組共用工具

load_predicted_ages: 載入預測年齡 JSON，回傳 {ID: mean_age}（僅依賴標準函式庫）。
load_age_error:      指定 cohort 的逐受試者年齡誤差 DataFrame[ID, age_error]
                     （延遲載入 pandas / cohort，import 本模組本身仍只依賴標準函式庫）。
"""

import json
from pathlib import Path


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


def load_age_error(p_visit, p_score, hc_visit, hc_score, *,
                   predictions_file=None):
    """指定 cohort 的逐受試者年齡誤差，回 DataFrame[ID, age_error]。

    age_error = real_age − predicted_age。real_age 取自 cohort（cohort_list 的
    Age，已轉數值），predicted_age 取自 predictions_file（預設 PREDICTED_AGES_FILE）。
    無預測值的受試者已 dropna。其餘 metadata（group / 分數等）請消費端自行從
    cohort_list 取，再以 ID join。
    """
    import pandas as pd

    from src.config import PREDICTED_AGES_FILE
    from src.common.cohort import cohort_list

    if predictions_file is None:
        predictions_file = PREDICTED_AGES_FILE
    preds = load_predicted_ages(predictions_file)
    df = cohort_list(p_visit, p_score, hc_visit, hc_score)
    real_age = pd.to_numeric(df["Age"], errors="coerce")
    predicted_age = df["ID"].map(preds)
    out = pd.DataFrame({"ID": df["ID"], "age_error": real_age - predicted_age})
    return out.dropna(subset=["age_error"]).reset_index(drop=True)
