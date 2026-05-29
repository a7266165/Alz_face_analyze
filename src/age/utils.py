"""
年齡模組共用工具

load_predicted_ages: 載入預測年齡 JSON，回傳 {ID: mean_age}。
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
