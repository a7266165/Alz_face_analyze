"""
頭部旋轉統計特徵提取

從角度序列 (pitch/yaw/roll) 提取統計特徵，用於下游分類分析。
"""

from typing import Dict

import numpy as np

from .angle_calc import SequenceResult


def extract_rotation_features(result: SequenceResult) -> Dict[str, float]:
    """
    從角度序列提取統計特徵。

    Args:
        result: SequenceResult，包含 pitch/yaw/roll 列表

    Returns:
        dict，包含每個軸的 mean, std, range, min, max, median, iqr
    """
    features = {}

    for axis_name, values in [
        ("pitch", result.pitch_list),
        ("yaw", result.yaw_list),
        ("roll", result.roll_list),
    ]:
        arr = np.array(values)
        if len(arr) == 0:
            for stat in ["mean", "std", "range", "min", "max", "median", "iqr"]:
                features[f"{axis_name}_{stat}"] = np.nan
            continue

        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)

        features[f"{axis_name}_mean"] = float(np.mean(arr))
        features[f"{axis_name}_std"] = float(np.std(arr))
        features[f"{axis_name}_range"] = float(np.ptp(arr))
        features[f"{axis_name}_min"] = float(np.min(arr))
        features[f"{axis_name}_max"] = float(np.max(arr))
        features[f"{axis_name}_median"] = float(np.median(arr))
        features[f"{axis_name}_iqr"] = float(q3 - q1)

    features["n_valid_frames"] = result.length
    features["method"] = result.method

    return features
