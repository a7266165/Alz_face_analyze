"""頭部旋轉統計特徵：從角度序列 (pitch/yaw/roll) 抽每軸統計量供下游分類。"""

from typing import Dict

import numpy as np

from .angle_calc import SequenceResult

_STATS = ("mean", "std", "range", "min", "max", "median", "iqr")


def extract_rotation_features(result: SequenceResult) -> Dict[str, float]:
    """每軸抽 mean/std/range/min/max/median/iqr，附 n_valid_frames 與 method。

    空序列的該軸統計量填 NaN。
    """
    features: Dict[str, float] = {}

    for axis, values in (("pitch", result.pitch_list),
                         ("yaw", result.yaw_list),
                         ("roll", result.roll_list)):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            features.update({f"{axis}_{s}": np.nan for s in _STATS})
            continue
        features[f"{axis}_mean"] = float(np.mean(arr))
        features[f"{axis}_std"] = float(np.std(arr))
        features[f"{axis}_range"] = float(np.ptp(arr))
        features[f"{axis}_min"] = float(np.min(arr))
        features[f"{axis}_max"] = float(np.max(arr))
        features[f"{axis}_median"] = float(np.median(arr))
        features[f"{axis}_iqr"] = float(np.percentile(arr, 75) - np.percentile(arr, 25))

    features["n_valid_frames"] = result.length
    features["method"] = result.method
    return features
