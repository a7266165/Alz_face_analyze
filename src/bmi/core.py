"""BMI 迴歸的共用核心：評估指標 + demographics→(ids, bmi, base_ids) 載入。

三條建模路線（ArcFace / 影像 CNN / MeFEm）共用本檔。GroupKFold 一律以 base_id（人）
分組，防同一人不同 visit 跨折 leakage。
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """回 MAE / RMSE / R² / Pearson r（含 p）/ n。"""
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r, p = stats.pearsonr(y_true, y_pred)
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "n": len(y_true),
    }


def load_bmi_subjects() -> Tuple[List[str], np.ndarray, List[str]]:
    """所有有 BMI 的 visit → (ids, bmi, base_ids)。ids 為完整鍵、base_ids 為人鍵。"""
    from src.common.cohort import load_demographics
    demo = load_demographics()
    if "BMI" not in demo.columns:
        raise RuntimeError("demographics 無 BMI 欄")
    demo = demo[["ID", "base_id", "BMI"]].dropna(subset=["BMI"])
    return demo["ID"].tolist(), demo["BMI"].to_numpy(dtype=np.float64), demo["base_id"].tolist()


def encode_groups(base_ids: List[str]) -> np.ndarray:
    """base_id 串 → GroupKFold 用的整數群組（依排序穩定編碼）。"""
    mapping = {b: i for i, b in enumerate(sorted(set(base_ids)))}
    return np.array([mapping[b] for b in base_ids])
