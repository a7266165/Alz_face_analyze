"""整臉幾何不對稱分析器。

以 MediaPipe Face Mesh 468 點的左右對應配對，算四種不對稱指標：點 X 差、
點 Y 差、線段長差、三角形面積差。吃預先算好的 landmarks，不自行初始化
FaceMesh。左右配對索引的單一真相在 src/common/mediapipe_utils.py。
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.mediapipe_utils import (
    LEFT_FACE_INDICES,
    LEFT_LINE_INDICES,
    LEFT_TRIANGLE_INDICES,
    RIGHT_FACE_INDICES,
    RIGHT_LINE_INDICES,
    RIGHT_TRIANGLE_INDICES,
)

logger = logging.getLogger(__name__)


class LandmarkAsymmetryAnalyzer:
    """整臉幾何不對稱分數器：點 X/Y 差、線段長差、三角形面積差。"""

    def __init__(self):
        self.point_pairs = list(zip(RIGHT_FACE_INDICES, LEFT_FACE_INDICES))
        self.line_pairs = list(zip(RIGHT_LINE_INDICES, LEFT_LINE_INDICES))
        self.triangle_pairs = list(zip(RIGHT_TRIANGLE_INDICES, LEFT_TRIANGLE_INDICES))

    @staticmethod
    def normalize_landmarks(
        landmarks: np.ndarray,
        target_width: float = 500.0,
    ) -> np.ndarray:
        """以點 234 的 x、點 10 的 y 為原點，依點 454 的 x 縮放至 target_width。

        與 extractor.normalize_landmarks（bbox min/max）為不同基準、不可互換：
        calculate_asymmetry 的點差以半寬（target_width/2）為對稱軸，與此綁定。
        """
        landmarks = landmarks.copy().astype(float)
        origin_x = landmarks[234, 0]
        origin_y = landmarks[10, 1]
        landmarks[:, 0] -= origin_x
        landmarks[:, 1] -= origin_y

        right_x = landmarks[454, 0]
        if right_x > 0:
            landmarks *= target_width / right_x

        return landmarks

    @staticmethod
    def _line_len(
        x: Dict[int, float], y: Dict[int, float], idxs: Tuple[int, int]
    ) -> float:
        """線段長度。"""
        i, j = idxs
        return np.hypot(x[i] - x[j], y[i] - y[j])

    @staticmethod
    def _tri_area(
        x: Dict[int, float], y: Dict[int, float], idxs: Tuple[int, int, int]
    ) -> float:
        """三角形面積（叉積公式）。"""
        i, j, k = idxs
        return abs((x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j])) / 2)

    def calculate_asymmetry(
        self,
        landmarks: np.ndarray,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """單筆 landmarks 的不對稱指標。

        Args:
            landmarks: (468, 2) 或 (468, 3)。
            normalize: 是否先 normalize_landmarks（對稱軸固定為半寬 250）。

        Returns:
            sum_point_x_diff / sum_point_y_diff / sum_line_diff /
            sum_triangle_area_diff，及四者和 total_asymmetry。
        """
        if normalize:
            landmarks = self.normalize_landmarks(landmarks)

        x = {i: landmarks[i, 0] for i in range(468)}
        y = {i: landmarks[i, 1] for i in range(468)}

        baseline_x = abs(x[234] - x[454])
        baseline_y = abs(y[10] - y[152])

        total_pt_x = 0.0
        for right_idx, left_idx in self.point_pairs:
            diff = abs(abs(x[left_idx] - 250) - abs(x[right_idx] - 250))
            if baseline_x > 0:
                diff /= baseline_x
            total_pt_x += diff

        total_pt_y = 0.0
        for right_idx, left_idx in self.point_pairs:
            diff = abs(y[left_idx] - y[right_idx])
            if baseline_y > 0:
                diff /= baseline_y
            total_pt_y += diff

        total_line = 0.0
        for right_line, left_line in self.line_pairs:
            ld = self._line_len(x, y, left_line)
            rd = self._line_len(x, y, right_line)
            if (ld + rd) > 0:
                total_line += abs(ld - rd) / (ld + rd)

        total_tri = 0.0
        for right_tri, left_tri in self.triangle_pairs:
            la = self._tri_area(x, y, left_tri)
            ra = self._tri_area(x, y, right_tri)
            if (la + ra) > 0:
                total_tri += abs(la - ra) / (la + ra)

        return {
            "sum_point_x_diff": total_pt_x,
            "sum_point_y_diff": total_pt_y,
            "sum_line_diff": total_line,
            "sum_triangle_area_diff": total_tri,
            "total_asymmetry": total_pt_x + total_pt_y + total_line + total_tri,
        }

    def calculate_batch(
        self,
        landmarks_list: List[np.ndarray],
        names: List[str] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """批次版 calculate_asymmetry，每列一筆；給 names 時附 name 欄。"""
        results = []
        for i, lm in enumerate(landmarks_list):
            metrics = self.calculate_asymmetry(lm, normalize=normalize)
            if names:
                metrics["name"] = names[i]
            results.append(metrics)
        return pd.DataFrame(results)
