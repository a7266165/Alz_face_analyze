"""
面部不對稱性分析器

基於 MediaPipe Face Mesh 468 點的雙側 landmark 計算不對稱性指標：
- 點 X/Y 差值（對稱性差異）
- 線段長度差值
- 三角形面積差值

接受預計算的 normalized landmarks 作為輸入，不自行初始化 FaceMesh。
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.core.mediapipe_utils import (
    FACEMESH_MID_LINE,
    LEFT_FACE_INDICES,
    LEFT_LINE_INDICES,
    LEFT_TRIANGLE_INDICES,
    MIDLINE_POINTS,
    RIGHT_FACE_INDICES,
    RIGHT_LINE_INDICES,
    RIGHT_TRIANGLE_INDICES,
)

logger = logging.getLogger(__name__)


class LandmarkAsymmetryAnalyzer:
    """
    面部不對稱性分析器

    使用 MediaPipe 468 個 landmark 的左右對稱配對，
    計算四種不對稱指標：點X差、點Y差、線段長差、三角形面積差。
    """

    def __init__(self):
        self.point_pairs = list(zip(RIGHT_FACE_INDICES, LEFT_FACE_INDICES))
        self.line_pairs = list(zip(RIGHT_LINE_INDICES, LEFT_LINE_INDICES))
        self.triangle_pairs = list(zip(RIGHT_TRIANGLE_INDICES, LEFT_TRIANGLE_INDICES))

    @staticmethod
    def mid_line_angle_4_points(
        landmarks: np.ndarray,
        midline_points: Tuple[int, ...] = MIDLINE_POINTS,
    ) -> float:
        """
        計算 4 點中軸角度總和（用於選擇最正面的臉部影像）。

        Args:
            landmarks: shape (468, 2) or (468, 3), 像素座標
            midline_points: 4 個中軸線 landmark index

        Returns:
            角度總和（度），越小越正面
        """
        pts = [landmarks[idx][:2] for idx in midline_points]
        dot1 = np.array([pts[0][0], pts[0][1], 0])
        dot2 = np.array([pts[1][0], pts[1][1], 0])
        dot3 = np.array([pts[2][0], pts[2][1], 0])
        dot4 = np.array([pts[3][0], pts[3][1], 0])

        v1 = dot2 - dot1
        v2 = dot3 - dot2
        v3 = dot4 - dot3

        norm1 = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle1 = np.arccos(np.clip(np.dot(v1, v2) / norm1, -1, 1)) if norm1 > 0 else 0

        norm2 = np.linalg.norm(v2) * np.linalg.norm(v3)
        angle2 = np.arccos(np.clip(np.dot(v2, v3) / norm2, -1, 1)) if norm2 > 0 else 0

        return np.degrees(angle1) + np.degrees(angle2)

    @staticmethod
    def mid_line_angle_all_points(
        landmarks: np.ndarray,
    ) -> float:
        """
        計算完整中軸線的平均傾斜角度。

        Args:
            landmarks: shape (468, 2) or (468, 3), 像素座標

        Returns:
            平均角度（度）
        """
        angles = []
        for idx1, idx2 in FACEMESH_MID_LINE:
            pt1 = landmarks[idx1][:2]
            pt2 = landmarks[idx2][:2]
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            if dy == 0:
                angles.append(90.0)
            else:
                angles.append(np.degrees(np.arctan(dx / dy)))
        return np.mean(angles) if angles else 0.0

    @staticmethod
    def normalize_landmarks(
        landmarks: np.ndarray,
        target_width: float = 500.0,
    ) -> np.ndarray:
        """
        正規化 landmarks：以左臉頰(234)為原點，固定寬度為 target_width。

        Args:
            landmarks: shape (468, 2) or (468, 3), 像素座標
            target_width: 正規化後的臉寬

        Returns:
            正規化後的 landmarks
        """
        landmarks = landmarks.copy().astype(float)
        # 以 landmark 234 的 x, landmark 10 的 y 為原點
        origin_x = landmarks[234, 0]
        origin_y = landmarks[10, 1]
        landmarks[:, 0] -= origin_x
        landmarks[:, 1] -= origin_y

        # 縮放到固定寬度
        right_x = landmarks[454, 0]
        if right_x > 0:
            scale = target_width / right_x
            landmarks *= scale

        return landmarks

    @staticmethod
    def _line_len(
        x: Dict[int, float], y: Dict[int, float], idxs: Tuple[int, int]
    ) -> float:
        """計算線段長度。"""
        i, j = idxs
        return np.hypot(x[i] - x[j], y[i] - y[j])

    @staticmethod
    def _tri_area(
        x: Dict[int, float], y: Dict[int, float], idxs: Tuple[int, int, int]
    ) -> float:
        """計算三角形面積（使用叉積公式）。"""
        i, j, k = idxs
        return abs((x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j])) / 2)

    def calculate_asymmetry(
        self,
        landmarks: np.ndarray,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        計算單筆 landmark 的不對稱性指標。

        Args:
            landmarks: shape (468, 2) or (468, 3)
            normalize: 是否先正規化 landmarks

        Returns:
            dict with keys:
            - sum_point_x_diff: 點 X 差值總和（正規化）
            - sum_point_y_diff: 點 Y 差值總和（正規化）
            - sum_line_diff: 線段長差值總和（正規化）
            - sum_triangle_area_diff: 三角形面積差值總和（正規化）
            - total_asymmetry: 上述四項總和
        """
        if normalize:
            landmarks = self.normalize_landmarks(landmarks)

        x = {i: landmarks[i, 0] for i in range(468)}
        y = {i: landmarks[i, 1] for i in range(468)}

        # 基線距離
        baseline_x = abs(x[234] - x[454])
        baseline_y = abs(y[10] - y[152])

        # 點對 X 差值
        total_pt_x = 0.0
        for right_idx, left_idx in self.point_pairs:
            diff = abs(abs(x[left_idx] - 250) - abs(x[right_idx] - 250))
            if baseline_x > 0:
                diff /= baseline_x
            total_pt_x += diff

        # 點對 Y 差值
        total_pt_y = 0.0
        for right_idx, left_idx in self.point_pairs:
            diff = abs(y[left_idx] - y[right_idx])
            if baseline_y > 0:
                diff /= baseline_y
            total_pt_y += diff

        # 線段長度差值
        total_line = 0.0
        for right_line, left_line in self.line_pairs:
            ld = self._line_len(x, y, left_line)
            rd = self._line_len(x, y, right_line)
            if (ld + rd) > 0:
                total_line += abs(ld - rd) / (ld + rd)

        # 三角形面積差值
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
        """
        批次計算多筆 landmarks 的不對稱性。

        Args:
            landmarks_list: list of (468, 2) or (468, 3) arrays
            names: 對應的名稱列表
            normalize: 是否正規化

        Returns:
            DataFrame with asymmetry metrics
        """
        results = []
        for i, lm in enumerate(landmarks_list):
            metrics = self.calculate_asymmetry(lm, normalize=normalize)
            if names:
                metrics["name"] = names[i]
            results.append(metrics)
        return pd.DataFrame(results)
