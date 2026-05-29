"""
臉部對齊器（pure station，純旋轉）

依中軸線傾斜角把臉轉正使其垂直。
去背（凸包遮罩）已拆到 masker.FaceMasker。

純函式：影像 / landmarks 進 → 角度 / 影像 出，不碰路徑與 I/O。
"""

import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FaceStraightener:
    """旋轉影像使臉部中線垂直。"""

    def __init__(self, midline_points: Tuple[int, ...] = (10, 168, 4, 2)):
        """
        Args:
            midline_points: 臉部中軸線特徵點索引
        """
        self.midline_points = midline_points

    def calculate_midline_tilt(self, landmarks: np.ndarray) -> float:
        """
        計算臉部中軸線相對於垂直線的傾斜角度

        Args:
            landmarks: 468 個特徵點座標 (468, 2)

        Returns:
            角度（度）- 正值表示向右傾斜，負值表示向左傾斜
        """
        angles = []

        for i in range(len(self.midline_points) - 1):
            x1, y1 = landmarks[self.midline_points[i]]
            x2, y2 = landmarks[self.midline_points[i + 1]]

            dx = x2 - x1
            dy = y2 - y1

            if abs(dy) < 1e-8:
                angles.append(90.0 if dx > 0 else -90.0)
            else:
                angles.append(np.degrees(np.arctan(dx / dy)))

        return np.mean(angles) if angles else 0.0

    def rotate_to_vertical(self, image: np.ndarray, tilt: float) -> np.ndarray:
        """
        依中線傾斜角把影像轉正（繞影像中心旋轉 -tilt）

        Args:
            image: 輸入影像（已去背或保留背景皆可）
            tilt: calculate_midline_tilt 算出的傾斜角（度）

        Returns:
            轉正後的影像
        """
        aligned = self.rotate_image(image, -tilt)
        logger.debug(f"對齊完成，傾斜角度: {tilt:.2f}°")
        return aligned

    def rotate_image(
        self,
        image: np.ndarray,
        angle: float,
        center: Tuple[float, float] = None,
    ) -> np.ndarray:
        """
        旋轉影像

        Args:
            image: 輸入影像
            angle: 旋轉角度（度）
            center: 旋轉中心，預設為影像中心

        Returns:
            旋轉後的影像
        """
        h, w = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
