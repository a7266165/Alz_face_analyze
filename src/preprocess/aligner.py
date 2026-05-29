"""
臉部對齊（pure functions，純旋轉）

依中軸線傾斜角把臉轉正使其垂直。去背（凸包遮罩）見 masker.py。
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def calculate_midline_tilt(landmarks: np.ndarray,
                           midline_points: Tuple[int, ...] = (10, 168, 4, 2)) -> float:
    """中軸線相對垂直線的傾斜角（度）；正值=向右傾，負值=向左傾。"""
    angles = []
    for i in range(len(midline_points) - 1):
        x1, y1 = landmarks[midline_points[i]]
        x2, y2 = landmarks[midline_points[i + 1]]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 1e-8:
            angles.append(90.0 if dx > 0 else -90.0)
        else:
            angles.append(np.degrees(np.arctan(dx / dy)))
    return np.mean(angles) if angles else 0.0


def rotate_to_vertical(image: np.ndarray, tilt: float) -> np.ndarray:
    """依傾斜角把影像轉正（繞影像中心旋轉 -tilt）。"""
    aligned = _rotate(image, -tilt)
    logger.debug(f"對齊完成，傾斜角度: {tilt:.2f}°")
    return aligned


def _rotate(image: np.ndarray, angle: float,
            center: Tuple[float, float] = None) -> np.ndarray:
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))
