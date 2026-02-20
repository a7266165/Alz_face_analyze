"""
臉部對齊器

負責旋轉校正臉部使其垂直，並建立臉部遮罩
"""

import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FaceAligner:
    """
    臉部對齊器

    套用遮罩並旋轉影像使臉部中線垂直
    """

    def __init__(self, midline_points: Tuple[int, ...] = (10, 168, 4, 2)):
        """
        初始化對齊器

        Args:
            midline_points: 臉部中軸線特徵點索引
        """
        self.midline_points = midline_points

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        對齊臉部

        1. 建立臉部遮罩
        2. 計算中線傾斜角度
        3. 旋轉影像使中線垂直

        Args:
            image: 輸入影像 (BGR)
            landmarks: 468 個特徵點座標 (468, 2)

        Returns:
            對齊後的影像
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # Step 1: 建立並套用遮罩
        mask = self.build_face_mask(image.shape, landmarks)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Step 2: 計算傾斜角度
        tilt = self.calculate_midline_tilt(landmarks)

        # Step 3: 旋轉影像
        M = cv2.getRotationMatrix2D(center, -tilt, 1.0)
        aligned_image = cv2.warpAffine(masked_image, M, (w, h))

        logger.debug(f"對齊完成，傾斜角度: {tilt:.2f}°")
        return aligned_image

    def build_face_mask(
        self,
        img_shape: Tuple[int, int, ...],
        face_points: np.ndarray
    ) -> np.ndarray:
        """
        建立臉部凸包遮罩

        Args:
            img_shape: 影像尺寸 (H, W, C)
            face_points: 臉部特徵點 (N, 2)

        Returns:
            二值遮罩 (H, W)
        """
        mask = np.zeros(img_shape[:2], dtype=np.uint8)

        if face_points.shape[0] == 0:
            return mask

        # 計算凸包
        hull = cv2.convexHull(face_points.astype(np.int32))

        # 填充凸包區域
        cv2.fillConvexPoly(mask, hull, 255)

        return mask

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

    def rotate_image(
        self,
        image: np.ndarray,
        angle: float,
        center: Tuple[float, float] = None
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
