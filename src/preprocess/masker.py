"""
臉部去背遮罩（pure functions）

以 468/478 landmarks 的凸包建立遮罩，把臉部以外區域塗黑。
"""

from typing import Tuple

import cv2
import numpy as np


def _build_mask(img_shape: Tuple[int, ...], landmarks: np.ndarray) -> np.ndarray:
    """臉部凸包二值遮罩 (H, W)；landmarks 為空時回傳全黑。"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if landmarks.shape[0] == 0:
        return mask
    hull = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def apply_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """去背：臉部凸包以外塗黑。"""
    mask = _build_mask(image.shape, landmarks)
    return cv2.bitwise_and(image, image, mask=mask)
