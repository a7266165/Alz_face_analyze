"""
臉部去背遮罩（pure station）

以 468 landmarks 的凸包建立遮罩，把臉部以外區域塗黑。
原本內嵌在 aligner.FaceStraightener，現拆成獨立一站。

純函式：影像 + landmarks 進 → 影像 / 遮罩 出，不碰路徑與 I/O。
"""

import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FaceMasker:
    """以 landmarks 凸包去背。"""

    def build_mask(
        self,
        img_shape: Tuple[int, int, ...],
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """
        建立臉部凸包遮罩

        Args:
            img_shape: 影像尺寸 (H, W, C)
            landmarks: 臉部特徵點 (N, 2)

        Returns:
            二值遮罩 (H, W)
        """
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        if landmarks.shape[0] == 0:
            return mask
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def apply(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        去背：臉部凸包以外塗黑

        Args:
            image: 輸入影像 (BGR)
            landmarks: 468 個特徵點 (468, 2)

        Returns:
            去背後影像（臉外為黑）
        """
        mask = self.build_mask(image.shape, landmarks)
        return cv2.bitwise_and(image, image, mask=mask)
