"""
鏡射生成器

負責生成左右臉鏡射影像
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MirrorGenerator:
    """
    鏡射生成器

    支援兩種鏡射方法：
    - midline: 沿臉部中線鏡射（較精確）
    - flip: 簡單水平翻轉（較快速）
    """

    def __init__(
        self,
        method: str = "midline",
        mirror_size: Tuple[int, int] = (512, 512),
        feather_px: int = 2,
        margin: float = 0.08,
        midline_points: Tuple[int, ...] = (10, 168, 4, 2),
    ):
        """
        初始化鏡射生成器

        Args:
            method: 鏡射方法 ("midline" 或 "flip")
            mirror_size: 輸出影像大小 (H, W)
            feather_px: 邊緣羽化像素
            margin: 畫布邊緣留白比例
            midline_points: 臉部中軸線特徵點索引
        """
        self.method = method
        self.mirror_size = mirror_size
        self.feather_px = feather_px
        self.margin = margin
        self.midline_points = midline_points

    def generate(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成左右臉鏡射

        Args:
            image: 輸入影像（已套用遮罩）
            landmarks: 468 個特徵點

        Returns:
            (左臉鏡射, 右臉鏡射)
        """
        if self.method == "flip":
            return self._create_flip_mirrors(image, landmarks)
        else:
            return self._create_midline_mirrors(image, landmarks)

    def _create_flip_mirrors(
        self,
        image: np.ndarray,
        _landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        簡單水平翻轉

        Args:
            image: 輸入影像
            _landmarks: 未使用，保留以維持介面一致

        Returns:
            (原圖, 水平翻轉圖)
        """
        H, W = self.mirror_size

        # 原圖縮放置中
        original = self._resize_and_center(image, (H, W))

        # 水平翻轉
        flipped = cv2.flip(original, 1)

        return original, flipped

    def _create_midline_mirrors(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        以臉部中線鏡射半臉

        使用 PCA 估計臉部中線，然後沿中線進行精確的鏡射

        Args:
            image: 輸入影像
            landmarks: 468 個特徵點

        Returns:
            (左臉鏡射, 右臉鏡射)
        """
        p0, n = self._estimate_midline(landmarks)

        left_mirror = self._align_to_canvas_premul(
            image, p0, n,
            side="left",
            out_size=self.mirror_size,
        )

        right_mirror = self._align_to_canvas_premul(
            image, p0, n,
            side="right",
            out_size=self.mirror_size,
        )

        return left_mirror, right_mirror

    def _estimate_midline(
        self,
        face_points: np.ndarray,
        midline_indices: Optional[Tuple[int, ...]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 PCA 估計臉部中線

        Args:
            face_points: 臉部特徵點 (468, 2)
            midline_indices: 中線相關的特徵點索引

        Returns:
            (中線上的點 p0, 法向量 n)
        """
        if midline_indices is None:
            midline_indices = self.midline_points

        # 提取中線特徵點
        idx = np.array(midline_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < face_points.shape[0])]

        if idx.size == 0:
            ml_pts = face_points
        else:
            ml_pts = face_points[idx, :]

        # PCA 找主方向
        p0 = ml_pts.mean(axis=0)
        X = ml_pts - p0

        # 處理退化情況
        if not np.isfinite(X).all() or np.allclose(X, 0):
            xs = face_points[:, 0]
            mid_x = 0.5 * (xs.min() + xs.max())
            p0 = np.array([mid_x, face_points[:, 1].mean()], dtype=np.float64)
            n = np.array([1.0, 0.0], dtype=np.float64)
            return p0, n

        # SVD 分解
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        u = Vt[0]
        u = u / (np.linalg.norm(u) + 1e-12)

        # 法向量（垂直於主方向）
        n = np.array([-u[1], u[0]], dtype=np.float64)

        # 確保 n 指向右側
        if n[0] < 0:
            n = -n

        return p0, n

    def _align_to_canvas_premul(
        self,
        img_bgr: np.ndarray,
        p0: np.ndarray,
        n: np.ndarray,
        side: str,
        out_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        沿中線鏡射並置中到畫布

        Args:
            img_bgr: 輸入影像（已套用遮罩，背景為黑色）
            p0: 中線上的點
            n: 法向量
            side: 'left' 或 'right'
            out_size: 輸出尺寸 (H, W)

        Returns:
            鏡射影像
        """
        H, W = out_size
        h, w = img_bgr.shape[:2]

        # 計算每個像素到中線的有號距離
        X, Y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32)
        )
        d = (X - p0[0]) * n[0] + (Y - p0[1]) * n[1]

        # 計算反射座標
        Xr = X - 2.0 * d * n[0]
        Yr = Y - 2.0 * d * n[1]

        # 建立半臉遮罩（基於中線）
        if side == "left":
            half_mask = (d < 0).astype(np.uint8) * 255
        else:
            half_mask = (d > 0).astype(np.uint8) * 255

        # 羽化邊緣
        if self.feather_px > 0:
            kernel_size = self.feather_px * 2 + 1
            half_mask = cv2.GaussianBlur(half_mask, (kernel_size, kernel_size), 0)

        alpha_f = half_mask.astype(np.float32) / 255.0

        # 反射另一半
        reflected = cv2.remap(img_bgr, Xr, Yr, cv2.INTER_LINEAR)
        reflected_alpha = cv2.remap(alpha_f, Xr, Yr, cv2.INTER_LINEAR)

        # 預乘 alpha 合成
        img_f = img_bgr.astype(np.float32) / 255.0
        result_f = (
            img_f * alpha_f[..., None]
            + (reflected.astype(np.float32) / 255.0) * reflected_alpha[..., None]
        )

        final_alpha = np.clip(alpha_f + reflected_alpha, 0, 1)

        # 除以 alpha 還原顏色
        eps = 1e-6
        result_f = np.where(
            final_alpha[..., None] > eps,
            result_f / final_alpha[..., None],
            0
        )

        result = np.clip(result_f * 255, 0, 255).astype(np.uint8)

        # 縮放並置中
        return self._resize_and_center(result, out_size)

    def _resize_and_center(
        self,
        image: np.ndarray,
        out_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        縮放影像並置中到指定大小的畫布

        Args:
            image: 輸入影像
            out_size: 輸出尺寸 (H, W)

        Returns:
            置中後的影像
        """
        H, W = out_size
        h, w = image.shape[:2]

        if h == 0 or w == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        # 找出非黑色區域
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray > 0)

        if len(xs) == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # 裁切非黑色區域
        cropped = image[y0:y1+1, x0:x1+1]
        face_h, face_w = cropped.shape[:2]

        # 計算縮放比例（保留邊緣空間）
        available_w = W * (1 - 2 * self.margin)
        available_h = H * (1 - 2 * self.margin)
        scale = min(available_w / face_w, available_h / face_h, 1.0)

        # 縮放
        new_w = int(face_w * scale)
        new_h = int(face_h * scale)

        if new_w <= 0 or new_h <= 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 置中到畫布
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        start_x = (W - new_w) // 2
        start_y = (H - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

        return canvas
