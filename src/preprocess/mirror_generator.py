"""
產生左右臉鏡射。

兩種方法：
- midline: 沿臉部中線鏡射
- flip:    水平翻轉
"""

from typing import Tuple

import cv2
import numpy as np

from src.config import MirrorConfig


def generate_mirrors(image: np.ndarray, landmarks: np.ndarray,
                     cfg: MirrorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """產生 (左臉鏡射, 右臉鏡射)。"""
    if cfg.mirror_method == "flip":
        return _flip_mirrors(image, cfg.mirror_size, cfg.margin)
    return _midline_mirrors(image, landmarks, cfg)


def _flip_mirrors(image: np.ndarray, mirror_size: Tuple[int, int],
                  margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """原圖縮放置中 + 水平翻轉。"""
    original = _resize_and_center(image, mirror_size, margin)
    flipped = cv2.flip(original, 1)
    return original, flipped


def _midline_mirrors(image: np.ndarray, landmarks: np.ndarray,
                     cfg: MirrorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """以 PCA 估臉部中線，沿中線精確鏡射半臉。"""
    p0, n = _estimate_midline(landmarks, cfg.midline_points)
    left = _align_to_canvas_premul(image, p0, n, "left",
                                   cfg.mirror_size, cfg.feather_px, cfg.margin)
    right = _align_to_canvas_premul(image, p0, n, "right",
                                    cfg.mirror_size, cfg.feather_px, cfg.margin)
    return left, right


def _estimate_midline(face_points: np.ndarray,
                      midline_indices: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """以 PCA 估臉部中線，回傳 (中線上的點 p0, 法向量 n)。"""
    idx = np.array(midline_indices, dtype=int)
    idx = idx[(idx >= 0) & (idx < face_points.shape[0])]

    if idx.size == 0:
        ml_pts = face_points
    else:
        ml_pts = face_points[idx, :]

    p0 = ml_pts.mean(axis=0)
    X = ml_pts - p0

    # 退化情況：用左右極值取中線
    if not np.isfinite(X).all() or np.allclose(X, 0):
        xs = face_points[:, 0]
        mid_x = 0.5 * (xs.min() + xs.max())
        p0 = np.array([mid_x, face_points[:, 1].mean()], dtype=np.float64)
        n = np.array([1.0, 0.0], dtype=np.float64)
        return p0, n

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]
    u = u / (np.linalg.norm(u) + 1e-12)

    # 法向量（垂直於主方向），確保指向右側
    n = np.array([-u[1], u[0]], dtype=np.float64)
    if n[0] < 0:
        n = -n
    return p0, n


def _align_to_canvas_premul(img_bgr: np.ndarray, p0: np.ndarray, n: np.ndarray,
                            side: str, out_size: Tuple[int, int],
                            feather_px: int, margin: float) -> np.ndarray:
    """沿中線鏡射並置中到畫布（預乘 alpha 合成）。"""
    h, w = img_bgr.shape[:2]

    # 每個像素到中線的有號距離
    X, Y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    d = (X - p0[0]) * n[0] + (Y - p0[1]) * n[1]

    # 反射座標
    Xr = X - 2.0 * d * n[0]
    Yr = Y - 2.0 * d * n[1]

    # 半臉遮罩（基於中線）
    if side == "left":
        half_mask = (d < 0).astype(np.uint8) * 255
    else:
        half_mask = (d > 0).astype(np.uint8) * 255

    # 羽化邊緣
    if feather_px > 0:
        kernel_size = feather_px * 2 + 1
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
        0,
    )

    result = np.clip(result_f * 255, 0, 255).astype(np.uint8)
    return _resize_and_center(result, out_size, margin)


def _resize_and_center(image: np.ndarray, out_size: Tuple[int, int],
                       margin: float) -> np.ndarray:
    """縮放影像並置中到指定大小的畫布（保留邊緣留白）。"""
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
    cropped = image[y0:y1 + 1, x0:x1 + 1]
    face_h, face_w = cropped.shape[:2]

    # 縮放比例（保留邊緣空間）
    available_w = W * (1 - 2 * margin)
    available_h = H * (1 - 2 * margin)
    scale = min(available_w / face_w, available_h / face_h, 1.0)

    new_w = int(face_w * scale)
    new_h = int(face_h * scale)
    if new_w <= 0 or new_h <= 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 置中到畫布
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    start_x = (W - new_w) // 2
    start_y = (H - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    return canvas
