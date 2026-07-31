"""
選最正面的 n 張。兩種準則：

    select_most_frontal   MediaPipe 中線頂點夾角總和（VAS）升冪取前 n
    select_by_head_pose   OpenFace 2.2 逐幀頭部姿態，絕對閾值閘門 + 最小 |yaw|
"""

import logging
from typing import List, Optional, Sequence

import numpy as np

from .detector import FaceInfo

logger = logging.getLogger(__name__)


def select_most_frontal(face_infos: List[FaceInfo], n_select: int = 10) -> List[FaceInfo]:
    """按頂點夾角總和（恆非負，越小越正面）升冪取前 n 張。"""
    if not face_infos:
        return []
    sorted_faces = sorted(face_infos, key=lambda x: x.vertex_angle_sum)
    selected = sorted_faces[:min(n_select, len(sorted_faces))]
    logger.info(f"從 {len(face_infos)} 張中選擇了 {len(selected)} 張最正面的臉部")
    return selected


def _median_filter(x: np.ndarray, win: int) -> np.ndarray:
    """一維中位數濾波（邊緣用 edge padding）。win 需為奇數，<=1 時原樣回傳。"""
    if win <= 1:
        return x
    if win % 2 == 0:
        win += 1
    half = win // 2
    padded = np.pad(x, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, win), axis=-1)


def _local_minima(x: np.ndarray) -> np.ndarray:
    """回傳局部極小的位置索引（<= 兩側鄰居，且至少一側嚴格小於）。含兩端點。"""
    if x.size < 3:
        return np.arange(x.size)
    left, mid, right = x[:-2], x[1:-1], x[2:]
    is_min = ((mid <= left) & (mid < right)) | ((mid < left) & (mid <= right))
    idx = np.flatnonzero(is_min) + 1
    # 端點若比唯一的鄰居小也算
    ends = []
    if x[0] < x[1]:
        ends.append(0)
    if x[-1] < x[-2]:
        ends.append(x.size - 1)
    return np.sort(np.concatenate([idx, np.array(ends, dtype=int)])) if ends else idx


def select_by_head_pose(
    frame_index: Sequence[int],
    yaw: Sequence[float],
    pitch: Sequence[float],
    roll: Sequence[float],
    confidence: Sequence[float],
    n_select: int = 10,
    pitch_max: float = 15.0,
    roll_max: float = 10.0,
    conf_min: float = 0.725,
    medfilt_win: int = 5,
    min_gap: int = 13,
    trajectory: bool = True,
) -> List[int]:
    """依 OpenFace 頭部姿態挑幀，回傳選中的「原始幀號」（非列序）。

    角度單位一律為度（OpenFace 的 pose_R* 是弧度，呼叫端負責轉換）。

    絕對閾值閘門：confidence > conf_min 且 |pitch| <= pitch_max 且 |roll| <= roll_max。
    這是與舊 VAS 準則最大的差別 —— 舊法是相對排名，候選越多挑到的越正面，
    因此 1200 張與 30 張兩種拍攝版本得到的品質系統性不同。

    trajectory=True （1200 張版本，30 fps 連續轉頭約 44 秒）
        |yaw| 先做中位數濾波，取局部極小值當候選，再按 |yaw| 升冪貪婪選取，
        彼此至少間隔 min_gap 幀。局部極小對應「頭部轉動經過正面時的停留」。

    trajectory=False（30 張版本，約 0.6 秒的正面連拍）
        30 張近乎同一姿勢，沒有軌跡可言 —— 濾波、局部極小、時間間距全部無意義，
        直接在通過閘門的幀中取 |yaw| 最小的 n 張。

    不足 n_select 時就回傳實際通過的數量（硬閘門，不逐案放寬），
    讓所有 visit 適用同一個絕對品質標準；產出率由呼叫端統計回報。
    """
    frame_index = np.asarray(frame_index, dtype=int)
    yaw = np.asarray(yaw, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    roll = np.asarray(roll, dtype=float)
    confidence = np.asarray(confidence, dtype=float)

    if frame_index.size == 0:
        return []

    order = np.argsort(frame_index)          # 一律按真實時序處理
    frame_index, yaw, pitch, roll, confidence = (
        a[order] for a in (frame_index, yaw, pitch, roll, confidence))

    abs_yaw = np.abs(yaw)
    gate = (
        (confidence > conf_min)
        & (np.abs(pitch) <= pitch_max)
        & (np.abs(roll) <= roll_max)
    )
    if not gate.any():
        return []

    if trajectory:
        smoothed = _median_filter(abs_yaw, medfilt_win)
        cand = np.intersect1d(_local_minima(smoothed), np.flatnonzero(gate))
        gap = min_gap
    else:
        cand = np.flatnonzero(gate)
        gap = 0

    if cand.size == 0:
        return []

    # 按 |yaw| 升冪貪婪選取，維持最小間距
    chosen: List[int] = []
    for i in cand[np.argsort(abs_yaw[cand], kind="stable")]:
        if gap and any(abs(int(frame_index[i]) - f) < gap for f in chosen):
            continue
        chosen.append(int(frame_index[i]))
        if len(chosen) >= n_select:
            break

    return sorted(chosen)
