"""
臉部偵測（pure functions + 一個資源工廠）

open_face_mesh()  開一個 MediaPipe Face Mesh（建構昂貴、需 close），用 with 管生命週期。
detect_faces()    純函式：吃 face_mesh handle + 影像 → FaceInfo 清單。

mediapipe 的初始化/釋放收在這支；呼叫端只透過 open_face_mesh 拿不透明 handle，
不必自己 import mediapipe（就像 `with open(...) as f` 把 file handle 傳給函式）。
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.common.mediapipe_utils import MIDLINE_POINTS

logger = logging.getLogger(__name__)


@dataclass
class FaceInfo:
    """單張臉部資訊"""
    image: np.ndarray
    vertex_angle_sum: float  # 中軸線頂點夾角總和（度），越小越正面
    landmarks: np.ndarray  # 特徵點座標 (N, 2)
    path: Optional[Path] = None  # 原始檔案路徑


@contextmanager
def open_face_mesh(detection_confidence: float = 0.5):
    """開一個 MediaPipe Face Mesh；with 區塊結束自動 close。"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=detection_confidence,
    )
    try:
        yield face_mesh
    finally:
        face_mesh.close()


def detect_faces(
    face_mesh,
    images: List[np.ndarray],
    paths: Optional[List[Path]] = None,
    midline_points: Tuple[int, ...] = MIDLINE_POINTS,
) -> List[FaceInfo]:
    """批次偵測：每張影像跑 FaceMesh，回傳成功偵測到臉的 FaceInfo 清單。"""
    face_infos = []
    for i, image in enumerate(images):
        info = _detect_single(
            face_mesh, image,
            path=paths[i] if paths else None,
            index=i, midline_points=midline_points,
        )
        if info:
            face_infos.append(info)
    return face_infos


def _detect_single(face_mesh, image: np.ndarray, path: Optional[Path] = None,
                   index: int = 0,
                   midline_points: Tuple[int, ...] = MIDLINE_POINTS) -> Optional[FaceInfo]:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        logger.debug(f"第 {index} 張影像未偵測到臉部")
        return None

    landmarks = results.multi_face_landmarks[0]
    points = _landmarks_to_array(landmarks, image.shape)
    return FaceInfo(
        image=image,
        vertex_angle_sum=_vertex_angle_sum(points, midline_points),
        landmarks=points,
        path=path,
    )


def _landmarks_to_array(landmarks, image_shape: Tuple[int, int]) -> np.ndarray:
    """MediaPipe 特徵點 → numpy 陣列 (N, 2)。"""
    h, w = image_shape[:2]
    points = [[lm.x * w, lm.y * h] for lm in landmarks.landmark]
    return np.array(points, dtype=np.float64)


def _vertex_angle_sum(points: np.ndarray,
                      midline_points: Tuple[int, ...]) -> float:
    """中軸線頂點夾角總和（度）：折線各頂點相鄰線段夾角之和，越小越正面。

    points 為已縮放的 (N, 2) 特徵點陣列（見 _landmarks_to_array）。
    """
    dots = [points[i] for i in midline_points]
    vector1 = dots[1] - dots[0]
    vector2 = dots[2] - dots[1]
    vector3 = dots[3] - dots[2]

    def vector_angle(v1, v2):
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(np.clip(dot / (norm + 1e-8), -1.0, 1.0))

    angle1 = vector_angle(vector1, vector2)
    angle2 = vector_angle(vector2, vector3)
    return np.degrees(angle1) + np.degrees(angle2)
