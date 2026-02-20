"""
臉部偵測器

負責使用 MediaPipe 偵測臉部並提取特徵點
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FaceInfo:
    """單張臉部資訊"""
    image: np.ndarray
    vertex_angle_sum: float  # 中軸線頂點夾角總和（度），越小越正面
    confidence: float  # 偵測信心度
    landmarks: np.ndarray  # 468個特徵點座標
    path: Optional[Path] = None  # 原始檔案路徑
    index: int = 0  # 在批次中的索引


class FaceDetector:
    """
    臉部偵測器

    使用 MediaPipe Face Mesh 偵測臉部並提取 468 個特徵點
    """

    def __init__(
        self,
        detection_confidence: float = 0.5,
        midline_points: Tuple[int, ...] = (10, 168, 4, 2),
    ):
        """
        初始化偵測器

        Args:
            detection_confidence: MediaPipe 偵測信心度閾值
            midline_points: 臉部中軸線特徵點索引
        """
        self.detection_confidence = detection_confidence
        self.midline_points = midline_points
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        self._init_face_mesh()

    def _init_face_mesh(self):
        """初始化 MediaPipe Face Mesh"""
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.detection_confidence,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """釋放資源"""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None

    def detect_batch(
        self,
        images: List[np.ndarray],
        paths: Optional[List[Path]] = None
    ) -> List[FaceInfo]:
        """
        批次偵測臉部

        Args:
            images: 影像列表 (BGR 格式)
            paths: 對應的檔案路徑列表（可選）

        Returns:
            偵測到的臉部資訊列表
        """
        face_infos = []

        for i, image in enumerate(images):
            face_info = self.detect_single(
                image,
                path=paths[i] if paths else None,
                index=i
            )
            if face_info:
                face_infos.append(face_info)

        return face_infos

    def detect_single(
        self,
        image: np.ndarray,
        path: Optional[Path] = None,
        index: int = 0
    ) -> Optional[FaceInfo]:
        """
        偵測單張影像的臉部

        Args:
            image: BGR 格式的影像
            path: 檔案路徑（可選）
            index: 索引

        Returns:
            臉部資訊，若未偵測到則返回 None
        """
        # 轉換色彩空間
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            logger.debug(f"第 {index} 張影像未偵測到臉部")
            return None

        # 提取特徵點
        landmarks = results.multi_face_landmarks[0]
        landmarks_array = self._landmarks_to_array(landmarks, image.shape)

        # 計算中軸角度
        vertex_angle_sum = self._calculate_vertex_angle_sum(landmarks, image.shape)

        return FaceInfo(
            image=image,
            vertex_angle_sum=vertex_angle_sum,
            confidence=1.0,  # MediaPipe 不直接提供信心度
            landmarks=landmarks_array,
            path=path,
            index=index,
        )

    def redetect_landmarks(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        重新偵測影像的臉部特徵點

        用於影像經過處理（如旋轉）後需要更新特徵點

        Args:
            image: BGR 格式的影像

        Returns:
            特徵點陣列 (468, 2)，若未偵測到則返回 None
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        return self._landmarks_to_array(
            results.multi_face_landmarks[0],
            image.shape
        )

    def _landmarks_to_array(
        self,
        landmarks,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        將 MediaPipe 特徵點轉換為 numpy 陣列

        Args:
            landmarks: MediaPipe 特徵點
            image_shape: 影像尺寸

        Returns:
            特徵點陣列 (468, 2)
        """
        h, w = image_shape[:2]
        points = []

        for lm in landmarks.landmark:
            x = lm.x * w
            y = lm.y * h
            points.append([x, y])

        return np.array(points, dtype=np.float64)

    def _calculate_vertex_angle_sum(
        self,
        landmarks,
        image_shape: Tuple[int, int]
    ) -> float:
        """
        計算中軸線頂點夾角總和

        將 midline_points 定義的特徵點連成折線，
        計算各頂點處相鄰線段的夾角總和。
        數值越小表示中軸線越直，臉部越正面。

        Args:
            landmarks: MediaPipe 特徵點
            image_shape: 影像尺寸 (H, W)

        Returns:
            頂點夾角總和（度）
        """
        h, w = image_shape[:2]

        dots = []
        for idx in self.midline_points:
            point = landmarks.landmark[idx]
            dots.append(np.array([point.x * w, point.y * h]))

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
