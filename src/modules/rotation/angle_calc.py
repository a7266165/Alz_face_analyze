"""
頭部旋轉角度計算

包含兩種方法：
1. VectorAngleCalculator — 向量幾何法
2. PnPAngleCalculator — PnP + 旋轉矩陣法
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Optional, List

from src.core.mediapipe_utils import (
    VECTOR_LANDMARKS,
    PNP_LANDMARKS,
    PNP_FACE_3D_MODEL,
)


@dataclass
class AngleResult:
    """儲存單張影像的角度計算結果"""
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        return all(v is not None for v in [self.pitch, self.yaw, self.roll])


@dataclass
class SequenceResult:
    """儲存整個序列的角度計算結果"""
    pitch_list: List[float]
    yaw_list: List[float]
    roll_list: List[float]
    folder_name: str
    method: str

    @property
    def length(self) -> int:
        return len(self.pitch_list)


# ============================================================
#  Base Calculator
# ============================================================
class BaseAngleCalculator(ABC):
    """角度計算器的抽象基底類別"""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    @abstractmethod
    def calculate(self, image: np.ndarray) -> AngleResult:
        """計算單張影像的角度"""
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """返回方法名稱"""
        pass

    def process_folder(self, folder_path: Path, verbose: bool = False) -> SequenceResult:
        """處理整個資料夾的影像"""
        pitch_list, yaw_list, roll_list = [], [], []

        image_files = sorted(
            folder_path.glob("*.jpg"),
            key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
        )

        total = len(image_files)
        for idx, img_path in enumerate(image_files):
            if verbose and idx % 100 == 0:
                print(f"\r  處理中: {idx+1}/{total}", end="", flush=True)

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            result = self.calculate(image)
            if result.is_valid:
                pitch_list.append(result.pitch)
                yaw_list.append(result.yaw)
                roll_list.append(result.roll)

        if verbose:
            print(f"\r  完成: {len(pitch_list)}/{total} 張有效")

        return SequenceResult(
            pitch_list=pitch_list,
            yaw_list=yaw_list,
            roll_list=roll_list,
            folder_name=folder_path.name,
            method=self.method_name
        )

    def close(self):
        """釋放資源"""
        self.face_mesh.close()


# ============================================================
#  Vector Angle Calculator — 向量幾何法
# ============================================================
class VectorAngleCalculator(BaseAngleCalculator):
    """
    使用向量幾何法計算頭部旋轉角度

    原理：
    1. 取得臉部關鍵點建立垂直向量 (額頭→下巴) 和水平向量 (右臉頰→左臉頰)
    2. 透過外積計算臉部法向量
    3. 使用 arctan2 計算各軸角度
    """

    LANDMARKS = VECTOR_LANDMARKS

    @property
    def method_name(self) -> str:
        return "Vector"

    def calculate(self, image: np.ndarray) -> AngleResult:
        """使用向量法計算角度"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return AngleResult()

        landmarks = results.multi_face_landmarks[0].landmark

        def get_point(name: str) -> np.ndarray:
            idx = self.LANDMARKS[name]
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h, lm.z * w])

        forehead = get_point('forehead')
        chin = get_point('chin')
        left_cheek = get_point('left_cheek')
        right_cheek = get_point('right_cheek')

        vector1 = chin - forehead
        vector2 = left_cheek - right_cheek
        vector3 = np.cross(vector1, vector2)

        v1_unit = vector1 / np.linalg.norm(vector1)
        v2_unit = vector2 / np.linalg.norm(vector2)
        v3_unit = vector3 / np.linalg.norm(vector3)

        pitch = np.arctan2(v3_unit[1], v3_unit[2]) * (180 / np.pi)
        yaw = np.arctan2(-v3_unit[0], np.sqrt(v3_unit[1]**2 + v3_unit[2]**2)) * (180 / np.pi)
        roll = np.arctan2(v2_unit[1], v1_unit[1]) * (180 / np.pi)

        return AngleResult(pitch=pitch, yaw=yaw, roll=roll)


# ============================================================
#  PnP Angle Calculator — PnP + 旋轉矩陣法
# ============================================================
class PnPAngleCalculator(BaseAngleCalculator):
    """
    使用 PnP 算法計算頭部旋轉角度

    原理：
    1. 建立標準 3D 人臉模型
    2. 從影像取得對應的 2D 關鍵點
    3. 使用 cv2.solvePnP 求解旋轉向量
    4. 將旋轉向量轉換為歐拉角
    """

    LANDMARKS = PNP_LANDMARKS
    FACE_3D_MODEL = PNP_FACE_3D_MODEL

    def __init__(self):
        super().__init__()

        self.camera_matrix = np.array([
            [385.73846, 0, 324.16235],
            [0, 385.73846, 241.51932],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((4, 1))

        self.object_points = np.array(
            [self.FACE_3D_MODEL[key] for key in self.LANDMARKS.keys()],
            dtype=np.float32
        )

    @property
    def method_name(self) -> str:
        return "PnP"

    def calculate(self, image: np.ndarray) -> AngleResult:
        """使用 PnP 法計算角度"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return AngleResult()

        landmarks = results.multi_face_landmarks[0].landmark
        image_points = []

        for key in self.LANDMARKS.keys():
            idx = self.LANDMARKS[key]
            lm = landmarks[idx]
            image_points.append([lm.x * w, lm.y * h])

        image_points = np.array(image_points, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return AngleResult()

        r = R.from_rotvec(rvec.flatten())
        angles = r.as_euler('xyz', degrees=True)

        return AngleResult(pitch=angles[0], yaw=angles[1], roll=angles[2])
