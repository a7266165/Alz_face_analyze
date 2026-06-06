"""頭部旋轉角度計算：向量幾何法與 PnP 旋轉矩陣法。

兩種計算器共用 BaseAngleCalculator（FaceMesh 偵測 + 逐資料夾批次），各自實作
calculate()。關鍵點索引的單一真相在 src/common/mediapipe_utils.py。cv2/mediapipe
為 producer-only 重依賴，採 lazy import。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.common.mediapipe_utils import (
    PNP_FACE_3D_MODEL,
    PNP_LANDMARKS,
    VECTOR_LANDMARKS,
)


@dataclass
class AngleResult:
    """單張影像的角度結果；三軸皆有值才 is_valid。"""
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        return all(v is not None for v in (self.pitch, self.yaw, self.roll))


@dataclass
class SequenceResult:
    """整個序列的角度結果（三軸 list + 來源資料夾 + 方法名）。"""
    pitch_list: List[float]
    yaw_list: List[float]
    roll_list: List[float]
    folder_name: str
    method: str

    @property
    def length(self) -> int:
        return len(self.pitch_list)


class BaseAngleCalculator(ABC):
    """角度計算器基底：建 FaceMesh、逐資料夾跑 calculate()、收有效幀。"""

    def __init__(self):
        import mediapipe as mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @abstractmethod
    def calculate(self, image: np.ndarray) -> AngleResult:
        """單張 BGR 影像 → 角度；偵測失敗回空 AngleResult。"""
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        ...

    def process_folder(self, folder_path: Path, verbose: bool = False) -> SequenceResult:
        """跑完資料夾內所有 .jpg（檔名數字排序），只收 is_valid 的幀。"""
        import cv2

        pitch_list, yaw_list, roll_list = [], [], []
        image_files = sorted(
            folder_path.glob("*.jpg"),
            key=lambda x: int("".join(filter(str.isdigit, x.stem)) or 0),
        )

        total = len(image_files)
        for idx, img_path in enumerate(image_files):
            if verbose and idx % 100 == 0:
                print(f"\r  處理中: {idx + 1}/{total}", end="", flush=True)

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
            method=self.method_name,
        )

    def close(self):
        self.face_mesh.close()


class VectorAngleCalculator(BaseAngleCalculator):
    """向量幾何法：以額頭→下巴、右頰→左頰兩向量的外積（臉部法向量）算三軸角度。"""

    LANDMARKS = VECTOR_LANDMARKS

    @property
    def method_name(self) -> str:
        return "Vector"

    def calculate(self, image: np.ndarray) -> AngleResult:
        import cv2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return AngleResult()

        landmarks = results.multi_face_landmarks[0].landmark

        def get_point(name: str) -> np.ndarray:
            lm = landmarks[self.LANDMARKS[name]]
            return np.array([lm.x * w, lm.y * h, lm.z * w])

        forehead = get_point("forehead")
        chin = get_point("chin")
        left_cheek = get_point("left_cheek")
        right_cheek = get_point("right_cheek")

        vector1 = chin - forehead
        vector2 = left_cheek - right_cheek
        vector3 = np.cross(vector1, vector2)

        v1_unit = vector1 / np.linalg.norm(vector1)
        v2_unit = vector2 / np.linalg.norm(vector2)
        v3_unit = vector3 / np.linalg.norm(vector3)

        pitch = np.arctan2(v3_unit[1], v3_unit[2]) * (180 / np.pi)
        yaw = np.arctan2(-v3_unit[0], np.sqrt(v3_unit[1] ** 2 + v3_unit[2] ** 2)) * (180 / np.pi)
        roll = np.arctan2(v2_unit[1], v1_unit[1]) * (180 / np.pi)

        return AngleResult(pitch=pitch, yaw=yaw, roll=roll)


class PnPAngleCalculator(BaseAngleCalculator):
    """PnP 法：標準 3D 人臉模型 + 2D 關鍵點 → cv2.solvePnP 求旋轉向量 → 歐拉角。"""

    LANDMARKS = PNP_LANDMARKS
    FACE_3D_MODEL = PNP_FACE_3D_MODEL

    def __init__(self):
        super().__init__()
        self.camera_matrix = np.array([
            [385.73846, 0, 324.16235],
            [0, 385.73846, 241.51932],
            [0, 0, 1],
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
        self.object_points = np.array(
            [self.FACE_3D_MODEL[key] for key in self.LANDMARKS],
            dtype=np.float32,
        )

    @property
    def method_name(self) -> str:
        return "PnP"

    def calculate(self, image: np.ndarray) -> AngleResult:
        import cv2
        from scipy.spatial.transform import Rotation as R

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return AngleResult()

        landmarks = results.multi_face_landmarks[0].landmark
        image_points = np.array(
            [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in self.LANDMARKS.values()],
            dtype=np.float32,
        )

        success, rvec, _ = cv2.solvePnP(
            self.object_points, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return AngleResult()

        angles = R.from_rotvec(rvec.flatten()).as_euler("xyz", degrees=True)
        return AngleResult(pitch=angles[0], yaw=angles[1], roll=angles[2])
