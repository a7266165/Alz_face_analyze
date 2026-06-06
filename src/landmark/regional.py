"""分區 landmark 不對稱特徵。

以 MediaPipe 468 點的左右對應點，分區（眼/鼻/唇/臉輪廓）算座標差與面積差。
兩步：extract_and_save_landmarks 取點存 .npy；compute_regional_features 從
.npy 算特徵出 CSV。路徑由呼叫端（producer）決定，本層不碰 config。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
#  左右對應點定義 (right_idx, left_idx)
# ============================================================

EYE_PAIRS: List[Tuple[int, int]] = [
    (33, 362), (155, 382), (154, 381), (153, 380),
    (145, 374), (144, 373), (163, 390), (7, 249),
    (133, 263), (246, 466), (161, 388), (160, 387),
    (159, 386), (158, 385), (157, 384), (173, 398),
]

NOSE_PAIRS: List[Tuple[int, int]] = [
    (122, 351), (174, 399), (198, 420), (129, 358),
    (196, 419), (3, 248), (236, 456), (51, 281),
    (134, 363), (131, 360), (45, 275), (220, 440),
    (115, 344), (48, 278),
]

# 唇部輪廓 18 pairs（外上/外下/內上/內下）
MOUTH_PAIRS: List[Tuple[int, int]] = [
    (61, 291), (185, 409), (40, 270), (39, 269), (37, 267),
    (146, 375), (91, 321), (181, 405), (84, 314),
    (78, 308), (191, 415), (80, 310), (81, 311), (82, 312),
    (95, 324), (88, 318), (178, 402), (87, 317),
]

FACE_OVAL_PAIRS: List[Tuple[int, int]] = [
    (109, 338), (67, 297), (103, 332), (54, 284),
    (21, 251), (162, 389), (127, 356), (234, 454),
    (93, 323), (132, 361), (58, 288), (172, 397),
    (136, 365), (150, 379), (149, 378),
    (148, 377), (176, 400),
]

ALL_PAIRS: Dict[str, List[Tuple[int, int]]] = {
    "eye": EYE_PAIRS,
    "nose": NOSE_PAIRS,
    "mouth": MOUTH_PAIRS,
    "face_oval": FACE_OVAL_PAIRS,
}

# 各區依序的輪廓點 (right_contour, left_contour)，供 Shoelace 算面積
AREA_CONTOURS: Dict[str, Tuple[List[int], List[int]]] = {
    "eye": (
        [33, 161, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7, 133, 246],
        [362, 388, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249, 263, 466],
    ),
    "nose": (
        [122, 174, 198, 129, 196, 3, 236, 51, 134, 131, 45, 220, 115, 48],
        [351, 399, 420, 358, 419, 248, 456, 281, 363, 360, 275, 440, 344, 278],
    ),
    "mouth": (
        [61, 185, 40, 39, 37, 84, 181, 91, 146],
        [291, 409, 270, 269, 267, 314, 405, 321, 375],
    ),
    "face_oval": (
        [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 148, 176, 152],
        [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
    ),
}


# ============================================================
#  核心運算
# ============================================================

def normalize_landmarks(landmarks: np.ndarray, target_width: float = 500.0) -> np.ndarray:
    """以 bbox 正規化：最左點→X=0、最頂點→Y=0、臉寬→target_width。

    與 analyzer.LandmarkAsymmetryAnalyzer.normalize_landmarks（點 234/454 基準）
    為不同基準、不可互換。
    """
    lm = landmarks.copy()
    x_min, x_max = lm[:, 0].min(), lm[:, 0].max()
    y_min = lm[:, 1].min()
    face_width = x_max - x_min
    if face_width < 1e-6:
        return lm
    scale = target_width / face_width
    lm[:, 0] = (lm[:, 0] - x_min) * scale
    lm[:, 1] = (lm[:, 1] - y_min) * scale
    return lm


def shoelace_area(points: np.ndarray) -> float:
    """多邊形面積（Shoelace 公式）；少於 3 點回 0。"""
    n = len(points)
    if n < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def extract_landmarks_from_image(image_path: Path, face_mesh) -> Optional[np.ndarray]:
    """單張影像 → 正規化 (468, 2) landmarks；偵測失敗回 None。"""
    import cv2  # lazy：只有 producer 端（含 mediapipe 的環境）用得到
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    h, w = image.shape[:2]
    lm = results.multi_face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm.landmark[:468]])
    return normalize_landmarks(pts)


def compute_pair_features(
    landmarks: np.ndarray,
    pairs: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    area_contours: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
) -> Dict[str, float]:
    """單筆 (468, 2) → 各 pair 的 (Δx, Δy) 與各區面積差。預設用 ALL_PAIRS / AREA_CONTOURS。"""
    if pairs is None:
        pairs = ALL_PAIRS
    if area_contours is None:
        area_contours = AREA_CONTOURS

    features: Dict[str, float] = {}

    for region_name, pair_list in pairs.items():
        for i, (r_idx, l_idx) in enumerate(pair_list, 1):
            features[f"{region_name}_x_pair_{i}"] = landmarks[l_idx, 0] - landmarks[r_idx, 0]
            features[f"{region_name}_y_pair_{i}"] = landmarks[l_idx, 1] - landmarks[r_idx, 1]

    for region_name, (contour_r, contour_l) in area_contours.items():
        area_r = shoelace_area(landmarks[contour_r])
        area_l = shoelace_area(landmarks[contour_l])
        features[f"{region_name}_area_diff"] = area_l - area_r

    return features


# ============================================================
#  Step 1：取點存 .npy
# ============================================================

def extract_and_save_landmarks(aligned_dir: Path, output_dir: Path) -> Tuple[int, int]:
    """逐受試者從 aligned 影像取正規化 landmarks，每人存一個 (n_images, 468, 2) .npy。

    Returns:
        (success_count, failed_count)。
    """
    import mediapipe as mp

    output_dir.mkdir(parents=True, exist_ok=True)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )

    subject_dirs = sorted([d for d in aligned_dir.iterdir() if d.is_dir()])
    logger.info(f"Extracting landmarks for {len(subject_dirs)} subjects...")

    success, failed = 0, 0

    for i, subject_dir in enumerate(subject_dirs):
        subject_id = subject_dir.name
        images = sorted(subject_dir.glob("*.png"))
        if not images:
            failed += 1
            continue

        all_landmarks = [
            pts for img_path in images
            if (pts := extract_landmarks_from_image(img_path, face_mesh)) is not None
        ]
        if not all_landmarks:
            failed += 1
            continue

        arr = np.stack(all_landmarks).astype(np.float32)
        np.save(output_dir / f"{subject_id}.npy", arr)
        success += 1

        if (i + 1) % 500 == 0:
            logger.info(f"  {i + 1}/{len(subject_dirs)} (success={success}, failed={failed})")

    face_mesh.close()
    logger.info(f"Extraction done: {success} saved, {failed} failed → {output_dir}")
    return success, failed


# ============================================================
#  Step 2：從 .npy 算特徵
# ============================================================

def compute_regional_features(
    landmarks_dir: Path,
    output_path: Path,
    pairs: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    area_contours: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
) -> pd.DataFrame:
    """讀 .npy landmarks，逐影像算 pair 特徵後取受試者平均，輸出 CSV。"""
    if pairs is None:
        pairs = ALL_PAIRS
    if area_contours is None:
        area_contours = AREA_CONTOURS

    npy_files = sorted(landmarks_dir.glob("*.npy"))
    logger.info(f"Computing features for {len(npy_files)} subjects...")

    results = []
    for npy_file in npy_files:
        arr = np.load(npy_file)  # (n_images, 468, 2)
        img_features = [compute_pair_features(arr[i], pairs, area_contours)
                        for i in range(arr.shape[0])]
        avg = pd.DataFrame(img_features).mean().to_dict()
        avg["subject_id"] = npy_file.stem
        results.append(avg)

    df = pd.DataFrame(results)
    df = df[["subject_id"] + [c for c in df.columns if c != "subject_id"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_features = sum(len(p) for p in pairs.values()) * 2 + len(area_contours)
    logger.info(f"Done: {len(df)} subjects, {n_features} features → {output_path}")
    return df
