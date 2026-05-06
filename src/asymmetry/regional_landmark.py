"""
區域式 Landmark-based 面部不對稱性

用 MediaPipe Face Mesh 468 landmarks 的左右對應點，
分區域（眼、鼻、唇、臉部輪廓）計算座標差異和面積差異。

兩步驟設計：
  1. extract_and_save_landmarks(): 提取 + 正規化 + 存 .npy
  2. compute_regional_features(): 從 .npy 計算 pair 差值 + 面積差 → CSV

Landmark 正規化方式：最左點→X=0, 最頂點→Y=0, 臉寬→500
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
#  Landmark Pair Definitions (right_idx, left_idx)
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

# 標準唇部輪廓 18 pairs
MOUTH_PAIRS: List[Tuple[int, int]] = [
    # Outer upper lip
    (61, 291), (185, 409), (40, 270), (39, 269), (37, 267),
    # Outer lower lip
    (146, 375), (91, 321), (181, 405), (84, 314),
    # Inner upper lip
    (78, 308), (191, 415), (80, 310), (81, 311), (82, 312),
    # Inner lower lip
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

# Ordered contour points for Shoelace area calculation
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
#  Core Functions
# ============================================================

def normalize_landmarks(landmarks: np.ndarray, target_width: float = 500.0) -> np.ndarray:
    """正規化：最左點→X=0, 最頂點→Y=0, 臉寬→target_width"""
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
    """Shoelace formula for polygon area."""
    n = len(points)
    if n < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def extract_landmarks_from_image(image_path: Path, face_mesh) -> Optional[np.ndarray]:
    """Extract and normalize 468 landmarks from a single image.

    Args:
        image_path: Path to image file
        face_mesh: Initialized MediaPipe FaceMesh instance

    Returns:
        (468, 2) normalized landmark array, or None if detection failed
    """
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
    """Compute (Δx, Δy) for all pairs + area diffs from a (468, 2) landmark array.

    Args:
        landmarks: (468, 2) normalized coordinates
        pairs: Region → pair list mapping. Defaults to ALL_PAIRS.
        area_contours: Region → (right_contour, left_contour). Defaults to AREA_CONTOURS.

    Returns:
        Dict of feature_name → value
    """
    if pairs is None:
        pairs = ALL_PAIRS
    if area_contours is None:
        area_contours = AREA_CONTOURS

    features: Dict[str, float] = {}

    for region_name, pair_list in pairs.items():
        for i, (r_idx, l_idx) in enumerate(pair_list, 1):
            dx = landmarks[l_idx, 0] - landmarks[r_idx, 0]
            dy = landmarks[l_idx, 1] - landmarks[r_idx, 1]
            features[f"{region_name}_x_pair_{i}"] = dx
            features[f"{region_name}_y_pair_{i}"] = dy

    for region_name, (contour_r, contour_l) in area_contours.items():
        area_r = shoelace_area(landmarks[contour_r])
        area_l = shoelace_area(landmarks[contour_l])
        features[f"{region_name}_area_diff"] = area_l - area_r

    return features


# ============================================================
#  Step 1: Extract & Save
# ============================================================

def extract_and_save_landmarks(
    aligned_dir: Path,
    output_dir: Path,
) -> Tuple[int, int]:
    """Extract normalized landmarks from all aligned images, save as .npy per subject.

    Args:
        aligned_dir: Directory containing subject subdirectories with aligned .png images
        output_dir: Directory to save {subject_id}.npy files, each with shape (n_images, 468, 2)

    Returns:
        (success_count, failed_count)
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

        all_landmarks = []
        for img_path in images:
            pts = extract_landmarks_from_image(img_path, face_mesh)
            if pts is not None:
                all_landmarks.append(pts)

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
#  Step 2: Compute Features from Saved Landmarks
# ============================================================

def compute_regional_features(
    landmarks_dir: Path,
    output_path: Path,
    pairs: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    area_contours: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
) -> pd.DataFrame:
    """Read saved .npy landmarks, compute pair features, output CSV.

    Args:
        landmarks_dir: Directory containing {subject_id}.npy files
        output_path: Path to save output CSV
        pairs: Custom pair definitions (defaults to ALL_PAIRS)
        area_contours: Custom area contours (defaults to AREA_CONTOURS)

    Returns:
        DataFrame with features
    """
    if pairs is None:
        pairs = ALL_PAIRS
    if area_contours is None:
        area_contours = AREA_CONTOURS

    npy_files = sorted(landmarks_dir.glob("*.npy"))
    logger.info(f"Computing features for {len(npy_files)} subjects...")

    results = []
    for npy_file in npy_files:
        subject_id = npy_file.stem
        arr = np.load(npy_file)  # (n_images, 468, 2)

        img_features = []
        for img_idx in range(arr.shape[0]):
            feats = compute_pair_features(arr[img_idx], pairs, area_contours)
            img_features.append(feats)

        avg = pd.DataFrame(img_features).mean().to_dict()
        avg["subject_id"] = subject_id
        results.append(avg)

    df = pd.DataFrame(results)
    cols = ["subject_id"] + [c for c in df.columns if c != "subject_id"]
    df = df[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_pairs = sum(len(p) for p in pairs.values())
    n_areas = len(area_contours)
    logger.info(f"Done: {len(df)} subjects, {n_pairs * 2 + n_areas} features → {output_path}")
    return df
