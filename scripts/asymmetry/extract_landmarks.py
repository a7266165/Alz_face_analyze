"""
Landmark-based 面部不對稱性分析 — 兩步驟

Step 1 (extract_landmarks):
  用 MediaPipe Face Mesh 提取 468 landmarks，正規化後存 .npy

Step 2 (compute_features):
  從 .npy 讀取座標，計算 pair 差值 + 面積差，輸出 CSV

用法:
  python extract_landmark_asymmetry.py extract   # Step 1
  python extract_landmark_asymmetry.py compute   # Step 2
  python extract_landmark_asymmetry.py all       # 兩步都跑
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Paths ===
ALIGNED_DIR = PROJECT_ROOT / "workspace" / "preprocess" / "aligned"
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "asymmetry"
LANDMARKS_DIR = OUTPUT_DIR / "landmarks"

# === Landmark Pair Definitions ===
# (right_side_idx, left_side_idx)

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

MOUTH_PAIRS: List[Tuple[int, int]] = [
    # 標準唇部輪廓 18 pairs
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
    # 底部兩組（已修正對應）
    (148, 377), (176, 400),
]

ALL_PAIRS = {
    "eye": EYE_PAIRS,
    "nose": NOSE_PAIRS,
    "mouth": MOUTH_PAIRS,
    "face_oval": FACE_OVAL_PAIRS,
}

# For area calculation: ordered contour points per region
EYE_CONTOUR_R = [33, 161, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7, 133, 246]
EYE_CONTOUR_L = [362, 388, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249, 263, 466]
NOSE_CONTOUR_R = [122, 174, 198, 129, 196, 3, 236, 51, 134, 131, 45, 220, 115, 48]
NOSE_CONTOUR_L = [351, 399, 420, 358, 419, 248, 456, 281, 363, 360, 275, 440, 344, 278]
MOUTH_CONTOUR_R = [61, 185, 40, 39, 37, 84, 181, 91, 146]
MOUTH_CONTOUR_L = [291, 409, 270, 269, 267, 314, 405, 321, 375]
FACE_OVAL_CONTOUR_R = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 148, 176, 152]
FACE_OVAL_CONTOUR_L = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]

AREA_CONTOURS = {
    "eye": (EYE_CONTOUR_R, EYE_CONTOUR_L),
    "nose": (NOSE_CONTOUR_R, NOSE_CONTOUR_L),
    "mouth": (MOUTH_CONTOUR_R, MOUTH_CONTOUR_L),
    "face_oval": (FACE_OVAL_CONTOUR_R, FACE_OVAL_CONTOUR_L),
}


# ============================================================
# Step 1: Extract & save raw landmarks
# ============================================================

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """最左點→X=0, 最頂點→Y=0, 臉寬→500"""
    lm = landmarks.copy()
    x_min, x_max = lm[:, 0].min(), lm[:, 0].max()
    y_min = lm[:, 1].min()
    face_width = x_max - x_min
    if face_width < 1e-6:
        return lm
    scale = 500.0 / face_width
    lm[:, 0] = (lm[:, 0] - x_min) * scale
    lm[:, 1] = (lm[:, 1] - y_min) * scale
    return lm


def extract_landmarks():
    """Step 1: Extract normalized 468 landmarks from all aligned images, save as .npy per subject."""
    import mediapipe as mp

    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )

    subject_dirs = sorted([d for d in ALIGNED_DIR.iterdir() if d.is_dir()])
    logger.info(f"Step 1: Extracting landmarks for {len(subject_dirs)} subjects...")

    success, failed = 0, 0

    for i, subject_dir in enumerate(subject_dirs):
        subject_id = subject_dir.name
        images = sorted(subject_dir.glob("*.png"))
        if not images:
            failed += 1
            continue

        all_landmarks = []
        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                continue

            h, w = image.shape[:2]
            lm = results.multi_face_landmarks[0]
            pts = np.array([[p.x * w, p.y * h] for p in lm.landmark[:468]])
            pts_norm = normalize_landmarks(pts)
            all_landmarks.append(pts_norm)

        if not all_landmarks:
            failed += 1
            continue

        # Save: (n_images, 468, 2)
        arr = np.stack(all_landmarks).astype(np.float32)
        np.save(LANDMARKS_DIR / f"{subject_id}.npy", arr)
        success += 1

        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(subject_dirs)} (success={success}, failed={failed})")

    face_mesh.close()
    logger.info(f"Step 1 done: {success} saved, {failed} failed → {LANDMARKS_DIR}")


# ============================================================
# Step 2: Compute pair features from saved landmarks
# ============================================================

def shoelace_area(points: np.ndarray) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_features_from_landmarks(landmarks: np.ndarray) -> Dict[str, float]:
    """Compute pair diffs + area diffs from a single (468,2) landmark array."""
    features = {}

    for region_name, pairs in ALL_PAIRS.items():
        for i, (r_idx, l_idx) in enumerate(pairs, 1):
            dx = landmarks[l_idx, 0] - landmarks[r_idx, 0]
            dy = landmarks[l_idx, 1] - landmarks[r_idx, 1]
            features[f"{region_name}_x_pair_{i}"] = dx
            features[f"{region_name}_y_pair_{i}"] = dy

    for region_name, (contour_r, contour_l) in AREA_CONTOURS.items():
        area_r = shoelace_area(landmarks[contour_r])
        area_l = shoelace_area(landmarks[contour_l])
        features[f"{region_name}_area_diff"] = area_l - area_r

    return features


def compute_features():
    """Step 2: Read .npy landmarks, compute pair features, output CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(LANDMARKS_DIR.glob("*.npy"))
    logger.info(f"Step 2: Computing features for {len(npy_files)} subjects...")

    results = []
    for npy_file in npy_files:
        subject_id = npy_file.stem
        arr = np.load(npy_file)  # (n_images, 468, 2)

        # Compute features per image, then average
        img_features = []
        for img_idx in range(arr.shape[0]):
            feats = compute_features_from_landmarks(arr[img_idx])
            img_features.append(feats)

        avg = pd.DataFrame(img_features).mean().to_dict()
        avg["subject_id"] = subject_id
        results.append(avg)

    df = pd.DataFrame(results)
    cols = ["subject_id"] + [c for c in df.columns if c != "subject_id"]
    df = df[cols]

    output_path = OUTPUT_DIR / "landmark_features.csv"
    df.to_csv(output_path, index=False)

    n_pairs = sum(len(p) for p in ALL_PAIRS.values())
    n_areas = len(AREA_CONTOURS)
    logger.info(f"Step 2 done: {len(df)} subjects, {n_pairs*2 + n_areas} features → {output_path}")

    print(f"\nFeature breakdown:")
    for region, pairs in ALL_PAIRS.items():
        print(f"  {region}: {len(pairs)} pairs × 2 (x,y) = {len(pairs)*2}")
    print(f"  area diffs: {n_areas}")
    print(f"  Total: {n_pairs} pairs × 2 + {n_areas} = {n_pairs*2 + n_areas}")


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_landmark_asymmetry.py [extract|compute|all]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "extract":
        extract_landmarks()
    elif cmd == "compute":
        compute_features()
    elif cmd == "all":
        extract_landmarks()
        compute_features()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
