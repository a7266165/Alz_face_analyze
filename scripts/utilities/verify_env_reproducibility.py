"""
End-to-end model-forward test for each new extract env.

Strict np.allclose reproducibility against saved workspace `.npy` would require
re-running the full alignment + frontality selection + mirror split pipeline
(crosses env boundaries: asymmetry env does preprocess, embedding_other env
does feature extract). That's a separate end-to-end task.

This script does the next-best validation: for each tool, load 10 cached
selected images for subject P1-2, run the model's forward pass, and verify
the output is structurally sane (correct shape, dtype, finite, non-trivial).
Stronger than `import` smoke test, weaker than np.allclose reproducibility.

Usage (run with the env that owns the tool):
    "<env>/python.exe" scripts/utilities/verify_env_reproducibility.py --tool arcface
    ...                                                                 --tool topofr
    ...                                                                 --tool dlib
    ...                                                                 --tool landmark
    ...                                                                 --tool age
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SUBJECT = "P1-2"
SELECTED_DIR = PROJECT_ROOT / "workspace" / "preprocess" / "selected" / SUBJECT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_selected_images():
    paths = sorted(SELECTED_DIR.glob("selected_*.png"))
    if not paths:
        raise FileNotFoundError(f"No selected images at {SELECTED_DIR}")
    images = [cv2.imread(str(p)) for p in paths]
    return images, paths


def check_output(arr, expected_shape, label, allow_nan=False):
    ok = True
    if arr.shape != expected_shape:
        logger.error(f"  [{label}] SHAPE MISMATCH: got {arr.shape}, expected {expected_shape}")
        ok = False
    if not allow_nan and not np.isfinite(arr).all():
        logger.error(f"  [{label}] CONTAINS NaN/Inf")
        ok = False
    if np.allclose(arr, 0):
        logger.error(f"  [{label}] ALL ZEROS — model likely didn't run")
        ok = False
    if ok:
        logger.info(f"  [{label}] shape={arr.shape} dtype={arr.dtype} "
                    f"|mean|={np.abs(arr).mean():.4f} std={arr.std():.4f}")
    return ok


def test_arcface(images):
    from src.extractor.features.embedding.arcface_extractor import ArcFaceExtractor
    ext = ArcFaceExtractor()
    assert ext.is_available(), "ArcFace not available"
    feats = np.stack([ext.extract(img) for img in images]).astype(np.float32)
    return check_output(feats, (10, 512), "arcface")


def test_topofr(images):
    from src.extractor.features.embedding.topofr_extractor import TopoFRExtractor
    ext = TopoFRExtractor()
    assert ext.is_available(), "TopoFR not available"
    feats = np.stack([ext.extract(img) for img in images]).astype(np.float32)
    return check_output(feats, (10, 512), "topofr")


def test_dlib(images):
    from src.extractor.features.embedding.dlib_extractor import DlibExtractor
    ext = DlibExtractor()
    assert ext.is_available(), "Dlib not available"
    feats = np.stack([ext.extract(img) for img in images]).astype(np.float32)
    return check_output(feats, (10, 128), "dlib")


def test_vggface(images):
    from src.extractor.features.embedding.vggface_extractor import VGGFaceExtractor
    ext = VGGFaceExtractor()
    assert ext.is_available(), "VGGFace not available"
    feats = np.stack([ext.extract(img) for img in images]).astype(np.float32)
    return check_output(feats, (10, ext.feature_dim), "vggface")


def test_landmark(images):
    import mediapipe as mp
    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5,
    )
    out = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        if not res.multi_face_landmarks:
            logger.warning("  no landmarks for one image")
            return False
        lm = res.multi_face_landmarks[0]
        h, w = img.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
        out.append(pts)
    feats = np.stack(out)
    return check_output(feats, (10, 468, 2), "landmark")


def test_rotation(images):
    """Mediapipe + OpenCV PnP for head rotation. Output: roll/pitch/yaw per image."""
    import mediapipe as mp
    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5,
    )
    angles = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        if not res.multi_face_landmarks:
            return False
        h, w = img.shape[:2]
        lm = res.multi_face_landmarks[0]
        pts2d = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
        # Use 6 stable points for PnP: nose tip, chin, left/right eye outer, mouth corners
        idx_2d = [1, 152, 33, 263, 61, 291]
        model_3d = np.array([
            [0.0, 0.0, 0.0], [0.0, -63.6, -12.5], [-43.3, 32.7, -26.0],
            [43.3, 32.7, -26.0], [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1],
        ], dtype=np.float32)
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
        ok, rvec, _ = cv2.solvePnP(model_3d, pts2d[idx_2d], cam, np.zeros(4))
        if not ok:
            return False
        R, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        angles.append([roll, pitch, yaw])
    feats = np.array(angles, dtype=np.float32)
    return check_output(feats, (10, 3), "rotation (roll/pitch/yaw)")


def test_age(images):
    """MiVOLO age prediction via the project's MiVOLOPredictor (HF auto-download)."""
    from src.extractor.features.age import MiVOLOPredictor
    pred = MiVOLOPredictor()
    pred.initialize()
    ages = pred.predict(images)
    ages_arr = np.array(ages, dtype=np.float32)
    if len(ages_arr) == 0:
        logger.error("  no faces detected")
        return False
    logger.info(f"  [age] mean={ages_arr.mean():.1f}, range=[{ages_arr.min():.1f}, {ages_arr.max():.1f}], n={len(ages_arr)}")
    return 30 < ages_arr.mean() < 100


def _test_emotion(images, ExtractorCls, label, expected_min_keys=4):
    ext = ExtractorCls()
    if hasattr(ext, "is_available") and not ext.is_available():
        logger.error(f"  [{label}] not available")
        return False
    out = []
    for img in images:
        d = ext.extract_frame(img)
        if d is None:
            return False
        out.append(d)
    if not out:
        return False
    keys = list(out[0].keys())
    arr = np.array([[d.get(k, np.nan) for k in keys] for d in out], dtype=np.float32)
    logger.info(f"  [{label}] n_frames={len(out)} n_features={len(keys)} "
                f"sample_keys={keys[:5]}... |mean|={np.nanmean(np.abs(arr)):.4f}")
    return len(keys) >= expected_min_keys and np.isfinite(arr).any()


def test_openface(images):
    from src.extractor.features.emotion.extractor.openface import OpenFaceExtractor
    return _test_emotion(images, OpenFaceExtractor, "openface")


def test_libreface(images):
    from src.extractor.features.emotion.extractor.libreface import LibreFaceExtractor
    return _test_emotion(images, LibreFaceExtractor, "libreface")


def test_pyfeat(images):
    from src.extractor.features.emotion.extractor.pyfeat import PyFeatExtractor
    return _test_emotion(images, PyFeatExtractor, "pyfeat")


def test_dan(images):
    from src.extractor.features.emotion.extractor.dan import DANExtractor
    return _test_emotion(images, DANExtractor, "dan")


TOOLS = {
    "arcface": test_arcface, "topofr": test_topofr, "dlib": test_dlib,
    "vggface": test_vggface, "landmark": test_landmark,
    "rotation": test_rotation, "age": test_age,
    "openface": test_openface, "libreface": test_libreface,
    "pyfeat": test_pyfeat, "dan": test_dan,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, choices=list(TOOLS.keys()))
    args = parser.parse_args()

    images, _ = load_selected_images()
    logger.info(f"loaded {len(images)} images for {SUBJECT}")
    result = TOOLS[args.tool](images)
    logger.info(f"=== {args.tool}: {'PASS' if result else 'FAIL'} ===")
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
