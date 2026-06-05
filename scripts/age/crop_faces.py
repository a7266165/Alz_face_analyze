"""
用 MediaPipe Face Mesh 從對齊影像裁出人臉,供 MiVOLO 年齡預測使用。

對齊影像本來就是 preprocess 階段用 MediaPipe 挑出的「最正臉」,故此處再偵測幾乎
100% 命中。裁切 = 468 個 landmark 的外接框,以框長邊為基準外擴 margin(預設 35%);
偵測不到臉時 fallback 用整張對齊圖(臉已置中)。取代舊的 Haar+面積守門(誤抓衣物)。

mediapipe 不在 age env,故本步驟在 asymmetry/rotation env 跑;裁切存檔後由
scripts/age/predict.py(age env)讀取預測。

輸出格式比照 input/<ID>/<原檔名>,可直接當 predict.py 的 --aligned-dir。
"""

import sys
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import preprocess_dir, WORKSPACE_REFACTOR_DIR
from src.common.image_io import iter_subject_dirs, load_subject
from src.common.cohort import load_demographics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = WORKSPACE_REFACTOR_DIR / "age" / "predictions" / "1_MiVOLO" / "input"


def _imwrite_unicode(path: Path, img) -> None:
    """Unicode-safe 影像寫檔(對應 image_io.imread_unicode)。"""
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    else:
        logger.warning(f"影像編碼失敗,未寫出: {path}")


def crop_to_landmarks(image: np.ndarray, landmarks: np.ndarray, margin: float) -> np.ndarray:
    """landmarks(N,2 像素座標)外接框 + margin 外擴後裁切;margin 以框長邊為基準。"""
    h, w = image.shape[:2]
    x1, y1 = landmarks[:, 0].min(), landmarks[:, 1].min()
    x2, y2 = landmarks[:, 0].max(), landmarks[:, 1].max()
    m = margin * max(x2 - x1, y2 - y1)
    X1, Y1 = int(max(0, x1 - m)), int(max(0, y1 - m))
    X2, Y2 = int(min(w, x2 + m)), int(min(h, y2 + m))
    return image[Y1:Y2, X1:X2]


def load_valid_ids() -> set:
    """demographics 中年齡有效的受試者 ID(與 predict.py 一致,只裁這些)。"""
    df = load_demographics()
    return set(df["ID"][df["Age"].notna()])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=0.35,
                    help="臉框外擴比例(以框長邊為基準;預設 0.35)")
    ap.add_argument("--aligned-dir", type=Path, default=None,
                    help="對齊影像來源;留空用含背景的對齊影像(aligned, background)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"裁切輸出目錄(預設 {DEFAULT_OUTPUT})")
    ap.add_argument("--subject-prefix", default=None,
                    help="只處理 ID 開頭符合 prefix 的受試者")
    ap.add_argument("--max-images", type=int, default=10,
                    help="每位受試者最多裁幾張(預設 10,與 predict.py 一致)")
    args = ap.parse_args()

    aligned_dir = args.aligned_dir or preprocess_dir("aligned", background=True)
    logger.info("=" * 60)
    logger.info("MediaPipe 人臉裁切")
    logger.info(f"來源: {aligned_dir}")
    logger.info(f"輸出: {args.output_dir}  | margin={args.margin}")
    logger.info("=" * 60)

    valid_ids = load_valid_ids()
    subjects = iter_subject_dirs(aligned_dir, include_prefix=args.subject_prefix)
    logger.info(f"受試者目錄 {len(subjects)} 個,demographics 有效 ID {len(valid_ids)} 個")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )

    n_img = n_miss = n_subj = 0
    try:
        for subject_dir in subjects:
            sid = subject_dir.name
            if sid not in valid_ids:
                continue
            pairs = load_subject(subject_dir, with_path=True, max_images=args.max_images)
            if not pairs:
                logger.warning(f"{sid}: 無影像")
                continue
            out_sub = args.output_dir / sid
            out_sub.mkdir(parents=True, exist_ok=True)
            for fp, img in pairs:
                res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if res.multi_face_landmarks:
                    h, w = img.shape[:2]
                    lm = res.multi_face_landmarks[0].landmark
                    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
                    crop = crop_to_landmarks(img, pts, args.margin)
                else:
                    crop = img  # fallback: 整張對齊圖(臉已置中)
                    n_miss += 1
                _imwrite_unicode(out_sub / fp.name, crop)
                n_img += 1
            n_subj += 1
            if n_subj % 200 == 0:
                logger.info(f"  已處理 {n_subj} 受試者 / {n_img} 張(miss={n_miss})")
    finally:
        face_mesh.close()

    logger.info(f"完成: {n_subj} 受試者 / {n_img} 張裁切,mediapipe miss={n_miss}")
    logger.info(f"輸出於: {args.output_dir}")


if __name__ == "__main__":
    main()
