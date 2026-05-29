"""
scripts/preprocess/run_preprocess.py
預處理 pipeline：raw 影像資料夾 → selected / aligned / mirrors。

遞迴走訪指定子樹，每個「含影像」的資料夾當一個 subject，依序跑五站：
    detect（偵測臉 + landmarks）
      → select（選最正面 n 張）
      → 去背（mask，可選）
      → align（轉正）
      → mirror（左右鏡射，可選）

bg / mirror 是 toggle，一次出齊：
    --backgrounds no_background background   要產哪些背景變體（預設兩者都產）
    --no-mirror                              不產鏡射（只到 aligned 為止）

mediapipe 的 FaceMesh 由 open_face_mesh() 在 main 開一次、with 區塊結束自動釋放；
五站皆為 pure function，直接 import 呼叫，不必當物件穿過來穿過去。

下游「影像 → 特徵」見 scripts/embedding/extract_mirror_features.py。

用法：
    "C:/Users/4080/anaconda3/envs/Alz_face_rotation/python.exe" scripts/preprocess/run_preprocess.py
    python scripts/preprocess/run_preprocess.py --backgrounds no_background --no-mirror
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import RAW_IMAGES_DIR, preprocess_dir, PreprocessConfig
from src.preprocess import (
    open_face_mesh,
    detect_faces,
    select_most_frontal,
    apply_mask,
    calculate_midline_tilt,
    rotate_to_vertical,
    generate_mirrors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUFFIXES = {".jpg", ".jpeg", ".png"}
# 預設掃描的三塊子樹（過濾掉 patient/bad、health/NAD資料檢查、NAD_to_P 等）
DEFAULT_SUBTREES = ["health/ACS", "health/NAD", "patient/good"]


def limit_cpu(n_cores):
    if n_cores is None:
        return
    logger.info(f"CPU 核心數: 限制為 {n_cores}")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_cores)
    try:
        cv2.setNumThreads(n_cores)
    except Exception:
        pass


def already_done(subject_id: str, variants: List[Tuple[str, bool]],
                 mirror: bool) -> bool:
    """所有要求的背景變體都已有產出 → 跳過（斷點續傳）。

    產鏡射時看 mirrors/，否則看 aligned/。
    """
    stage = "mirrors" if mirror else "aligned"
    pattern = "*_left.png" if mirror else "*.png"
    for _, is_bg in variants:
        d = preprocess_dir(stage, background=is_bg) / subject_id
        if not (d.is_dir() and any(d.glob(pattern))):
            return False
    return True


def process_subject(subject_dir: Path, face_mesh, cfg: PreprocessConfig,
                    variants: List[Tuple[str, bool]], mirror: bool) -> bool:
    """raw 資料夾 → detect → select → (每 variant: 去背→align→存；可選鏡射)。"""
    subject_id = subject_dir.name

    images, paths = [], []
    for p in sorted(subject_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUFFIXES:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(img)
                paths.append(p)
    if not images:
        logger.warning(f"{subject_id}: 沒有可讀的影像")
        return False

    faces = detect_faces(face_mesh, images, paths, cfg.midline_points)
    if not faces:
        logger.warning(f"{subject_id}: 未偵測到臉部")
        return False
    selected = select_most_frontal(faces, cfg.n_select)
    if not selected:
        logger.warning(f"{subject_id}: select 後沒有臉部")
        return False

    # 存 selected（原始、未去背未轉正）→ no_background/selected
    sel_dir = preprocess_dir("selected") / subject_id
    sel_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(selected):
        cv2.imwrite(str(sel_dir / f"selected_{i:03d}_vas_{f.vertex_angle_sum:.1f}.png"),
                    f.image)

    # 每張臉算一次 tilt；no_bg 與 bg 共用同一旋轉角，差別只在轉正前是否去背
    for i, face in enumerate(selected):
        tilt = calculate_midline_tilt(face.landmarks, cfg.midline_points)
        stem = face.path.stem if face.path else None
        for _, is_bg in variants:
            src = face.image if is_bg else apply_mask(face.image, face.landmarks)
            aligned = rotate_to_vertical(src, tilt)

            al_dir = preprocess_dir("aligned", background=is_bg) / subject_id
            al_dir.mkdir(parents=True, exist_ok=True)
            al_name = f"{stem}_aligned.png" if stem else f"aligned_{i:03d}.png"
            cv2.imwrite(str(al_dir / al_name), aligned)

            if mirror:
                if cfg.mirror.mirror_method == "flip":
                    lm = face.landmarks  # flip 忽略 landmarks，免一次 re-detect
                else:
                    # midline：對齊後重新偵測，找不到才退回對齊前 landmark
                    redet = detect_faces(face_mesh, [aligned],
                                         midline_points=cfg.midline_points)
                    lm = redet[0].landmarks if redet else face.landmarks
                left, right = generate_mirrors(aligned, lm, cfg.mirror)
                mr_dir = preprocess_dir("mirrors", background=is_bg) / subject_id
                mr_dir.mkdir(parents=True, exist_ok=True)
                base = stem if stem else f"face_{i:03d}"
                cv2.imwrite(str(mr_dir / f"{base}_left.png"), left)
                cv2.imwrite(str(mr_dir / f"{base}_right.png"), right)
    return True


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", type=Path, default=None,
                    help="覆寫 RAW_IMAGES_DIR；留空沿用 data/path.txt 設定")
    ap.add_argument("--subtrees", nargs="+", default=None,
                    help=f"要遞迴走訪的子樹（相對 input-root）；留空用預設 {DEFAULT_SUBTREES}")
    ap.add_argument("--n-select", type=int, default=10)
    ap.add_argument("--max-cpu-cores", type=int, default=2)
    ap.add_argument("--backgrounds", nargs="+",
                    choices=["no_background", "background"],
                    default=["no_background", "background"],
                    help="要產哪些背景變體（預設兩者都產）")
    ap.add_argument("--no-mirror", action="store_true",
                    help="不產鏡射（只到 aligned 為止）")
    args = ap.parse_args()

    limit_cpu(args.max_cpu_cores)

    raw_root = args.input_root if args.input_root is not None else RAW_IMAGES_DIR
    variants = [(name, name == "background") for name in args.backgrounds]
    mirror = not args.no_mirror

    # 遞迴走訪各子樹，收集所有「含影像」的資料夾
    subjects: List[Path] = []
    for sub in (args.subtrees or DEFAULT_SUBTREES):
        root = raw_root / sub
        if not root.exists():
            logger.warning(f"找不到子樹，略過: {root}")
            continue
        for d in sorted(root.rglob("*")):
            if d.is_dir() and any(f.suffix.lower() in SUFFIXES
                                  for f in d.iterdir() if f.is_file()):
                subjects.append(d)

    logger.info("=" * 70)
    logger.info(f"預處理：raw → selected / aligned{' / mirrors' if mirror else ''}")
    logger.info(f"影像來源: {raw_root}  子樹: {args.subtrees or DEFAULT_SUBTREES}")
    logger.info(f"背景變體: {[n for n, _ in variants]}  產鏡射: {mirror}")
    logger.info(f"找到 {len(subjects)} 個受試者")
    logger.info("=" * 70)
    if not subjects:
        logger.error("沒有找到任何受試者目錄")
        return

    cfg = PreprocessConfig(n_select=args.n_select)
    start = datetime.now()
    n_ok = n_fail = n_skip = 0
    with open_face_mesh(cfg.detection_confidence) as fm:
        for subject_dir in tqdm(subjects, desc="預處理"):
            subject_id = subject_dir.name
            if already_done(subject_id, variants, mirror):
                n_skip += 1
                continue
            try:
                if process_subject(subject_dir, fm, cfg, variants, mirror):
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                n_fail += 1
                logger.error(f"✗ {subject_id}: {e}")
                import traceback
                traceback.print_exc()

    logger.info("=" * 70)
    logger.info(f"完成：成功 {n_ok}、失敗 {n_fail}、跳過(斷點) {n_skip}"
                f"、共 {len(subjects)}；耗時 {datetime.now() - start}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
