"""
掃描母資料夾所有個案的人臉相片資料夾，依序進行：
1. 臉部偵測
2. 選擇最正面相片
3. 去背（可選）
4. 轉正
5. 鏡射（可選）

並將中間過程存至workspace
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
    """斷點檢查"""
    sel = preprocess_dir("selected") / subject_id
    if not sel.is_dir():
        return False
    k = sum(1 for _ in sel.glob("*.png"))
    if k == 0:
        return False
    for _, is_bg in variants:
        al = preprocess_dir("aligned", background=is_bg) / subject_id
        if sum(1 for _ in al.glob("*.png")) != k:
            return False
        if mirror:
            mr = preprocess_dir("mirrors", background=is_bg) / subject_id
            if sum(1 for _ in mr.glob("*_left.png")) != k:
                return False
    return True


def process_subject(subject_dir: Path, face_mesh, cfg: PreprocessConfig,
                    variants: List[Tuple[str, bool]], mirror: bool) -> bool:
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

    # 合併兩迴圈 already_done 會失效
    sel_dir = preprocess_dir("selected") / subject_id
    sel_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(selected):
        cv2.imwrite(str(sel_dir / f"selected_{i:03d}_vas_{f.vertex_angle_sum:.1f}.png"),
                    f.image)

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
                    lm = face.landmarks
                else:
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
                    help=f"要掃描的子樹（相對 input-root），取其直接子目錄為 subject；留空用預設 {DEFAULT_SUBTREES}")
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

    # 不用rglob，結構目錄默認 raw_root/DEFAULT_SUBTREES/subject_ID/figs.jpg，
    subjects: List[Path] = []
    for sub in (args.subtrees or DEFAULT_SUBTREES):
        root = raw_root / sub
        if not root.exists():
            logger.warning(f"找不到子樹，略過: {root}")
            continue
        for d in sorted(root.iterdir()):
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
