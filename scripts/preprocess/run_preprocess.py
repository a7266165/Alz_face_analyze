"""
scripts/preprocess/run_preprocess.py
預處理 pipeline：影像進、影像出。直接串接 src/preprocess 的五個獨立站。

職責邊界（到「鏡射」為止，不碰特徵）：
  一個 subject 的 raw 相片資料夾
    → detect（偵測臉 + landmarks）
    → select（選最正面 n 張）
    → 去背（mask，可選）
    → align（轉正）
    → mirror（左右鏡射，可選）
    → 輸出 selected/ aligned/ mirrors/ 影像資料夾

bg / mirror 是 toggle，一次出齊：
  --backgrounds no_background background   要產哪些背景變體（預設兩者都產）
  --mirror / --no-mirror                   要不要產鏡射（預設產）

同一張臉的 detect / select / tilt 只算一次，no_background 與 background
共用同一個旋轉角；差別只在「轉正前有沒有先去背」。

下游「影像 → 特徵」請見 scripts/embedding/extract_mirror_features.py。

用法：
  conda run -n <preprocess-env> python scripts/preprocess/run_preprocess.py
  python scripts/preprocess/run_preprocess.py --backgrounds no_background --no-mirror
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import RAW_IMAGES_DIR, PREPROCESSING_DIR, preprocess_dir, PreprocessConfig
from src.preprocess import (
    FaceDetector,
    FaceSelector,
    FaceMasker,
    FaceStraightener,
    MirrorGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class PreprocessRunner:
    """raw 影像 → 影像 的預處理 runner（detect / select / 去背 / align / mirror）。"""

    def __init__(
        self,
        raw_images_dir: Path,
        n_select: int = 10,
        backgrounds: Optional[List[str]] = None,
        mirror: bool = True,
        input_groups: Optional[List[str]] = None,
        max_cpu_cores: Optional[int] = None,
    ):
        self.raw_images_dir = Path(raw_images_dir)
        self.input_groups = input_groups
        self.mirror = mirror

        # 背景變體：name → is_background
        names = backgrounds or ["no_background", "background"]
        self.variants: List[Tuple[str, bool]] = [
            (name, name == "background") for name in names
        ]

        self._setup_cpu_limit(max_cpu_cores)

        if not self.raw_images_dir.exists():
            raise FileNotFoundError(f"原始影像目錄不存在: {self.raw_images_dir}")

        # 五站（純元件，無資源者於此建立；detector 持有 mediapipe，於 run() 建立/釋放）
        cfg = PreprocessConfig(n_select=n_select)
        self.config = cfg
        self.selector = FaceSelector(n_select=cfg.n_select)
        self.masker = FaceMasker()
        self.straightener = FaceStraightener(midline_points=cfg.midline_points)
        mc = cfg.mirror
        self.mirror_gen = MirrorGenerator(
            method=mc.mirror_method,
            mirror_size=mc.mirror_size,
            feather_px=mc.feather_px,
            margin=mc.margin,
            midline_points=mc.midline_points,
        )
        self.detector: Optional[FaceDetector] = None

        self.stats = {
            "total_subjects": 0,
            "successful_subjects": 0,
            "failed_subjects": 0,
            "skipped_subjects": 0,
            "total_images": 0,
            "backgrounds": [n for n, _ in self.variants],
            "mirror": self.mirror,
            "start_time": None,
            "end_time": None,
        }

    # ------------------------------------------------------------------
    def _setup_cpu_limit(self, max_cpu_cores: Optional[int]):
        if max_cpu_cores is None:
            logger.info("CPU 核心數: 不限制")
            return
        logger.info(f"CPU 核心數: 限制為 {max_cpu_cores} 核心")
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(max_cpu_cores)
        try:
            cv2.setNumThreads(max_cpu_cores)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def run(self):
        logger.info("=" * 70)
        logger.info("預處理（raw → selected / aligned"
                    f"{' / mirrors' if self.mirror else ''}）")
        logger.info("=" * 70)
        logger.info(f"影像來源: {self.raw_images_dir}")
        logger.info(f"背景變體: {[n for n, _ in self.variants]}")
        logger.info(f"產鏡射: {self.mirror}")

        self.stats["start_time"] = datetime.now()

        subject_dirs = self._scan_subjects()
        self.stats["total_subjects"] = len(subject_dirs)
        logger.info(f"找到 {len(subject_dirs)} 個受試者")
        if not subject_dirs:
            logger.error("沒有找到任何受試者目錄")
            return

        # 斷點續傳：所有要求的變體都已有產出的 subject 跳過
        processed = self._get_processed_subjects()
        if processed:
            logger.info(f"發現 {len(processed)} 個已處理的受試者，將跳過")
            self.stats["skipped_subjects"] = len(processed)
        remaining = [d for d in subject_dirs if d.name not in processed]
        logger.info(f"待處理受試者數: {len(remaining)}")
        if not remaining:
            logger.info("所有受試者已處理完成！")
            self._print_statistics()
            return

        self.detector = FaceDetector(
            detection_confidence=self.config.detection_confidence,
            midline_points=self.config.midline_points,
        )
        try:
            with tqdm(remaining, desc="預處理") as pbar:
                for subject_dir in pbar:
                    subject_id = subject_dir.name
                    pbar.set_description(f"處理 {subject_id}")
                    try:
                        ok = self._process_subject(subject_dir)
                        if ok:
                            self.stats["successful_subjects"] += 1
                            logger.info(f"✓ {subject_id}: 已儲存")
                        else:
                            self.stats["failed_subjects"] += 1
                            logger.warning(f"✗ {subject_id}: 處理失敗")
                    except Exception as e:
                        self.stats["failed_subjects"] += 1
                        logger.error(f"✗ {subject_id}: {e}")
                        import traceback
                        traceback.print_exc()
        finally:
            if self.detector is not None:
                self.detector.close()
                self.detector = None

        self.stats["end_time"] = datetime.now()
        self._print_statistics()
        self._save_statistics()
        logger.info("\n預處理完成！")

    # ------------------------------------------------------------------
    def _process_subject(self, subject_dir: Path) -> bool:
        """raw → detect → select → (每 variant: 去背→align→存 aligned→鏡射→存 mirrors)。"""
        subject_id = subject_dir.name
        images, paths = self._load_images_from_subject(subject_dir)
        if not images:
            logger.warning(f"{subject_id}: 沒有找到影像")
            return False
        self.stats["total_images"] += len(images)

        # Step 1: detect
        face_infos = self.detector.detect_face_batch(images, paths)
        if not face_infos:
            logger.warning(f"{subject_id}: 未偵測到任何臉部")
            return False

        # Step 2: select
        selected = self.selector.select_most_frontal(face_infos)
        if not selected:
            logger.warning(f"{subject_id}: select 後沒有臉部")
            return False
        self._save_selected(subject_id, selected)

        # Step 3-5: 每張臉 × 每個背景變體
        for i, face in enumerate(selected):
            tilt = self.straightener.calculate_midline_tilt(face.landmarks)
            stem = face.path.stem if face.path else None
            for variant_name, is_bg in self.variants:
                # 去背（background 變體保留原背景）
                src = face.image if is_bg else self.masker.apply(
                    face.image, face.landmarks)
                # align（與另一變體共用 tilt）
                aligned = self.straightener.rotate_to_vertical(src, tilt)
                self._save_aligned(subject_id, i, stem, aligned, is_bg)
                # mirror（可選）
                if self.mirror:
                    self._mirror_and_save(
                        subject_id, i, stem, aligned, face.landmarks, is_bg)
        return True

    def _mirror_and_save(self, subject_id, idx, stem, aligned, fallback_lm, is_bg):
        """對齊後重新偵測 landmark（找不到則沿用對齊前的），鏡射並儲存。"""
        redet = self._redetect_landmarks(aligned)
        lm = redet if redet is not None else fallback_lm
        left, right = self.mirror_gen.generate(aligned, lm)
        base = stem if stem else f"face_{idx:03d}"
        save_dir = preprocess_dir("mirrors", background=is_bg) / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{base}_left.png"), left)
        cv2.imwrite(str(save_dir / f"{base}_right.png"), right)

    def _redetect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        infos = self.detector.detect_face_batch([image])
        return infos[0].landmarks if infos else None

    # ------------------------------------------------------------------
    def _save_selected(self, subject_id, faces):
        """選中的（原始、未去背未轉正）影像，存到 no_background/selected。"""
        save_dir = preprocess_dir("selected") / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, face in enumerate(faces):
            fn = f"selected_{i:03d}_vas_{face.vertex_angle_sum:.1f}.png"
            cv2.imwrite(str(save_dir / fn), face.image)

    def _save_aligned(self, subject_id, idx, stem, aligned, is_bg):
        save_dir = preprocess_dir("aligned", background=is_bg) / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        fn = f"{stem}_aligned.png" if stem else f"aligned_{idx:03d}.png"
        cv2.imwrite(str(save_dir / fn), aligned)

    # ------------------------------------------------------------------
    def _get_processed_subjects(self) -> Set[str]:
        """所有要求變體都有產出才算完成（取交集）。

        產鏡射時看 mirrors/，否則看 aligned/。
        """
        stage = "mirrors" if self.mirror else "aligned"
        result: Optional[Set[str]] = None
        for _, is_bg in self.variants:
            d = preprocess_dir(stage, background=is_bg)
            if not d.exists():
                return set()
            if stage == "mirrors":
                subs = {p.name for p in d.iterdir()
                        if p.is_dir() and any(p.glob("*_left.png"))}
            else:
                subs = {p.name for p in d.iterdir()
                        if p.is_dir() and any(p.glob("*.png"))}
            result = subs if result is None else (result & subs)
        return result or set()

    def _scan_subjects(self) -> List[Path]:
        subject_dirs = []
        if self.input_groups is not None:
            # 自訂 group：每個 group 是 raw_images_dir 底下直接子目錄，內含 subject folder
            for group_name in self.input_groups:
                group_path = self.raw_images_dir / group_name
                if group_path.exists():
                    for d in sorted(group_path.iterdir()):
                        if d.is_dir() and self._has_images(d):
                            subject_dirs.append(d)
                else:
                    logger.warning(f"找不到目錄: {group_path}")
            return subject_dirs

        # 預設：內部 cohort 結構 health/ACS、health/NAD、patient/good
        for group_name in ["ACS", "NAD"]:
            group_path = self.raw_images_dir / "health" / group_name
            if group_path.exists():
                for d in sorted(group_path.iterdir()):
                    if d.is_dir() and self._has_images(d):
                        subject_dirs.append(d)
            else:
                logger.warning(f"找不到目錄: {group_path}")
        patient_path = self.raw_images_dir / "patient" / "good"
        if patient_path.exists():
            for d in sorted(patient_path.iterdir()):
                if d.is_dir() and self._has_images(d):
                    subject_dirs.append(d)
        else:
            logger.warning(f"找不到目錄: {patient_path}")
        return subject_dirs

    def _has_images(self, directory: Path) -> bool:
        return any(f.is_file() and f.suffix.lower() in VALID_IMAGE_SUFFIXES
                   for f in directory.iterdir())

    def _load_images_from_subject(self, subject_dir: Path):
        images, paths = [], []
        for file_path in sorted(subject_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                img = cv2.imread(str(file_path))
                if img is not None:
                    images.append(img)
                    paths.append(file_path)
        return images, paths

    # ------------------------------------------------------------------
    def _print_statistics(self):
        s = self.stats
        logger.info("\n" + "=" * 70)
        logger.info("預處理統計")
        logger.info("=" * 70)
        logger.info(f"總受試者數: {s['total_subjects']}")
        logger.info(f"已跳過（斷點）: {s['skipped_subjects']}")
        logger.info(f"成功處理: {s['successful_subjects']}")
        logger.info(f"處理失敗: {s['failed_subjects']}")
        logger.info(f"總影像數: {s['total_images']}")
        if s["start_time"] and s["end_time"]:
            logger.info(f"\n總耗時: {s['end_time'] - s['start_time']}")

    def _save_statistics(self):
        stats_file = PREPROCESSING_DIR / "preprocess_stats.json"
        out = self.stats.copy()
        for k in ("start_time", "end_time"):
            if out[k]:
                out[k] = out[k].isoformat()
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logger.info(f"統計資訊已儲存: {stats_file}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", type=Path, default=None,
                    help="覆寫 RAW_IMAGES_DIR；留空沿用 data/path.txt 設定")
    ap.add_argument("--input-groups", nargs="+", default=None,
                    help="自訂 group 子目錄名；留空沿用內部預設 health/ACS, health/NAD, patient/good")
    ap.add_argument("--n-select", type=int, default=10)
    ap.add_argument("--max-cpu-cores", type=int, default=2)
    ap.add_argument("--backgrounds", nargs="+",
                    choices=["no_background", "background"],
                    default=["no_background", "background"],
                    help="要產哪些背景變體（預設兩者都產）")
    ap.add_argument("--no-mirror", action="store_true",
                    help="不產鏡射（只到 aligned 為止）")
    args = ap.parse_args()

    raw_dir = args.input_root if args.input_root is not None else RAW_IMAGES_DIR
    try:
        runner = PreprocessRunner(
            raw_images_dir=raw_dir,
            n_select=args.n_select,
            backgrounds=args.backgrounds,
            mirror=not args.no_mirror,
            input_groups=args.input_groups,
            max_cpu_cores=args.max_cpu_cores,
        )
        runner.run()
    except Exception as e:
        logger.error(f"預處理執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
