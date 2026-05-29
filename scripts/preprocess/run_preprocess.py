"""
scripts/preprocess/run_preprocess.py
預處理 pipeline：影像進、影像出。

職責邊界（到「鏡射」為止，不碰特徵）：
  一個 subject 的 raw 相片資料夾
    → 偵測臉 → 選最正面 n 張 → 轉正 → 鏡射
    → 輸出 aligned/ 與 mirrors/ 影像資料夾（含 background 變體）

兩種模式：
  預設（raw）       raw 影像 → aligned/(+aligned_background/) + mirrors/
  --from-aligned    從已對齊影像開始，只補做鏡射（aligned/ → mirrors/）

下游「影像 → 特徵」請見 scripts/embedding/extract_mirror_features.py。

用法：
  conda run -n <preprocess-env> python scripts/preprocess/run_preprocess.py
  python scripts/preprocess/run_preprocess.py --from-aligned --bg-variant background
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import RAW_IMAGES_DIR, preprocess_dir, AnalyzeConfig
from src.preprocess import (
    PreprocessPipeline,
    ProcessedFace,
    MirrorGenerator,
    FaceDetector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PreprocessRunner:
    """影像 → 影像 的預處理 runner（偵測 / 選照 / 對齊 / 鏡射）。"""

    def __init__(
        self,
        raw_images_dir: Path,
        n_select: int = 10,
        save_aligned_background: bool = True,
        bg_variant: str = "no_background",
        input_groups: Optional[List[str]] = None,
        max_cpu_cores: Optional[int] = None,
        from_aligned: bool = False,
    ):
        self.raw_images_dir = Path(raw_images_dir)
        self.bg_variant = bg_variant
        self.input_groups = input_groups
        self.from_aligned = from_aligned

        # 依 bg_variant 決定對齊 / 鏡射的讀寫路徑
        bg = (bg_variant == "background")
        self.aligned_dir = preprocess_dir("aligned", background=bg)
        self.mirrors_dir = preprocess_dir("mirrors", background=bg)

        self._setup_cpu_limit(max_cpu_cores)

        if from_aligned:
            if not self.aligned_dir.exists():
                raise FileNotFoundError(f"對齊影像目錄不存在: {self.aligned_dir}")
        else:
            if not self.raw_images_dir.exists():
                raise FileNotFoundError(f"原始影像目錄不存在: {self.raw_images_dir}")

        self.preprocess_config = AnalyzeConfig(
            n_select=n_select,
            save_intermediate=True,
            also_save_aligned_background=save_aligned_background,
        )
        mc = self.preprocess_config.mirror
        self.mirror_gen = MirrorGenerator(
            method=mc.mirror_method,
            mirror_size=mc.mirror_size,
            feather_px=mc.feather_px,
            margin=mc.margin,
            midline_points=mc.midline_points,
        )

        self.stats = {
            "total_subjects": 0,
            "successful_subjects": 0,
            "failed_subjects": 0,
            "skipped_subjects": 0,
            "total_images": 0,
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
        logger.info(
            f"預處理（{'aligned→mirror' if self.from_aligned else 'raw→aligned+mirror'}"
            f"，{self.bg_variant}）")
        logger.info("=" * 70)
        src_dir = self.aligned_dir if self.from_aligned else self.raw_images_dir
        logger.info(f"影像來源: {src_dir}")
        logger.info(f"鏡射輸出: {self.mirrors_dir}")

        self.stats["start_time"] = datetime.now()

        # 掃描 subject
        if self.from_aligned:
            subject_dirs = sorted(
                d for d in self.aligned_dir.iterdir() if d.is_dir())
        else:
            subject_dirs = self._scan_subjects()
        self.stats["total_subjects"] = len(subject_dirs)
        logger.info(f"找到 {len(subject_dirs)} 個受試者")
        if not subject_dirs:
            logger.error("沒有找到任何受試者目錄")
            return

        # 斷點續傳：已有 mirror 產出的 subject 跳過
        processed = self._get_processed_subjects()
        if processed:
            logger.info(f"發現 {len(processed)} 個已處理（已有 mirrors）的受試者，將跳過")
            self.stats["skipped_subjects"] = len(processed)
        remaining = [d for d in subject_dirs if d.name not in processed]
        logger.info(f"待處理受試者數: {len(remaining)}")
        if not remaining:
            logger.info("所有受試者已處理完成！")
            self._print_statistics()
            return

        detector = FaceDetector() if self.from_aligned else None
        try:
            with tqdm(remaining, desc="預處理") as pbar:
                for subject_dir in pbar:
                    subject_id = subject_dir.name
                    pbar.set_description(f"處理 {subject_id}")
                    try:
                        if self.from_aligned:
                            ok = self._mirror_from_aligned(subject_dir, detector)
                        else:
                            ok = self._process_subject(subject_dir)
                        if ok:
                            self.stats["successful_subjects"] += 1
                            logger.info(f"✓ {subject_id}: mirrors 已儲存")
                        else:
                            self.stats["failed_subjects"] += 1
                            logger.warning(f"✗ {subject_id}: 處理失敗")
                    except Exception as e:
                        self.stats["failed_subjects"] += 1
                        logger.error(f"✗ {subject_id}: {e}")
                        import traceback
                        traceback.print_exc()
        finally:
            if detector is not None:
                detector.close()

        self.stats["end_time"] = datetime.now()
        self._print_statistics()
        self._save_statistics()
        logger.info("\n預處理完成！")

    # ------------------------------------------------------------------
    def _process_subject(self, subject_dir: Path) -> bool:
        """raw → 偵測/選照/對齊（PreprocessPipeline 內含存 aligned）→ 鏡射 → 存 mirrors。"""
        subject_id = subject_dir.name
        images, paths = self._load_images_from_subject(subject_dir)
        if not images:
            logger.warning(f"{subject_id}: 沒有找到影像")
            return False
        self.stats["total_images"] += len(images)

        subject_config = AnalyzeConfig(
            n_select=self.preprocess_config.n_select,
            save_intermediate=self.preprocess_config.save_intermediate,
            also_save_aligned_background=self.preprocess_config.also_save_aligned_background,
            subject_id=subject_id,
        )
        with PreprocessPipeline(subject_config) as preprocessor:
            processed_faces: List[ProcessedFace] = preprocessor.process(images, paths)
        if not processed_faces:
            logger.warning(f"{subject_id}: 預處理後沒有有效臉部")
            return False

        for i, face in enumerate(processed_faces):
            left, right = self.mirror_gen.generate(face.aligned, face.landmarks)
            base = face.metadata.get("path")
            base = Path(base).stem if base else f"face_{i:03d}"
            self._save_mirror_pair(subject_id, base, left, right)
        return True

    def _mirror_from_aligned(self, subject_dir: Path,
                             detector: FaceDetector) -> bool:
        """aligned → 偵測 landmark（重用 FaceDetector）→ 鏡射 → 存 mirrors。"""
        subject_id = subject_dir.name
        images, paths = self._load_images_from_subject(subject_dir)
        if not images:
            logger.warning(f"{subject_id}: 沒有找到影像")
            return False
        self.stats["total_images"] += len(images)

        face_infos = detector.detect_face_batch(images, paths)
        if not face_infos:
            logger.warning(f"{subject_id}: 未偵測到臉部")
            return False
        for fi in face_infos:
            left, right = self.mirror_gen.generate(fi.image, fi.landmarks)
            base = fi.path.stem if fi.path else f"face_{fi.index:03d}"
            self._save_mirror_pair(subject_id, base, left, right)
        return True

    def _save_mirror_pair(self, subject_id, base_name, left, right):
        save_dir = self.mirrors_dir / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{base_name}_left.png"), left)
        cv2.imwrite(str(save_dir / f"{base_name}_right.png"), right)

    # ------------------------------------------------------------------
    def _get_processed_subjects(self) -> Set[str]:
        if not self.mirrors_dir.exists():
            return set()
        return {d.name for d in self.mirrors_dir.iterdir()
                if d.is_dir() and any(d.glob("*_left.png"))}

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
        valid = {".jpg", ".jpeg", ".png"}
        return any(f.is_file() and f.suffix.lower() in valid
                   for f in directory.iterdir())

    def _load_images_from_subject(self, subject_dir: Path):
        images, paths = [], []
        valid = {".jpg", ".jpeg", ".png"}
        for file_path in sorted(subject_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in valid:
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
        stats_file = self.mirrors_dir / "preprocess_stats.json"
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
    ap.add_argument("--no-aligned-background", action="store_true",
                    help="不產出 aligned_background/（debug 用；預設會產）")
    ap.add_argument("--from-aligned", action="store_true",
                    help="從已對齊影像開始，只補做鏡射（aligned/ → mirrors/）")
    ap.add_argument("--bg-variant", choices=["no_background", "background"],
                    default="no_background", help="背景變體（控制讀寫路徑）")
    args = ap.parse_args()

    raw_dir = args.input_root if args.input_root is not None else RAW_IMAGES_DIR
    try:
        runner = PreprocessRunner(
            raw_images_dir=raw_dir,
            n_select=args.n_select,
            save_aligned_background=not args.no_aligned_background,
            bg_variant=args.bg_variant,
            input_groups=args.input_groups,
            max_cpu_cores=args.max_cpu_cores,
            from_aligned=args.from_aligned,
        )
        runner.run()
    except Exception as e:
        logger.error(f"預處理執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
