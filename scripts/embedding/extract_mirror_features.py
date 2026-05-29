"""
scripts/embedding/extract_mirror_features.py
影像 → 特徵：讀預處理產出的鏡射影像，抽 embedding 並算左右不對稱特徵。

職責（preprocess 的下游）：
  mirrors/<subject>/*_left.png, *_right.png
    → 各 embedding 模型抽特徵（FeatureExtractor）
    → 左右配對算差異/平均/相對等（calculate_differences）
    → 存 EMBEDDING_FEATURES_DIR/<model>/<bg_variant>/<ftype>/<subject>.npy

鏡射影像由 scripts/preprocess/run_preprocess.py 產出。本步驟不需 mediapipe。

用法：
  conda run -n <embedding-env> python scripts/embedding/extract_mirror_features.py
  python scripts/embedding/extract_mirror_features.py --bg-variant background
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    EMBEDDING_FEATURES_DIR,
    MIRRORS_DIR as _MIRRORS_DIR,
    MIRRORS_BACKGROUND_DIR as _MIRRORS_BACKGROUND_DIR,
)
from src.embedding import FeatureExtractor
from src.asymmetry import calculate_differences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FTYPE_TO_METHOD = {
    "difference": "differences",
    "absolute_difference": "absolute_differences",
    "average": "averages",
    "relative_differences": "relative_differences",
    "absolute_relative_differences": "absolute_relative_differences",
}


class MirrorFeaturePipeline:
    """鏡射影像 → embedding 不對稱特徵。"""

    def __init__(
        self,
        output_dir: Path,
        embedding_models: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        bg_variant: str = "no_background",
        max_cpu_cores: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.bg_variant = bg_variant
        self.mirrors_dir = (_MIRRORS_BACKGROUND_DIR if bg_variant == "background"
                            else _MIRRORS_DIR)
        self.embedding_models = embedding_models or ["arcface", "dlib", "topofr"]
        self.feature_types = feature_types or list(FTYPE_TO_METHOD)

        self._setup_cpu_limit(max_cpu_cores)
        if not self.mirrors_dir.exists():
            raise FileNotFoundError(
                f"鏡射影像目錄不存在: {self.mirrors_dir}（先跑 run_preprocess.py）")
        self._setup_output_dirs()

        self.stats = {
            "total_subjects": 0,
            "successful_subjects": 0,
            "failed_subjects": 0,
            "skipped_subjects": 0,
            "total_images": 0,
            "models_extracted": {},
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
        try:
            import torch
            torch.set_num_threads(max_cpu_cores)
            torch.set_num_interop_threads(max_cpu_cores)
        except Exception:
            pass

    def _setup_output_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for model in self.embedding_models:
            for ftype in self.feature_types:
                (self.output_dir / model / self.bg_variant / ftype).mkdir(
                    parents=True, exist_ok=True)

    def _get_processed_subjects(self) -> Set[str]:
        """所有 model × ftype 都有 .npy 才算完成（取交集）。"""
        subject_sets = []
        for model in self.embedding_models:
            for ftype in self.feature_types:
                d = self.output_dir / model / self.bg_variant / ftype
                subject_sets.append({f.stem for f in d.glob("*.npy")}
                                    if d.exists() else set())
        if not subject_sets:
            return set()
        processed = subject_sets[0]
        for s in subject_sets[1:]:
            processed = processed & s
        return processed

    # ------------------------------------------------------------------
    def run(self):
        logger.info("=" * 70)
        logger.info(f"鏡射影像 → 不對稱特徵（{self.bg_variant}）")
        logger.info("=" * 70)
        logger.info(f"影像來源: {self.mirrors_dir}")
        logger.info(f"輸出目錄: {self.output_dir}")
        logger.info(f"嵌入模型: {self.embedding_models}")
        logger.info(f"特徵類型: {self.feature_types}")

        self.stats["start_time"] = datetime.now()

        subject_dirs = sorted(d for d in self.mirrors_dir.iterdir() if d.is_dir())
        self.stats["total_subjects"] = len(subject_dirs)
        logger.info(f"找到 {len(subject_dirs)} 個受試者")
        if not subject_dirs:
            logger.error("沒有找到任何鏡射受試者目錄")
            return

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

        logger.info("初始化特徵提取器...")
        extractor = FeatureExtractor()

        with tqdm(remaining, desc="抽特徵") as pbar:
            for subject_dir in pbar:
                subject_id = subject_dir.name
                pbar.set_description(f"處理 {subject_id}")
                try:
                    features = self._process_subject(subject_id, extractor)
                    if features:
                        self._save_subject_features(subject_id, features)
                        self.stats["successful_subjects"] += 1
                        logger.info(f"✓ {subject_id}: 特徵已儲存")
                    else:
                        self.stats["failed_subjects"] += 1
                        logger.warning(f"✗ {subject_id}: 處理失敗")
                except Exception as e:
                    self.stats["failed_subjects"] += 1
                    logger.error(f"✗ {subject_id}: {e}")
                    import traceback
                    traceback.print_exc()

        self.stats["end_time"] = datetime.now()
        self._print_statistics()
        self._save_statistics()
        logger.info("\n特徵提取完成！")

    # ------------------------------------------------------------------
    def _process_subject(self, subject_id: str,
                         extractor: FeatureExtractor) -> Optional[Dict]:
        mirror_dir = self.mirrors_dir / subject_id
        left_files = sorted(mirror_dir.glob("*_left.png"))
        right_files = sorted(mirror_dir.glob("*_right.png"))
        if not left_files or len(left_files) != len(right_files):
            logger.warning(
                f"{subject_id}: mirror 檔案不完整 "
                f"(left={len(left_files)}, right={len(right_files)})")
            return None

        left_images = [im for im in (cv2.imread(str(f)) for f in left_files)
                       if im is not None]
        right_images = [im for im in (cv2.imread(str(f)) for f in right_files)
                        if im is not None]
        if not left_images or len(left_images) != len(right_images):
            logger.warning(f"{subject_id}: 無法讀取完整 mirror 影像")
            return None
        self.stats["total_images"] += len(left_images) + len(right_images)

        return self._extract_and_compute_asymmetry(
            subject_id, left_images, right_images, extractor)

    def _extract_and_compute_asymmetry(
        self, subject_id, left_images, right_images, extractor,
    ) -> Optional[Dict]:
        subject_features = {}
        for model in self.embedding_models:
            if model not in extractor.available_models:
                continue
            model_features = {}
            try:
                left_dict = extractor.extract_features(left_images, model)
                right_dict = extractor.extract_features(right_images, model)
                if model not in left_dict or model not in right_dict:
                    logger.warning(f"{subject_id}: {model} 特徵提取失敗")
                    continue
                valid_pairs = [
                    (l, r) for l, r in zip(left_dict[model], right_dict[model])
                    if l is not None and r is not None
                ]
                if not valid_pairs:
                    logger.warning(f"{subject_id}: {model} 沒有有效特徵")
                    continue
                left_array = np.array([p[0] for p in valid_pairs])
                right_array = np.array([p[1] for p in valid_pairs])
                for ftype in self.feature_types:
                    if ftype not in FTYPE_TO_METHOD:
                        logger.warning(f"未知的特徵類型: {ftype}")
                        continue
                    result = calculate_differences(
                        left_array, right_array,
                        methods=[FTYPE_TO_METHOD[ftype]])
                    model_features[ftype] = result
                if model_features:
                    subject_features[model] = model_features
                    self.stats["models_extracted"][model] = (
                        self.stats["models_extracted"].get(model, 0) + 1)
            except Exception as e:
                logger.warning(f"{subject_id}: {model} 特徵提取失敗 - {e}")
                continue
        return subject_features or None

    def _save_subject_features(self, subject_id: str, features: Dict):
        for model in self.embedding_models:
            if model not in features:
                continue
            for ftype in self.feature_types:
                if ftype not in features[model]:
                    continue
                npy_path = (self.output_dir / model / self.bg_variant / ftype
                            / f"{subject_id}.npy")
                np.save(npy_path, features[model][ftype])

    # ------------------------------------------------------------------
    def _print_statistics(self):
        s = self.stats
        logger.info("\n" + "=" * 70)
        logger.info("特徵提取統計")
        logger.info("=" * 70)
        logger.info(f"總受試者數: {s['total_subjects']}")
        logger.info(f"已跳過（斷點）: {s['skipped_subjects']}")
        logger.info(f"成功處理: {s['successful_subjects']}")
        logger.info(f"處理失敗: {s['failed_subjects']}")
        logger.info(f"總影像數: {s['total_images']}")
        logger.info("\n各模型提取統計:")
        for model, count in s["models_extracted"].items():
            logger.info(f"  {model}: {count} 個受試者")
        if s["start_time"] and s["end_time"]:
            logger.info(f"\n總耗時: {s['end_time'] - s['start_time']}")

    def _save_statistics(self):
        stats_file = self.output_dir / "feature_extraction_stats.json"
        out = self.stats.copy()
        for k in ("start_time", "end_time"):
            if out[k]:
                out[k] = out[k].isoformat()
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logger.info(f"統計資訊已儲存: {stats_file}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="覆寫 EMBEDDING_FEATURES_DIR；留空用預設")
    ap.add_argument("--bg-variant", choices=["no_background", "background"],
                    default="no_background")
    ap.add_argument("--max-cpu-cores", type=int, default=2)
    ap.add_argument("--embedding-models", nargs="+", default=None,
                    help="覆寫預設嵌入模型列表")
    args = ap.parse_args()

    out_dir = args.output_dir if args.output_dir is not None else EMBEDDING_FEATURES_DIR
    try:
        pipeline = MirrorFeaturePipeline(
            output_dir=out_dir,
            embedding_models=args.embedding_models,
            bg_variant=args.bg_variant,
            max_cpu_cores=args.max_cpu_cores,
        )
        pipeline.run()
    except Exception as e:
        logger.error(f"特徵提取執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
