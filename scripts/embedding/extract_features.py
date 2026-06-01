"""
scripts/embedding/extract_features.py
影像 → embedding 特徵：統一萃取入口，依 --source 切換兩種來源。

  --source original : aligned/<subj>/*.{jpg,png}  → 每張圖的原始 embedding
        存 {model}/{bg_variant}/original/<subj>.npy   形狀 (n_images, dim)

  --source mirror   : mirrors/<subj>/*_left.png, *_right.png  → 左右配對不對稱特徵
        存 {model}/{bg_variant}/<ftype>/<subj>.npy    ftype = 5 種，裸 (n_pairs, dim) 陣列

兩種來源共用：CPU 限制、subject 掃描、斷點續傳（model × ftype 取交集）、tqdm、stats。
輸入影像由 scripts/preprocess/run_preprocess.py 產出；本步驟不需 mediapipe。

用法：
  conda run -n <embedding-env> python scripts/embedding/extract_features.py --source original
  python scripts/embedding/extract_features.py --source mirror --bg-variant background
"""
import os
import sys
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import EMBEDDING_FEATURES_DIR, preprocess_dir
from src.embedding import get_extractor
from src.asymmetry import calculate_differences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["arcface", "dlib", "topofr"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# mirror ftype → calculate_differences 的 method 名
FTYPE_TO_METHOD = {
    "difference": "differences",
    "absolute_difference": "absolute_differences",
    "average": "averages",
    "relative_differences": "relative_differences",
    "absolute_relative_differences": "absolute_relative_differences",
}


def setup_cpu_limit(max_cores: Optional[int]):
    """限制 CPU 核心數；max_cores 為 None 表示不限制。"""
    if max_cores is None:
        logger.info("CPU 核心數: 不限制")
        return
    logger.info(f"CPU 核心數: 限制為 {max_cores} 核心")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(max_cores)
    try:
        cv2.setNumThreads(max_cores)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(max_cores)
        torch.set_num_interop_threads(max_cores)
    except Exception:
        pass


def _load_images(subject_dir: Path) -> List[np.ndarray]:
    """載入受試者目錄下所有影像（BGR）。"""
    images = []
    for fp in sorted(subject_dir.iterdir()):
        if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(str(fp))
            if img is not None:
                images.append(img)
    return images


# =============================================================================
# Feature sources：唯一因來源而異的部分（輸入形狀 + 後處理）
# =============================================================================
class FeatureSource(ABC):
    """一種特徵來源：負責「讀一個 subject 的 payload」與「把 payload 經 extractor 算成 {ftype: 落地值}」。"""

    name: str
    ftypes: List[str]

    def input_dir(self, background: bool) -> Path:
        raise NotImplementedError

    @abstractmethod
    def load(self, subject_dir: Path) -> Optional[object]:
        """讀一個 subject 的影像；無法使用回 None（每個 subject 只讀一次，跨模型共用）。"""

    @abstractmethod
    def compute(self, payload: object, extractor) -> Dict[str, object]:
        """payload + 單一模型 extractor → {ftype: 要存進 .npy 的值}；無有效特徵回空 dict。"""


class OriginalSource(FeatureSource):
    """aligned 影像 → 每張圖的原始 embedding，堆疊成 (n_images, dim)。"""

    name = "original"
    ftypes = ["original"]

    def input_dir(self, background: bool) -> Path:
        return preprocess_dir("aligned", background=background)

    def load(self, subject_dir: Path) -> Optional[List[np.ndarray]]:
        images = _load_images(subject_dir)
        if not images:
            logger.warning(f"{subject_dir.name}: 沒有影像")
            return None
        return images

    def compute(self, payload, extractor) -> Dict[str, object]:
        valid = [f for f in extractor.extract_batch(payload) if f is not None]
        return {"original": np.array(valid)} if valid else {}


class MirrorSource(FeatureSource):
    """mirrors 左右影像 → 配對算不對稱特徵（5 種），存裸 (n_pairs, dim) 陣列。"""

    name = "mirror"
    ftypes = list(FTYPE_TO_METHOD)

    def input_dir(self, background: bool) -> Path:
        return preprocess_dir("mirrors", background=background)

    def load(self, subject_dir: Path):
        left_files = sorted(subject_dir.glob("*_left.png"))
        right_files = sorted(subject_dir.glob("*_right.png"))
        if not left_files or len(left_files) != len(right_files):
            logger.warning(
                f"{subject_dir.name}: mirror 檔案不完整 "
                f"(left={len(left_files)}, right={len(right_files)})")
            return None
        left_images = [im for im in (cv2.imread(str(f)) for f in left_files)
                       if im is not None]
        right_images = [im for im in (cv2.imread(str(f)) for f in right_files)
                        if im is not None]
        if not left_images or len(left_images) != len(right_images):
            logger.warning(f"{subject_dir.name}: 無法讀取完整 mirror 影像")
            return None
        return left_images, right_images

    def compute(self, payload, extractor) -> Dict[str, object]:
        left_images, right_images = payload
        left_feats = extractor.extract_batch(left_images)
        right_feats = extractor.extract_batch(right_images)
        valid_pairs = [(l, r) for l, r in zip(left_feats, right_feats)
                       if l is not None and r is not None]
        if not valid_pairs:
            return {}
        left_array = np.array([p[0] for p in valid_pairs])
        right_array = np.array([p[1] for p in valid_pairs])
        # calculate_differences 是多-method API，回傳 {method: arr}；此處每次只算
        # 單一 method，取出該 arr 存裸陣列（與 original 格式一致）。
        return {
            ftype: next(iter(calculate_differences(
                left_array, right_array,
                methods=[FTYPE_TO_METHOD[ftype]]).values()))
            for ftype in self.ftypes
        }


SOURCES = {"original": OriginalSource, "mirror": MirrorSource}


# =============================================================================
# 共用 runner
# =============================================================================
def _processed_subjects(output_dir: Path, models: List[str],
                        ftypes: List[str], bg_variant: str) -> Set[str]:
    """所有 model × ftype 都有 .npy 才算完成（取交集）。"""
    subject_sets = []
    for model in models:
        for ftype in ftypes:
            d = output_dir / model / bg_variant / ftype
            subject_sets.append({f.stem for f in d.glob("*.npy")}
                                if d.exists() else set())
    if not subject_sets:
        return set()
    processed = subject_sets[0]
    for s in subject_sets[1:]:
        processed &= s
    return processed


def _save_stats(stats: dict, source: FeatureSource, output_dir: Path):
    logger.info("=" * 70)
    logger.info(f"完成 — source={source.name}")
    logger.info(f"總受試者: {stats['total']} | 跳過: {stats['skipped']} | "
                f"成功: {stats['success']} | 失敗: {stats['fail']}")
    if stats.get("start") and stats.get("end"):
        logger.info(f"總耗時: {stats['end'] - stats['start']}")
    logger.info("=" * 70)

    out = {k: stats[k] for k in ("total", "skipped", "success", "fail")}
    out["start_time"] = stats["start"].isoformat() if stats.get("start") else None
    out["end_time"] = stats["end"].isoformat() if stats.get("end") else None
    stats_file = output_dir / f"feature_extraction_stats_{source.name}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"統計已儲存: {stats_file}")


def run_extraction(source: FeatureSource, models: List[str], bg_variant: str,
                   max_cpu_cores: Optional[int], output_dir: Path):
    setup_cpu_limit(max_cpu_cores)
    input_dir = source.input_dir(bg_variant == "background")

    logger.info("=" * 70)
    logger.info(f"embedding 特徵提取 — source={source.name}（{bg_variant}）")
    logger.info("=" * 70)
    logger.info(f"影像來源: {input_dir}")
    logger.info(f"輸出目錄: {output_dir} / {{model}} / {bg_variant} / {{ftype}}")
    logger.info(f"嵌入模型: {models}")
    logger.info(f"特徵類型: {source.ftypes}")

    if not input_dir.exists():
        logger.error(f"找不到影像來源目錄: {input_dir}")
        sys.exit(1)

    for model in models:
        for ftype in source.ftypes:
            (output_dir / model / bg_variant / ftype).mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    logger.info(f"找到 {len(subject_dirs)} 個受試者")
    if not subject_dirs:
        logger.error("沒有找到任何受試者目錄")
        return

    processed = _processed_subjects(output_dir, models, source.ftypes, bg_variant)
    if processed:
        logger.info(f"跳過 {len(processed)} 個已處理的受試者")
    remaining = [d for d in subject_dirs if d.name not in processed]
    logger.info(f"待處理: {len(remaining)} 個受試者")
    if not remaining:
        logger.info("所有受試者已處理完成")
        return

    stats = {"total": len(subject_dirs), "skipped": len(processed),
             "success": 0, "fail": 0, "start": datetime.now(), "end": None}

    with tqdm(remaining, desc=f"抽特徵({source.name})") as pbar:
        for subject_dir in pbar:
            subject_id = subject_dir.name
            pbar.set_description(f"處理 {subject_id}")
            try:
                payload = source.load(subject_dir)
                if payload is None:
                    stats["fail"] += 1
                    continue

                saved_any = False
                for model in models:
                    extractor = get_extractor(model)
                    if extractor is None:
                        continue
                    feats = source.compute(payload, extractor)
                    if not feats:
                        logger.warning(f"{subject_id}: {model} 沒有有效特徵")
                        continue
                    for ftype, value in feats.items():
                        out_path = (output_dir / model / bg_variant / ftype
                                    / f"{subject_id}.npy")
                        np.save(out_path, value)
                    saved_any = True

                stats["success" if saved_any else "fail"] += 1
            except Exception as e:
                logger.error(f"{subject_id}: {e}")
                stats["fail"] += 1
                import traceback
                traceback.print_exc()

    stats["end"] = datetime.now()
    _save_stats(stats, source, output_dir)


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=list(SOURCES),
                    help="original: aligned 原始 embedding；mirror: 鏡射不對稱特徵")
    ap.add_argument("--bg-variant", choices=["no_background", "background"],
                    default="no_background")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help=f"嵌入模型 (預設: {DEFAULT_MODELS})")
    ap.add_argument("--max-cpu-cores", type=int, default=2,
                    help="限制 CPU 核心數；傳 0 表示不限制")
    ap.add_argument("--output-dir", type=Path, default=EMBEDDING_FEATURES_DIR,
                    help="覆寫 EMBEDDING_FEATURES_DIR")
    args = ap.parse_args()

    source = SOURCES[args.source]()
    max_cpu = None if args.max_cpu_cores == 0 else args.max_cpu_cores
    run_extraction(source, args.models, args.bg_variant, max_cpu, args.output_dir)


if __name__ == "__main__":
    main()
