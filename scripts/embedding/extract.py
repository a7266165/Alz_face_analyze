"""
embedding extractor入口。

1. 將每張原始人臉相片 embedding 後存至 {model}/{bg_variant}/original/<subj>.npy，形狀 (n_images, dim)
2. 將每對鏡射人臉相片 embedding 後存至 {model}/{bg_variant}/<ftype>/<subj>.npy，形狀 (n_pairs, dim)
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
from src.common.image_io import (
    batch_apply,
    imread_unicode,
    iter_subject_dirs,
    load_subject,
)
from src.embedding import get_extractor
from src.asymmetry import calculate_differences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["arcface", "dlib", "topofr"]

# mirror ftype → calculate_differences 的 method 名
FTYPE_TO_METHOD = {
    "difference": "differences",
    "absolute_difference": "absolute_differences",
    "relative_differences": "relative_differences",
    "absolute_relative_differences": "absolute_relative_differences",
}


def setup_cpu_limit(max_cores: Optional[int]):
    """限制 CPU 核心數；max_cores 為 None 表示不限制。"""
    if max_cores is None:
        logger.info("CPU 核心數: 不限制")
        return
    logger.info(f"CPU 核心數: 限制為 {max_cores} 核心")
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
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


# =============================================================================
# Feature sources
# =============================================================================
class FeatureSource(ABC):
    """一種特徵來源：負責「讀一個 subject 的 payload」與「把 payload 經 extractor 算成 {ftype: 落地值}」。"""

    name: str
    ftypes: List[str]

    def input_dir(self, background: bool) -> Path:
        raise NotImplementedError

    @abstractmethod
    def load(self, subject_dir: Path) -> Optional[object]:
        """讀一個 subject 的影像；無法使用回 None。"""

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
        images = load_subject(subject_dir)
        if not images:
            logger.warning(f"{subject_dir.name}: 沒有影像")
            return None
        return images

    def compute(self, payload, extractor) -> Dict[str, object]:
        feats = batch_apply(extractor.extract, payload, label=extractor.model_name)
        valid = [f for f in feats if f is not None]
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
                f"(left={len(left_files)}, right={len(right_files)})"
            )
            return None
        left_images = [
            im for im in (imread_unicode(f) for f in left_files) if im is not None
        ]
        right_images = [
            im for im in (imread_unicode(f) for f in right_files) if im is not None
        ]
        if not left_images or len(left_images) != len(right_images):
            logger.warning(f"{subject_dir.name}: 無法讀取完整 mirror 影像")
            return None
        return left_images, right_images

    def compute(self, payload, extractor) -> Dict[str, object]:
        left_images, right_images = payload
        left_feats = batch_apply(
            extractor.extract, left_images, label=extractor.model_name
        )
        right_feats = batch_apply(
            extractor.extract, right_images, label=extractor.model_name
        )
        valid_pairs = [
            (l, r)
            for l, r in zip(left_feats, right_feats)
            if l is not None and r is not None
        ]
        if not valid_pairs:
            return {}
        left_array = np.array([p[0] for p in valid_pairs])
        right_array = np.array([p[1] for p in valid_pairs])
        # calculate_differences 一次算齊所有 method，回傳 {"embedding_<method>": arr}；
        # 依 ftype→method 取出對應 arr 存裸陣列（與 original 格式一致）。
        results = calculate_differences(
            left_array, right_array, methods=list(FTYPE_TO_METHOD.values())
        )
        return {
            ftype: results[f"embedding_{method}"]
            for ftype, method in FTYPE_TO_METHOD.items()
        }


SOURCES = {"original": OriginalSource, "mirror": MirrorSource}


# =============================================================================
# 共用 runner
# =============================================================================
def _processed_subjects(
    output_dir: Path, models: List[str], ftypes: List[str], bg_variant: str
) -> Set[str]:
    """檢查個案處理狀態"""
    subject_sets = []
    # 檢查一名個案所有 model × ftype 是否有 .npy
    for model in models:
        for ftype in ftypes:
            d = output_dir / model / bg_variant / ftype
            subject_sets.append(
                {f.stem for f in d.glob("*.npy")} if d.exists() else set()
            )
    if not subject_sets:
        return set()
    processed = subject_sets[0]
    for s in subject_sets[1:]:
        processed &= s
    return processed


def _save_stats(stats: dict, source: FeatureSource, output_dir: Path):
    logger.info("=" * 70)
    logger.info(f"完成 — source={source.name}")
    logger.info(
        f"總受試者: {stats['total']} | 跳過: {stats['skipped']} | "
        f"成功: {stats['success']} | 失敗: {stats['fail']}"
    )
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


def run_extraction(
    source: FeatureSource,
    models: List[str],
    bg_variant: str,
    max_cpu_cores: Optional[int],
    output_dir: Path,
):
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

    subject_dirs = iter_subject_dirs(input_dir)
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

    # 依序載入模型，載入失敗則跳過。
    extractors = {}
    for model in models:
        ext = get_extractor(model)
        if ext is None:
            logger.warning(f"{model} 不可用（未安裝或權重缺失），跳過")
            continue
        try:
            ext.initialize()
        except Exception as e:
            logger.error(f"{model} 初始化失敗，跳過: {e}")
            continue
        extractors[model] = ext
    if not extractors:
        logger.error("沒有任何可用模型，結束")
        return

    stats = {
        "total": len(subject_dirs),
        "skipped": len(processed),
        "success": 0,
        "fail": 0,
        "start": datetime.now(),
        "end": None,
    }

    # 提取進度條
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
                for model, extractor in extractors.items():
                    feats = source.compute(payload, extractor)
                    if not feats:
                        logger.warning(f"{subject_id}: {model} 沒有有效特徵")
                        continue
                    for ftype, value in feats.items():
                        out_path = (
                            output_dir
                            / model
                            / bg_variant
                            / ftype
                            / f"{subject_id}.npy"
                        )
                        np.save(out_path, value)
                    saved_any = True

                stats["success" if saved_any else "fail"] += 1
            except Exception:
                logger.exception(f"{subject_id}: extraction failed")
                stats["fail"] += 1

    stats["end"] = datetime.now()
    _save_stats(stats, source, output_dir)


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source",
        required=True,
        choices=list(SOURCES),
        help="original: aligned 原始 embedding；mirror: 鏡射不對稱特徵",
    )
    ap.add_argument(
        "--bg-variant", choices=["no_background", "background"], default="no_background"
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"嵌入模型 (預設: {DEFAULT_MODELS})",
    )
    ap.add_argument(
        "--max-cpu-cores", type=int, default=2, help="限制 CPU 核心數；傳 0 表示不限制"
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=EMBEDDING_FEATURES_DIR,
        help="覆寫 EMBEDDING_FEATURES_DIR",
    )
    args = ap.parse_args()

    source = SOURCES[args.source]()
    max_cpu = None if args.max_cpu_cores == 0 else args.max_cpu_cores
    run_extraction(source, args.models, args.bg_variant, max_cpu, args.output_dir)


if __name__ == "__main__":
    main()
