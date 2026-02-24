"""
scripts/extract_original_features.py
從已對齊的臉部影像提取原始嵌入向量（不做鏡射）。

功能：
1. 讀取 workspace/preprocessing/aligned/ 下各受試者影像
2. 使用 ArcFace / TopoFR 等提取器取得嵌入向量
3. 儲存為 workspace/features/{model}/original/{subject_id}.npy
4. 支援斷點續傳

使用方式:
    poetry run python scripts/extract_original_features.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np
from tqdm import tqdm

# 專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import FEATURES_DIR, WORKSPACE_DIR
from src.core.extractor import FeatureExtractor

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ========== 設定 ==========
ALIGNED_DIR = WORKSPACE_DIR / "preprocessing" / "aligned"
OUTPUT_DIR = FEATURES_DIR  # workspace/features
EMBEDDING_MODELS = ["arcface", "topofr", "dlib"]
MAX_CPU_CORES = 2


def setup_cpu_limit(max_cores: int):
    """限制 CPU 核心數"""
    os.environ["OMP_NUM_THREADS"] = str(max_cores)
    os.environ["MKL_NUM_THREADS"] = str(max_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_cores)
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


def get_processed_subjects(output_dir: Path, models: List[str]) -> Set[str]:
    """取得所有模型都已處理完成的受試者集合"""
    subject_sets = []
    for model in models:
        feature_dir = output_dir / model / "original"
        if feature_dir.exists():
            subjects = {f.stem for f in feature_dir.glob("*.npy")}
            subject_sets.append(subjects)
        else:
            subject_sets.append(set())

    if not subject_sets:
        return set()

    # 取交集
    processed = subject_sets[0]
    for s in subject_sets[1:]:
        processed = processed & s
    return processed


def load_images(subject_dir: Path) -> List[np.ndarray]:
    """載入受試者目錄下所有影像"""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    images = []
    for file_path in sorted(subject_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            img = cv2.imread(str(file_path))
            if img is not None:
                images.append(img)
    return images


def main():
    """主程式"""
    setup_cpu_limit(MAX_CPU_CORES)

    logger.info("=" * 60)
    logger.info("原始臉部特徵提取")
    logger.info("=" * 60)
    logger.info(f"影像來源: {ALIGNED_DIR}")
    logger.info(f"輸出目錄: {OUTPUT_DIR}")
    logger.info(f"嵌入模型: {EMBEDDING_MODELS}")

    if not ALIGNED_DIR.exists():
        logger.error(f"找不到對齊影像目錄: {ALIGNED_DIR}")
        sys.exit(1)

    # 建立輸出目錄
    for model in EMBEDDING_MODELS:
        (OUTPUT_DIR / model / "original").mkdir(parents=True, exist_ok=True)

    # 掃描受試者
    subject_dirs = sorted([
        d for d in ALIGNED_DIR.iterdir()
        if d.is_dir()
    ])
    logger.info(f"找到 {len(subject_dirs)} 個受試者")

    # 檢查斷點
    processed = get_processed_subjects(OUTPUT_DIR, EMBEDDING_MODELS)
    if processed:
        logger.info(f"跳過 {len(processed)} 個已處理的受試者")

    remaining = [d for d in subject_dirs if d.name not in processed]
    logger.info(f"待處理: {len(remaining)} 個受試者")

    if not remaining:
        logger.info("所有受試者已處理完成")
        return

    # 初始化提取器
    extractor = FeatureExtractor()

    success = 0
    fail = 0

    with tqdm(remaining, desc="提取原始特徵") as pbar:
        for subject_dir in pbar:
            subject_id = subject_dir.name
            pbar.set_description(f"處理 {subject_id}")

            try:
                images = load_images(subject_dir)
                if not images:
                    logger.warning(f"{subject_id}: 沒有影像")
                    fail += 1
                    continue

                # 提取各模型的特徵
                results = extractor.extract_features(
                    images, models=EMBEDDING_MODELS
                )

                for model in EMBEDDING_MODELS:
                    if model not in results:
                        continue

                    features = results[model]
                    # 過濾 None
                    valid = [f for f in features if f is not None]
                    if not valid:
                        logger.warning(f"{subject_id}: {model} 沒有有效特徵")
                        continue

                    feature_array = np.array(valid)  # (n_images, feature_dim)
                    out_path = OUTPUT_DIR / model / "original" / f"{subject_id}.npy"
                    np.save(out_path, feature_array)

                success += 1

            except Exception as e:
                logger.error(f"{subject_id}: {e}")
                fail += 1

    logger.info("=" * 60)
    logger.info(f"完成: 成功 {success}, 失敗 {fail}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
