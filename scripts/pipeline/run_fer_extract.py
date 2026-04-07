"""
FER (justinshenk/fer) 獨立提取腳本

在 fer_env 環境中執行，輸出格式與其他工具的 raw CSV 一致。
Usage:
    conda activate fer_env
    python scripts/run_fer_extract.py
"""

import csv
import logging
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fer.fer import FER
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.config import ALIGNED_DIR

# ── 路徑配置 ──
RAW_OUTPUT_DIR = PROJECT_ROOT / "workspace" / "au_features" / "raw" / "fer"

# FER 輸出的 7 種表情 → harmonized 名稱對映
# fer 預設 key: angry, disgust, fear, happy, sad, surprise, neutral
EMO_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}
HARMONIZED_COLS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_subject_dirs(input_dir: Path) -> List[Path]:
    dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    return dirs


def extract_subject(detector: FER, subject_dir: Path) -> list:
    image_paths = sorted(
        [p for p in subject_dir.iterdir()
         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")],
        key=lambda p: p.name,
    )
    if not image_paths:
        return []

    results = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        try:
            # FER detect_emotions 回傳 list of dicts
            # 對 aligned face 通常只有一張臉，取第一個
            detections = detector.detect_emotions(image)
            if not detections:
                continue
            emo_scores = detections[0]["emotions"]
            row = {"frame": img_path.stem}
            for fer_key, harmonized_key in EMO_MAP.items():
                row[harmonized_key] = float(emo_scores.get(fer_key, 0.0))
            results.append(row)
        except Exception:
            continue

    return results


def main():
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_dirs = get_subject_dirs(ALIGNED_DIR)
    logger.info(f"共 {len(subject_dirs)} 個受試者待處理")

    # 初始化 FER detector（使用 MTCNN 以提高準確度）
    detector = FER(mtcnn=True)
    logger.info("FER detector 初始化完成 (MTCNN=True)")

    success, skip, fail = 0, 0, 0

    for subject_dir in tqdm(subject_dirs, desc="fer"):
        output_file = RAW_OUTPUT_DIR / f"{subject_dir.name}.csv"
        if output_file.exists():
            skip += 1
            continue

        try:
            rows = extract_subject(detector, subject_dir)
            if rows:
                with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=["frame"] + HARMONIZED_COLS)
                    writer.writeheader()
                    writer.writerows(rows)
                success += 1
            else:
                fail += 1
        except Exception as e:
            logger.error(f"  {subject_dir.name}: {e}")
            fail += 1

    logger.info(f"FER: 成功={success}, 跳過={skip}, 失敗={fail}")


if __name__ == "__main__":
    main()
