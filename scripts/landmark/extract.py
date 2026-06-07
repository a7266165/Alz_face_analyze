"""Landmark 幾何不對稱特徵 — 兩步 CLI。

  extract  aligned 影像 → 正規化 468 landmarks → 每人一個 .npy
  compute  .npy → 分區 pair 差值 + 面積差 → pair_features.csv
  all      兩步都跑

路徑取自 src.config 的 ASYMMETRY_*；幾何運算全來自 src.landmark.extractor。
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ASYMMETRY_LANDMARKS_DIR,
    ASYMMETRY_PAIR_FEATURES_FILE,
    preprocess_dir,
)
from src.landmark.extractor import compute_regional_features, extract_and_save_landmarks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ALIGNED_DIR = preprocess_dir("aligned")  # workspace/preprocess/no_background/aligned


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "compute", "all"])
    args = ap.parse_args()

    if args.step in ("extract", "all"):
        extract_and_save_landmarks(ALIGNED_DIR, ASYMMETRY_LANDMARKS_DIR)
    if args.step in ("compute", "all"):
        compute_regional_features(ASYMMETRY_LANDMARKS_DIR, ASYMMETRY_PAIR_FEATURES_FILE)


if __name__ == "__main__":
    main()
