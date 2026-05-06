"""
scripts/predict_ages.py
遍歷原始影像目錄，用 MiVOLO 預測年齡並儲存
"""

import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT
project_root = PROJECT_ROOT

from src.config import ALIGNED_DIR, AGE_PREDICTION_DIR
from src.extractor.features.age import MiVOLOPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scan_subjects(aligned_dir: Path) -> list:
    """掃描所有受試者目錄"""
    subjects = sorted(
        d for d in aligned_dir.iterdir() if d.is_dir()
    )
    return subjects


def load_images(subject_dir: Path, max_images: int = 10) -> list:
    """載入前 N 張影像"""
    valid_ext = {'.jpg', '.jpeg', '.png'}
    images = []
    
    for f in sorted(subject_dir.iterdir()):
        if f.suffix.lower() in valid_ext:
            img = cv2.imread(str(f))
            if img is not None:
                images.append(img)
            if len(images) >= max_images:
                break
    
    return images


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned-dir", type=Path, default=None,
                    help="覆寫對齊影像目錄；留空用 ALIGNED_DIR (內部 cohort 預設)")
    ap.add_argument("--output-file", type=Path, default=None,
                    help="覆寫輸出路徑；留空用 AGE_PREDICTION_DIR/predicted_ages_2.json")
    ap.add_argument("--subject-prefix", default=None,
                    help="只處理 ID 開頭符合 prefix 的受試者 (e.g. EACS_)；留空全處理")
    ap.add_argument("--merge", action="store_true",
                    help="若輸出檔已存在，將新結果 merge 進去而非覆寫")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("MiVOLO 年齡預測")
    logger.info("=" * 60)

    aligned_dir = args.aligned_dir if args.aligned_dir is not None else ALIGNED_DIR
    output_file = args.output_file if args.output_file is not None else (
        AGE_PREDICTION_DIR / "predicted_ages_2.json"
    )

    logger.info(f"影像目錄: {aligned_dir}")
    if args.subject_prefix:
        logger.info(f"Subject filter prefix: {args.subject_prefix}")

    logger.info("初始化 MiVOLO...")
    predictor = MiVOLOPredictor()
    predictor.initialize()

    subjects = scan_subjects(aligned_dir)
    if args.subject_prefix:
        subjects = [s for s in subjects if s.name.startswith(args.subject_prefix)]
    logger.info(f"找到 {len(subjects)} 個受試者")

    # Pre-load existing results if merging
    results = {}
    if args.merge and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info(f"merge 模式：既有 {len(results)} 個 id")

    for subject_dir in tqdm(subjects, desc="預測年齡"):
        subject_id = subject_dir.name
        images = load_images(subject_dir)

        if not images:
            logger.warning(f"{subject_id}: 無影像")
            continue

        ages = predictor.predict(images)

        if ages:
            avg_age = sum(ages) / len(ages)
            results[subject_id] = round(avg_age, 2)
        else:
            logger.warning(f"{subject_id}: 預測失敗")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ 完成: {len(results)} 個 id 總計 (本次 {len(subjects)} 個)")
    logger.info(f"儲存至: {output_file}")


if __name__ == "__main__":
    main()