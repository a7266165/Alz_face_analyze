"""
scripts/age/predict.py
遍歷對齊影像目錄，用指定模型預測年齡並儲存

輸出格式 (JSON):
  {
    "subject_id": {
      "actual_age": 40.15,
      "predicted_ages": [31.39, 32.10, 30.85, ...]
    },
    ...
  }
每個 subject 保留最多 10 張影像的逐張預測值。

模型選項:
  mivolo (預設)   → PREDICTED_AGES_FILE（供 pipeline 下游使用）
  insightface     → AGE_PREDICTIONS_DIR/2_InsightFace/predicted_ages.json
  deepface        → AGE_PREDICTIONS_DIR/3_DeepFace/predicted_ages.json
  fairface        → AGE_PREDICTIONS_DIR/4_FairFace/predicted_ages.json
  opencv_dnn      → AGE_PREDICTIONS_DIR/5_OpenCV_DNN/predicted_ages.json
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

from src.config import (
    ALIGNED_BACKGROUND_DIR,
    AGE_BENCHMARK_DIR,
    DEMOGRAPHICS_DIR,
)
from src.age import PREDICTOR_MAP, BENCHMARK_DIR_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scan_subjects(aligned_dir: Path) -> list:
    """掃描所有受試者目錄"""
    return sorted(d for d in aligned_dir.iterdir() if d.is_dir())


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


def load_actual_ages() -> dict:
    """從 demographics CSV 讀取真實年齡 {ID: age}"""
    import pandas as pd
    ages = {}
    for csv_name in ["ACS.csv", "NAD.csv", "P.csv"]:
        csv_path = DEMOGRAPHICS_DIR / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        id_col = "ID" if "ID" in df.columns else df.columns[0]
        age_col = next((c for c in df.columns if c.lower() == "age"), None)
        if age_col is None:
            continue
        for _, row in df.iterrows():
            sid = str(row[id_col]).strip()
            try:
                ages[sid] = float(row[age_col])
            except (ValueError, TypeError):
                pass
    return ages


def default_output(model_name: str) -> Path:
    """依模型名稱決定預設輸出路徑"""
    dir_name = BENCHMARK_DIR_NAMES[model_name]
    return AGE_BENCHMARK_DIR / dir_name / "predicted_ages.json"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PREDICTOR_MAP),
                    default="mivolo", help="年齡預測模型 (預設: mivolo)")
    ap.add_argument("--aligned-dir", type=Path, default=None,
                    help="覆寫對齊影像目錄；留空用 ALIGNED_BACKGROUND_DIR")
    ap.add_argument("--output-file", type=Path, default=None,
                    help="覆寫輸出路徑；留空依模型自動決定")
    ap.add_argument("--subject-prefix", default=None,
                    help="只處理 ID 開頭符合 prefix 的受試者")
    ap.add_argument("--merge", action="store_true",
                    help="若輸出檔已存在，將新結果 merge 進去而非覆寫")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info(f"年齡預測 — 模型: {args.model}")
    logger.info("=" * 60)

    aligned_dir = args.aligned_dir or ALIGNED_BACKGROUND_DIR
    output_file = args.output_file or default_output(args.model)

    logger.info(f"影像目錄: {aligned_dir}")
    logger.info(f"輸出路徑: {output_file}")
    if args.subject_prefix:
        logger.info(f"Subject filter prefix: {args.subject_prefix}")

    logger.info(f"初始化 {args.model}...")
    predictor_cls = PREDICTOR_MAP[args.model]
    predictor = predictor_cls()
    predictor.initialize()

    subjects = scan_subjects(aligned_dir)
    if args.subject_prefix:
        subjects = [s for s in subjects if s.name.startswith(args.subject_prefix)]
    logger.info(f"找到 {len(subjects)} 個受試者")

    actual_ages = load_actual_ages()
    logger.info(f"demographics 載入 {len(actual_ages)} 個真實年齡")

    results = {}
    if args.merge and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info(f"merge 模式：既有 {len(results)} 個 id")

    skipped_no_demo = 0
    for subject_dir in tqdm(subjects, desc="預測年齡"):
        subject_id = subject_dir.name

        # 只預測清理後 demographics 中存在的受試者（已要求 Age + BMI 皆有效）。
        # 不在其中者（例如 BMI 缺失而被清理掉）為無效樣本，一律捨棄。
        if subject_id not in actual_ages:
            skipped_no_demo += 1
            continue

        images = load_images(subject_dir)
        if not images:
            logger.warning(f"{subject_id}: 無影像")
            continue

        ages = predictor.predict(images)

        if ages:
            results[subject_id] = {
                "predicted_ages": [round(a, 2) for a in ages],
                "actual_age": actual_ages[subject_id],
            }
        else:
            logger.warning(f"{subject_id}: 預測失敗")

    if skipped_no_demo:
        logger.info(f"跳過 {skipped_no_demo} 個不在清理後 demographics 的受試者（無效樣本）")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"完成: {len(results)} 個 id（本次 {len(subjects)} 個受試者）")
    logger.info(f"儲存至: {output_file}")


if __name__ == "__main__":
    main()
