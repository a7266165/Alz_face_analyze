"""
scripts/age/predict_ages.py
遍歷對齊影像目錄，用指定模型預測年齡並儲存

模型選項:
  mivolo (預設)   → JSON 輸出至 PREDICTED_AGES_FILE（供 pipeline 下游使用）
  insightface     → CSV 輸出至 AGE_BENCHMARK_DIR/2_InsightFace/
  deepface        → CSV 輸出至 AGE_BENCHMARK_DIR/3_DeepFace/
  fairface        → CSV 輸出至 AGE_BENCHMARK_DIR/4_FairFace/
  opencv_dnn      → CSV 輸出至 AGE_BENCHMARK_DIR/5_OpenCV_DNN/
"""

import sys
import csv
import json
import logging
from pathlib import Path
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT
project_root = PROJECT_ROOT

from src.config import (
    ALIGNED_DIR,
    AGE_BENCHMARK_DIR,
    DEMOGRAPHICS_DIR,
    PREDICTED_AGES_FILE,
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
    ages = {}
    for csv_name in ["ACS.csv", "NAD.csv", "P.csv"]:
        csv_path = DEMOGRAPHICS_DIR / csv_name
        if not csv_path.exists():
            continue
        import pandas as pd
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
    if model_name == "mivolo":
        return PREDICTED_AGES_FILE
    dir_name = BENCHMARK_DIR_NAMES[model_name]
    return AGE_BENCHMARK_DIR / dir_name / "predicted_ages.csv"


def save_json(results: dict, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_csv(results: dict, actual_ages: dict, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Actual_Age", "Predicted_Age"])
        for sid in sorted(results):
            actual = actual_ages.get(sid, "")
            writer.writerow([sid, actual, results[sid]])


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PREDICTOR_MAP),
                    default="mivolo", help="年齡預測模型 (預設: mivolo)")
    ap.add_argument("--aligned-dir", type=Path, default=None,
                    help="覆寫對齊影像目錄；留空用 ALIGNED_DIR")
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

    aligned_dir = args.aligned_dir or ALIGNED_DIR
    output_file = args.output_file or default_output(args.model)
    is_json = output_file.suffix.lower() == ".json"

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

    results = {}
    if args.merge and output_file.exists():
        if is_json:
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

    if is_json:
        save_json(results, output_file)
    else:
        actual_ages = load_actual_ages()
        save_csv(results, actual_ages, output_file)

    logger.info(f"完成: {len(results)} 個 id（本次 {len(subjects)} 個受試者）")
    logger.info(f"儲存至: {output_file}")


if __name__ == "__main__":
    main()
