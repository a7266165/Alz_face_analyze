"""
scripts/age/predict.py
遍歷對齊影像目錄，用指定模型預測年齡並儲存

輸出格式 (JSON):
  {
    "subject_id": {
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
import os
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

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


def _imread_unicode(path: Path):
    """Unicode-safe 影像讀取。

    cv2.imread 在 Windows 走 ANSI API，路徑含非 ASCII（例如中文）字元時會靜默
    回傳 None。改用 np.fromfile 讀進 bytes、再 cv2.imdecode 解碼，含中文的路徑也能讀。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_images(subject_dir: Path, max_images: int = 10) -> list:
    """載入前 N 張影像，回傳 (檔名, 影像) 配對清單"""
    valid_ext = {'.jpg', '.jpeg', '.png'}
    images = []

    for f in sorted(subject_dir.iterdir()):
        if f.suffix.lower() in valid_ext:
            img = _imread_unicode(f)
            if img is not None:
                images.append((f.name, img))
            if len(images) >= max_images:
                break

    return images


def _imwrite_unicode(path: Path, img) -> None:
    """Unicode-safe 影像寫檔（對應 _imread_unicode）。"""
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    else:
        logger.warning(f"影像編碼失敗，未寫出: {path}")


def save_results(results: dict, output_file: Path) -> None:
    """原子寫出 JSON：先寫 .tmp 再 os.replace，避免寫到一半當機損毀原檔。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(output_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_file)


def load_demographic_ids() -> set:
    """從 demographics CSV 讀取「年齡有效」的受試者 ID 集合。

    predict 已不再輸出 actual_age，這裡只用於成員判斷（決定哪些受試者要預測），
    故回傳 set 而非 {ID: age}；無法轉成 float 的年齡視為無效、不納入。
    """
    # 唯一讀取點：cohort.load_demographics() 已組好 ID(完整鍵) 並解析 Age。
    from src.common.cohort import load_demographics
    df = load_demographics()
    return set(df["ID"][df["Age"].notna()])


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
    ap.add_argument("--save-input", action="store_true",
                    help="把實際餵入模型的人臉裁切存到 <輸出目錄>/input/<ID>/"
                         "（格式比照 preprocess，便於查驗；僅支援有 face_crop 的模型如 mivolo）")
    ap.add_argument("--resume", action="store_true",
                    help="載入既有輸出檔並略過已完成的受試者（搭配 checkpoint，當機後可接續）")
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="每處理 N 個受試者就寫一次 JSON checkpoint（預設 200；0 表關閉）")
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

    valid_ids = load_demographic_ids()
    logger.info(f"demographics 載入 {len(valid_ids)} 個有效受試者 ID")

    results = {}
    if (args.merge or args.resume) and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        # predict 不再輸出 actual_age；既有檔若有殘留也順手清掉，保持一致
        for entry in results.values():
            if isinstance(entry, dict):
                entry.pop("actual_age", None)
        mode = "resume" if args.resume else "merge"
        logger.info(f"{mode} 模式：載入既有 {len(results)} 個 id")

    skipped_no_demo = 0
    processed = 0
    for subject_dir in tqdm(subjects, desc="預測年齡"):
        subject_id = subject_dir.name

        # 只預測清理後 demographics 中存在的受試者（已要求 Age + BMI 皆有效）。
        # 不在其中者（例如 BMI 缺失而被清理掉）為無效樣本，一律捨棄。
        if subject_id not in valid_ids:
            skipped_no_demo += 1
            continue

        # resume：已完成者直接略過（接續上次 checkpoint）
        if args.resume and subject_id in results:
            continue

        pairs = load_images(subject_dir)
        if not pairs:
            logger.warning(f"{subject_id}: 無影像")
            continue

        images = [img for _, img in pairs]

        # 存下實際餵入模型的人臉裁切（含最小臉框守門後的結果），格式比照 preprocess。
        # 裁切只算一次：存檔後直接餵給模型（predict_cropped），避免對同一張圖
        # 重跑兩次人臉偵測，也確保「存下來的」與「實際推論的」是同一份裁切。
        if args.save_input and hasattr(predictor, "face_crop"):
            crop_dir = output_file.parent / "input" / subject_id
            crop_dir.mkdir(parents=True, exist_ok=True)
            crops = []
            for name, img in pairs:
                crop = predictor.face_crop(img)
                _imwrite_unicode(crop_dir / name, crop)
                crops.append(crop)
            if hasattr(predictor, "predict_cropped"):
                ages = predictor.predict_cropped(crops)
            else:
                ages = predictor.predict(images)
        else:
            ages = predictor.predict(images)

        if ages:
            results[subject_id] = {
                "predicted_ages": [round(a, 2) for a in ages],
            }
        else:
            logger.warning(f"{subject_id}: 預測失敗")

        # 定期 checkpoint：每 N 個受試者寫一次 JSON，當機後最多只損失最後一段
        processed += 1
        if args.checkpoint_every and processed % args.checkpoint_every == 0:
            save_results(results, output_file)
            logger.info(f"checkpoint: 已寫出 {len(results)} 個 id（本次處理 {processed}）")

    if skipped_no_demo:
        logger.info(f"跳過 {skipped_no_demo} 個不在清理後 demographics 的受試者（無效樣本）")

    save_results(results, output_file)
    logger.info(f"完成: {len(results)} 個 id（本次處理 {processed} 個受試者）")
    logger.info(f"儲存至: {output_file}")


if __name__ == "__main__":
    main()
