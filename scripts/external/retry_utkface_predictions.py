"""
Retry MiVOLO 年齡預測 for UTKFace subjects，優先用 UTKFace 原本就提供的
`{filename}_aligned.png`（裁切好的 face），以規避 in-the-wild 構圖問題。

Blacklist：UTKFace race label 誤標的 subjects（明顯非亞裔），直接剔除。

輸入：
  external/public_face_datasets/filtered/asian_elderly_60plus/EACS_UTKFace_*
  workspace/age/age_prediction/predicted_ages.json
輸出：
  更新 predicted_ages.json（只動 EACS_UTKFace_* 條目；
    blacklist 的 id 從檔中移除）
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    DEMOGRAPHICS_DIR,
    EXTERNAL_FILTERED_DIR,
    PREDICTED_AGES_FILE,
)
from src.extractor.features.age import MiVOLOPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# UTKFace race=2 的 2 張明顯誤標（人工辨識確認非亞裔）
RACE_BLACKLIST = {
    "EACS_UTKFace_00191-1",  # real=80 非洲男性
    "EACS_UTKFace_00234-1",  # real=85 白人老伯
}

# LibreFace MediaPipe 偵測失敗（沒 _aligned.png）+ Haar 也抓不到臉
# → MiVOLO fallback 到整張圖 → 預測嚴重偏低且無救
QUALITY_BLACKLIST = {
    "EACS_UTKFace_00162-1",  # real=76, pred=25.9 — iStock 浮水印遮臉
    "EACS_UTKFace_00151-1",  # real=75, pred=26.0 — 尼泊爾紅帽老人，低對比
    "EACS_UTKFace_00235-1",  # real=86, pred=39.4 — 奶奶+蔥堆全身照，臉小
}

UTKFACE_POOL = EXTERNAL_FILTERED_DIR / "asian_elderly_60plus"


def pick_image(folder: Path):
    """回傳優先順序：_aligned.png → 第一張 .jpg/.png。"""
    aligned = sorted(folder.glob("*_aligned.png"))
    if aligned:
        return aligned[0], "aligned"
    raw = sorted(p for p in folder.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                 and not p.name.endswith("_aligned.png"))
    if raw:
        return raw[0], "raw"
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(PREDICTED_AGES_FILE, "r", encoding="utf-8") as f:
        preds = json.load(f)
    logger.info(f"loaded {len(preds)} existing predictions")

    # 掃描 UTKFace subject folders
    utkface_subjects = sorted(
        [d for d in UTKFACE_POOL.iterdir()
         if d.is_dir() and d.name.startswith("EACS_UTKFace_")]
    )
    logger.info(f"UTKFace folders found: {len(utkface_subjects)}")

    # 初始化 MiVOLO（與原 predict_ages.py 相同）
    logger.info("init MiVOLO...")
    predictor = MiVOLOPredictor()
    predictor.initialize()

    # 重跑
    changes = []
    skipped = []
    stats = {"aligned": 0, "raw": 0, "none": 0, "blacklist": 0,
             "fail": 0, "updated": 0, "removed": 0}

    for folder in utkface_subjects:
        sid = folder.name
        if sid in RACE_BLACKLIST or sid in QUALITY_BLACKLIST:
            stats["blacklist"] += 1
            reason = "race_blacklist" if sid in RACE_BLACKLIST else "quality_blacklist"
            if sid in preds:
                skipped.append((sid, reason, preds.get(sid)))
            continue

        img_path, kind = pick_image(folder)
        stats[kind] += 1
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            stats["fail"] += 1
            continue

        ages = predictor.predict([img])
        if not ages:
            stats["fail"] += 1
            continue
        new_pred = round(sum(ages) / len(ages), 2)
        old_pred = preds.get(sid)
        if old_pred is None or abs(new_pred - old_pred) > 0.01:
            changes.append((sid, old_pred, new_pred, kind))
            stats["updated"] += 1
        if not args.dry_run:
            preds[sid] = new_pred

    # 移除 blacklist（race + quality）
    for sid in RACE_BLACKLIST | QUALITY_BLACKLIST:
        if sid in preds:
            if not args.dry_run:
                del preds[sid]
            stats["removed"] += 1

    # Summary
    logger.info(f"stats: {stats}")
    logger.info(f"Sample changes (first 15):")
    for sid, old, new, kind in changes[:15]:
        old_s = f"{old:5.1f}" if old is not None else "  n/a"
        logger.info(f"  {sid:35s}  {old_s} -> {new:5.1f}  ({kind})")

    # 大改動（差 > 10 歲）特別列出
    big = [c for c in changes if c[1] is not None and abs(c[2] - c[1]) > 10]
    logger.info(f"\nBig changes (|Δ|>10): {len(big)}")
    for sid, old, new, kind in big:
        logger.info(f"  {sid:35s}  {old:5.1f} -> {new:5.1f}  Δ{new-old:+5.1f}  ({kind})")

    if not args.dry_run:
        with open(PREDICTED_AGES_FILE, "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        logger.info(f"saved {PREDICTED_AGES_FILE}")
    else:
        logger.info("DRY RUN — predicted_ages.json 未更新")


if __name__ == "__main__":
    main()
