"""
計算情緒分數

使用 MT-EmotiEffNet 模型對每個受試者的照片計算情緒分數
輸出 8 種情緒機率 + Valence/Arousal 連續值
"""

import sys
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import WORKSPACE_DIR

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 路徑設定
ALIGNED_DIR = WORKSPACE_DIR / "preprocessing" / "aligned"
OUTPUT_FILE = WORKSPACE_DIR / "emotion_score.csv"

# 情緒類別（MTL 模型輸出順序）
EMOTION_COLUMNS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
    "Valence",
    "Arousal",
]


def load_model():
    """載入 MT-EmotiEffNet 模型"""
    from emotiefflib.facial_analysis import EmotiEffLibRecognizerOnnx

    logger.info("載入 MT-EmotiEffNet 模型 (enet_b0_8_va_mtl)...")
    model = EmotiEffLibRecognizerOnnx("enet_b0_8_va_mtl")

    if not model.is_mtl:
        raise ValueError("載入的模型不是 MTL 模型，無法輸出 Valence/Arousal")

    logger.info("模型載入完成")
    return model


def compute_scores_for_subject(model, subject_dir: Path) -> dict:
    """
    計算單一受試者的情緒分數

    Args:
        model: EmotiEffLibRecognizerOnnx 模型
        subject_dir: 受試者照片目錄

    Returns:
        包含 subject_id 和 10 個情緒分數的字典
    """
    subject_id = subject_dir.name
    images = sorted(subject_dir.glob("*.png"))

    if not images:
        logger.warning(f"  {subject_id}: 沒有找到照片")
        return None

    # 載入所有照片
    face_imgs = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            # 轉換為 RGB（emotiefflib 預期 RGB）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_imgs.append(img_rgb)
        else:
            logger.warning(f"  無法載入: {img_path.name}")

    if not face_imgs:
        logger.warning(f"  {subject_id}: 沒有成功載入任何照片")
        return None

    # 批次預測（傳入 list 可一次處理多張）
    _, scores = model.predict_emotions(face_imgs, logits=False)
    # scores shape: (n_images, 10)

    # 計算平均分數
    avg_scores = np.mean(scores, axis=0)

    result = {"subject_id": subject_id}
    for i, col in enumerate(EMOTION_COLUMNS):
        result[col] = float(avg_scores[i])

    return result


def main():
    """主程式"""
    logger.info("=" * 60)
    logger.info("情緒分數計算")
    logger.info("=" * 60)
    logger.info(f"輸入目錄: {ALIGNED_DIR}")
    logger.info(f"輸出檔案: {OUTPUT_FILE}")

    # 檢查輸入目錄
    if not ALIGNED_DIR.exists():
        logger.error(f"輸入目錄不存在: {ALIGNED_DIR}")
        return

    # 載入模型
    model = load_model()

    # 取得所有子資料夾
    subject_dirs = sorted(
        [d for d in ALIGNED_DIR.iterdir() if d.is_dir()]
    )
    logger.info(f"找到 {len(subject_dirs)} 個受試者")

    # 計算每個受試者的分數
    results = []
    for subject_dir in tqdm(subject_dirs, desc="計算情緒分數"):
        result = compute_scores_for_subject(model, subject_dir)
        if result:
            results.append(result)

    # 建立 DataFrame 並儲存
    df = pd.DataFrame(results)

    # 確保欄位順序
    columns = ["subject_id"] + EMOTION_COLUMNS
    df = df[columns]

    # 儲存 CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info(f"完成！共處理 {len(results)} 個受試者")
    logger.info(f"結果已儲存至: {OUTPUT_FILE}")
    logger.info("=" * 60)

    # 顯示統計摘要
    logger.info("\n情緒分數統計摘要:")
    logger.info("-" * 40)
    for col in EMOTION_COLUMNS:
        mean_val = df[col].mean()
        std_val = df[col].std()
        logger.info(f"  {col:12s}: {mean_val:.4f} +/- {std_val:.4f}")


if __name__ == "__main__":
    main()
