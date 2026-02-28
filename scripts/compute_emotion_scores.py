"""
計算情緒分數

使用 EmoNet 8-class 模型對每個受試者的照片計算情緒分數
輸出 8 種情緒機率 + Valence/Arousal 連續值
"""

import sys
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external" / "emonet"))

from src.config import WORKSPACE_DIR

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 路徑設定
ALIGNED_DIR = WORKSPACE_DIR / "preprocessing" / "aligned"
OUTPUT_FILE = WORKSPACE_DIR / "emotion_score_EmoNet.csv"
WEIGHTS_PATH = project_root / "external" / "emonet" / "pretrained" / "emonet_8.pth"

# EmoNet 輸出索引 → 下游欄位名稱映射
EMONET_INDEX_TO_COLUMN = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
}

# 情緒類別（維持與下游 loader 一致的順序）
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
    """載入 EmoNet 8-class 模型"""
    from emonet.models import EmoNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"載入 EmoNet 8-class 模型 (device={device})...")

    model = EmoNet(n_expression=8).to(device)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"找不到 EmoNet 權重檔: {WEIGHTS_PATH}")

    state_dict = torch.load(str(WEIGHTS_PATH), map_location=device, weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logger.info("模型載入完成")
    return model, device


def compute_scores_for_subject(model, device, subject_dir: Path) -> dict:
    """
    計算單一受試者的情緒分數

    Args:
        model: EmoNet 模型
        device: torch device
        subject_dir: 受試者照片目錄

    Returns:
        包含 subject_id 和 10 個情緒分數的字典
    """
    subject_id = subject_dir.name
    images = sorted(subject_dir.glob("*.png"))

    if not images:
        logger.warning(f"  {subject_id}: 沒有找到照片")
        return None

    emo_probs_list = []
    valence_list = []
    arousal_list = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"  無法載入: {img_path.name}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)
            probs = F.softmax(out["expression"], dim=1).squeeze().cpu().numpy()
            emo_probs_list.append(probs)
            valence_list.append(out["valence"].item())
            arousal_list.append(out["arousal"].item())

    if not emo_probs_list:
        logger.warning(f"  {subject_id}: 沒有成功載入任何照片")
        return None

    avg_probs = np.mean(emo_probs_list, axis=0)
    avg_valence = np.mean(valence_list)
    avg_arousal = np.mean(arousal_list)

    result = {"subject_id": subject_id}
    for idx, col_name in EMONET_INDEX_TO_COLUMN.items():
        result[col_name] = float(avg_probs[idx])
    result["Valence"] = float(avg_valence)
    result["Arousal"] = float(avg_arousal)

    return result


def main():
    """主程式"""
    logger.info("=" * 60)
    logger.info("情緒分數計算 (EmoNet 8-class)")
    logger.info("=" * 60)
    logger.info(f"輸入目錄: {ALIGNED_DIR}")
    logger.info(f"輸出檔案: {OUTPUT_FILE}")

    # 檢查輸入目錄
    if not ALIGNED_DIR.exists():
        logger.error(f"輸入目錄不存在: {ALIGNED_DIR}")
        return

    # 載入模型
    model, device = load_model()

    # 取得所有子資料夾
    subject_dirs = sorted(
        [d for d in ALIGNED_DIR.iterdir() if d.is_dir()]
    )
    logger.info(f"找到 {len(subject_dirs)} 個受試者")

    # 計算每個受試者的分數
    results = []
    for subject_dir in tqdm(subject_dirs, desc="計算情緒分數"):
        result = compute_scores_for_subject(model, device, subject_dir)
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
