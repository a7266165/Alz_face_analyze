"""
DAN / HSEmotion / ViT (trpakov) 統一提取腳本

在 fer_models 環境中執行:
    conda activate fer_models
    python scripts/run_new_models_extract.py --tools dan hsemotion vit
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.config import ALIGNED_DIR

# ── 路徑 ──
RAW_OUTPUT_DIR = PROJECT_ROOT / "workspace" / "au_features" / "raw"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

HARMONIZED_COLS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# DAN
# =============================================================================
class DANExtractor:
    """DAN (Distract Your Attention Network) — RAF-DB trained"""

    # RAF-DB label order: 0=surprise, 1=fear, 2=disgust, 3=happiness, 4=sadness, 5=anger, 6=neutral
    RAFDB_INDEX = {0: "surprise", 1: "fear", 2: "disgust", 3: "happiness", 4: "sadness", 5: "anger", 6: "neutral"}

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _init_model(self):
        if self.model is not None:
            return
        # Add DAN to path
        dan_dir = PROJECT_ROOT / "models" / "DAN"
        if str(dan_dir) not in sys.path:
            sys.path.insert(0, str(dan_dir))
        from networks.dan import DAN

        model = DAN(num_class=7, num_head=4, pretrained=False)
        checkpoint_path = WEIGHTS_DIR / "dan" / "rafdb_epoch21_acc0.8970_bacc0.8272.pth"
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(self.device)
        model.eval()
        self.model = model
        logger.info(f"DAN 模型載入完成 (device={self.device})")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        self._init_model()
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out, _, _ = self.model(tensor)
            probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
            return {self.RAFDB_INDEX[i]: float(probs[i]) for i in range(7)}
        except Exception:
            return None


# =============================================================================
# HSEmotion
# =============================================================================
class HSEmotionExtractor:
    """HSEmotion — EfficientNet-based emotion recognition"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fer = None

    def _init_model(self):
        if self.fer is not None:
            return
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self.fer = HSEmotionRecognizer(model_name="enet_b2_7", device=self.device)
        logger.info(f"HSEmotion 模型載入完成 (device={self.device})")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        self._init_model()
        try:
            # HSEmotion expects BGR and does its own face detection
            # But since our images are already aligned faces, we pass directly
            # The library's predict method works on face crops
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            emotion, scores = self.fer.predict_emotions(rgb, logits=False)
            # enet_b2_7 outputs 7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
            emo_names = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
            return {emo_names[i]: float(scores[i]) for i in range(7)}
        except Exception:
            return None


# =============================================================================
# ViT (trpakov/vit-face-expression)
# =============================================================================
class ViTExtractor:
    """HuggingFace ViT fine-tuned for facial expression recognition"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        # Model label order: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
        self.label_map = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness", 4: "neutral", 5: "sadness", 6: "surprise"}

    def _init_model(self):
        if self.model is not None:
            return
        from transformers import ViTImageProcessor, ViTForImageClassification
        model_name = "trpakov/vit-face-expression"
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"ViT 模型載入完成 (device={self.device})")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        self._init_model()
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_img = Image.fromarray(rgb)
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
            return {self.label_map[i]: float(probs[i]) for i in range(7)}
        except Exception:
            return None


# =============================================================================
# 共用提取邏輯
# =============================================================================
EXTRACTORS = {
    "dan": DANExtractor,
    "hsemotion": HSEmotionExtractor,
    "vit": ViTExtractor,
}


def get_subject_dirs(input_dir: Path) -> List[Path]:
    return sorted([d for d in input_dir.iterdir() if d.is_dir()])


def extract_tool(tool_name: str, device: str):
    output_dir = RAW_OUTPUT_DIR / tool_name
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = get_subject_dirs(ALIGNED_DIR)
    logger.info(f"[{tool_name}] 共 {len(subject_dirs)} 個受試者")

    extractor = EXTRACTORS[tool_name](device=device)

    success, skip, fail = 0, 0, 0
    for subject_dir in tqdm(subject_dirs, desc=tool_name):
        output_file = output_dir / f"{subject_dir.name}.csv"
        if output_file.exists():
            skip += 1
            continue

        image_paths = sorted(
            [p for p in subject_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")],
            key=lambda p: p.name,
        )
        if not image_paths:
            fail += 1
            continue

        rows = []
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            result = extractor.extract_frame(image)
            if result is not None:
                result["frame"] = img_path.stem
                rows.append(result)

        if rows:
            with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["frame"] + HARMONIZED_COLS)
                writer.writeheader()
                writer.writerows([{k: row.get(k, 0.0) for k in ["frame"] + HARMONIZED_COLS} for row in rows])
            success += 1
        else:
            fail += 1

    logger.info(f"[{tool_name}] 成功={success}, 跳過={skip}, 失敗={fail}")


def main():
    parser = argparse.ArgumentParser(description="DAN / HSEmotion / ViT 表情提取")
    parser.add_argument("--tools", nargs="+", default=["dan", "hsemotion", "vit"],
                        choices=["dan", "hsemotion", "vit"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    for tool in args.tools:
        logger.info(f"\n{'='*60}")
        logger.info(f"開始提取: {tool}")
        logger.info(f"{'='*60}")
        extract_tool(tool, args.device)


if __name__ == "__main__":
    main()
