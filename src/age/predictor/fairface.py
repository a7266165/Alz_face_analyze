"""
FairFace (ResNet34) 年齡預測器
"""

import logging
from typing import Optional

import cv2
import numpy as np

from src.config import EXTERNAL_DIR
from .base import BasePredictor

logger = logging.getLogger(__name__)


class FairFacePredictor(BasePredictor):
    """FairFace (ResNet34) 年齡預測器"""

    AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39",
                "40-49", "50-59", "60-69", "70+"]
    AGE_MIDPOINTS = [1.0, 6.0, 14.5, 24.5, 34.5,
                     44.5, 54.5, 64.5, 75.0]
    WEIGHT_FILENAME = "fairface_alldata_20191111.pt"

    def __init__(self):
        self._model = None
        self._transform = None
        self._device = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            from torchvision import models  # noqa: F401
        except ImportError:
            return False
        weight_dir = EXTERNAL_DIR / "age" / "fairface"
        return ((weight_dir / self.WEIGHT_FILENAME).exists()
                or (weight_dir / "res34_fair_align_multi_7_20190809.pt").exists())

    def initialize(self):
        import torch
        from torchvision import transforms, models

        weight_dir = EXTERNAL_DIR / "age" / "fairface"
        weight_path = weight_dir / self.WEIGHT_FILENAME
        # 也接受舊版權重名稱
        if not weight_path.exists():
            alt = weight_dir / "res34_fair_align_multi_7_20190809.pt"
            if alt.exists():
                weight_path = alt
        if not weight_path.exists():
            raise FileNotFoundError(
                f"FairFace 權重不存在: {weight_path}\n"
                f"請從 https://drive.google.com/drive/folders/"
                f"1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu 下載並放到 {weight_dir}/"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # FairFace uses ResNet34 with 18 outputs: 7 race + 2 gender + 9 age
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 18)
        state = torch.load(str(weight_path), map_location=self._device,
                           weights_only=True)
        model.load_state_dict(state)
        model.to(self._device).eval()
        self._model = model

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"FairFace 初始化完成 ({self._device.upper()})")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        import torch

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self._transform(image_rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                out = self._model(tensor)
            # age logits are the last 9 outputs
            age_logits = out[0, 9:18]
            probs = torch.softmax(age_logits, dim=0).cpu().numpy()
            midpoints = np.array(self.AGE_MIDPOINTS)
            return float((probs * midpoints).sum())
        except Exception as e:
            logger.debug(f"FairFace 預測失敗: {e}")
        return None
