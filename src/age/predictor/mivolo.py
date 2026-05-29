"""
MiVOLO v2 年齡預測器
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from .base import BasePredictor

logger = logging.getLogger(__name__)


class MiVOLOPredictor(BasePredictor):
    """MiVOLO v2 年齡預測器"""

    # Haar 偵測框面積佔整張圖比例低於此值，視為誤判（常誤抓背景/衣物的小區塊），
    # 改用整張圖預測。實測：真臉框佔 12~40%、誤判框佔 0.4~5%，8% 可乾淨區隔。
    MIN_FACE_AREA_FRAC = 0.08

    def __init__(self):
        self.model = None
        self.processor = None
        self.face_detector = None
        self.device = None

    def initialize(self):
        """載入模型"""
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.model = AutoModelForImageClassification.from_pretrained(
                "iitolstykh/mivolo_v2",
                trust_remote_code=True,
                dtype=dtype
            )
            self.processor = AutoImageProcessor.from_pretrained(
                "iitolstykh/mivolo_v2",
                trust_remote_code=True
            )
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            if torch.cuda.is_available():
                self.model = self.model.cuda()

            self.model.eval()
            logger.info(f"✓ MiVOLO 初始化完成 ({self.device.upper()})")

        except Exception as e:
            raise RuntimeError(f"MiVOLO 初始化失敗: {e}")

    def face_crop(self, image: np.ndarray) -> np.ndarray:
        """回傳實際餵入模型的人臉裁切。

        用 Haar 偵測最大臉框並外擴 30% margin。但 Haar 常在背景/衣物誤抓
        ~1% 的小框（裁出來不是臉，會被估成年輕人），故加最小臉框守門：
        偵測不到臉、或最大框面積佔比 < MIN_FACE_AREA_FRAC 時，一律改用整張圖
        （影像已對齊、臉在中央，整張圖預測仍正確）。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            return image

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        area_frac = (w * h) / (image.shape[0] * image.shape[1])
        if area_frac < self.MIN_FACE_AREA_FRAC:
            return image  # 框過小，視為誤判 → 整張圖

        margin = int(max(w, h) * 0.3)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
        return image[y1:y2, x1:x2]

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        """預測單張影像的年齡（先做人臉裁切再推論）"""
        return self._predict_on_crop(self.face_crop(image))

    def _predict_on_crop(self, face_crop: np.ndarray) -> Optional[float]:
        """對「已裁切好的人臉」做推論，不再重跑人臉偵測。"""
        import torch

        try:
            # 預處理
            inputs = self.processor(images=[face_crop])["pixel_values"]
            inputs = inputs.to(dtype=self.model.dtype, device=self.model.device)

            # 推論
            with torch.no_grad():
                outputs = self.model(faces_input=inputs, body_input=inputs)

            if hasattr(outputs, 'age_output'):
                return outputs.age_output[0].item()

        except Exception as e:
            logger.debug(f"預測失敗: {e}")

        return None

    def predict_cropped(self, crops: List[np.ndarray]) -> List[float]:
        """對一批「已裁切好的人臉」做推論。

        供 ``--save-input`` 重用先前算好的 face_crop 結果，避免對同一張圖
        重跑兩次人臉偵測，同時確保「存下來的裁切」與「實際餵入模型的裁切」
        完全一致。
        """
        return [a for c in crops if (a := self._predict_on_crop(c)) is not None]
