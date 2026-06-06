"""OpenFace 3.0 AU/emotion 提取器:openface-test API（FaceDetector + MultitaskPredictor），輸出 8 AU intensity、8 emotion、2D gaze。"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import logging

from .base import EmoAUExtractor
from ._torch_patch import patched_torch_load
from src.common.image_io import temp_image_png
from src.emo_au.extractor.au_config import (
    OPENFACE_AU_INDEX,
    OPENFACE_EMOTION_INDEX,
    OPENFACE_GAZE_COLUMNS,
    OPENFACE_WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)


class OpenFaceExtractor(EmoAUExtractor):
    """OpenFace 3.0:FaceDetector(RetinaFace 偵測+裁切) → MultitaskPredictor(EfficientNet 多任務) → 8 AU(DISFA)、8 emotion、2D gaze(yaw/pitch)。"""

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self._weights_dir = Path(weights_dir or OPENFACE_WEIGHTS_DIR)
        self._device = device or "cuda"
        self._detector = None
        self._predictor = None

    @property
    def model_name(self) -> str:
        return "openface"

    @property
    def output_columns(self) -> List[str]:
        # 8 AU + 8 emotion(含 contempt)+ 2D gaze，順序照 _parse_outputs 寫入序
        return (list(OPENFACE_AU_INDEX.values())
                + list(OPENFACE_EMOTION_INDEX.values())
                + list(OPENFACE_GAZE_COLUMNS))

    def _probe(self) -> bool:
        if not self._probe_import("openface", "pip install openface-test --no-deps"):
            return False
        retina = self._weights_dir / "Alignment_RetinaFace.pth"
        mtl = self._weights_dir / "MTL_backbone.pth"
        if retina.exists() and mtl.exists():
            return True
        logger.warning(f"OpenFace 權重不存在: {self._weights_dir}；"
                       f"請執行: openface download --output {self._weights_dir}")
        return False

    def _load(self) -> None:
        """載入 OpenFace 3.0 FaceDetector + MultitaskPredictor（含 mobilenet 路徑重導）。"""
        from openface.face_detection import FaceDetector
        from openface.multitask_model import MultitaskPredictor

        retina_path = str(self._weights_dir / "Alignment_RetinaFace.pth")
        mtl_path = str(self._weights_dir / "MTL_backbone.pth")

        # RetinaFace 內部 hardcode "./weights/mobilenetV1X0.25_pretrain.tar"，建構時重導。
        mobilenet_path = str(self._weights_dir / "mobilenetV1X0.25_pretrain.tar")

        logger.info(f"載入 OpenFace 3.0 模型（device={self._device}）...")
        with patched_torch_load({"mobilenetV1X0.25_pretrain.tar": mobilenet_path}):
            self._detector = FaceDetector(
                model_path=retina_path, device=self._device
            )

        self._predictor = MultitaskPredictor(
            model_path=mtl_path, device=self._device
        )
        logger.info("OpenFace 3.0 模型載入完成")

    def _extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """FaceDetector.get_face() 是 path-only API，故經 temp_image_png 暫存後餵入。"""
        with temp_image_png(image) as path:
            return self._do_extract(path)

    def _do_extract(self, image_path: str) -> Optional[Dict[str, float]]:
        """從檔案路徑提取 AU / emotion / gaze 特徵；偵測不到臉回 None。"""
        cropped_face, dets = self._detector.get_face(image_path)
        if cropped_face is None or dets is None:
            return None

        emotion_output, gaze_output, au_output = (
            self._predictor.predict(cropped_face)
        )
        return self._parse_outputs(emotion_output, gaze_output, au_output)

    def _parse_outputs(
        self,
        emotion_output: torch.Tensor,
        gaze_output: torch.Tensor,
        au_output: torch.Tensor,
    ) -> Dict[str, float]:
        """
        將模型 tensor 輸出轉換為特徵字典

        - AU: raw cosine similarity → sigmoid → [0, 1]
        - Emotion: logits → softmax → probability [0, 1]
        - Gaze: raw yaw/pitch（弧度）
        """
        result = {}

        # AU：sigmoid 正規化到 [0, 1]
        au_probs = torch.sigmoid(au_output).squeeze(0).cpu().numpy()
        for idx, au_name in OPENFACE_AU_INDEX.items():
            result[au_name] = float(au_probs[idx])

        # Emotion：softmax 轉機率
        emotion_probs = torch.softmax(emotion_output, dim=1).squeeze(0).cpu().numpy()
        for idx, emo_name in OPENFACE_EMOTION_INDEX.items():
            result[emo_name] = float(emotion_probs[idx])

        # Gaze：yaw, pitch
        gaze = gaze_output.squeeze(0).cpu().numpy()
        result["gaze_yaw"] = float(gaze[0])
        result["gaze_pitch"] = float(gaze[1])

        return result
