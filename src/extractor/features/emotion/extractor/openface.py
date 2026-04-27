"""
OpenFace 3.0 AU 特徵提取器

使用 openface-test Python API（FaceDetector + MultitaskPredictor）
提取 8 AU intensity、8 emotions、2D gaze
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import logging

from .base import BaseAUExtractor
from src.extractor.features.emotion.extractor.au_config import (
    OPENFACE_AU_INDEX,
    OPENFACE_EMOTION_INDEX,
    OPENFACE_GAZE_COLUMNS,
    AUExtractionConfig,
)

logger = logging.getLogger(__name__)


class OpenFaceExtractor(BaseAUExtractor):
    """
    OpenFace 3.0 AU 特徵提取器

    使用 Python API：
    - FaceDetector: RetinaFace 臉部偵測 + 裁切
    - MultitaskPredictor: EfficientNet backbone 多任務預測
      → 8 AU (DISFA), 8 emotions, 2D gaze (yaw/pitch)
    """

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        config = AUExtractionConfig()
        self._weights_dir = Path(weights_dir or config.openface_weights_dir)
        self._device = device or config.openface_device
        self._detector = None
        self._predictor = None
        self._available = None

    @property
    def tool_name(self) -> str:
        return "openface"

    @property
    def au_columns(self) -> List[str]:
        return list(OPENFACE_AU_INDEX.values())

    @property
    def emotion_columns(self) -> List[str]:
        return list(OPENFACE_EMOTION_INDEX.values())

    @property
    def extra_columns(self) -> List[str]:
        return OPENFACE_GAZE_COLUMNS

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import openface  # noqa: F401
            retina_path = self._weights_dir / "Alignment_RetinaFace.pth"
            mtl_path = self._weights_dir / "MTL_backbone.pth"
            if not retina_path.exists() or not mtl_path.exists():
                logger.warning(
                    f"OpenFace 權重不存在: {self._weights_dir}\n"
                    f"  請執行: openface download --output {self._weights_dir}"
                )
                self._available = False
            else:
                self._available = True
        except ImportError:
            logger.warning("openface-test 未安裝（pip install openface-test --no-deps）")
            self._available = False
        return self._available

    def _init_models(self):
        """延遲載入模型（首次呼叫時才初始化）"""
        if self._detector is not None:
            return

        from openface.face_detection import FaceDetector
        from openface.multitask_model import MultitaskPredictor

        retina_path = str(self._weights_dir / "Alignment_RetinaFace.pth")
        mtl_path = str(self._weights_dir / "MTL_backbone.pth")

        # openface 的 RetinaFace 內部 hardcode 了
        # "./weights/mobilenetV1X0.25_pretrain.tar"，
        # monkey-patch torch.load 重導到實際路徑
        mobilenet_path = str(self._weights_dir / "mobilenetV1X0.25_pretrain.tar")
        original_torch_load = torch.load

        def patched_load(f, *args, **kwargs):
            if isinstance(f, str) and "mobilenetV1X0.25_pretrain.tar" in f:
                f = mobilenet_path
            return original_torch_load(f, *args, **kwargs)

        logger.info(f"載入 OpenFace 3.0 模型（device={self._device}）...")
        try:
            torch.load = patched_load
            self._detector = FaceDetector(
                model_path=retina_path, device=self._device
            )
        finally:
            torch.load = original_torch_load

        self._predictor = MultitaskPredictor(
            model_path=mtl_path, device=self._device
        )
        logger.info("OpenFace 3.0 模型載入完成")

    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        對單一影像（numpy BGR）提取特徵

        FaceDetector.get_face() 需要檔案路徑，
        所以將 numpy array 暫存為臨時檔案
        """
        self._init_models()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
            cv2.imwrite(tmp_path, image)

        try:
            return self._do_extract(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _extract_from_path(self, image_path: Path) -> Optional[Dict[str, float]]:
        """直接使用檔案路徑提取，避免不必要的暫存"""
        self._init_models()
        return self._do_extract(str(image_path))

    def _do_extract(self, image_path: str) -> Optional[Dict[str, float]]:
        """
        從檔案路徑提取 AU / emotion / gaze 特徵

        流程：FaceDetector.get_face() → MultitaskPredictor.predict()
        """
        try:
            cropped_face, dets = self._detector.get_face(image_path)

            if cropped_face is None or dets is None:
                return None

            emotion_output, gaze_output, au_output = (
                self._predictor.predict(cropped_face)
            )

            return self._parse_outputs(emotion_output, gaze_output, au_output)

        except Exception as e:
            logger.debug(f"  提取失敗 {Path(image_path).name}: {e}")
            return None

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
