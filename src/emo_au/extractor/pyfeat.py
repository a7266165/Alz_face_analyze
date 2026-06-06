"""Py-Feat AU/emotion 提取器:feat.Detector API，輸出 20 AU + 7 emotion 機率（皆 [0,1]）。"""

from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import logging

try:
    from torch.utils.data import default_collate
except ImportError:  # 舊版 torch
    from torch.utils.data._utils.collate import default_collate

from .base import EmoAUExtractor
from src.emo_au.extractor.au_config import PYFEAT_AU_MAP, PYFEAT_EMOTION_MAP

logger = logging.getLogger(__name__)


class PyFeatExtractor(EmoAUExtractor):
    """feat.Detector 偵測人臉並分析 AU/情緒，輸出 20 AU + 7 emotion 機率。"""

    def __init__(self, device: str = "cpu", **kwargs):
        self._detector = None
        self._device = device

    @property
    def model_name(self) -> str:
        return "pyfeat"

    @property
    def output_columns(self) -> List[str]:
        # 落地序照 extract:AU probabilities → emotion probabilities
        return list(PYFEAT_AU_MAP.keys()) + list(PYFEAT_EMOTION_MAP.keys())

    def _probe(self) -> bool:
        return self._probe_import("feat", "pip install py-feat")

    def _load(self) -> None:
        """載入 Py-Feat Detector。

        Monkey-patch Fex.compute_identities 成 no-op —
        pyfeat 0.6 在 Windows 跑 identity embeddings (FaceNet) 會有 GIL
        violation fatal crash（>5000 張時必現）。我們只要 AU + emotion，
        不需要 identity 聚合。
        """
        from feat import Detector
        from feat.data import Fex
        if not getattr(Fex.compute_identities, "_patched", False):
            def _noop_compute_identities(self, *args, **kwargs):
                # inplace=True 時原版返回 None；維持同語義
                if kwargs.get("inplace", False):
                    return None
                return self
            _noop_compute_identities._patched = True
            Fex.compute_identities = _noop_compute_identities
        logger.info("載入 Py-Feat Detector（compute_identities 已 no-op）...")
        self._detector = Detector(device=self._device)
        logger.info(f"Py-Feat Detector 載入完成 (device={self._device})")

    def _extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """從單一影像（BGR ndarray）提取 AU 和情緒特徵（假設已 initialize()）。

        直接把 array 餵進 py-feat，不寫 temp 檔。py-feat 的 detect_image 在
        output_size=None（我們的用法）時，ImageDataset 只做 read_image（讀成 RGB
        uint8 tensor、Scale=1、不 resize/pad），真正的前處理在各 detect_* 內部。
        故此處複製 detect_image 的 waterfall，只把「讀檔成 tensor」換成「由傳入
        array 轉成等價 tensor」，輸出與讀檔路徑 byte-identical。
        """
        fex = self._detect_array(image)
        if fex is None or len(fex) == 0:
            return None

        row = fex.iloc[0]
        features = {}
        for pf_col in PYFEAT_AU_MAP:
            if pf_col in fex.columns:
                features[pf_col] = float(row[pf_col])
        for pf_col in PYFEAT_EMOTION_MAP:
            if pf_col in fex.columns:
                features[pf_col] = float(row[pf_col])

        return features if features else None

    def _detect_array(self, image: np.ndarray):
        """把 BGR ndarray 跑完 py-feat 偵測 waterfall，回傳 Fex（與 detect_image 同路徑）。

        Image tensor 構造成與 torchvision.io.read_image 等價（RGB、uint8、CHW），
        batch_data 用 default_collate（與 detect_image 內 DataLoader(batch_size=1) 的
        collate 相同），故 _run_detection_waterfall / _create_fex 收到的輸入與讀檔
        路徑完全一致。
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # read_image() 回 (C,H,W) uint8 RGB;對齊其 dtype / 通道序 / 維度排列
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        item = {
            "Image": img_tensor,
            "Scale": 1.0,
            "Padding": {"Left": 0, "Top": 0, "Right": 0, "Bottom": 0},
            "FileNames": "<in_memory>",
        }
        batch_data = default_collate([item])

        d = self._detector
        # 與 detect_image 內部呼叫一致:6 個 model-kwargs 皆用空 dict（預設）。
        faces, landmarks, poses, aus, emotions, identities = (
            d._run_detection_waterfall(
                batch_data, 0.5, {}, {}, {}, {}, {}, {})
        )
        return d._create_fex(
            faces, landmarks, poses, aus, emotions, identities,
            batch_data["FileNames"], 0)
