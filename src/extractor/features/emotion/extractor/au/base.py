"""
AU 特徵提取器基礎類別

定義所有 AU 提取器的共同介面
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseAUExtractor(ABC):
    """
    AU 特徵提取器抽象基底類別

    所有 AU 提取器（OpenFace, Py-Feat, LibreFace）繼承此類
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """工具名稱識別符"""
        ...

    @property
    @abstractmethod
    def au_columns(self) -> List[str]:
        """此工具輸出的所有 AU 欄位名稱（原始名稱）"""
        ...

    @property
    @abstractmethod
    def emotion_columns(self) -> List[str]:
        """此工具輸出的情緒欄位名稱（原始名稱）"""
        ...

    @property
    @abstractmethod
    def extra_columns(self) -> List[str]:
        """此工具輸出的額外欄位（如 gaze, pose）"""
        ...

    @abstractmethod
    def extract_frame(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        從單一影像中提取 AU 特徵

        Args:
            image: BGR 格式的影像

        Returns:
            特徵字典 {column_name: value}，提取失敗返回 None
        """
        ...

    def extract_subject(self, subject_dir: Path) -> Optional[pd.DataFrame]:
        """
        提取單一受試者所有影像的 AU 特徵

        Args:
            subject_dir: 受試者影像目錄

        Returns:
            DataFrame (n_frames, n_features)，每列一幀
        """
        image_paths = sorted(
            [p for p in subject_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")],
            key=lambda p: p.name,
        )

        if not image_paths:
            logger.warning(f"  {subject_dir.name}: 沒有找到影像")
            return None

        results = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"  無法載入: {img_path.name}")
                continue

            frame_data = self.extract_frame(img)
            if frame_data is not None:
                frame_data["frame"] = img_path.stem
                results.append(frame_data)

        if not results:
            logger.warning(f"  {subject_dir.name}: 沒有成功提取任何幀")
            return None

        df = pd.DataFrame(results)
        # 將 frame 移到第一欄
        cols = ["frame"] + [c for c in df.columns if c != "frame"]
        return df[cols]

    def is_available(self) -> bool:
        """檢查此提取器是否可用（依賴已安裝）"""
        return True
