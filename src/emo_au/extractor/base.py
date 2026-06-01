"""
emotion / AU 特徵提取器基礎類別

定義所有 emotion / AU 提取器的共同介面。與 src/embedding/extractor 的
EmbeddingExtractor 互為鏡像:

  EmbeddingExtractor.extract(img) -> ndarray      EmoAUExtractor.extract(img) -> dict
  EmbeddingExtractor.extract_batch(images)        EmoAUExtractor.extract_batch(paths)

兩處刻意不對稱(honest asymmetry):
  - embedding 用單一 int `feature_dim` 描述 schema;emo_au 特徵具名且跨工具變長,
    故改用單一有序 `output_columns`(raw CSV 欄序的唯一真實來源,不含 frame)。
  - embedding 批次吃 arrays(mirror/original 需在記憶體配對/堆疊);emo_au 各幀獨立,
    且 OpenFace / LibreFace 的 API 只吃檔案路徑,故批次吃 image paths,並透過
    extract_from_path() 把「array-native vs path-native」差異封裝在 extractor 內。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmoAUExtractor(ABC):
    """
    emotion / AU 特徵提取器抽象基底類別

    所有提取器（OpenFace, Py-Feat, LibreFace, POSTER++, DAN, EmoNet, FER,
    HSEmotion, ViT）繼承此類。
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """工具名稱識別符"""
        ...

    @property
    @abstractmethod
    def output_columns(self) -> List[str]:
        """此工具 raw CSV 的有序欄位（不含 frame）。

        為 schema 的唯一真實來源:producer 以此 reindex 落地，保證欄序穩定。
        必須等於該工具實際輸出的欄序（含 AU / emotion / 額外欄如 gaze、valence）。
        """
        ...

    @abstractmethod
    def extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        從單一影像（numpy BGR）提取特徵

        Args:
            image: BGR 格式的影像

        Returns:
            特徵字典 {column_name: value}，提取失敗返回 None
        """
        ...

    def extract_from_path(self, image_path: Path) -> Optional[Dict[str, float]]:
        """
        從檔案路徑提取單幀特徵（producer 呼叫的統一入口）。

        預設:cv2.imread → extract。若 API 原生吃檔案路徑（如 OpenFace, LibreFace），
        子類可覆寫此方法以避免不必要的 decode→encode→暫存檔來回。
        """
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"  無法載入: {Path(image_path).name}")
            return None
        return self.extract(img)

    def extract_batch(
        self,
        image_paths: List[Path],
        verbose: bool = False,
    ) -> List[Optional[Dict[str, float]]]:
        """
        批次提取特徵（鏡像 EmbeddingExtractor.extract_batch）

        Args:
            image_paths: 影像檔案路徑列表
            verbose: 是否顯示進度

        Returns:
            特徵字典列表（與 image_paths 等長，失敗者為 None）
        """
        results: List[Optional[Dict[str, float]]] = []
        for i, image_path in enumerate(image_paths):
            if verbose and i % 10 == 0:
                logger.info(f"{self.model_name} 處理進度: {i}/{len(image_paths)}")
            try:
                results.append(self.extract_from_path(image_path))
            except Exception as e:
                logger.error(f"{self.model_name} 提取失敗: {e}")
                results.append(None)

        success = sum(1 for f in results if f is not None)
        logger.info(f"{self.model_name}: {success}/{len(image_paths)} 成功")
        return results

    def is_available(self) -> bool:
        """檢查此提取器是否可用（依賴已安裝）"""
        return True
