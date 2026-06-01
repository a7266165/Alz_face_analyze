"""特徵提取器。

EmbeddingExtractor 定義單一模型的契約；各模型實作於同層檔案。
EXTRACTORS 是「名稱 → 類別」對照表，get_extractor() 負責建構 + is_available 探測 + 快取
（回傳未載入的 extractor，呼叫端自行 initialize()）。
"""
import logging
from typing import Dict, List, Optional, Type

from .base import EmbeddingExtractor
from .dlib import DlibExtractor
from .arcface import ArcFaceExtractor
from .topofr import TopoFRExtractor
from .vggface import VGGFaceExtractor

logger = logging.getLogger(__name__)

# 名稱 → 類別（加新模型只要在這裡多一行）
EXTRACTORS: Dict[str, Type[EmbeddingExtractor]] = {
    "dlib": DlibExtractor,
    "arcface": ArcFaceExtractor,
    "topofr": TopoFRExtractor,
    "vggface": VGGFaceExtractor,
}

_cache: Dict[str, Optional[EmbeddingExtractor]] = {}


def get_extractor(name: str) -> Optional[EmbeddingExtractor]:
    """取得提取器（建構 + is_available 探測 + 快取）。未知/不可用一律回 None。

    回傳的是「尚未載入權重」的 extractor;呼叫端需自行 initialize()（eager 載入）。
    """
    if name in _cache:
        return _cache[name]
    if name not in EXTRACTORS:
        logger.warning(f"未知的提取器: {name}")
        _cache[name] = None
        return None
    try:
        ext: Optional[EmbeddingExtractor] = EXTRACTORS[name]()
        if not ext.is_available():
            logger.warning(f"✗ {name} 不可用（依賴未安裝或權重缺失）")
            ext = None
    except Exception as e:
        logger.warning(f"✗ {name} 建立失敗: {e}")
        ext = None
    _cache[name] = ext
    return ext


def available_extractors(names: Optional[List[str]] = None) -> List[str]:
    """實際可用的模型名稱（會觸發 is_available 探測，不載入權重）。"""
    return [n for n in (names or EXTRACTORS) if get_extractor(n) is not None]


__all__ = [
    "EmbeddingExtractor",
    "DlibExtractor",
    "ArcFaceExtractor",
    "TopoFRExtractor",
    "VGGFaceExtractor",
    "EXTRACTORS",
    "get_extractor",
    "available_extractors",
]
