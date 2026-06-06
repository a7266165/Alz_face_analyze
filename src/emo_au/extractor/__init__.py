"""emotion / AU 特徵提取器。

EmoAUExtractor 定義單一工具的契約；各工具實作於同層檔案。
EXTRACTORS 是「名稱 → 模組:類別」對照表，get_extractor() 負責懶載入 + 快取。

與 src/embedding/extractor 的差異:embedding 在 __init__ eager import 全部 class
（其模型把重依賴延遲到方法內，import class 很便宜）。emo_au 的模型多在 top-level
import torch/torchvision，且工具分散在不同 conda env（見 memory project_settings_env），
若 eager import 會在 import 本套件時就拉起 torch，於 fer_env 等環境直接失敗。故改用
importlib 懶載入:只有被點名的工具才會 import 其模組與依賴，保證 env 隔離。
"""
import importlib
import logging
from typing import Dict, Optional

from .base import EmoAUExtractor

logger = logging.getLogger(__name__)

# 名稱 → "模組:類別"（字串形式，不在 import 時觸發重依賴）
EXTRACTORS: Dict[str, str] = {
    "openface": "openface:OpenFaceExtractor",
    "libreface": "libreface:LibreFaceExtractor",
    "pyfeat": "pyfeat:PyFeatExtractor",
    "poster_pp": "poster_pp:PosterPPExtractor",
    "dan": "dan:DANExtractor",
    "emonet": "emonet:EmoNetExtractor",
    "fer": "fer:FERExtractor",
    "hsemotion": "hsemotion:HSEmotionExtractor",
    "vit": "vit:ViTExtractor",
}

_cache: Dict[tuple, Optional[EmoAUExtractor]] = {}


def get_extractor(name: str, device: str = "cuda") -> Optional[EmoAUExtractor]:
    """取得提取器（import 模組 + 建構 + is_available 探測 + 快取）。未知/不可用回 None。

    以 (name, device) 為快取鍵；只 import 被點名工具的模組，保證跨 env 隔離。
    回傳的是「尚未載入權重」的 extractor;呼叫端需自行 initialize()（eager 載入）。
    """
    key = (name, device)
    if key in _cache:
        return _cache[key]
    if name not in EXTRACTORS:
        logger.warning(f"未知的提取器: {name}")
        _cache[key] = None
        return None
    try:
        mod_name, cls_name = EXTRACTORS[name].split(":")
        module = importlib.import_module(f".{mod_name}", __package__)
        ext: Optional[EmoAUExtractor] = getattr(module, cls_name)(device=device)
        if not ext.is_available():
            logger.warning(f"✗ {name} 不可用（依賴未安裝或權重缺失）")
            ext = None
    except Exception as e:
        logger.warning(f"✗ {name} 建立失敗: {e}")
        ext = None
    _cache[key] = ext
    return ext


__all__ = [
    "EmoAUExtractor",
    "EXTRACTORS",
    "get_extractor",
]
