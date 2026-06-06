"""
emotion / AU 特徵提取器基礎類別

定義所有 emotion / AU 提取器的共同介面。與 src/embedding/extractor 的
EmbeddingExtractor、src/age/predictor 的 BasePredictor 互為鏡像:皆為
「純函數 (已讀好的 image: ndarray) → result」—— 讀檔、遍歷、批次迴圈都是
producer 的工作（見 src/common/image_io），extractor 只負責對單張臉做事。

  EmbeddingExtractor.extract(img) -> ndarray      EmoAUExtractor.extract(img) -> dict

生命週期（三家族一致）:建構（瑣碎）→ is_available()（便宜探測，不載入）→
initialize()（eager 載入，producer 顯式呼叫一次）→ extract()（純推論，假設已 initialize）。
共同骨架（探測結果快取、冪等載入、失敗回 None）由本基底以 template method 提供，子類
只實作差異 hook:_probe() / _load() / _extract()。

唯一刻意不對稱(honest asymmetry):embedding 用單一 int `feature_dim` 描述 schema；
emo_au 特徵具名且跨工具變長，故用單一有序 `output_columns`（該工具「有哪些欄」的真實
來源；落地的統一物理欄序另由 au_config.canonical_order 決定，跨工具一致）。
"""

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmoAUExtractor(ABC):
    """emotion / AU 特徵提取器抽象基底。

    子類（OpenFace, Py-Feat, LibreFace, POSTER++, DAN, EmoNet, FER, HSEmotion, ViT）
    宣告 model_name / output_columns 並實作 _extract()，視需要覆寫 _probe()（依賴/權重
    探測）與 _load()（eager 載入）。
    """

    _available: Optional[bool] = None
    _initialized: bool = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """工具名稱識別符。"""
        ...

    @property
    @abstractmethod
    def output_columns(self) -> List[str]:
        """此工具輸出的有序欄位（不含 frame）—— 該工具「有哪些欄」的唯一真實來源。

        欄名與值以此為準；producer 落地時會再套用全庫統一物理序
        （au_config.canonical_order），故 npz / schema 的欄序是跨工具一致的標準序，
        不一定等於這裡的原生序。必須涵蓋該工具實際輸出的所有欄（AU / emotion /
        額外欄如 gaze、valence）。
        """
        ...

    # ── 生命週期 template method（子類只填差異 hook）────────────────────────────

    def initialize(self) -> None:
        """eager 載入模型權重等資源（producer 取得 extractor 後呼叫一次，冪等）。

        實際載入交給子類 _load()；對「由套件內部自管載入」的工具（如 LibreFace），
        _load() 為預設 no-op。is_available() 已先行篩選，真正載入失敗應於 _load() raise。
        """
        if self._initialized:
            return
        self._load()
        self._initialized = True

    def _load(self) -> None:
        """載入權重/建構模型（子類覆寫；預設 no-op，供自管載入的工具用）。"""

    def extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """從單一影像提取特徵（假設已 initialize()）。

        Args:
            image: BGR 格式的影像。

        Returns:
            特徵字典 {column_name: value}，偵測/推論失敗回 None。
        """
        try:
            return self._extract(image)
        except Exception as e:
            logger.debug(f"  {self.model_name} 提取失敗: {e}")
            return None

    @abstractmethod
    def _extract(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """子類實作:對單張 BGR 影像做純推論回特徵 dict（不必自包 try/except）。"""
        ...

    # ── 可用性探測 template method ───────────────────────────────────────────

    def is_available(self) -> bool:
        """此提取器是否可用（依賴已安裝、權重存在；結果快取）。registry 據此篩選、不載入。"""
        if self._available is None:
            self._available = self._probe()
        return self._available

    def _probe(self) -> bool:
        """探測依賴/權重是否就緒（子類覆寫；預設 True）。可用 _probe_import / _probe_weights。"""
        return True

    def _probe_import(self, module: str, hint: str) -> bool:
        """探測 module 是否可 import；不可用時 warn（附安裝提示）並回 False。"""
        try:
            importlib.import_module(module)
            return True
        except ImportError:
            logger.warning(f"{self.model_name} 需要 {module}；{hint}")
            return False

    def _probe_weights(self, *paths) -> bool:
        """探測權重檔是否都存在；任一缺失即 warn 並回 False。"""
        missing = [p for p in paths if not p.exists()]
        for p in missing:
            logger.warning(f"{self.model_name} 權重不存在: {p}")
        return not missing
