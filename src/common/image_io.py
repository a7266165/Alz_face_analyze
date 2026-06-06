"""影像讀取與個案資料夾遍歷的共用工具。

讀檔、遍歷、批次迴圈是 producer 的關注點 —— extractor/predictor 只吃「單張已讀好的
ndarray」。本模組是這些 I/O 的唯一真實來源:

  - imread_unicode   : unicode-safe 讀單張（Windows 上裸 cv2.imread 遇非 ASCII 路徑會靜默回 None）
  - iter_subject_dirs: 列舉個案子目錄（exclude / include prefix 過濾）
  - load_subject     : 個案資料夾 → List[img] 或 List[(Path, img)]
  - batch_apply      : 對 items 逐項套用 fn（per-item try/except→None + 進度 log）
  - temp_image_png   : BGR ndarray → 暫存 .png 路徑（給只吃路徑的第三方 API 當 adapter）

本模組需要 cv2 / numpy，由 producer（scripts/）import；extractor / predictor 的 base
不得 import 本模組，以維持其 lazy / cv2-free 的 import。
"""
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 各 modality 副檔名集合的聯集
IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def imread_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Unicode-safe 影像讀取（回傳 BGR ndarray，讀不到回 None）。

    cv2.imread 在 Windows 走 ANSI API，路徑含非 ASCII（例如中文）字元時會靜默回 None。
    改用 np.fromfile 讀進 bytes、再 cv2.imdecode 解碼，含中文的路徑也能讀。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def iter_subject_dirs(
    root: Path,
    *,
    exclude_prefix: Optional[str] = None,
    include_prefix: Optional[str] = None,
) -> List[Path]:
    """列舉 root 下的個案子目錄（依名稱排序）。

    include_prefix: 只保留 name 以此開頭者（例如 --subject-prefix EACS_）。
    exclude_prefix: 排除 name 以此開頭者（例如排除 ACS）。
    給定 include_prefix 時略過 exclude_prefix（前者較專一）。
    """
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if include_prefix:
        return [d for d in dirs if d.name.startswith(include_prefix)]
    if exclude_prefix:
        return [d for d in dirs if not d.name.startswith(exclude_prefix)]
    return dirs


def load_subject(
    subject_dir: Path,
    *,
    exts: Tuple[str, ...] = IMAGE_EXTS,
    with_path: bool = False,
    max_images: Optional[int] = None,
) -> Union[List[np.ndarray], List[Tuple[Path, np.ndarray]]]:
    """讀取個案資料夾內的影像（依檔名排序），讀不到的檔案略過。

    with_path=False -> List[ndarray]
    with_path=True  -> List[(Path, ndarray)]（producer 自取 .stem 當 frame、.name 存檔）
    max_images: 限制張數（None 為不限）。
    """
    out: list = []
    for fp in sorted(subject_dir.iterdir()):
        if not (fp.is_file() and fp.suffix.lower() in exts):
            continue
        img = imread_unicode(fp)
        if img is None:
            continue
        out.append((fp, img) if with_path else img)
        if max_images is not None and len(out) >= max_images:
            break
    return out


def batch_apply(
    fn: Callable[[Any], Any],
    items: list,
    *,
    verbose: bool = False,
    label: str = "",
) -> List[Optional[Any]]:
    """對 items 逐項套用 fn，回傳等長 list（保留 None、對齊索引）。

    單項拋例外 → 該項記為 None（不中斷整批）。取代各 extractor base 先前重複的批次迴圈。
    """
    results: List[Optional[Any]] = []
    n = len(items)
    for i, item in enumerate(items):
        if verbose and i % 10 == 0:
            logger.info(f"{label} 進度: {i}/{n}")
        try:
            results.append(fn(item))
        except Exception as e:
            logger.error(f"{label} 第 {i} 項失敗: {e}")
            results.append(None)
    if verbose:
        ok = sum(1 for r in results if r is not None)
        logger.info(f"{label}: {ok}/{n} 成功")
    return results


@contextmanager
def temp_image_png(image: np.ndarray):
    """把 BGR ndarray 寫成暫存 .png、yield 其路徑，離開時刪檔（給只吃路徑的第三方 API 當 adapter）。

    用 .png 無損暫存:與「直接讀原始對齊 PNG」像素一致，避免 JPEG 重壓縮改變模型輸出。
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
        cv2.imwrite(tmp_path, image)
    try:
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)
