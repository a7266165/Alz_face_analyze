"""
emotion / AU 特徵提取入口（embedding extractor 的姊妹 producer）。

對 aligned 影像逐幀跑各工具，把每名受試者的逐幀特徵堆疊成 (n_frames, n_dim) 存成
{tool}/<subj>.npz，並維護一份描述各工具欄位的 _schema.json。

- 輸入沿用 embedding / age 的標準來源 preprocess_dir("aligned")（每人 ≤ n_select 張，
  預設 10），故輸出與 embedding original / age 一致：每人一個 (n_frames, n_dim) 陣列。
- 各工具輸出的欄不一致，但落地一律套用全庫統一物理欄序（au_config.canonical_order）：
  共有情緒 → contempt → AU(依編號) → 其他額外欄。每個工具只填自己有的欄。
- 存成 .npz（自描述）：data=(n_frames, n_dim) float32，columns=該工具的統一欄序；缺欄補
  NaN（下游聚合 ~isnan 濾掉，不會被當成真實 0）。_schema.json 為跨工具索引，跨多次執行
  累積合併（各工具在自己的 conda env 跑完後逐步補齊）。

Usage:
    python extract.py --tools openface --device cuda
    python extract.py --tools pyfeat dan vit          # 同一 env 內多工具
    python extract.py --tools openface --bg-variant background
    python extract.py --tools dan --output-dir <自訂輸出根>
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import EMO_AU_FEATURES_DIR, preprocess_dir
from src.common.image_io import batch_apply, iter_subject_dirs, load_subject
from src.emo_au.extractor import EXTRACTORS, get_extractor
from src.emo_au.extractor.au_config import HARMONIZED_EMOTIONS, canonical_order

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全部 9 個已實作工具（= registry EXTRACTORS）。前 8 個對應 workspace/emo_au/features
# 既有方法；emonet 額外提供 valence / arousal（[-1,1]），故一併納入預設。各工具分屬不同
# conda env，當前 env 取不到的會被跳過。
DEFAULT_TOOLS = [
    "openface", "libreface", "pyfeat", "dan",
    "hsemotion", "vit", "poster_pp", "fer", "emonet",
]


def setup_cpu_limit(max_cores: Optional[int]):
    """限制 CPU 核心數；max_cores 為 None 表示不限制。"""
    if max_cores is None:
        logger.info("CPU 核心數: 不限制")
        return
    logger.info(f"CPU 核心數: 限制為 {max_cores} 核心")
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(max_cores)
    try:
        cv2.setNumThreads(max_cores)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(max_cores)
        torch.set_num_interop_threads(max_cores)
    except Exception:
        pass


def _extract_subject(
    extractor, subject_dir: Path, cols: List[str]
) -> Optional[np.ndarray]:
    """個案資料夾 → (n_frames, n_dim) 陣列；無有效幀回 None。

    讀檔/遍歷由 common.image_io.load_subject 負責，extractor 只吃單張 ndarray。
    每幀 dict 依統一欄序 cols reindex（缺欄補 NaN，誠實標記「無此資料」，下游聚合會
    ~isnan 濾掉，不會被當成真實 0 拉低統計）。
    """
    images = load_subject(subject_dir)
    if not images:
        logger.warning(f"  {subject_dir.name}: 沒有找到影像")
        return None

    feats = batch_apply(extractor.extract, images, label=extractor.model_name)
    rows = [
        [float(frame.get(c, np.nan)) for c in cols]
        for frame in feats
        if frame is not None
    ]
    if not rows:
        logger.warning(f"  {subject_dir.name}: 沒有成功提取任何幀")
        return None
    return np.asarray(rows, dtype=np.float32)


def _update_schema(
    output_dir: Path, produced: Dict[str, List[str]], bg_variant: str, input_dir: Path
):
    """把本次跑過的工具欄位合併進 output_dir/_schema.json（跨執行累積）。

    columns 為統一物理欄序（au_config.canonical_order）；每個 .npz 已自帶同序的 columns，
    schema 僅作為「免逐檔開啟」的方法/欄位索引。另記 bg_variant / input_dir 作為來源
    溯源（輸出路徑不含 bg 層，靠這裡標明這批特徵來自去背或保留背景的相片）。
    """
    schema_file = output_dir / "_schema.json"
    if schema_file.exists():
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    else:
        schema = {}

    # 輸出路徑扁平、不含 bg 層；若既有 schema 是另一種 bg，混進同一目錄會污染。
    prev_bg = schema.get("bg_variant")
    if prev_bg is not None and prev_bg != bg_variant:
        logger.warning(
            f"此目錄既有 bg_variant={prev_bg}，本次={bg_variant}；"
            f"扁平路徑無法分流，混用會污染 —— 請改用不同 --output-dir。"
        )

    methods = schema.get("methods", {})
    for tool, cols in produced.items():
        methods[tool] = {"columns": list(cols), "n_dim": len(cols)}

    schema["bg_variant"] = bg_variant
    schema["input_dir"] = str(input_dir)
    schema["shared_emotions"] = list(HARMONIZED_EMOTIONS)
    schema["methods"] = methods
    schema["comment"] = (
        "Each <method>/<subject>.npz holds data=(n_frames, n_dim) float32 + columns "
        "(str). `columns` follows the project-wide unified physical order "
        "(au_config.canonical_order): shared emotions, then contempt, then AUs by "
        "number, then other extras. Missing cells are NaN. This file is a "
        "convenience index; each npz is self-describing via its own `columns`."
    )
    schema_file.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"schema 已更新: {schema_file}")


def run_extraction(
    tools: List[str],
    bg_variant: str,
    device: str,
    max_cpu_cores: Optional[int],
    output_dir: Path,
):
    setup_cpu_limit(max_cpu_cores)
    input_dir = preprocess_dir("aligned", background=bg_variant == "background")

    logger.info("=" * 70)
    logger.info(f"emotion / AU 特徵提取（{bg_variant}）")
    logger.info("=" * 70)
    logger.info(f"影像來源: {input_dir}")
    logger.info(f"輸出目錄: {output_dir} / {{tool}} / <subj>.npz")
    logger.info(f"工具: {tools}")

    if not input_dir.exists():
        logger.error(f"找不到影像來源目錄: {input_dir}")
        sys.exit(1)

    subject_dirs = iter_subject_dirs(input_dir)
    logger.info(f"找到 {len(subject_dirs)} 個受試者")
    if not subject_dirs:
        logger.error("沒有找到任何受試者目錄")
        return

    produced: Dict[str, List[str]] = {}
    start = datetime.now()
    for tool in tools:
        extractor = get_extractor(tool, device=device)
        if extractor is None:
            logger.warning(f"{tool} 不可用（未安裝或權重缺失），跳過")
            continue
        # 顯式 eager 載入：fail-fast，載入錯誤在進 subject 迴圈前就爆，不會被逐
        # subject 的 try/except 吞成「每個受試者都失敗」。單一 tool 失敗只跳過該 tool。
        try:
            extractor.initialize()
        except Exception as e:
            logger.error(f"{tool} 初始化失敗，跳過: {e}")
            continue

        cols = canonical_order(extractor.output_columns)  # 全庫統一物理欄序
        tool_dir = output_dir / tool
        tool_dir.mkdir(parents=True, exist_ok=True)
        produced[tool] = cols

        logger.info(f"\n--- {tool} (n_dim={len(cols)}) ---")
        success, skip, fail = 0, 0, 0
        for subject_dir in tqdm(subject_dirs, desc=tool):
            out_path = tool_dir / f"{subject_dir.name}.npz"
            if out_path.exists():  # checkpoint
                skip += 1
                continue
            try:
                arr = _extract_subject(extractor, subject_dir, cols)
                if arr is not None:
                    np.savez(out_path, data=arr, columns=np.asarray(cols))
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                logger.error(f"  {subject_dir.name}: {e}")
                fail += 1
        logger.info(f"{tool}: 成功={success}, 跳過={skip}, 失敗={fail}")

    if not produced:
        logger.error("沒有任何可用工具，未產生輸出")
        return

    _update_schema(output_dir, produced, bg_variant, input_dir)
    logger.info("=" * 70)
    logger.info(f"完成 — 工具={list(produced)} | 總耗時: {datetime.now() - start}")
    logger.info("=" * 70)


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tools",
        nargs="+",
        default=DEFAULT_TOOLS,
        choices=list(EXTRACTORS),
        help=f"提取工具（預設: {DEFAULT_TOOLS}）",
    )
    ap.add_argument(
        "--bg-variant", choices=["no_background", "background"], default="no_background"
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--max-cpu-cores", type=int, default=2, help="限制 CPU 核心數；傳 0 表示不限制"
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=EMO_AU_FEATURES_DIR,
        help="覆寫輸出根（預設 EMO_AU_FEATURES_DIR）",
    )
    args = ap.parse_args()

    max_cpu = None if args.max_cpu_cores == 0 else args.max_cpu_cores
    run_extraction(
        args.tools, args.bg_variant, args.device, max_cpu, args.output_dir
    )


if __name__ == "__main__":
    main()
