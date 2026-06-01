"""
AU 特徵 Pipeline: 提取 → 統一化 → 聚合

Usage:
    python run_au_pipeline.py --tools openface pyfeat libreface --device cuda
    python run_au_pipeline.py --tools openface --steps extract harmonize aggregate
    python run_au_pipeline.py --steps harmonize aggregate   # 跳過提取，只做後處理
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.config import preprocess_dir
from src.emo_au.extractor import get_extractor
from src.emo_au.extractor.au_config import AU_RAW_DIR, AUExtractionConfig

ALIGNED_DIR = preprocess_dir("aligned")
from src.emo_au.postprocess.harmonizer import AUHarmonizer
from src.emo_au.postprocess.aggregator import TemporalAggregator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALL_STEPS = ["extract", "harmonize", "aggregate"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ═══════════════════════════════════════════════════════════
# Step 1: 提取
# ═══════════════════════════════════════════════════════════

def _extract_subject(extractor, subject_dir: Path) -> Optional[pd.DataFrame]:
    """掃描 subject 目錄所有影像 → 每幀一列的 DataFrame。

    （此邏輯原在 EmoAUExtractor.extract_subject，依鏡像 embedding 的設計移回 producer。）
    欄序由 extractor.output_columns 決定（reindex），保證落地穩定;缺欄補 0.0。
    """
    image_paths = sorted(
        [p for p in subject_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )
    if not image_paths:
        logger.warning(f"  {subject_dir.name}: 沒有找到影像")
        return None

    rows = []
    for img_path in image_paths:
        frame_data = extractor.extract_from_path(img_path)
        if frame_data is not None:
            frame_data["frame"] = img_path.stem
            rows.append(frame_data)

    if not rows:
        logger.warning(f"  {subject_dir.name}: 沒有成功提取任何幀")
        return None

    df = pd.DataFrame(rows)
    return df.reindex(columns=["frame"] + extractor.output_columns, fill_value=0.0)


def get_subject_dirs(input_dir: Path, exclude_acs: bool = True,
                     subject_prefix: str = None) -> List[Path]:
    dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if subject_prefix:
        dirs = [d for d in dirs if d.name.startswith(subject_prefix)]
        logger.info(f"prefix={subject_prefix} 過濾後剩餘 {len(dirs)} 個受試者")
        return dirs
    if exclude_acs:
        dirs = [d for d in dirs if not d.name.startswith("ACS")]
        logger.info(f"排除 ACS 後剩餘 {len(dirs)} 個受試者")
    return dirs


def run_extract(tools: List[str], config: AUExtractionConfig, device: str):
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: AU 特徵提取")
    logger.info("=" * 60)

    if not config.input_dir.exists():
        logger.error(f"輸入目錄不存在: {config.input_dir}")
        return

    subject_dirs = get_subject_dirs(
        config.input_dir, config.exclude_acs,
        subject_prefix=getattr(config, "_subject_prefix", None),
    )
    logger.info(f"共 {len(subject_dirs)} 個受試者待處理")

    for tool_name in tools:
        tool_output_dir = config.output_dir / tool_name
        tool_output_dir.mkdir(parents=True, exist_ok=True)

        extractor = get_extractor(tool_name, device=device)
        if extractor is None:
            logger.error(f"{tool_name} 未知或不可用，跳過")
            continue

        logger.info(f"\n--- {tool_name} ---")
        success, skip, fail = 0, 0, 0

        for subject_dir in tqdm(subject_dirs, desc=f"{tool_name}"):
            output_file = tool_output_dir / f"{subject_dir.name}.csv"
            if output_file.exists():
                skip += 1
                continue
            try:
                df = _extract_subject(extractor, subject_dir)
                if df is not None and len(df) > 0:
                    df.to_csv(output_file, index=False, encoding="utf-8-sig")
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                logger.error(f"  {subject_dir.name}: {e}")
                fail += 1

        logger.info(f"{tool_name}: 成功={success}, 跳過={skip}, 失敗={fail}")


# ═══════════════════════════════════════════════════════════
# Step 2: 統一化
# ═══════════════════════════════════════════════════════════

def run_harmonize(tools: List[str]):
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: AU 特徵 Harmonization")
    logger.info("=" * 60)

    harmonizer = AUHarmonizer()
    for tool in tools:
        count = harmonizer.harmonize_all(tool)
        logger.info(f"{tool}: {count} 個受試者完成")


# ═══════════════════════════════════════════════════════════
# Step 3: 聚合
# ═══════════════════════════════════════════════════════════

def run_aggregate(tools: List[str], min_frames: int = 3):
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: AU 特徵時序聚合")
    logger.info("=" * 60)

    aggregator = TemporalAggregator(min_frames=min_frames)
    for tool in tools:
        for fs in ["harmonized", "extended"]:
            df = aggregator.aggregate_tool(tool, feature_set=fs)
            if df is not None:
                logger.info(f"  {tool}_{fs}: {df.shape}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AU 特徵 Pipeline")
    parser.add_argument(
        "--tools", nargs="+", default=None,
        choices=["openface", "pyfeat", "libreface", "poster_pp",
                 "dan", "emonet", "fer", "hsemotion", "vit"],
        help="指定工具（預設依 config）",
    )
    parser.add_argument(
        "--steps", nargs="+", default=ALL_STEPS,
        choices=ALL_STEPS,
        help="指定步驟（預設全部）",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--include-acs", action="store_true",
        help="包含 ACS 受試者（預設排除）",
    )
    parser.add_argument(
        "--aligned-dir", type=Path, default=None,
        help="覆寫對齊影像目錄；留空用 config.input_dir (ALIGNED_DIR)",
    )
    parser.add_argument(
        "--subject-prefix", default=None,
        help="只處理 ID 開頭符合 prefix 的受試者 (e.g. EACS_)；覆寫 --include-acs",
    )
    args = parser.parse_args()

    config = AUExtractionConfig()
    if args.include_acs:
        config.exclude_acs = False
    if args.aligned_dir is not None:
        config.input_dir = args.aligned_dir
    if args.subject_prefix:
        # prefix 走自訂路徑，關掉 exclude_acs 保證不會誤殺 EACS_ 之類
        config.exclude_acs = False
        config._subject_prefix = args.subject_prefix
    tools = args.tools or config.tools

    logger.info(f"工具: {tools}")
    logger.info(f"步驟: {args.steps}")

    if "extract" in args.steps:
        run_extract(tools, config, args.device)
    if "harmonize" in args.steps:
        run_harmonize(tools)
    if "aggregate" in args.steps:
        run_aggregate(tools, config.min_frames)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline 完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
