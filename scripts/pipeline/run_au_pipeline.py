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
from typing import List

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.config import ALIGNED_DIR
from src.modules.emotion.extractor.au_config import AU_RAW_DIR, AUExtractionConfig
from src.modules.emotion.postprocess.harmonizer import AUHarmonizer
from src.modules.emotion.postprocess.aggregator import TemporalAggregator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALL_STEPS = ["extract", "harmonize", "aggregate"]


# ═══════════════════════════════════════════════════════════
# Step 1: 提取
# ═══════════════════════════════════════════════════════════

def _get_extractor_class(tool_name: str):
    if tool_name == "openface":
        from src.modules.emotion.extractor.au.openface import OpenFaceExtractor
        return OpenFaceExtractor
    elif tool_name == "pyfeat":
        from src.modules.emotion.extractor.au.pyfeat import PyFeatExtractor
        return PyFeatExtractor
    elif tool_name == "libreface":
        from src.modules.emotion.extractor.au.libreface import LibreFaceExtractor
        return LibreFaceExtractor
    elif tool_name == "poster_pp":
        from src.modules.emotion.extractor.au.poster_pp import PosterPPExtractor
        return PosterPPExtractor
    return None


def get_subject_dirs(input_dir: Path, exclude_acs: bool = True) -> List[Path]:
    dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
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

    subject_dirs = get_subject_dirs(config.input_dir, config.exclude_acs)
    logger.info(f"共 {len(subject_dirs)} 個受試者待處理")

    for tool_name in tools:
        tool_output_dir = config.output_dir / tool_name
        tool_output_dir.mkdir(parents=True, exist_ok=True)

        extractor_cls = _get_extractor_class(tool_name)
        if extractor_cls is None:
            logger.error(f"未知工具: {tool_name}")
            continue

        extractor = extractor_cls(device=device)
        if not extractor.is_available():
            logger.error(f"{tool_name} 不可用，跳過")
            continue

        logger.info(f"\n--- {tool_name} ---")
        success, skip, fail = 0, 0, 0

        for subject_dir in tqdm(subject_dirs, desc=f"{tool_name}"):
            output_file = tool_output_dir / f"{subject_dir.name}.csv"
            if output_file.exists():
                skip += 1
                continue
            try:
                df = extractor.extract_subject(subject_dir)
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
        choices=["openface", "pyfeat", "libreface", "poster_pp", "fer", "dan", "hsemotion", "vit"],
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
    args = parser.parse_args()

    config = AUExtractionConfig()
    if args.include_acs:
        config.exclude_acs = False
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
