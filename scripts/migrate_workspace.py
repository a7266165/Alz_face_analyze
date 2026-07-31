"""
把 workspace 下的子系統搬成 <subsystem>/<selection>/ 佈局。

背景：新增「選幀準則」這個軸（mediapipe / openface_p10r5 / openface_p15r10），
每個準則各自擁有一整棵完整的下游樹。既有結果全部歸屬 mediapipe。

    workspace/embedding/…         →  workspace/embedding/mediapipe/…
    （age / meta / emo_au / deploy 同理）

preprocess 由 scripts/migrate_preprocess_layout.py 另外處理：它的 selection 層在
去背變體之下（preprocess/{no_background|background}/selector_<selection>/）。

rotation 與 overview 不搬：前者直讀 RAW_IMAGES_DIR、與選幀無關；
後者是跨準則的比較輸出，不屬於任何一棵樹。

搬移一律在同一磁碟區內進行，是 O(1) 的 metadata 操作，與資料量無關。
因為不能把目錄直接搬進自己的子目錄，每個子系統走兩步：
    X → __migrate_tmp__X → X/<selection>

預設只印計畫不動手，確認無誤後加 --execute。

用法：
    python scripts/migrate_workspace.py                # 乾跑
    python scripts/migrate_workspace.py --stat         # 乾跑 + 統計檔數與容量（較慢）
    python scripts/migrate_workspace.py --execute      # 實際搬移
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, NamedTuple, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import WORKSPACE_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 需要搬進 <selection>/ 的子系統
# preprocess 不在此列：它的 selection 層在去背變體「之下」
# （preprocess/{no_background|background}/selector_<selection>/），見
# scripts/migrate_preprocess_layout.py。
MOVE_SUBSYSTEMS = ["embedding", "age", "meta", "emo_au", "deploy"]
# 與選幀無關，留在 workspace/ 底下
KEEP_SUBSYSTEMS = ["rotation", "overview", "preprocess"]

DEFAULT_SELECTION = "mediapipe"
TMP_PREFIX = "__migrate_tmp__"


class Move(NamedTuple):
    name: str
    src: Path   # workspace/<name>
    tmp: Path   # workspace/__migrate_tmp__<name>
    dst: Path   # workspace/<name>/<selection>


def build_plan(ws: Path, selection: str) -> List[Move]:
    """列出要搬的子系統（不存在的直接略過）。"""
    plan = []
    for name in MOVE_SUBSYSTEMS:
        src = ws / name
        if not src.is_dir():
            logger.info(f"  略過 {name}：不存在")
            continue
        plan.append(Move(
            name=name,
            src=src,
            tmp=ws / f"{TMP_PREFIX}{name}",
            dst=src / selection,
        ))
    return plan


def preflight(ws: Path, selection: str, plan: List[Move]) -> List[str]:
    """回傳阻擋搬移的問題清單，空 list 代表可以動手。"""
    problems = []

    if not ws.is_dir():
        return [f"workspace 不存在：{ws}"]

    if not plan:
        problems.append("沒有任何可搬的子系統")

    for mv in plan:
        # 已經搬過了？
        if mv.dst.exists():
            problems.append(f"{mv.name}：目標已存在，可能已搬過 → {mv.dst}")
        # 上次中斷留下的暫存目錄？
        if mv.tmp.exists():
            problems.append(f"{mv.name}：暫存目錄殘留，請先手動處理 → {mv.tmp}")
        # 目錄底下已經只剩一個 selection 名稱的子目錄 = 疑似搬過
        children = [c.name for c in mv.src.iterdir()]
        if children == [selection]:
            problems.append(f"{mv.name}：底下只有 {selection}/，看起來已完成搬移")

    return problems


def stat_dir(path: Path) -> Tuple[int, int]:
    """回傳 (檔案數, 總位元組)。走 os.scandir，345k 檔約需數十秒。"""
    n_files = 0
    n_bytes = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        n_files += 1
                        n_bytes += e.stat().st_size
        except OSError as exc:
            logger.warning(f"    無法讀取 {cur}: {exc}")
    return n_files, n_bytes


def human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024


def execute(plan: List[Move]) -> None:
    """兩步搬移。任何一步失敗都直接拋出，不做部分回捲。"""
    for mv in plan:
        logger.info(f"  {mv.name} …")
        mv.src.rename(mv.tmp)                      # X → __migrate_tmp__X
        mv.src.mkdir(parents=False, exist_ok=False)  # 重建空的 X
        mv.tmp.rename(mv.dst)                      # __migrate_tmp__X → X/<selection>

        # 驗證：X 底下必須剛好只有 <selection>
        children = [c.name for c in mv.src.iterdir()]
        if children != [mv.dst.name]:
            raise RuntimeError(
                f"{mv.name} 搬移後狀態異常，{mv.src} 底下是 {children}，預期 ['{mv.dst.name}']")
        logger.info(f"    OK  {mv.src} → {mv.dst}")


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", default=DEFAULT_SELECTION,
                    help=f"既有結果要歸屬的選幀準則名稱（預設 {DEFAULT_SELECTION}）")
    ap.add_argument("--execute", action="store_true",
                    help="實際搬移；未指定時只印計畫")
    ap.add_argument("--stat", action="store_true",
                    help="統計每個子系統的檔數與容量（走遍所有檔案，較慢）")
    args = ap.parse_args()

    ws = WORKSPACE_DIR
    logger.info("=" * 70)
    logger.info(f"workspace: {ws}")
    logger.info(f"選幀準則 : {args.selection}")
    logger.info("=" * 70)

    plan = build_plan(ws, args.selection)
    problems = preflight(ws, args.selection, plan)

    logger.info("")
    logger.info("要搬移：")
    for mv in plan:
        extra = ""
        if args.stat:
            n_files, n_bytes = stat_dir(mv.src)
            extra = f"   [{n_files:,} 檔, {human(n_bytes)}]"
        logger.info(f"  {mv.src.relative_to(ws)}/  →  {mv.dst.relative_to(ws)}/{extra}")

    logger.info("")
    logger.info("保持原位（與選幀無關）：")
    for name in KEEP_SUBSYSTEMS:
        d = ws / name
        mark = "" if d.is_dir() else "   （不存在）"
        logger.info(f"  {name}/{mark}")

    other = sorted(
        c.name for c in ws.iterdir()
        if c.is_dir()
        and c.name not in MOVE_SUBSYSTEMS
        and c.name not in KEEP_SUBSYSTEMS
    )
    if other:
        logger.info("")
        logger.info("未納入計畫的目錄（請確認是否需要處理）：")
        for name in other:
            logger.info(f"  {name}/")

    if problems:
        logger.info("")
        logger.error("阻擋項目：")
        for p in problems:
            logger.error(f"  - {p}")
        logger.error("")
        logger.error("已中止，未做任何變更。")
        sys.exit(1)

    if not args.execute:
        logger.info("")
        logger.info("以上為乾跑結果，未做任何變更。確認無誤後加 --execute 實際搬移。")
        logger.info("")
        logger.info("搬移前請確認：")
        logger.info("  1. 沒有任何 python 程序正在讀寫 workspace")
        logger.info("     特別是 repo 外的 C:/tmp/exp/full_sweep_driver.py（Tier C sweep）——")
        logger.info("     它手上的路徑物件仍指舊位置，搬移後會重建舊路徑繼續寫，造成 split-brain。")
        logger.info("  2. 搬完要同步改 src/config.py，兩者必須一起生效。")
        return

    logger.info("")
    logger.info("開始搬移…")
    execute(plan)
    logger.info("")
    logger.info("完成。接下來：")
    logger.info("  1. 改 src/config.py（加上 selection 軸），否則所有路徑都會找不到。")
    logger.info("  2. 跑一次既有的 evaluate 或 meta aggregate，確認舊結果數值完全重現。")


if __name__ == "__main__":
    main()
