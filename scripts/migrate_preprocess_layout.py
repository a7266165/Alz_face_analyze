"""
把 workspace/preprocess 帶到目前的佈局。冪等——只做還沒做的部分。

目標佈局照管線的分岔點排：

    讀原圖 → FaceMesh → 選 10 張 ──┬─→ apply_mask → 轉正 → no_background/…/aligned
                          ↑        └─→ （不遮罩） → 轉正 → background/…/aligned
                       selected

    preprocess/selected/selector_<selection>/<visit>/                    分岔前，兩變體共用
    preprocess/{no_background|background}/selector_<selection>/{aligned,mirrors}/<visit>/

兩個歷史佈局都能收斂到這裡：

    A) preprocess/<selection>/{no_background|background}/{selected,aligned,mirrors}/
    B) preprocess/{no_background|background}/selector_<selection>/{selected,aligned,mirrors}/

每次搬移都是同磁碟區 rename，O(1) metadata 操作，與底下多少檔案無關。

selected 在去背分岔之前，兩個變體拿到的是同一批影像。舊佈局若在兩個變體下各留了
一份，那是目錄重整複製出來的重複資料：本腳本保留 no_background 那份抬出去，其餘的
在 --drop-duplicate-selected 時逐檔比對 SHA-256 確認完全相同後才刪除。

預設只印計畫不動手，確認無誤後加 --execute。

用法：
    python scripts/migrate_preprocess_layout.py
    python scripts/migrate_preprocess_layout.py --execute
    python scripts/migrate_preprocess_layout.py --execute --drop-duplicate-selected
"""

import sys
import shutil
import hashlib
import logging
from pathlib import Path
from typing import List, NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import PREPROCESSING_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VARIANTS = ("no_background", "background")
SELECTED_ROOT = "selected"
SELECTOR_PREFIX = "selector_"
META_FILES = ("_config.json", "_yield.csv")
TOP_LEVEL = VARIANTS + (SELECTED_ROOT,)


class Move(NamedTuple):
    what: str
    src: Path
    dst: Path


def find_legacy_selections(root: Path) -> List[str]:
    """佈局 A 殘留：preprocess 底下不是 background/no_background/selected 的都是準則。"""
    return sorted(c.name for c in root.iterdir()
                  if c.is_dir() and c.name not in TOP_LEVEL)


def find_selectors(root: Path) -> List[str]:
    """已在變體之下的 selector_* 準則名。"""
    names = set()
    for variant in VARIANTS:
        d = root / variant
        if d.is_dir():
            names |= {c.name[len(SELECTOR_PREFIX):] for c in d.iterdir()
                      if c.is_dir() and c.name.startswith(SELECTOR_PREFIX)}
    return sorted(names)


def build_plan(root: Path):
    """回傳 (step1 搬移, step2 抬 selected, 重複的 selected 目錄)。"""
    step1: List[Move] = []
    for sel in find_legacy_selections(root):
        for variant in VARIANTS:
            src = root / sel / variant
            if src.is_dir():
                step1.append(Move(
                    what=f"{sel}/{variant}",
                    src=src,
                    dst=root / variant / f"{SELECTOR_PREFIX}{sel}"))

    # step1 跑完後會有哪些準則落在變體之下
    selections = sorted(set(find_selectors(root))
                        | {m.what.split("/")[0] for m in step1})

    step2: List[Move] = []
    dups: List[Path] = []
    for sel in selections:
        dst = root / SELECTED_ROOT / f"{SELECTOR_PREFIX}{sel}"
        # 來源優先序：no_background 那份為準（新版程式碼只寫這一份）
        srcs = [p for p in
                (_selected_src(root, sel, v, step1) for v in VARIANTS)
                if p is not None]
        if not srcs:
            continue
        if dst.exists():          # 已抬過，剩下的都是重複
            dups.extend(srcs)
            continue
        step2.append(Move(what=f"{sel}", src=srcs[0], dst=dst))
        dups.extend(srcs[1:])
    return step1, step2, dups


def _selected_src(root: Path, sel: str, variant: str,
                  step1: List[Move]) -> Optional[Path]:
    """selected 在 step1 之後會落在哪裡（step1 尚未執行時也算得出來）。"""
    post = root / variant / f"{SELECTOR_PREFIX}{sel}" / SELECTED_ROOT
    if post.is_dir():
        return post
    pre = root / sel / variant / SELECTED_ROOT
    if pre.is_dir() and any(m.src == pre.parent for m in step1):
        return post          # step1 會把它搬到 post
    return None


def dirs_identical(a: Path, b: Path) -> bool:
    """逐檔比對相對路徑與 SHA-256。任何差異都回 False。"""
    fa = {p.relative_to(a): p for p in a.rglob("*") if p.is_file()}
    fb = {p.relative_to(b): p for p in b.rglob("*") if p.is_file()}
    if fa.keys() != fb.keys():
        logger.warning(f"    檔案清單不同：{len(fa)} vs {len(fb)}")
        return False
    for rel, pa in fa.items():
        if _sha256(pa) != _sha256(fb[rel]):
            logger.warning(f"    內容不同：{rel}")
            return False
    return True


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def move_meta(root: Path, step1: List[Move]) -> None:
    """佈局 A 的 selection 根目錄殘留 _config.json/_yield.csv，複製到各變體後清空。"""
    for sel in sorted({m.what.split("/")[0] for m in step1}):
        old_root = root / sel
        if not old_root.is_dir():
            continue
        dsts = [m.dst for m in step1 if m.what.startswith(f"{sel}/")]
        for name in META_FILES:
            f = old_root / name
            if not f.is_file():
                continue
            for dst in dsts:
                shutil.copy2(f, dst / name)
            f.unlink()
            logger.info(f"    {sel}/{name} → {len(dsts)} 個變體目錄")
        leftover = [c.name for c in old_root.iterdir()]
        if leftover:
            logger.warning(f"    {old_root} 還留有 {leftover}，未刪除")
        else:
            old_root.rmdir()
            logger.info(f"    移除空目錄 {sel}/")


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true",
                    help="實際搬移；未指定時只印計畫")
    ap.add_argument("--drop-duplicate-selected", action="store_true",
                    help="逐檔驗證完全相同後，刪除多餘的 selected 副本")
    args = ap.parse_args()

    root = PREPROCESSING_DIR
    logger.info("=" * 70)
    logger.info(f"preprocess: {root}")
    logger.info("=" * 70)
    logger.info("")

    step1, step2, dups = build_plan(root)

    logger.info(f"[1] 選幀準則搬進去背變體之下（{len(step1)} 項）")
    for mv in step1:
        logger.info(f"    {mv.what}/  →  {mv.dst.relative_to(root)}/")
    if not step1:
        logger.info("    （無，已是新佈局）")

    logger.info("")
    logger.info(f"[2] selected 抬到變體之上（{len(step2)} 項）")
    for mv in step2:
        logger.info(f"    {mv.src.relative_to(root)}/  →  {mv.dst.relative_to(root)}/")
    if not step2:
        logger.info("    （無）")

    logger.info("")
    logger.info(f"[3] 多餘的 selected 副本（{len(dups)} 項）")
    for d in dups:
        logger.info(f"    {d.relative_to(root)}/"
                    + ("" if args.drop_duplicate_selected else "   （保留，未加 --drop-duplicate-selected）"))
    if not dups:
        logger.info("    （無）")

    if not (step1 or step2 or dups):
        logger.info("")
        logger.info("已經是目標佈局，無事可做。")
        return

    if not args.execute:
        logger.info("")
        logger.info("以上為乾跑結果，未做任何變更。確認無誤後加 --execute。")
        logger.info("搬移前請確認沒有任何 python 程序正在讀寫 workspace/preprocess。")
        return

    logger.info("")
    logger.info("開始…")

    for mv in step1:
        mv.dst.parent.mkdir(parents=True, exist_ok=True)
        mv.src.rename(mv.dst)
        logger.info(f"    OK  {mv.what}  →  {mv.dst.relative_to(root)}")
    move_meta(root, step1)

    for mv in step2:
        mv.dst.parent.mkdir(parents=True, exist_ok=True)
        mv.src.rename(mv.dst)
        logger.info(f"    OK  {mv.src.relative_to(root)}  →  {mv.dst.relative_to(root)}")

    if args.drop_duplicate_selected:
        for d in dups:
            sel = d.parent.name[len(SELECTOR_PREFIX):]
            keeper = root / SELECTED_ROOT / f"{SELECTOR_PREFIX}{sel}"
            logger.info(f"    驗證 {d.relative_to(root)} 與 {keeper.relative_to(root)} …")
            if not dirs_identical(d, keeper):
                logger.error(f"    不完全相同，保留不刪：{d}")
                continue
            shutil.rmtree(d)
            logger.info(f"    已刪除 {d.relative_to(root)}")

    logger.info("")
    logger.info("完成。")


if __name__ == "__main__":
    main()
