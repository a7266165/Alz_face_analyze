"""頭部旋轉角度分析 — producer。

遍歷各類別中相片數 == TARGET_COUNT 的資料夾，每個資料夾以向量法與 PnP 法各算一次
頭部旋轉角度，存訊號圖（ROTATION_FIG_DIR）與角度序列 npy（ROTATION_FEATURES_DIR）。
圖與 npy 都已存在的資料夾自動跳過。
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401  副作用：把 PROJECT_ROOT/scripts 加進 sys.path

from src.common.cohort import base_id_of
from src.config import RAW_IMAGES_DIR, ROTATION_FEATURES_DIR, ROTATION_FIG_DIR
from src.rotation import (
    AnglePlotter,
    PnPAngleCalculator,
    SequenceResult,
    VectorAngleCalculator,
)

CATEGORIES = {
    "ACS": RAW_IMAGES_DIR / "health" / "ACS",
    "NAD": RAW_IMAGES_DIR / "health" / "NAD",
    "Patient": RAW_IMAGES_DIR / "patient" / "good",
}

OUTPUT_DIR_PNP = ROTATION_FIG_DIR / "PnP"
OUTPUT_DIR_VECTOR = ROTATION_FIG_DIR / "vector_angle"
FEATURES_DIR_PNP = ROTATION_FEATURES_DIR / "PnP"
FEATURES_DIR_VECTOR = ROTATION_FEATURES_DIR / "vector_angle"

TARGET_COUNT = 1200


# ── coordinator：對單一資料夾跑兩法、存圖 + 角度 npy ──────────────────────────

def _save_angles_npy(result: SequenceResult, output_path: Path) -> None:
    """角度序列存 npy，shape (3, T)，列序 pitch/yaw/roll。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array([result.pitch_list, result.yaw_list, result.roll_list]))


def process_single_folder(folder_path: Path, verbose: bool = True) -> Tuple[SequenceResult, SequenceResult]:
    """資料夾分別用向量法、PnP 法計算，各存訊號圖與角度 npy，回 (vector, pnp)。"""
    name = folder_path.name

    if verbose:
        print("  [Vector] ", end="")
    vector_calc = VectorAngleCalculator()
    vector_result = vector_calc.process_folder(folder_path, verbose=verbose)
    vector_calc.close()
    AnglePlotter.plot_sequence(vector_result, OUTPUT_DIR_VECTOR / f"{name}.png")
    _save_angles_npy(vector_result, FEATURES_DIR_VECTOR / f"{name}.npy")

    if verbose:
        print("  [PnP] ", end="")
    pnp_calc = PnPAngleCalculator()
    pnp_result = pnp_calc.process_folder(folder_path, verbose=verbose)
    pnp_calc.close()
    AnglePlotter.plot_sequence(pnp_result, OUTPUT_DIR_PNP / f"{name}.png")
    _save_angles_npy(pnp_result, FEATURES_DIR_PNP / f"{name}.npy")

    return vector_result, pnp_result


# ── 資料夾挑選與斷點 ─────────────────────────────────────────────────────────

def get_qualified_folders(category_path: Path) -> List[Path]:
    """類別下相片數恰為 TARGET_COUNT 的資料夾，依名稱排序。"""
    if not category_path.exists():
        return []
    return sorted(
        (d for d in category_path.iterdir()
         if d.is_dir() and len(list(d.glob("*.jpg"))) == TARGET_COUNT),
        key=lambda d: d.name,
    )


def is_fully_processed(folder_name: str) -> bool:
    """兩法的訊號圖（.png）與角度 npy 是否都已存在。"""
    figs = (d / f"{folder_name}.png" for d in (OUTPUT_DIR_PNP, OUTPUT_DIR_VECTOR))
    npys = (d / f"{folder_name}.npy" for d in (FEATURES_DIR_PNP, FEATURES_DIR_VECTOR))
    return all(p.exists() for p in (*figs, *npys))


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"頭部旋轉角度分析  開始: {datetime.now():%Y-%m-%d %H:%M:%S}  目標張數: {TARGET_COUNT}")
    print("=" * 70)

    for d in (OUTPUT_DIR_PNP, OUTPUT_DIR_VECTOR, FEATURES_DIR_PNP, FEATURES_DIR_VECTOR):
        d.mkdir(parents=True, exist_ok=True)

    stats = {cat: {"persons": set(), "sessions": 0, "ok": 0, "fail": 0, "skip": 0}
             for cat in CATEGORIES}

    for category, category_path in CATEGORIES.items():
        print(f"\n{'─' * 70}\n類別: {category}  路徑: {category_path}\n{'─' * 70}")
        if not category_path.exists():
            print("  [WARN] 路徑不存在，跳過")
            continue

        folders = get_qualified_folders(category_path)
        if not folders:
            print(f"  [WARN] 沒有 {TARGET_COUNT} 張相片的資料夾")
            continue

        done = sum(is_fully_processed(f.name) for f in folders)
        print(f"  找到 {len(folders)} 個資料夾（已處理 {done}, 待處理 {len(folders) - done}）")

        for idx, folder in enumerate(folders, 1):
            s = stats[category]
            s["persons"].add(base_id_of(folder.name))
            s["sessions"] += 1

            if is_fully_processed(folder.name):
                s["skip"] += 1
                print(f"  [{idx}/{len(folders)}] {folder.name} - SKIP")
                continue

            print(f"\n  [{idx}/{len(folders)}] {folder.name}")
            try:
                vec, pnp = process_single_folder(folder)
                s["ok"] += 1
                print(f"  [OK] (Vector: {vec.length}, PnP: {pnp.length} frames)")
            except Exception as e:
                s["fail"] += 1
                print(f"  [FAIL] {e}")

    print("\n" + "=" * 70)
    print(f"{'類別':<10}{'人':<8}{'人次':<8}{'新處理':<8}{'已跳過':<8}{'失敗':<8}")
    print("-" * 70)
    for category, s in stats.items():
        print(f"{category:<10}{len(s['persons']):<8}{s['sessions']:<8}{s['ok']:<8}{s['skip']:<8}{s['fail']:<8}")
    print("=" * 70)
    print(f"輸出: 圖 → {ROTATION_FIG_DIR}  ；角度 npy → {ROTATION_FEATURES_DIR}")
    print(f"結束: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
