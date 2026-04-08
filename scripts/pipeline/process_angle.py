"""
process_angles.py
遍歷有 1200 張相片的資料夾，使用兩種方法計算頭部旋轉角度
並將訊號圖儲存到 workspace/PnP 與 workspace/vector_angle

功能：
- 自動跳過已處理過的資料夾（檢查輸出圖片是否存在）
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.extractor.features.rotation import VectorAngleCalculator, PnPAngleCalculator
from src.extractor.features.rotation.plotter import AnglePlotter, process_single_folder
from src.config import RAW_IMAGES_DIR


# ============================================================
#  設定
# ============================================================

# 三個類別的路徑
CATEGORIES = {
    "ACS": RAW_IMAGES_DIR / "health" / "ACS",
    "NAD": RAW_IMAGES_DIR / "health" / "NAD",
    "Patient": RAW_IMAGES_DIR / "patient" / "good",
}

# 輸出目錄
OUTPUT_DIR_PNP = PROJECT_ROOT / "workspace" / "PnP"
OUTPUT_DIR_VECTOR = PROJECT_ROOT / "workspace" / "vector_angle"

# 目標相片數量
TARGET_COUNT = 1200


# ============================================================
#  工具函數
# ============================================================

def count_images(folder: Path) -> int:
    """計算資料夾中的圖片數量"""
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")))


def get_qualified_folders(category_path: Path, target_count: int) -> list[Path]:
    """取得符合條件（圖片數量 = target_count）的資料夾"""
    if not category_path.exists():
        return []

    qualified = []
    for folder in category_path.iterdir():
        if folder.is_dir() and count_images(folder) == target_count:
            qualified.append(folder)

    return sorted(qualified, key=lambda x: x.name)


def extract_person_id(folder_name: str) -> str:
    """從資料夾名稱提取人員 ID (例如: ACS1-1 -> ACS1)"""
    if '-' in folder_name:
        return folder_name.rsplit('-', 1)[0]
    return folder_name


def check_already_processed(
    folder_name: str,
    category: str,
    output_dir_pnp: Path,
    output_dir_vector: Path
) -> dict:
    """
    檢查資料夾是否已經處理過

    Returns:
        dict: {'pnp': bool, 'vector': bool} 表示各方法是否已完成
    """
    pnp_output = output_dir_pnp / category / f"{folder_name}.png"
    vector_output = output_dir_vector / category / f"{folder_name}.png"

    return {
        'pnp': pnp_output.exists(),
        'vector': vector_output.exists()
    }


def is_fully_processed(
    folder_name: str,
    category: str,
    output_dir_pnp: Path,
    output_dir_vector: Path
) -> bool:
    """檢查資料夾是否兩種方法都已處理完成"""
    status = check_already_processed(folder_name, category, output_dir_pnp, output_dir_vector)
    return status['pnp'] and status['vector']


# ============================================================
#  主程式
# ============================================================

def main():
    print("=" * 70)
    print("頭部旋轉角度分析")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目標相片數量: {TARGET_COUNT}")
    print("=" * 70)

    # 建立輸出目錄
    OUTPUT_DIR_PNP.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_VECTOR.mkdir(parents=True, exist_ok=True)

    # 為每個類別建立子目錄
    for category in CATEGORIES.keys():
        (OUTPUT_DIR_PNP / category).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR_VECTOR / category).mkdir(parents=True, exist_ok=True)

    # 統計資訊
    stats = {cat: {"人": set(), "人次": 0, "處理成功": 0, "處理失敗": 0, "已跳過": 0}
             for cat in CATEGORIES.keys()}

    total_processed = 0
    total_failed = 0
    total_skipped = 0

    # 遍歷每個類別
    for category, category_path in CATEGORIES.items():
        print(f"\n{'─' * 70}")
        print(f"類別: {category}")
        print(f"路徑: {category_path}")
        print(f"{'─' * 70}")

        if not category_path.exists():
            print(f"  ⚠ 路徑不存在，跳過")
            continue

        # 取得符合條件的資料夾
        qualified_folders = get_qualified_folders(category_path, TARGET_COUNT)

        if not qualified_folders:
            print(f"  ⚠ 沒有找到 {TARGET_COUNT} 張相片的資料夾")
            continue

        # 先統計已處理數量
        already_done = sum(
            1 for f in qualified_folders
            if is_fully_processed(f.name, category, OUTPUT_DIR_PNP, OUTPUT_DIR_VECTOR)
        )
        print(f"  找到 {len(qualified_folders)} 個符合條件的資料夾 (已處理: {already_done}, 待處理: {len(qualified_folders) - already_done})")

        # 處理每個資料夾
        for idx, folder in enumerate(qualified_folders, 1):
            folder_name = folder.name
            person_id = extract_person_id(folder_name)

            # 檢查是否已處理
            if is_fully_processed(folder_name, category, OUTPUT_DIR_PNP, OUTPUT_DIR_VECTOR):
                stats[category]["已跳過"] += 1
                stats[category]["人"].add(person_id)
                stats[category]["人次"] += 1
                total_skipped += 1
                print(f"  [{idx}/{len(qualified_folders)}] {folder_name} - ⏭ 已存在，跳過")
                continue

            print(f"\n  [{idx}/{len(qualified_folders)}] {folder_name}")

            try:
                # 使用兩種方法處理
                vector_result, pnp_result = process_single_folder(
                    folder_path=folder,
                    output_dir_pnp=OUTPUT_DIR_PNP / category,
                    output_dir_vector=OUTPUT_DIR_VECTOR / category,
                    verbose=True
                )

                # 更新統計
                stats[category]["人"].add(person_id)
                stats[category]["人次"] += 1
                stats[category]["處理成功"] += 1
                total_processed += 1

                print(f"  ✓ 完成 (Vector: {vector_result.length}, PnP: {pnp_result.length} frames)")

            except Exception as e:
                stats[category]["處理失敗"] += 1
                total_failed += 1
                print(f"  ✗ 錯誤: {e}")

    # 輸出統計報告
    print("\n")
    print("=" * 70)
    print("統計報告")
    print("=" * 70)
    print(f"{'類別':<12} {'人':<8} {'人次':<8} {'新處理':<8} {'已跳過':<8} {'失敗':<8}")
    print("-" * 70)

    total_persons = 0
    total_sessions = 0

    for category, data in stats.items():
        person_count = len(data["人"])
        session_count = data["人次"]
        success_count = data["處理成功"]
        skip_count = data["已跳過"]
        fail_count = data["處理失敗"]

        print(f"{category:<12} {person_count:<8} {session_count:<8} {success_count:<8} {skip_count:<8} {fail_count:<8}")

        total_persons += person_count
        total_sessions += session_count

    print("-" * 70)
    print(f"{'合計':<12} {total_persons:<8} {total_sessions:<8} {total_processed:<8} {total_skipped:<8} {total_failed:<8}")
    print("=" * 70)

    # 輸出檔案位置
    print(f"\n輸出位置:")
    print(f"  PnP 方法:    {OUTPUT_DIR_PNP}")
    print(f"  Vector 方法: {OUTPUT_DIR_VECTOR}")

    print(f"\n結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
