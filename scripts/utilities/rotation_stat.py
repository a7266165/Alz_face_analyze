"""
統計相片數量為 1200 張的資料夾
分三個類別 (ACS, NAD, Patient)，統計「人」與「人次」
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

import os
from collections import defaultdict
from datetime import datetime

from src.config import RAW_IMAGES_DIR

# 設定根目錄路徑
ROOT_PATH = RAW_IMAGES_DIR

# 輸出目錄
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "pic_num_stat"

# 定義三個類別的路徑
CATEGORIES = {
    "ACS": ROOT_PATH / "health" / "ACS",
    "NAD": ROOT_PATH / "health" / "NAD",
    "Patient": ROOT_PATH / "patient" / "good",  # 假設只統計 good 資料夾
}

TARGET_COUNT = 1200  # 目標相片數量


def count_images(folder: Path) -> int:
    """計算資料夾中的圖片數量"""
    if not folder.exists():
        return 0
    return len([f for f in folder.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])


def extract_person_id(folder_name: str) -> str:
    """
    從資料夾名稱提取人員 ID
    例如: ACS1-1 -> ACS1, P1-2 -> P1, NAD10-1 -> NAD10
    """
    if '-' in folder_name:
        return folder_name.rsplit('-', 1)[0]
    return folder_name


def main():
    results = {}

    for category, path in CATEGORIES.items():
        if not path.exists():
            print(f"警告: 路徑不存在 - {path}")
            continue

        # 找出所有子資料夾
        folders = [f for f in path.iterdir() if f.is_dir()]

        # 篩選相片數量為 1200 的資料夾
        qualified_folders = []
        for folder in folders:
            img_count = count_images(folder)
            if img_count == TARGET_COUNT:
                qualified_folders.append(folder.name)

        # 統計人次 (資料夾數量)
        session_count = len(qualified_folders)

        # 統計人 (不重複的人員 ID)
        unique_persons = set(extract_person_id(f) for f in qualified_folders)
        person_count = len(unique_persons)

        results[category] = {
            "人": person_count,
            "人次": session_count,
            "資料夾列表": qualified_folders,
        }

    # 建立輸出資料夾
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 準備輸出內容
    lines = []
    lines.append("=" * 60)
    lines.append(f"統計結果：相片數量 = {TARGET_COUNT} 張的資料夾")
    lines.append(f"統計時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append(f"{'類別':<10} {'人':<10} {'人次':<10}")
    lines.append("-" * 60)

    total_person = 0
    total_session = 0

    for category, data in results.items():
        lines.append(f"{category:<10} {data['人']:<10} {data['人次']:<10}")
        total_person += data['人']
        total_session += data['人次']

    lines.append("-" * 60)
    lines.append(f"{'合計':<10} {total_person:<10} {total_session:<10}")
    lines.append("=" * 60)

    # 詳細列表
    lines.append("\n詳細資料夾列表：")
    for category, data in results.items():
        if data['資料夾列表']:
            lines.append(f"\n{category}:")
            for folder in sorted(data['資料夾列表']):
                lines.append(f"  - {folder}")

    # 輸出到終端
    output_text = "\n".join(lines)
    print(output_text)

    # 存到檔案
    summary_file = OUTPUT_DIR / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\n結果已儲存至: {summary_file}")

    # 另存 CSV 格式 (方便後續分析)
    csv_file = OUTPUT_DIR / "summary.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("類別,人,人次\n")
        for category, data in results.items():
            f.write(f"{category},{data['人']},{data['人次']}\n")
        f.write(f"合計,{total_person},{total_session}\n")
    print(f"CSV 已儲存至: {csv_file}")

    # 儲存各類別的詳細資料夾列表
    for category, data in results.items():
        if data['資料夾列表']:
            detail_file = OUTPUT_DIR / f"{category}_folders.txt"
            with open(detail_file, "w", encoding="utf-8") as f:
                for folder in sorted(data['資料夾列表']):
                    f.write(f"{folder}\n")
            print(f"{category} 資料夾列表已儲存至: {detail_file}")


if __name__ == "__main__":
    main()
