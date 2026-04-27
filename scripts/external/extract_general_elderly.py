"""
將 AgeDB + APPA-REAL 60 歲以上面孔複製到 filtered/general_elderly_60plus/。

這兩個 dataset 都沒有 race label，所以另開一個 folder 與 asian_elderly_60plus/ 並列。
命名規則與 extract_asian_elderly.py 一致：
    {dataset}_age{age}_{extra}_{stem}{suffix}
其中 extra 放 sex（AgeDB 有）或 split（APPA-REAL）。
"""
import csv
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import EXTERNAL_DATASETS_DIR, EXTERNAL_FILTERED_DIR

BASE = EXTERNAL_DATASETS_DIR
OUT = EXTERNAL_FILTERED_DIR / "general_elderly_60plus"
OUT.mkdir(parents=True, exist_ok=True)

stats = Counter()


def copy_file(src: Path, dataset_name: str, age: int, extra: str = ""):
    src = Path(src)
    suffix = src.suffix
    stem = src.stem
    tag = f"{extra}_" if extra else ""
    dest = OUT / f"{dataset_name}_age{age}_{tag}{stem}{suffix}"
    i = 1
    while dest.exists():
        dest = OUT / f"{dataset_name}_age{age}_{tag}{stem}_{i}{suffix}"
        i += 1
    shutil.copy2(src, dest)
    stats[dataset_name] += 1


# AgeDB: filename = {id}_{name}_{age}_{sex}.jpg, 名字可含 underscore / apostrophe
AGEDB_PAT = re.compile(r"^(\d+)_(.+)_(\d+)_([mfMF])\.jpg$", re.IGNORECASE)


def extract_agedb():
    root = BASE / "AgeDB"
    if not root.exists():
        return
    for fname in os.listdir(root):
        m = AGEDB_PAT.match(fname)
        if not m:
            continue
        age = int(m.group(3))
        sex = m.group(4).upper()
        if age >= 60:
            copy_file(root / fname, "AgeDB", age, sex)
    print(f"  AgeDB: {stats['AgeDB']} files")


def extract_appa_real():
    root = BASE / "appa-real-release"
    if not root.exists():
        return
    for split in ["train", "valid", "test"]:
        csv_path = root / f"gt_avg_{split}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    age = int(round(float(row["real_age"])))
                except (KeyError, ValueError, TypeError):
                    continue
                if age < 60:
                    continue
                fname = row["file_name"].strip()
                img_path = root / split / fname
                if img_path.exists():
                    copy_file(img_path, "APPA-REAL", age, split)
    print(f"  APPA-REAL: {stats['APPA-REAL']} files")


def main():
    print(f"Extracting general elderly (60+) faces to: {OUT}")
    print()
    extract_agedb()
    extract_appa_real()

    total = sum(stats.values())
    print(f"\n{'=' * 50}")
    print(f"Total: {total} files copied to {OUT}")
    print(f"{'=' * 50}")
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {name:>12s}: {count:>5,}")


if __name__ == "__main__":
    main()
