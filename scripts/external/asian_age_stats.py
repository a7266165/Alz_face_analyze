"""
統計所有資料集中亞洲人面孔的年齡分布
"""
import os
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import EXTERNAL_DATASETS_DIR

BASE = EXTERNAL_DATASETS_DIR

# Age bins for summary
AGE_BINS = [
    (0, 9, "0-9"),
    (10, 19, "10-19"),
    (20, 29, "20-29"),
    (30, 39, "30-39"),
    (40, 49, "40-49"),
    (50, 59, "50-59"),
    (60, 69, "60-69"),
    (70, 79, "70-79"),
    (80, 89, "80-89"),
    (90, 199, "90+"),
]

def age_to_bin(age):
    for lo, hi, label in AGE_BINS:
        if lo <= age <= hi:
            return label
    return "unknown"


def count_afad():
    """AFAD: all Asian, folder = age, subfolder = gender (111=M, 112=F)"""
    age_counts = Counter()
    root = BASE / "AFAD" / "AFAD-Full"
    if not root.exists():
        return "AFAD", age_counts, 0
    total = 0
    for age_dir in root.iterdir():
        if age_dir.is_dir() and age_dir.name.isdigit():
            age = int(age_dir.name)
            count = 0
            for gender_dir in age_dir.iterdir():
                if gender_dir.is_dir():
                    count += len(list(gender_dir.glob("*.*")))
            age_counts[age] = count
            total += count
    return "AFAD (全亞洲)", age_counts, total


def count_fairface():
    """FairFace: CSV with race column, filter East Asian + Southeast Asian"""
    age_counts = Counter()
    total = 0
    for csv_file in ["fairface_label_train.csv", "fairface_label_val.csv"]:
        path = BASE / "FairFace" / csv_file
        if not path.exists():
            continue
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                race = row.get("race", "")
                if "East Asian" in race or "Southeast Asian" in race:
                    age_str = row.get("age", "")
                    # age format: "0-2", "3-9", "10-19", "20-29", etc.
                    if "-" in age_str:
                        parts = age_str.split("-")
                        try:
                            age_mid = (int(parts[0]) + int(parts[1])) // 2
                        except:
                            age_mid = -1
                    elif age_str == "more than 70":
                        age_mid = 75
                    else:
                        age_mid = -1
                    if age_mid >= 0:
                        age_counts[age_mid] += 1
                        total += 1
    return "FairFace (East/SE Asian)", age_counts, total


def count_utkface():
    """UTKFace: filename = age_gender_race_date.jpg, race=2 is Asian"""
    age_counts = Counter()
    total = 0
    root = BASE / "UTKFace" / "UTKface_inthewild"
    if not root.exists():
        return "UTKFace (Asian)", age_counts, 0
    for part_dir in root.iterdir():
        if not part_dir.is_dir():
            continue
        for f in part_dir.iterdir():
            if not f.is_file():
                continue
            parts = f.name.split("_")
            if len(parts) >= 4:
                try:
                    age = int(parts[0])
                    race = int(parts[2])
                    if race == 2:  # Asian
                        age_counts[age] += 1
                        total += 1
                except:
                    pass
    return "UTKFace (race=2 Asian)", age_counts, total


def count_megaage():
    """MegaAge-Asian: all Asian, age in list/train_age.txt"""
    age_counts = Counter()
    total = 0
    for split in ["train", "test"]:
        age_file = BASE / "MegaAge-Asian" / "megaage_asian" / "list" / f"{split}_age.txt"
        if not age_file.exists():
            continue
        with open(age_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        age = int(line)
                        age_counts[age] += 1
                        total += 1
                    except:
                        pass
    return "MegaAge-Asian (全亞洲)", age_counts, total


def count_diverse_asian():
    """Diverse Asian Facial Ages: all Asian, folder names contain age range"""
    age_counts = Counter()
    total = 0

    # Also check CSV for precise ages
    csv_path = BASE / "Diverse Asian Facial Ages" / "Data_label.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    age = int(row.get("Age", -1))
                    if age >= 0:
                        age_counts[age] += 1
                        total += 1
                except:
                    pass
        if total > 0:
            return "Diverse Asian Facial Ages (全亞洲)", age_counts, total

    # Fallback: count by folder
    data_dir = BASE / "Diverse Asian Facial Ages" / "Data"
    if not data_dir.exists():
        return "Diverse Asian Facial Ages (全亞洲)", age_counts, 0

    folder_age_map = {
        "từ nhỏ đến 5 tuổi": 3,
        "5 tới 10 tuổi": 7,
        "10 tuổi 15": 12,
        "15 tuổi 20": 17,
        "20 tuổi 25": 22,
        "25 tuổi 30": 27,
        "30 đến 40 tuổi": 35,
        "40 đến 50 tuổi": 45,
        "50 đến 60 tuổi": 55,
        "60 đến 70 tuổi": 65,
        "70 đến 80 tuổi": 75,
    }

    for folder_name, age_mid in folder_age_map.items():
        folder = data_dir / folder_name
        if folder.exists():
            count = len([f for f in folder.iterdir() if f.is_file()])
            age_counts[age_mid] += count
            total += count

    return "Diverse Asian Facial Ages (全亞洲)", age_counts, total


def count_szu():
    """SZU-EmoDage: all Chinese, aging faces have age variants 10-70"""
    age_counts = Counter()
    total = 0
    aging_dir = BASE / "SZU-EmoDage" / "Aging_faces" / "Aging_faces"
    if not aging_dir.exists():
        aging_dir = BASE / "SZU-EmoDage" / "Aging_faces"
    if aging_dir.exists():
        # Count all image files
        for f in aging_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".jpg", ".png", ".bmp"):
                # Try to extract age from filename
                name = f.stem.lower()
                for age in [10, 20, 30, 40, 50, 60, 70]:
                    if str(age) in name:
                        age_counts[age] += 1
                        total += 1
                        break
                else:
                    age_counts[-1] += 1
                    total += 1
    return "SZU-EmoDage (全中國人)", age_counts, total


def binned_summary(name, age_counts, total):
    """Convert exact ages to age bins"""
    binned = Counter()
    for age, count in age_counts.items():
        if age < 0:
            binned["unknown"] += count
        else:
            binned[age_to_bin(age)] += count
    return binned


def main():
    datasets = [
        count_afad(),
        count_megaage(),
        count_diverse_asian(),
        count_fairface(),
        count_utkface(),
        count_szu(),
    ]

    print("=" * 80)
    print("亞洲人面孔年齡分布統計")
    print("=" * 80)

    grand_total = 0
    grand_binned = Counter()

    for name, age_counts, total in datasets:
        binned = binned_summary(name, age_counts, total)
        grand_total += total
        for k, v in binned.items():
            grand_binned[k] += v

        print(f"\n{'─' * 60}")
        print(f"📊 {name}  (共 {total:,} 張)")
        print(f"{'─' * 60}")

        for _, _, label in AGE_BINS:
            count = binned.get(label, 0)
            if count > 0:
                bar = "█" * min(count // 100, 50) if count >= 100 else "▏"
                pct = count / total * 100 if total > 0 else 0
                print(f"  {label:>6s}: {count:>7,} ({pct:5.1f}%) {bar}")
        if binned.get("unknown", 0) > 0:
            print(f"  {'unkn':>6s}: {binned['unknown']:>7,}")

    # Grand total
    print(f"\n{'═' * 60}")
    print(f"🔢 所有亞洲人面孔合計: {grand_total:,} 張")
    print(f"{'═' * 60}")
    for _, _, label in AGE_BINS:
        count = grand_binned.get(label, 0)
        if count > 0:
            bar = "█" * min(count // 200, 50) if count >= 200 else "▏"
            pct = count / grand_total * 100 if grand_total > 0 else 0
            print(f"  {label:>6s}: {count:>7,} ({pct:5.1f}%) {bar}")
    if grand_binned.get("unknown", 0) > 0:
        print(f"  {'unkn':>6s}: {grand_binned['unknown']:>7,}")

    # Highlight elderly (60+)
    elderly = sum(grand_binned.get(label, 0) for _, _, label in AGE_BINS if int(label.split("-")[0].replace("+","")) >= 60)
    print(f"\n⚠️  60歲以上合計: {elderly:,} 張 ({elderly/grand_total*100:.1f}%)")
    print(f"   (這是你做阿茲海默研究最需要的年齡段)")


if __name__ == "__main__":
    main()
