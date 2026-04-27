"""
將所有資料集中 60 歲以上的亞洲人面孔複製到統一資料夾
"""
import os
import csv
import shutil
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import EXTERNAL_DATASETS_DIR, EXTERNAL_FILTERED_DIR

BASE = EXTERNAL_DATASETS_DIR
OUT = EXTERNAL_FILTERED_DIR / "asian_elderly_60plus"
OUT.mkdir(parents=True, exist_ok=True)

stats = Counter()


def copy_file(src, dataset_name, age, extra=""):
    """Copy file to output with unique naming: dataset_age_originalname"""
    src = Path(src)
    suffix = src.suffix
    stem = src.stem
    tag = f"{extra}_" if extra else ""
    dest_name = f"{dataset_name}_age{age}_{tag}{stem}{suffix}"
    dest = OUT / dest_name
    # Handle duplicates
    i = 1
    while dest.exists():
        dest = OUT / f"{dataset_name}_age{age}_{tag}{stem}_{i}{suffix}"
        i += 1
    shutil.copy2(src, dest)
    stats[dataset_name] += 1


def extract_afad():
    """AFAD: folder = age, subfolder = gender (111=M, 112=F)"""
    root = BASE / "AFAD" / "AFAD-Full"
    if not root.exists():
        return
    for age_dir in root.iterdir():
        if age_dir.is_dir() and age_dir.name.isdigit():
            age = int(age_dir.name)
            if age >= 60:
                for gender_dir in age_dir.iterdir():
                    if gender_dir.is_dir():
                        g = "M" if gender_dir.name == "111" else "F"
                        for f in gender_dir.iterdir():
                            if f.is_file():
                                copy_file(f, "AFAD", age, g)
    print(f"  AFAD: {stats['AFAD']} files")


def extract_fairface():
    """FairFace: CSV labels, filter East/SE Asian + age 60+"""
    img_base = BASE / "FairFace" / "images"
    for csv_file in ["fairface_label_train.csv", "fairface_label_val.csv"]:
        path = BASE / "FairFace" / csv_file
        if not path.exists():
            continue
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                race = row.get("race", "")
                if "East Asian" not in race and "Southeast Asian" not in race:
                    continue
                age_str = row.get("age", "")
                if age_str == "more than 70":
                    age_mid = 75
                elif "-" in age_str:
                    parts = age_str.split("-")
                    try:
                        lo = int(parts[0])
                    except:
                        continue
                    if lo < 60:
                        continue
                    age_mid = lo
                else:
                    continue

                if age_mid < 60:
                    continue

                img_path = img_base / row["file"]
                if img_path.exists():
                    r = race.replace(" ", "")
                    copy_file(img_path, "FairFace", age_str, r)
    print(f"  FairFace: {stats['FairFace']} files")


def extract_utkface():
    """UTKFace: filename = age_gender_race_date.jpg, race=2 is Asian"""
    root = BASE / "UTKFace" / "UTKface_inthewild"
    if not root.exists():
        return
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
                    if race == 2 and age >= 60:
                        gender = "M" if parts[1] == "0" else "F"
                        copy_file(f, "UTKFace", age, gender)
                except:
                    pass
    print(f"  UTKFace: {stats['UTKFace']} files")


def extract_megaage():
    """MegaAge-Asian: all Asian, age in list files"""
    for split in ["train", "test"]:
        age_file = BASE / "MegaAge-Asian" / "megaage_asian" / "list" / f"{split}_age.txt"
        name_file = BASE / "MegaAge-Asian" / "megaage_asian" / "list" / f"{split}_name.txt"
        if not age_file.exists() or not name_file.exists():
            continue
        ages = open(age_file).read().strip().split("\n")
        names = open(name_file).read().strip().split("\n")
        for age_str, fname in zip(ages, names):
            try:
                age = int(age_str.strip())
            except:
                continue
            if age >= 60:
                img_path = BASE / "MegaAge-Asian" / "megaage_asian" / split / fname.strip()
                if img_path.exists():
                    copy_file(img_path, "MegaAge", age)
    print(f"  MegaAge: {stats['MegaAge']} files")


def extract_diverse_asian():
    """Diverse Asian Facial Ages: CSV has age, or use folder names"""
    csv_path = BASE / "Diverse Asian Facial Ages" / "Data_label.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    age = int(row.get("Age", -1))
                except:
                    continue
                if age < 60:
                    continue
                img_name = row.get("Img", "").strip()
                # Try to find the image in Data_all or Data folders
                for subdir in ["Data_all", "Data"]:
                    img_path = BASE / "Diverse Asian Facial Ages" / subdir / f"{img_name}.jpg"
                    if img_path.exists():
                        copy_file(img_path, "DiverseAsian", age)
                        break
                    img_path = BASE / "Diverse Asian Facial Ages" / subdir / f"{img_name}.png"
                    if img_path.exists():
                        copy_file(img_path, "DiverseAsian", age)
                        break
                else:
                    # Search in subfolders
                    for p in (BASE / "Diverse Asian Facial Ages").rglob(f"{img_name}.*"):
                        if p.is_file() and p.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp"):
                            copy_file(p, "DiverseAsian", age)
                            break
    else:
        # Fallback: folder-based
        data_dir = BASE / "Diverse Asian Facial Ages" / "Data"
        folder_map = {
            "60 \u0111\u1ebfn 70 tu\u1ed5i": 65,
            "70 \u0111\u1ebfn 80 tu\u1ed5i": 75,
        }
        for folder_name, age in folder_map.items():
            folder = data_dir / folder_name
            if folder.exists():
                for f in folder.iterdir():
                    if f.is_file():
                        copy_file(f, "DiverseAsian", age)
    print(f"  DiverseAsian: {stats['DiverseAsian']} files")


def extract_szu():
    """SZU-EmoDage: Chinese faces, aging images with age 60/70 in filename"""
    aging_dir = BASE / "SZU-EmoDage" / "Aging_faces"
    if not aging_dir.exists():
        return
    for f in aging_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in (".jpg", ".png", ".bmp"):
            name = f.stem.lower()
            for age in [60, 70]:
                if str(age) in name:
                    copy_file(f, "SZU-EmoDage", age)
                    break
    print(f"  SZU-EmoDage: {stats['SZU-EmoDage']} files")


def main():
    print(f"Extracting Asian elderly (60+) faces to: {OUT}")
    print()
    extract_afad()
    extract_megaage()
    extract_diverse_asian()
    extract_fairface()
    extract_utkface()
    extract_szu()

    total = sum(stats.values())
    print(f"\n{'=' * 50}")
    print(f"Total: {total} files copied to {OUT}")
    print(f"{'=' * 50}")
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {name:>20s}: {count:>5,}")


if __name__ == "__main__":
    main()
