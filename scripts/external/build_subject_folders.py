"""
將 filtered pool 的 flat 影像重組成 pipeline 期待的 per-subject folder 結構

輸入：
  external/public_face_datasets/filtered/asian_elderly_60plus/*.jpg  (3,842 張)
  external/public_face_datasets/filtered/IMDB_60_plus/*.jpg           (17,872 張)

輸出：
  filtered/asian_elderly_60plus/EACS_{SRC}_{seq:05d}-1/{filename}.jpg
  filtered/IMDB_60_plus/EACS_IMDB_{nm_id}-{visit:02d}/{filename}.jpg
  filtered/manifest.csv  (subject_id, source_dataset, age, sex, n_images, first_filename, folder_path)

IMDB：依 nm-id 分組、(nm_id, photo_year) 作 visit → 同年照片落同一 visit 資料夾。
其他資料集：每張 = 1 subject（single-image，visit=1）。
"""
import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import EXTERNAL_FILTERED_DIR

POOL_ASIAN_ELDERLY = EXTERNAL_FILTERED_DIR / "asian_elderly_60plus"
POOL_IMDB = EXTERNAL_FILTERED_DIR / "IMDB_60_plus"
MANIFEST_PATH = EXTERNAL_FILTERED_DIR / "manifest.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Regex per source — pool asian_elderly_60plus
RE_AFAD = re.compile(r"^AFAD_age(?P<age>\d+)_(?P<sex>[FM])_(?P<rest>.+)$")
RE_MEGAAGE = re.compile(r"^MegaAge_age(?P<age>\d+)_(?P<rest>.+)$")
RE_FAIRFACE = re.compile(r"^FairFace_age(?P<age_lo>\d+)-(?P<age_hi>\d+)_(?P<race>[A-Za-z]+)_(?P<rest>.+)$")
RE_FAIRFACE_MORE = re.compile(r"^FairFace_agemore than (?P<age_lo>\d+)_(?P<race>[A-Za-z]+)_(?P<rest>.+)$")
RE_UTKFACE = re.compile(r"^UTKFace_age(?P<age>\d+)_(?P<sex>[FM])_(?P<rest>.+)$")
RE_DIVERSEASIAN = re.compile(r"^DiverseAsian_age(?P<age>\d+)_(?P<rest>.+)$")
RE_SZU = re.compile(r"^SZU-EmoDage_age(?P<age>\d+)_(?P<rest>.+)$")

# IMDB: {test|valid}_age60_F_nm0000263_rm1313717760_1951-1-12_2011.jpg
RE_IMDB = re.compile(
    r"^(?:test|valid|train)_age(?P<age_filename>\d+)_(?P<sex>[FM])_(?P<nm>nm\d+)_rm\d+_"
    r"(?P<birth_year>\d{4})-\d+-\d+_(?P<photo_year>\d{4})\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


def parse_flat_name(fname: str):
    """Return (src, age, sex) or None if no match. sex='U' if unknown."""
    for prefix, regex, default_sex in (
        ("AFAD", RE_AFAD, None),
        ("MegaAge", RE_MEGAAGE, "U"),
        ("FairFace", RE_FAIRFACE, "U"),
        ("FairFace", RE_FAIRFACE_MORE, "U"),
        ("UTKFace", RE_UTKFACE, None),
        ("DiverseAsian", RE_DIVERSEASIAN, "U"),
        ("SZU-EmoDage", RE_SZU, "U"),
    ):
        m = regex.match(fname)
        if not m:
            continue
        gd = m.groupdict()
        if prefix == "FairFace":
            if "age_hi" in gd:
                age = (int(gd["age_lo"]) + int(gd["age_hi"])) // 2
            else:
                age = int(gd["age_lo"]) + 5  # "more than 70" -> 75
        else:
            age = int(gd["age"])
        sex = gd.get("sex", default_sex) or default_sex
        return prefix, age, sex
    return None


def build_non_imdb(dry_run: bool):
    """Group asian_elderly_60plus pool → per-subject folders (each image = 1 subject)."""
    pool = POOL_ASIAN_ELDERLY
    files = sorted(p for p in pool.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    counters = defaultdict(int)
    manifest_rows = []

    for f in files:
        parsed = parse_flat_name(f.name)
        if parsed is None:
            print(f"  [skip unparsed] {f.name}")
            continue
        src, age, sex = parsed
        src_key = src.replace("-", "")  # SZU-EmoDage → SZUEmoDage
        counters[src_key] += 1
        seq = counters[src_key]
        subject_id = f"EACS_{src_key}_{seq:05d}-1"
        folder = pool / subject_id
        target = folder / f.name
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(target))
        manifest_rows.append({
            "subject_id": subject_id,
            "source_dataset": src,
            "age": age,
            "sex": sex,
            "n_images": 1,
            "first_filename": f.name,
            "folder_path": str(folder.relative_to(EXTERNAL_FILTERED_DIR)).replace("\\", "/"),
        })

    print(f"  asian_elderly_60plus: {len(manifest_rows)} subjects")
    for src_key, n in sorted(counters.items(), key=lambda x: -x[1]):
        print(f"    {src_key}: {n}")
    return manifest_rows


def build_imdb(dry_run: bool):
    """Group IMDB_60_plus pool by (nm_id, photo_year)."""
    pool = POOL_IMDB
    files = sorted(p for p in pool.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

    # Group by (nm_id, photo_year)
    groups: dict[tuple[str, int], list[Path]] = defaultdict(list)
    meta: dict[str, dict] = {}
    for f in files:
        m = RE_IMDB.match(f.name)
        if not m:
            print(f"  [skip unparsed IMDB] {f.name}")
            continue
        gd = m.groupdict()
        nm = gd["nm"]
        photo_year = int(gd["photo_year"])
        birth_year = int(gd["birth_year"])
        sex = gd["sex"]
        groups[(nm, photo_year)].append(f)
        if nm not in meta:
            meta[nm] = {"sex": sex, "birth_year": birth_year}

    # Assign visit numbers per nm_id (1-indexed by photo_year order)
    nm_to_years: dict[str, list[int]] = defaultdict(list)
    for nm, year in groups.keys():
        nm_to_years[nm].append(year)
    for nm in nm_to_years:
        nm_to_years[nm] = sorted(set(nm_to_years[nm]))
    nm_year_visit: dict[tuple[str, int], int] = {
        (nm, year): visit_idx + 1
        for nm, years in nm_to_years.items()
        for visit_idx, year in enumerate(years)
    }

    manifest_rows = []
    n_multi_visit = 0
    for (nm, year), fs in groups.items():
        visit = nm_year_visit[(nm, year)]
        age = year - meta[nm]["birth_year"]
        subject_id = f"EACS_IMDB_{nm}-{visit:02d}"
        folder = pool / subject_id
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)
            for f in fs:
                shutil.move(str(f), str(folder / f.name))
        manifest_rows.append({
            "subject_id": subject_id,
            "source_dataset": "IMDB",
            "age": age,
            "sex": meta[nm]["sex"],
            "n_images": len(fs),
            "first_filename": fs[0].name,
            "folder_path": str(folder.relative_to(EXTERNAL_FILTERED_DIR)).replace("\\", "/"),
        })

    n_multi_visit = sum(1 for years in nm_to_years.values() if len(years) >= 2)
    print(f"  IMDB_60_plus: {len(meta)} identities -> {len(manifest_rows)} subject-visits")
    print(f"    multi-visit identities (>=2 distinct photo_years): {n_multi_visit}")
    return manifest_rows


def write_manifest(rows):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "source_dataset", "age", "sex", "n_images", "first_filename", "folder_path"]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  manifest → {MANIFEST_PATH}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只掃描與計算，不實際移動檔案")
    args = ap.parse_args()

    if args.dry_run:
        print("=== DRY RUN ===")
    print(f"Pool asian_elderly: {POOL_ASIAN_ELDERLY}")
    print(f"Pool IMDB:          {POOL_IMDB}")
    print()

    rows = []
    if POOL_ASIAN_ELDERLY.exists():
        rows += build_non_imdb(args.dry_run)
    else:
        print(f"  [skip] pool not found: {POOL_ASIAN_ELDERLY}")

    if POOL_IMDB.exists():
        rows += build_imdb(args.dry_run)
    else:
        print(f"  [skip] pool not found: {POOL_IMDB}")

    if rows and not args.dry_run:
        write_manifest(rows)
    elif args.dry_run:
        print(f"\n  [dry-run] would write manifest with {len(rows)} rows")


if __name__ == "__main__":
    main()
