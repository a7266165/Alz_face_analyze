"""
把 filtered/general_elderly_60plus/ 的 flat pool 重組成 per-subject folders，
並把新 subjects append 到既有的 manifest.csv（保留 asian + IMDB 原內容）。

與 build_subject_folders.py 的 build_non_imdb 同樣邏輯：
  每張 = 1 subject（single-image），subject_id = EACS_{src_key}_{seq:05d}-1

支援兩個 source：AgeDB、APPA-REAL。
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

POOL_GENERAL = EXTERNAL_FILTERED_DIR / "general_elderly_60plus"
MANIFEST_PATH = EXTERNAL_FILTERED_DIR / "manifest.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# AgeDB flat filename: AgeDB_age{age}_{sex}_{id}_{name}_{age}_{f/m}.jpg
RE_AGEDB = re.compile(r"^AgeDB_age(?P<age>\d+)_(?P<sex>[FM])_(?P<rest>.+)$")
# APPA-REAL: APPA-REAL_age{age}_{split}_{fname}.jpg
RE_APPA = re.compile(r"^APPA-REAL_age(?P<age>\d+)_(?P<split>train|valid|test)_(?P<rest>.+)$")


def parse_flat_name(fname: str):
    m = RE_AGEDB.match(fname)
    if m:
        return "AgeDB", int(m.group("age")), m.group("sex")
    m = RE_APPA.match(fname)
    if m:
        return "APPA-REAL", int(m.group("age")), "U"
    return None


def build_general(dry_run: bool):
    files = sorted(p for p in POOL_GENERAL.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)
    counters = defaultdict(int)
    manifest_rows = []

    for f in files:
        parsed = parse_flat_name(f.name)
        if parsed is None:
            print(f"  [skip unparsed] {f.name}")
            continue
        src, age, sex = parsed
        src_key = src.replace("-", "")  # APPA-REAL → APPAREAL
        counters[src_key] += 1
        seq = counters[src_key]
        subject_id = f"EACS_{src_key}_{seq:05d}-1"
        folder = POOL_GENERAL / subject_id
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

    print(f"  general_elderly_60plus: {len(manifest_rows)} subjects")
    for src_key, n in sorted(counters.items(), key=lambda x: -x[1]):
        print(f"    {src_key}: {n}")
    return manifest_rows


def append_manifest(new_rows: list[dict]):
    """Read existing manifest, append new rows, write back."""
    fieldnames = ["subject_id", "source_dataset", "age", "sex",
                  "n_images", "first_filename", "folder_path"]
    existing = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
        print(f"  existing manifest: {len(existing)} rows")
    existing_ids = {r["subject_id"] for r in existing}

    # Dedupe: if same subject_id already exists, skip
    to_add = [r for r in new_rows if r["subject_id"] not in existing_ids]
    dup = len(new_rows) - len(to_add)
    if dup:
        print(f"  skipping {dup} duplicate subject_ids")

    merged = existing + [
        {**{k: r.get(k, "") for k in fieldnames}} for r in to_add
    ]

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  manifest → {MANIFEST_PATH}  ({len(merged)} rows; +{len(to_add)} new)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print("=== DRY RUN ===")
    print(f"Pool: {POOL_GENERAL}")
    print()

    if not POOL_GENERAL.exists():
        print(f"  [skip] pool not found: {POOL_GENERAL}")
        return

    rows = build_general(args.dry_run)
    if rows and not args.dry_run:
        append_manifest(rows)
    elif args.dry_run:
        print(f"\n  [dry-run] would append {len(rows)} rows")


if __name__ == "__main__":
    main()
