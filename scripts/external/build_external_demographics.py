"""
從 manifest.csv 產 data/demographics/EACS.csv

schema 比照 ACS.csv（ID, Sex, Age, Photo_Session, Has_Missing, Photo_Date,
NPT_Date, NPT_Session, Diff_Days, MMSE, CASI），再加一欄 `Source` 記錄資料集來源。

Photo_Date 欄：
  - IMDB 從 subject folder 的第一張檔案名回推 photo_year，寫 `{year}-06-15`（mid-year 近似）
  - 其他 single-image source 留空（無需縱向）
MMSE/CASI 留空（external 無認知評估）；loader 用 Source 判斷 strict HC bypass。
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DEMOGRAPHICS_DIR, EXTERNAL_FILTERED_DIR

MANIFEST_PATH = EXTERNAL_FILTERED_DIR / "manifest.csv"
OUTPUT_CSV = DEMOGRAPHICS_DIR / "EACS.csv"

RE_SUBJECT = re.compile(r"^(?P<id>.+)-(?P<visit>\d+)$")
# IMDB: (test|valid|train)_age##_[FM]_nm#####_rm#####_YYYY-M-D_YYYY.jpg
RE_IMDB_PHOTO = re.compile(
    r"_(?P<birth>\d{4})-\d+-\d+_(?P<photo_year>\d{4})\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


def _imdb_photo_year(folder_rel: str) -> str:
    """Read first filename in subject folder; return 'YYYY-06-15' or ''."""
    folder = EXTERNAL_FILTERED_DIR / folder_rel
    if not folder.exists():
        return ""
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            m = RE_IMDB_PHOTO.search(f.name)
            if m:
                return f"{m.group('photo_year')}-06-15"
    return ""


def main():
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found: {MANIFEST_PATH}")
        print("  跑 build_subject_folders.py 先")
        sys.exit(1)

    rows_out = []
    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row["subject_id"]
            m = RE_SUBJECT.match(subject_id)
            visit = int(m.group("visit")) if m else 1

            photo_date = ""
            if row["source_dataset"] == "IMDB":
                photo_date = _imdb_photo_year(row["folder_path"])

            rows_out.append({
                "ID": subject_id,
                "Sex": row["sex"],
                "Age": row["age"],
                "Photo_Session": visit,
                "Has_Missing": "FALSE",
                "Photo_Date": photo_date,
                "NPT_Date": "",
                "NPT_Session": "",
                "Diff_Days": "",
                "MMSE": "",
                "CASI": "",
                "Source": row["source_dataset"],
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ID", "Sex", "Age", "Photo_Session", "Has_Missing",
        "Photo_Date", "NPT_Date", "NPT_Session", "Diff_Days",
        "MMSE", "CASI", "Source",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # Summary
    print(f"Wrote {OUTPUT_CSV}  ({len(rows_out)} rows)")
    from collections import Counter
    src_counts = Counter(r["Source"] for r in rows_out)
    for src, n in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {n}")
    imdb_with_date = sum(1 for r in rows_out if r["Source"] == "IMDB" and r["Photo_Date"])
    print(f"  IMDB with Photo_Date: {imdb_with_date}")


if __name__ == "__main__":
    main()
