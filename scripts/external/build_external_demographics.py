"""
從 manifest.csv 產 data/demographics/EACS.csv

schema 比照 ACS.csv（ID, Sex, Age, Photo_Session, Has_Missing, Photo_Date,
NPT_Date, NPT_Session, Diff_Days, MMSE, CASI），再加一欄 `Source` 記錄資料集來源。
MMSE/CASI 留空（external 無認知評估）；loader 根據 Source 判斷是否 bypass strict HC。
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

            rows_out.append({
                "ID": subject_id,
                "Sex": row["sex"],
                "Age": row["age"],
                "Photo_Session": visit,
                "Has_Missing": "FALSE",
                "Photo_Date": "",
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


if __name__ == "__main__":
    main()
