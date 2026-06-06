"""
為 EACS 受試者計算 age_error（real_age - predicted_age）。

Prerequisites：
  - data/demographics/EACS.csv（含 Age 欄，從檔名回推）
  - workspace/age/predictions/1_MiVOLO/predicted_ages.json（原始預測值）

Outputs：
  workspace/age/predictions/<default-cohort>/eacs_age_error.csv
    欄：subject_id, Source, real_age, pred_age, age_error, abs_age_error

此 CSV 供下游 EACS age-error 統計讀取（以 subject_id 為鍵）。
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    DEMOGRAPHICS_DIR,
    AGE_STAT_DIR,
    PREDICTED_AGES_FILE,
)
from src.age.utils import load_predicted_ages

OUTPUT_CSV = AGE_STAT_DIR / "eacs_age_error.csv"


def load_eacs_ages():
    """Return dict of EACS subject_id -> (Age, Source)."""
    path = DEMOGRAPHICS_DIR / "EACS.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                age = float(row["Age"])
            except (ValueError, KeyError):
                continue
            out[row["ID"]] = (age, row.get("Source", "unknown"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    eacs = load_eacs_ages()
    if not PREDICTED_AGES_FILE.exists():
        raise FileNotFoundError(PREDICTED_AGES_FILE)
    preds = load_predicted_ages(PREDICTED_AGES_FILE)

    rows = []
    n_missing = 0
    for sid, (real, source) in sorted(eacs.items()):
        pred = preds.get(sid)
        if pred is None:
            n_missing += 1
            continue
        err = real - pred
        rows.append({
            "subject_id": sid,
            "Source": source,
            "real_age": round(real, 2),
            "pred_age": round(float(pred), 2),
            "age_error": round(err, 2),
            "abs_age_error": round(abs(err), 2),
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "Source", "real_age", "pred_age",
            "age_error", "abs_age_error"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_CSV}  ({len(rows)} rows; {n_missing} EACS ids missing predictions)")
    print(f"Source predictions: {PREDICTED_AGES_FILE.name}")
    if rows:
        import statistics
        errs = [r["age_error"] for r in rows]
        print(f"age_error  mean={statistics.mean(errs):+.2f}  "
              f"median={statistics.median(errs):+.2f}  "
              f"sd={statistics.pstdev(errs):.2f}")


if __name__ == "__main__":
    main()
