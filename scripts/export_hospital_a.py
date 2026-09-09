"""從收案主表 outcome_k.csv 匯出分析用 data/demographics/hospital_A.csv。

主表(AlzheimerResearch/_data/outcome_k.csv)是唯一真相;hospital_A.csv 是它的 14 欄子集
(不帶姓名等個資),加兩欄:
  Diff_Days  :|Photo_Date - NPT_Date| 天數(任一缺日期 → 空),由主表算
  NPT_Session:醫院端的 NPT 施測序號(含沒拍照的次數),主表沒有、算不出來;
               既有場次從現有 hospital_A.csv 沿用,新場次留空。分析程式不讀這欄。
分析程式(src/config.PHOTO_DATE_MAX)另有世代凍結,主表新增場次不影響既有結果。

用法:
  python scripts/export_hospital_a.py                 # 讀預設主表,寫 data/demographics/hospital_A.csv
  python scripts/export_hospital_a.py --check         # 只跟現有 hospital_A.csv 逐格比對,不寫檔
  python scripts/export_hospital_a.py --master PATH   # 指定主表
"""

import argparse
import csv
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = ROOT.parent / "AlzheimerResearch" / "_data" / "outcome_k.csv"
HOSPITAL_A = ROOT / "data" / "demographics" / "hospital_A.csv"

OUT_COLS = [
    "Group", "Number", "Photo_Session", "Photo_Date", "Birth_Date", "Sex", "Age",
    "BMI", "NPT_Date", "NPT_Session", "Diff_Days", "MMSE", "CASI", "Global_CDR",
    "Session_Version",
]


def parse_date(text):
    s = (text or "").strip().replace("-", "/")
    if not s:
        return None
    parts = s.split("/")
    if len(parts) != 3:
        return None
    try:
        y, m, d = (int(p) for p in parts)
    except ValueError:
        return None
    if y < 1900:
        y += 1911
    try:
        return date(y, m, d)
    except ValueError:
        return None


def fmt_date(d, pad):
    if d is None:
        return ""
    return f"{d.year}/{d.month:02d}/{d.day:02d}" if pad else f"{d.year}/{d.month}/{d.day}"


def row_key(r):
    return ((r.get("Group") or "").strip().upper(), (r.get("Number") or "").strip(),
            (r.get("Photo_Session") or "").strip())


def export_rows(master_rows, carry=None):
    """主表列 → hospital_A 列(依 Group, Number, Photo_Session 排序)。
    carry = 現有 hospital_A 的列(以 row_key 為鍵),只用來沿用 NPT_Session。"""
    carry = carry or {}
    out = []
    for r in master_rows:
        group, number, session = row_key(r)
        photo = parse_date(r.get("Photo_Date"))
        npt = parse_date(r.get("NPT_Date"))
        old = carry.get((group, number, session), {})
        diff = "" if (photo is None or npt is None) else f"{abs((photo - npt).days)}.0"
        out.append({
                "Group": group,
                "Number": number,
                "Photo_Session": session,
                "Photo_Date": fmt_date(photo, pad=False),
                "Birth_Date": fmt_date(parse_date(r.get("Birth_Date")), pad=True),
                "Sex": (r.get("Sex") or "").strip().upper(),
                "Age": (r.get("Age") or "").strip(),
                "BMI": (r.get("BMI") or "").strip(),
                "NPT_Date": fmt_date(npt, pad=False),
                "NPT_Session": (old.get("NPT_Session") or "").strip(),
                "Diff_Days": diff,
                "MMSE": (r.get("MMSE") or "").strip(),
                "CASI": (r.get("CASI") or "").strip(),
                "Global_CDR": (r.get("Global_CDR") or "").strip(),
                "Session_Version": (r.get("Session_Version") or "").strip(),
            })
    order = {"P": 0, "NAD": 1, "ACS": 2}
    out.sort(key=lambda r: (order.get(r["Group"], 9), int(r["Number"] or 0),
                            int(r["Photo_Session"] or 0)))
    return out


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _num_equal(a, b):
    a, b = (a or "").strip(), (b or "").strip()
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-6)


def check(new_rows, old_rows):
    """逐格比對(數值容忍浮點位數差),回傳 (只在新表, 只在舊表, 差異清單)。"""
    key = lambda r: (r["Group"], r["Number"], r["Photo_Session"])  # noqa: E731
    new, old = {key(r): r for r in new_rows}, {key(r): r for r in old_rows}
    only_new = sorted(set(new) - set(old))
    only_old = sorted(set(old) - set(new))
    diffs = []
    for k in sorted(set(new) & set(old)):
        for col in OUT_COLS:
            if col not in old[k]:
                continue
            a, b = new[k].get(col, ""), old[k].get(col, "")
            if col in ("Photo_Date", "Birth_Date", "NPT_Date"):
                same = parse_date(a) == parse_date(b)
            else:
                same = _num_equal(a, b)
            if not same:
                diffs.append((k, col, b, a))
    return only_new, only_old, diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    ap.add_argument("--out", type=Path, default=HOSPITAL_A)
    ap.add_argument("--check", action="store_true", help="只比對,不寫檔")
    a = ap.parse_args(argv)
    if not a.master.is_file():
        sys.exit(f"主表不存在:{a.master}")
    old_rows = read_csv(a.out) if a.out.is_file() else []
    new_rows = export_rows(read_csv(a.master), {row_key(r): r for r in old_rows})
    print(f"主表 {a.master}:{len(new_rows)} 場")
    if old_rows:
        only_new, only_old, diffs = check(new_rows, old_rows)
        by_col = {}
        for _, col, _, _ in diffs:
            by_col[col] = by_col.get(col, 0) + 1
        print(f"與現有 {a.out.name} 比對:只在主表 {len(only_new)} 場、"
              f"只在舊表 {len(only_old)} 場、既有場次值差異 {len(diffs)} 格 {by_col}")
        for k in only_new[:40]:
            print(f"  + {k[0]}{k[1]}-{k[2]}")
        for k in only_old[:40]:
            print(f"  - {k[0]}{k[1]}-{k[2]}")
        for k, col, b, a_ in diffs[:40]:
            print(f"  ~ {k[0]}{k[1]}-{k[2]} {col}: {b!r} -> {a_!r}")
    if a.check:
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(new_rows)
    print(f"已寫入 {a.out}")


if __name__ == "__main__":
    main()
