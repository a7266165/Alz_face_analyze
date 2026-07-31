"""三世代（P / NAD / ACS / HC）之 MMSE / CASI / CDR 分層計數 → 三分頁 Excel。

subject-level（每人取最早一次拍攝之量表值）。切點抽成常數，易改：
  MMSE：≥MMSE_HI ／ MMSE_LO–(MMSE_HI−1) ／ <MMSE_LO
  CASI：≥CASI_HI ／ CASI_LO–(CASI_HI−1) ／ <CASI_LO   （80/50 為暫定，非公認標準）
  CDR ：0 / 0.5 / 1 / 2 / 3 / 缺測

「拍攝（人次）」為分析世代之視次數（P 取首訪＝1,061；HC 保留全部）。
數字與 paper/draft/v4/dataset.md 之表 1a、及集合圖同源，可即時重算核對。

用法：python scripts/paper/cognitive_strata_table.py [-o out.xlsx]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

try:                                    # 讓含 ≥/≤ 的表格能印到非 UTF-8 主控台
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.common.cohort import load_demographics

# ── 可調切點 ──────────────────────────────────────────────────────────────────
MMSE_HI, MMSE_LO = 26, 18
CASI_HI, CASI_LO = 80, 50

# 分析世代視次數（p_first 對 P、hc_all 對 HC）；與 cohort_list 規模一致
VISITS = {"P": 1061, "NAD": 791, "ACS": 218}
GROUPS = [("P", ["P"]), ("NAD（SCD）", ["NAD"]), ("ACS", ["ACS"]),
          ("HC（NAD+ACS）", ["NAD", "ACS"])]

DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "draft" / "v4" / "cognitive_strata.xlsx"


def _subject_level():
    """每人取首訪值（去 Age 缺值後）。"""
    demo = load_demographics()
    demo = demo[demo["Age"].notna()].copy()
    return (demo.sort_values(["base_id", "visit"])
            .groupby("base_id", as_index=False).first())


def _bin(series, hi, lo):
    """回傳 (>=hi, [lo,hi), <lo, 缺測) 四段計數。"""
    x = pd.to_numeric(series, errors="coerce")
    return [int((x >= hi).sum()),
            int(((x >= lo) & (x < hi)).sum()),
            int((x < lo).sum()),
            int(x.isna().sum())]


def build_tables():
    sub = _subject_level()

    def rows(builder):
        out = {}
        for label, codes in GROUPS:
            d = sub[sub["Group"].isin(codes)]
            n = len(d)
            visits = sum(VISITS[c] for c in codes)
            out[label] = [n, visits] + builder(d)
        return out

    mmse = pd.DataFrame.from_dict(
        rows(lambda d: _bin(d["MMSE"], MMSE_HI, MMSE_LO)), orient="index",
        columns=["n（人）", "拍攝（人次）",
                 f"≥{MMSE_HI}", f"{MMSE_LO}–{MMSE_HI-1}", f"≤{MMSE_LO-1}", "缺測"])
    casi = pd.DataFrame.from_dict(
        rows(lambda d: _bin(d["CASI"], CASI_HI, CASI_LO)), orient="index",
        columns=["n（人）", "拍攝（人次）",
                 f"≥{CASI_HI}", f"{CASI_LO}–{CASI_HI-1}", f"≤{CASI_LO-1}", "缺測"])

    def cdr_row(d):
        c = pd.to_numeric(d["Global_CDR"], errors="coerce")
        return [int((c == v).sum()) for v in (0.0, 0.5, 1.0, 2.0, 3.0)] + [int(c.isna().sum())]
    cdr = pd.DataFrame.from_dict(
        rows(cdr_row), orient="index",
        columns=["n（人）", "拍攝（人次）", "CDR 0", "CDR 0.5", "CDR 1", "CDR 2", "CDR 3", "缺測"])

    return {"MMSE": mmse, "CASI": casi, "CDR": cdr}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    tables = build_tables()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name, index_label="組別")
        # 說明分頁
        notes = pd.DataFrame({"說明": [
            "資料層級：subject-level（每人取最早一次拍攝之量表值）。",
            f"MMSE 分界：≥{MMSE_HI} / {MMSE_LO}–{MMSE_HI-1} / ≤{MMSE_LO-1}（{MMSE_LO} 歸中組）。",
            f"CASI 分界：≥{CASI_HI} / {CASI_LO}–{CASI_HI-1} / ≤{CASI_LO-1}（暫定，非公認標準）。",
            "拍攝（人次）＝分析世代視次數：P 取首訪(=人數)，HC 保留全部視次。",
            "CDR 有值者：P 990/1,061、NAD 151/458、ACS 0/91。",
            "來源：hospital_A.csv + src/common/cohort.py；由 scripts/paper/cognitive_strata_table.py 產生。",
        ]})
        notes.to_excel(xw, sheet_name="說明", index=False)

    for name, df in tables.items():
        print(f"\n=== {name} ===")
        print(df.to_string())
    print("\nsaved ->", args.output)


if __name__ == "__main__":
    main()
