"""
從 data/demographics/legacy/{P,NAD,ACS}.csv 與去連結 Excel 的「生日」欄，
重建單一乾淨的 data/demographics/hospital_A.csv。

hospital_A.csv 規格（14 欄，正規化：不存可推導的 ID）：
  Group         P / NAD / ACS
  Number        個案編號（人；1, 2, 3…，只在同 Group 內唯一）
  Photo_Session 第幾次拍照
  Photo_Date, Birth_Date, Sex, Age, BMI,
  NPT_Date, NPT_Session(Int), Diff_Days, MMSE, CASI, Global_CDR

單筆唯一鍵 ID（"P1-2"，= .npy / predicted_ages.json 的鍵）不存檔，由唯一讀取點
src.common.cohort.load_demographics() 於讀入時組出：
    ID      = Group + str(Number) + "-" + str(Photo_Session)   # "P1-2"
    base_id = Group + str(Number)                              # "P1"（人鍵）

Birth_Date 來源：去連結 Excel 的三張「基本資料及NPT檢查」分頁（各含「生日」欄），
以 (Group, 個案編號數字) 對應；Excel 編號格式各異（P:"1"、NAD:"NAD-001"、
ACS:"ACS001"），統一抽數字後比對。

用法：
  conda run -n Alz_face_main_analysis python scripts/external/build_hospital_A.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DEMOGRAPHICS_DIR

LEGACY_DIR = DEMOGRAPHICS_DIR / "legacy"
EXCEL_PATH = DEMOGRAPHICS_DIR.parent / "TO中山大學AI收案 20250416去連結.xlsx"
OUTPUT_CSV = DEMOGRAPHICS_DIR / "hospital_A.csv"

COLS = ["Group", "Number", "Photo_Session", "Photo_Date", "Birth_Date",
        "Sex", "Age", "BMI", "NPT_Date", "NPT_Session", "Diff_Days",
        "MMSE", "CASI", "Global_CDR"]


def _digits(x):
    """抽出字串中的數字並轉 int；無數字回 None。"""
    d = re.sub(r"\D", "", str(x))
    return int(d) if d else None


def _group_of_sheet(name: str):
    """由分頁名判斷組別；Non-Dementia 須先於 Dementia 判斷。"""
    if "Non-Dementia" in name:
        return "NAD"
    if "Dementia" in name:
        return "P"
    if "ACS" in name:
        return "ACS"
    return None


def build_birth_map() -> dict:
    """掃描 Excel 含「生日」欄的分頁，回 {(group, id_int): 'YYYY/MM/DD'}。"""
    birth_map = {}
    xl = pd.ExcelFile(EXCEL_PATH)
    for sheet in xl.sheet_names:
        grp = _group_of_sheet(sheet)
        if grp is None:
            continue
        df = pd.read_excel(xl, sheet)
        birth_col = next((c for c in df.columns if "生日" in str(c)), None)
        id_col = next((c for c in df.columns if str(c).startswith("編號")), None)
        if birth_col is None or id_col is None:
            continue
        for _, row in df.iterrows():
            sid = _digits(row[id_col])
            if sid is None or pd.isna(row[birth_col]):
                continue
            raw = re.sub(r"/+", "/", str(row[birth_col]).strip())  # 修「1954/05//07」
            if re.fullmatch(r"\d{4,6}", raw):  # Excel 序列日期（自 1899-12-30 起算）
                dt = pd.to_datetime(int(raw), unit="D", origin="1899-12-30",
                                    errors="coerce")
            else:
                dt = pd.to_datetime(raw, errors="coerce")
            if pd.isna(dt):
                continue
            birth_map[(grp, sid)] = dt.strftime("%Y/%m/%d")
    return birth_map


def main():
    birth_map = build_birth_map()

    frames = []
    for grp in ("P", "NAD", "ACS"):
        df = pd.read_csv(LEGACY_DIR / f"{grp}.csv")
        df["Group"] = grp
        # 來源完整 ID "P1-2" → 個案編號 Number=1（ID 不存檔，讀入時再組）。
        df["Number"] = df["ID"].str.extract(r"^[A-Za-z]+(\d+)-")[0].astype(int)
        df["Birth_Date"] = df["Number"].apply(lambda s: birth_map.get((grp, int(s))))
        for c in COLS:
            if c not in df.columns:
                df[c] = pd.NA  # ACS 無 Global_CDR
        frames.append(df[COLS])

    out = pd.concat(frames, ignore_index=True)
    out["NPT_Session"] = out["NPT_Session"].astype("Int64")
    out.to_csv(OUTPUT_CSV, index=False)

    cov = out.groupby("Group")["Birth_Date"].apply(
        lambda s: f"{s.notna().sum()}/{len(s)}")
    print(f"hospital_A.csv: {len(out)} 列 → {OUTPUT_CSV}")
    print(f"Birth_Date 覆蓋: {dict(cov)}")


if __name__ == "__main__":
    main()
