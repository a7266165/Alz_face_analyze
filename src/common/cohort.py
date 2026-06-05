"""Cohort 定義與族群挑選：讀 hospital_A.csv，依 4-token 規格挑 AD-vs-HC 名單。

選取軸與回傳欄位見 cohort_list。
"""
import pandas as pd

from src.config import HOSPITAL_A_CSV, validate_cohort_tokens

_OUTPUT_COLS = ["ID", "Group", "Number", "Photo_Session", "Age", "MMSE",
                "CASI", "Global_CDR"]
_P_CDR_THR = {"p_cdr05": 0.5, "p_cdr1": 1.0, "p_cdr2": 2.0}  # Global_CDR >= 門檻


def load_demographics(groups=("P", "NAD", "ACS")):
    """讀取 hospital_A.csv，篩出指定 *groups*。"""
    demo = pd.read_csv(HOSPITAL_A_CSV)
    demo = demo[demo["Group"].isin(groups)].copy()
    for c in ("Age", "Global_CDR", "MMSE", "CASI"):
        if c in demo.columns:
            demo[c] = pd.to_numeric(demo[c], errors="coerce")
    demo["base_id"] = demo["Group"] + demo["Number"].astype(str)
    demo["ID"] = demo["base_id"] + "-" + demo["Photo_Session"].astype(str)
    demo["visit"] = pd.to_numeric(demo["Photo_Session"], errors="coerce")
    return demo


# ID 切分工具
def base_id_of(id_str) -> str:
    """ID(含 -session 尾)→ base_id(Group+Number),如 'ACS1-1' → 'ACS1'。"""
    return str(id_str).rsplit("-", 1)[0]


def group_of(base_id) -> str:
    """base_id(如 NAD1)→ Group(NAD):去掉尾端數字編號。"""
    return str(base_id).rstrip("0123456789")

# 個案篩選工具
def p_filter(df, p_score):
    """P 側 CDR 門檻：p_cdrall 不篩；p_cdr05/1/2 = Global_CDR >= 0.5/1/2。"""
    if p_score == "p_cdrall":
        return df
    cdr = pd.to_numeric(df.get("Global_CDR"), errors="coerce")
    return df[cdr >= _P_CDR_THR[p_score]]


def hc_filter(df, hc_score):
    """HC 側認知門檻：hc_cdrall_or_mmseall 不篩；
    hc_cdr0_or_mmse26 = CDR==0 或 (CDR 缺 & MMSE>=26)。"""
    if hc_score == "hc_cdrall_or_mmseall":
        return df
    cdr = pd.to_numeric(df.get("Global_CDR"), errors="coerce")
    mmse = pd.to_numeric(df.get("MMSE"), errors="coerce")
    return df[(cdr == 0) | (cdr.isna() & (mmse >= 26))]


def visit_selection(df, visit):
    """*_first → 每 base_id 最早（visit 編號最小）的 visit；*_all → 全部。"""
    df = df.sort_values(["base_id", "visit"])
    if visit in ("p_first", "hc_first"):
        return df.groupby("base_id", as_index=False).first()
    return df.reset_index(drop=True)

# 主入口
def cohort_list(p_visit, p_score, hc_visit, hc_score):
    """挑選 AD-vs-HC 族群，回傳 8 欄未配對名單。

    p_visit  ∈ {p_first, p_all}
    p_score  ∈ {p_cdrall, p_cdr05, p_cdr1, p_cdr2}
    hc_visit ∈ {hc_first, hc_all}
    hc_score ∈ {hc_cdrall_or_mmseall, hc_cdr0_or_mmse26}

    回傳欄位：ID, Group, Number, Photo_Session, Age, MMSE, CASI, Global_CDR。
    """
    validate_cohort_tokens(p_visit, p_score, hc_visit, hc_score)

    demo = load_demographics()
    demo = demo[demo["Age"].notna()].copy()

    p = visit_selection(p_filter(demo[demo["Group"] == "P"], p_score), p_visit)
    hc = visit_selection(hc_filter(demo[demo["Group"] != "P"], hc_score), hc_visit)

    return pd.concat([p, hc], ignore_index=True)[_OUTPUT_COLS].reset_index(drop=True)
