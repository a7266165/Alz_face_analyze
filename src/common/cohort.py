"""
Cohort 定義與族群挑選。

讀取人口學表 data/demographics/hospital_A.csv，依
  (1) visit 設定（first / all）
  (2) P 的 CDR 設定、HC 的 CDR / MMSE 設定
挑選研究族群，並回傳名單。

回傳 8 欄：ID, Group, Number, Photo_Session, Age, MMSE, CASI, Global_CDR。
  - ID      單筆唯一鍵（"P1-2"），即 .npy / predicted_ages.json 的鍵，下游直接用。
            此欄不存於 CSV，由 load_demographics() 於讀入時組出（正規化、零冗餘）。
  - Group   P / NAD / ACS
  - Number  個案編號（人；只在同 Group 內唯一）。人鍵 base_id = Group+Number（"P1"）。

load_demographics() 是讀人口學的唯一入口——含直接讀檔需求的模組都應走它，
ID 的組裝邏輯只存在這一處。group 直接用 Group 欄，base_id = Group + Number。
"""
import pandas as pd

from src.config import HOSPITAL_A_CSV

P_VISIT = {"p_first", "p_all"}
P_SCORE = {"p_cdrall", "p_cdr05", "p_cdr1", "p_cdr2"}
HC_VISIT = {"hc_first", "hc_all"}
HC_SCORE = {"hc_cdrall_or_mmseall", "hc_cdr0_or_mmse26"}

_OUTPUT_COLS = ["ID", "Group", "Number", "Photo_Session", "Age", "MMSE",
                "CASI", "Global_CDR"]
_P_CDR_THR = {"p_cdr05": 0.5, "p_cdr1": 1.0, "p_cdr2": 2.0}  # Global_CDR >= 門檻


def load_demographics(groups=("P", "NAD", "ACS")):
    """讀取 hospital_A.csv，篩出指定 *groups*。

    數值欄（Age / Global_CDR / MMSE / CASI）解析為數值；組出唯一鍵 ID（"P1-2"）與
    內部便利欄：base_id（= Group+Number，受試者層級，例 "P1"）、visit（= Photo_Session）。
    """
    demo = pd.read_csv(HOSPITAL_A_CSV)
    demo = demo[demo["Group"].isin(groups)].copy()
    for c in ("Age", "Global_CDR", "MMSE", "CASI"):
        if c in demo.columns:
            demo[c] = pd.to_numeric(demo[c], errors="coerce")
    demo["base_id"] = demo["Group"] + demo["Number"].astype(str)
    demo["ID"] = demo["base_id"] + "-" + demo["Photo_Session"].astype(str)
    demo["visit"] = pd.to_numeric(demo["Photo_Session"], errors="coerce")
    return demo


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
    """*_first → 每 base_id 最早（visit 編號最小）的 visit；*_all → 全部。

    需要 base_id 欄；呼叫端需先做好 Age / metric 的 notna 過濾。"""
    df = df.sort_values(["base_id", "visit"])
    if visit in ("p_first", "hc_first"):
        return df.groupby("base_id", as_index=False).first()
    return df.reset_index(drop=True)


def cohort_list(p_visit, p_score, hc_visit, hc_score):
    """挑選 AD-vs-HC 族群，回傳 7 欄未配對名單。

    p_visit  ∈ {p_first, p_all}
    p_score  ∈ {p_cdrall, p_cdr05, p_cdr1, p_cdr2}
    hc_visit ∈ {hc_first, hc_all}
    hc_score ∈ {hc_cdrall_or_mmseall, hc_cdr0_or_mmse26}

    回傳欄位：ID, Group, Number, Photo_Session, Age, MMSE, CASI, Global_CDR。
    """
    for v, vocab in [(p_visit, P_VISIT), (p_score, P_SCORE),
                     (hc_visit, HC_VISIT), (hc_score, HC_SCORE)]:
        if v not in vocab:
            raise ValueError(
                f"invalid token {v!r}; expected one of {sorted(vocab)}")

    demo = load_demographics()
    demo = demo[demo["Age"].notna()].copy()

    p = visit_selection(p_filter(demo[demo["Group"] == "P"], p_score), p_visit)
    hc = visit_selection(hc_filter(demo[demo["Group"] != "P"], hc_score), hc_visit)

    return pd.concat([p, hc], ignore_index=True)[_OUTPUT_COLS].reset_index(drop=True)
