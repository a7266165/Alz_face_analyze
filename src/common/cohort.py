"""
Cohort 定義與族群挑選。

讀取單一乾淨人口學表 data/demographics/hospital_A.csv，依
  (1) visit 設定（first / all）
  (2) P 的 CDR 設定、HC 的 CDR / MMSE 設定
挑選研究族群，並回傳名單。

回傳 7 欄：Group, ID, Photo_Session, Age, MMSE, CASI, Global_CDR。
  - Group         P / NAD / ACS（直接一欄，下游不需再用正則從 ID 拆 group）
  - ID            受試者數字（1, 2, …；同一人不同 visit 共用同一 ID）
  - Photo_Session 第幾次拍照

下游若需對應特徵檔（.npy）/ predicted_ages.json 的完整 ID（"P1-2"），自行組回：
    feature_id = Group + ID.astype(str) + "-" + Photo_Session.astype(str)
而 group / base_id 不必再用正則，直接：
    group   = Group
    base_id = Group + ID.astype(str)            # 例如 "P1"
"""
import pandas as pd

from src.config import HOSPITAL_A_CSV

P_VISIT = {"p_first", "p_all"}
P_SCORE = {"p_cdrall", "p_cdr05", "p_cdr1", "p_cdr2"}
HC_VISIT = {"hc_first", "hc_all"}
HC_SCORE = {"hc_cdrall_or_mmseall", "hc_cdr0_or_mmse26"}

_OUTPUT_COLS = ["Group", "ID", "Photo_Session", "Age", "MMSE", "CASI",
                "Global_CDR"]
_P_CDR_THR = {"p_cdr05": 0.5, "p_cdr1": 1.0, "p_cdr2": 2.0}  # Global_CDR >= 門檻


def load_demographics(groups=("P", "NAD", "ACS")):
    """讀取 hospital_A.csv，篩出指定 *groups*。

    數值欄（Age / Global_CDR / MMSE / CASI）解析為數值；另補兩個內部便利欄供
    挑選用：base_id（= Group+ID，受試者層級）、visit（= Photo_Session）。
    """
    demo = pd.read_csv(HOSPITAL_A_CSV)
    demo = demo[demo["Group"].isin(groups)].copy()
    for c in ("Age", "Global_CDR", "MMSE", "CASI"):
        if c in demo.columns:
            demo[c] = pd.to_numeric(demo[c], errors="coerce")
    demo["base_id"] = demo["Group"] + demo["ID"].astype(str)
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

    回傳欄位：Group, ID, Photo_Session, Age, MMSE, CASI, Global_CDR。
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
