"""
年齡 error 表：real vs predicted age 的逐受試者比較表，供 scripts/age/error/ 的
四個 consumer（stat / lines / scatter / violin）共用，取代原本各自重抄一遍的
「load preds → join ID → dropna → age_error = real − predicted」區塊。

四個 consumer 現在都走 canonical cohort（給 cohort_mode 套 CDR/MMSE/visit 篩選），
並各自輸出「完整 cohort」與「1by1matched」兩份結果；後者的 AD-vs-HC 年齡配對由
本模組的 matched_ad_vs_hc 統一委派給 canonical src.common.matching。
"""

import numpy as np
import pandas as pd

from src.config import PREDICTED_AGES_FILE, cohort_spec_from_name
from src.age.utils import load_predicted_ages

_SCORE_COLS = ["MMSE", "CASI", "Global_CDR"]


def load_age_error_table(cohort_mode=None, *, predictions_file=PREDICTED_AGES_FILE):
    """逐受試者的 real vs predicted age error 表。

    cohort_mode 非 None → 以 src.common.cohort.cohort_list 套 canonical 篩選
                          （stat / lines 用，與 histogram 同一條 gold-standard 篩法）；
    cohort_mode None    → 以 src.common.cohort.load_demographics 取完整人口學，不篩
                          （scatter / violin 用）。

    回傳欄位（在來源欄位之上補出）：
        group, real_age, predicted_age, age_error(=real−predicted), age_int, subject,
        以及 MMSE / CASI / Global_CDR（來源缺者補 NaN）。
        來源本身的欄位（ID, Age, Group, Number, BMI …）一併保留。
    無 predicted_age 或 real_age 的列已 dropna。
    """
    # cohort-core 函式 lazy import，避免 import-time 相依環。
    from src.common.cohort import cohort_list, load_demographics

    preds = load_predicted_ages(predictions_file)

    if cohort_mode is not None:
        spec = cohort_spec_from_name(cohort_mode)
        df = cohort_list(
            f"p_{spec.p_visit}", f"p_{spec.p_cdr}", f"hc_{spec.hc_visit}",
            "hc_cdr0_or_mmse26" if spec.hc_strict else "hc_cdrall_or_mmseall")
    else:
        df = load_demographics()

    df = df.copy()
    df["group"] = df["Group"]              # ID 已是完整鍵 "P1-2"
    df["real_age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["predicted_age"] = df["ID"].map(preds)
    df = df.dropna(subset=["real_age", "predicted_age"]).reset_index(drop=True)
    df["age_error"] = df["real_age"] - df["predicted_age"]
    df["age_int"] = df["real_age"].astype(int)
    df["subject"] = df["ID"].apply(lambda x: str(x).rsplit("-", 1)[0])
    for c in _SCORE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def matched_ad_vs_hc(df, *, controls=None, caliper=1.0):
    """取 AD(P) vs 對照組的年齡 1:1 配對子集，回傳 df 中被選中的列。

    配對委派給 canonical ``src.common.matching.match_cohort``（gold standard，
    embedding / meta / histogram 同一支），caliper 預設 1.0。本函式只負責「把
    error 表整理成配對所需的 roster → 取回被選中的 ID → 篩回原列」。

        controls=None       → 所有非 P 當對照（標準 AD-vs-HC）
        controls=["NAD"] 等 → 只取指定組當對照（P vs NAD / P vs ACS）

    df 需含 ID / group / real_age / MMSE / Global_CDR（load_age_error_table 的輸出
    皆具備）。回傳保留 df 的全部欄位，已 reset_index。
    """
    from src.common.matching import match_cohort

    roster = df[["ID", "group", "MMSE", "Global_CDR"]].copy()
    roster["Age"] = pd.to_numeric(df["real_age"], errors="coerce")
    ml = match_cohort(roster, controls=controls, caliper=caliper)
    ids = set(ml.case["ID"]) | set(ml.control["ID"])
    return df[df["ID"].isin(ids)].reset_index(drop=True)


def matched_by_score(df, score, *, cut="median", caliper=1.0):
    """依問卷分數切 high/low 兩臂，取年齡 1:1 配對子集，回傳 df 中被選中的列。

    配對委派給 canonical ``src.common.matching.match_by_score``。score ∈
    {MMSE, CASI, Global_CDR …}；cut="median" 用中位數，或給數值。若只想在某組內
    比 high/low（例如僅患者 P），先自行篩好 df 再傳入。

    df 需含 ID / real_age / <score>。回傳保留 df 的全部欄位，已 reset_index。
    """
    from src.common.matching import match_by_score as _match_by_score

    roster = df.copy()
    roster["Age"] = pd.to_numeric(roster["real_age"], errors="coerce")
    ml = _match_by_score(roster, score, cut=cut, caliper=caliper)
    ids = set(ml.case["ID"]) | set(ml.control["ID"])
    return df[df["ID"].isin(ids)].reset_index(drop=True)
