"""
年齡 error 表：real vs predicted age 的逐受試者比較表，供 scripts/age/error/ 的
四個 consumer（stat / lines / scatter / violin）共用，取代原本各自重抄一遍的
「load preds → join ID → dropna → age_error = real − predicted」區塊。

stat / lines 走 canonical cohort（套 CDR/MMSE/visit 篩選）；scatter / violin 走
完整 demographics（不篩）。差別只在 cohort_mode 給不給——把這個「篩/不篩」做成
明確參數，而不是散在四個檔裡的隱性分歧。
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
