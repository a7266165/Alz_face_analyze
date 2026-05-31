"""
年齡 error 表。
"""

import pandas as pd

from src.config import PREDICTED_AGES_FILE
from src.age.utils import load_predicted_ages


def load_age_error(p_visit, p_score, hc_visit, hc_score, *,
                   predictions_file=PREDICTED_AGES_FILE):
    """指定 cohort 的逐受試者年齡誤差，回 DataFrame[ID, age_error]。

    age_error = real_age − predicted_age。real_age 取自 cohort（cohort_list 的
    Age，已轉數值），predicted_age 取自 predictions_file（load_predicted_ages）。
    無預測值的受試者已 dropna。其餘 metadata（group / 分數等）請消費端自行從
    cohort_list 取，再以 ID join。
    """
    # cohort 核心 lazy import，避免 import-time 相依環。
    from src.common.cohort import cohort_list

    preds = load_predicted_ages(predictions_file)
    df = cohort_list(p_visit, p_score, hc_visit, hc_score)
    real_age = pd.to_numeric(df["Age"], errors="coerce")
    predicted_age = df["ID"].map(preds)
    out = pd.DataFrame({"ID": df["ID"], "age_error": real_age - predicted_age})
    return out.dropna(subset=["age_error"]).reset_index(drop=True)
