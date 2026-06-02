"""[LEGACY] Age-calibration cohort builder — 從 src/cohort.py 隔離。

對應已被移除的 age-calibration 子系統（commit 587afef），目前無 consumer。
保留於此僅供日後查考，不建議再使用。
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_COHORT_TOKENS
from src.common.legacy.feature_gate import (
    keep_visits_with_features,
    pick_first_visit_with_features,
)
from src.common.legacy.predicted_age import load_predicted_ages

logger = logging.getLogger(__name__)


def build_cohort_for_age_calibration(
    cohort=DEFAULT_COHORT_TOKENS,
    hc_source_mode="ACS",
    predicted_ages_file=None,
):
    """Build calibration cohort using unified cohort logic.

    Applies the same CDR/MMSE/visit filters as the embedding pipeline
    but returns three separate DataFrames for calibration use.

    Returns (df_acs, df_nad, df_p) each with columns:
        ID, subject, group, real_age, predicted_age, error, age_int
    """
    # cohort-core 函式 lazy import，避免 import-time 相依環。
    from src.common.cohort import hc_filter, p_filter
    from src.common.legacy.eacs import load_combined_demographics_with_eacs
    from src.common.legacy.predicted_age import apply_predicted_age_filter

    p_visit, p_score, hc_visit, hc_score = cohort

    if predicted_ages_file is not None:
        pred_ages = json.loads(Path(predicted_ages_file).read_text(
            encoding="utf-8"))
    else:
        pred_ages = load_predicted_ages()

    demo = load_combined_demographics_with_eacs(hc_source_mode)

    # --- P side ---
    p_all = demo[(demo["group"] == "P") & demo["Age"].notna()].copy()
    p_all = p_filter(p_all, p_score)
    p_all = p_all.sort_values(["base_id", "visit"])
    p_all = apply_predicted_age_filter(p_all, pred_ages=pred_ages)
    if p_visit == "p_first":
        p_df = pick_first_visit_with_features(p_all)
    else:
        p_df = keep_visits_with_features(p_all)

    # --- HC side (ACS + NAD separately) ---
    hc_frames = {}
    for grp in ("ACS", "NAD"):
        hc = demo[(demo["group"] == grp) & demo["Age"].notna()].copy()
        hc = hc_filter(hc, hc_score)
        hc = hc.sort_values(["base_id", "visit"])
        hc = apply_predicted_age_filter(hc, pred_ages=pred_ages)
        if hc_visit == "hc_first":
            hc = hc.groupby("base_id", as_index=False).first()
        hc_frames[grp] = hc

    # --- shared post-processing ---
    out = {}
    for grp, df in [("P", p_df), ("ACS", hc_frames.get("ACS", pd.DataFrame())),
                     ("NAD", hc_frames.get("NAD", pd.DataFrame()))]:
        if df.empty:
            out[grp] = pd.DataFrame(
                columns=["ID", "subject", "group", "real_age",
                         "predicted_age", "error", "age_int"])
            continue
        df = df.copy()
        df["group"] = grp
        df["subject"] = df["ID"].apply(lambda x: str(x).rsplit("-", 1)[0])
        df["predicted_age"] = df["ID"].map(pred_ages)
        df = df.dropna(subset=["predicted_age", "Age"])
        df["real_age"] = df["Age"]
        df["error"] = df["real_age"] - df["predicted_age"]
        df["age_int"] = df["real_age"].apply(lambda x: int(np.floor(x)))
        out[grp] = df[
            ["ID", "subject", "group", "real_age", "predicted_age",
             "error", "age_int"]
        ].copy()

    logger.info(
        f"build_cohort_for_age_calibration (cohort={cohort}): "
        f"ACS={len(out['ACS'])}, NAD={len(out['NAD'])}, P={len(out['P'])}")
    return out["ACS"], out["NAD"], out["P"]
