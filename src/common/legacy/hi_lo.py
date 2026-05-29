"""[LEGACY] AD 內部 MMSE/CASI hi-lo cohort — 從 cohort 移出（無外部 consumer）。

P 群依 metric 中位數切 hi/lo。保留於此供查考；現行主流程只用
``src.common.cohort.cohort_list``（AD-vs-HC）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HOSPITAL_A_CSV, DEFAULT_COHORT_MODE, cohort_spec_from_name
from src.common.cohort import load_demographics, p_filter, visit_selection

VALID_DESIGNS = ("cross_naive", "cross_matched")


def build_cohort_ad_hi_lo(
    design,
    cohort_mode=DEFAULT_COHORT_MODE,
    metric="MMSE",
    matched_features_csv=None,
):
    """AD-internal hi-lo cohort.  metric in {'MMSE', 'CASI'}."""
    if design not in VALID_DESIGNS:
        raise ValueError(
            f"design must be one of {VALID_DESIGNS}, got {design!r}")
    spec = cohort_spec_from_name(cohort_mode)
    metric_low = metric.lower()
    group_col = f"{metric_low}_group"

    if design == "cross_naive":
        demo = load_demographics(("P",))
        demo = demo[demo[metric].notna() & demo["Age"].notna()].copy()
        # base_id 由 load_demographics 提供；組回完整特徵 ID。
        demo["ID"] = demo["base_id"] + "-" + demo["Photo_Session"].astype(str)
        demo = p_filter(demo, f"p_{spec.p_cdr}")
        cohort = visit_selection(demo, f"p_{spec.p_visit}")
        med = cohort[metric].median()
        cohort[group_col] = np.where(cohort[metric] >= med, "high", "low")
        cohort["label"] = (cohort[group_col] == "high").astype(int)
        return cohort

    # design == 'cross_matched' — 讀入外部已配對 CSV
    if matched_features_csv is None:
        raise ValueError(
            "design='cross_matched' for hi-lo requires matched_features_csv")
    if not Path(matched_features_csv).exists():
        raise FileNotFoundError(
            f"Run run_cross_matched.py --comparison {metric_low}_hilo first; "
            f"missing: {matched_features_csv}")
    cohort = pd.read_csv(matched_features_csv)
    cohort["label"] = (cohort[group_col] == "high").astype(int)
    p_df = pd.read_csv(HOSPITAL_A_CSV)
    p_df = p_df[p_df["Group"] == "P"].copy()
    p_df["ID"] = (p_df["Group"] + p_df["ID"].astype(str)
                  + "-" + p_df["Photo_Session"].astype(str))
    keep_cog = [c for c in ["MMSE", "CASI", "Global_CDR", "CDR_SB"]
                if c in p_df.columns and c not in cohort.columns]
    if keep_cog:
        cohort = cohort.merge(
            p_df[["ID"] + keep_cog].drop_duplicates("ID"), on="ID", how="left")
        for c in keep_cog:
            cohort[c] = pd.to_numeric(cohort[c], errors="coerce")
    return cohort
