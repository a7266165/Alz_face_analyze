"""[LEGACY] 分組 / 額外篩選 helpers — 從 src/cohort.py 隔離。

- ``split_by_metric_median``：MMSE/CASI 中位數切 hi/lo，是「定義比較組」
  （對應 flowchart partition 區），不是族群挑選。
- ``apply_max_cdr_filter``：subject-level 取 max CDR 的篩選，目前無 consumer
  （死碼），暫存待清理。
"""
import numpy as np
import pandas as pd


def split_by_metric_median(cohort, tiebreak="high", metric="MMSE",
                           group_col=None):
    """Median split of cohort by metric.  Adds <metric_low>_group col."""
    if group_col is None:
        group_col = f"{metric.lower()}_group"
    median = cohort[metric].median()
    if tiebreak == "high":
        cohort[group_col] = np.where(cohort[metric] >= median, "high", "low")
    else:
        cohort[group_col] = np.where(cohort[metric] > median, "high", "low")
    return cohort, float(median)


def apply_max_cdr_filter(df, min_cdr=0.5, group_col="base_id",
                         cdr_col="Global_CDR"):
    """Subject-level CDR filter: keep subjects whose max CDR across visits
    >= *min_cdr*.  Different from ``apply_p_cdr_filter`` which is per-visit."""
    cdr = pd.to_numeric(df[cdr_col], errors="coerce")
    max_cdr = cdr.groupby(df[group_col]).max()
    keep = max_cdr[max_cdr >= min_cdr].index
    return df[df[group_col].isin(keep)].copy()
