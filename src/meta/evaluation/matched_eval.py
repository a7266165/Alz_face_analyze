"""
Meta Analysis 匹配評估模組

eval chain:
  eval_strategy (1by1matched / caliper_group)
  × match_level (subject_match / visit_match)
  × eval_unit (eval_by_subject / eval_by_visit)
  × match_strategy (no_priority / priority_acs / priority_nad)
  × partition (ad_vs_hc / ad_vs_nad / ad_vs_acs / mmse_hilo / casi_hilo)

Matching 只依賴 demographics (Age)，與 embedding model / features 無關，
所以用 build_matching_cache() 在整個 sweep 最外層做一次。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from src.common.cohort import cohort_list
from src.common.legacy.matching import match_1to1, build_caliper_group
from src.config import cohort_spec_from_name

logger = logging.getLogger(__name__)

PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs", "mmse_hilo", "casi_hilo"]
EVAL_STRATEGIES = ["1by1matched"]
MATCH_LEVELS = ["subject_match"]
EVAL_UNITS = ["eval_by_subject", "eval_by_visit"]
MATCH_STRATEGIES = ["no_priority", "priority_acs", "priority_nad"]

PARTITION_KEEP_GROUPS = {
    "ad_vs_hc": None,
    "ad_vs_nad": {"P", "NAD"},
    "ad_vs_acs": {"P", "ACS"},
    "mmse_hilo": None,
    "casi_hilo": None,
}

AD_VS_PARTITIONS = {"ad_vs_hc", "ad_vs_nad", "ad_vs_acs"}
HILO_PARTITIONS = {"mmse_hilo", "casi_hilo"}


def bootstrap_auc_ci(y_true, y_prob, n=100, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_clf_metrics(y_true, y_score, threshold=0.5, n_bootstrap=100,
                        seed=42):
    y_pred = (y_score >= threshold).astype(int)
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    else:
        tn = fp = fn = tp = 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = float(roc_auc_score(y_true, y_score)) \
        if len(np.unique(y_true)) > 1 else float("nan")
    ci_low, ci_high = (bootstrap_auc_ci(y_true, y_score, n=n_bootstrap, seed=seed)
                       if len(np.unique(y_true)) > 1
                       else (float("nan"), float("nan")))
    return {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "auc": auc,
        "auc_ci_low": float(ci_low) if not np.isnan(ci_low) else None,
        "auc_ci_high": float(ci_high) if not np.isnan(ci_high) else None,
        "balacc": float(balanced_accuracy_score(y_true, y_pred))
            if len(np.unique(y_true)) > 1 else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
            if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sens": float(sens) if not np.isnan(sens) else None,
        "spec": float(spec) if not np.isnan(spec) else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def paired_wilcoxon_by_pair(matched_with_score):
    wide = matched_with_score.pivot_table(
        index="pair_id", columns="label", values="y_score", aggfunc="first",
    ).dropna()
    n = len(wide)
    if n < 2 or (1 not in wide.columns) or (0 not in wide.columns):
        return {"W": float("nan"), "p": float("nan"),
                "n_pairs": int(n), "mean_diff": float("nan")}
    pos = wide[1].to_numpy(dtype=float)
    neg = wide[0].to_numpy(dtype=float)
    if np.allclose(pos, neg):
        return {"W": float("nan"), "p": float("nan"),
                "n_pairs": int(n), "mean_diff": float("nan")}
    res = stats.wilcoxon(pos, neg, zero_method="wilcox", alternative="two-sided")
    return {
        "W": float(res.statistic), "p": float(res.pvalue),
        "n_pairs": int(n), "mean_diff": float(np.mean(pos - neg)),
    }


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ====================================================================
# Global matching cache — call once, reuse across all runs
# ====================================================================

def build_matching_cache(
    cohort_mode: str,
    partitions: List[str] = None,
    match_levels: List[str] = None,
    match_strategies: List[str] = None,
    caliper: float = 1.0,
    seed: int = 42,
) -> dict:
    """Build all matching cohorts upfront. Returns cache dict.

    Uses cohort_list to get cohort-filtered demographics that respect
    the p_visit / hc_visit / CDR / MMSE settings.

    Only depends on demographics (Age), independent of embedding/features.
    AD partitions share one matching per (match_strat, match_level).
    Hilo partitions ignore match_strategy.
    """
    if partitions is None:
        partitions = list(PARTITIONS)
    if match_levels is None:
        match_levels = list(MATCH_LEVELS)
    if match_strategies is None:
        match_strategies = list(MATCH_STRATEGIES)

    spec = cohort_spec_from_name(cohort_mode)
    demo = cohort_list(
        f"p_{spec.p_visit}", f"p_{spec.p_cdr}", f"hc_{spec.hc_visit}",
        "hc_cdr0_or_mmse26" if spec.hc_strict else "hc_cdrall_or_mmseall")
    demo["group"] = demo["Group"]
    demo["base_id"] = demo["Group"] + demo["ID"].astype(str)
    demo["ID"] = demo["base_id"] + "-" + demo["Photo_Session"].astype(str)
    demo["label"] = (demo["group"] == "P").astype(int)

    cache = {"demo": demo}

    has_ad = bool(set(partitions) & AD_VS_PARTITIONS)
    has_hilo = [p for p in partitions if p in HILO_PARTITIONS]

    for ml_name in match_levels:
        ml_param = "subject" if ml_name == "subject_match" else "visit"

        if has_ad:
            for match_strat in match_strategies:
                priority = None
                if match_strat == "priority_acs":
                    priority = ["ACS"]
                elif match_strat == "priority_nad":
                    priority = ["NAD"]
                try:
                    full, matched, pairs = _build_ad_partition(
                        demo, caliper, seed, priority,
                        match_level=ml_param,
                    )
                    cache[("ad", match_strat, ml_name)] = (full, matched, pairs)
                    logger.info(f"  Matching: AD/{match_strat}/{ml_name} → {len(matched)} rows")
                except Exception as e:
                    logger.warning(f"  Matching 失敗: AD/{match_strat}/{ml_name}: {e}")

        for hilo_part in has_hilo:
            metric = "MMSE" if hilo_part == "mmse_hilo" else "CASI"
            try:
                full, matched, pairs = _build_hilo_partition(
                    demo, caliper, seed, metric,
                    match_level=ml_param,
                )
                cache[(hilo_part, ml_name)] = (full, matched, pairs)
                logger.info(f"  Matching: {hilo_part}/{ml_name} → {len(matched)} rows")
            except Exception as e:
                logger.warning(f"  Matching 失敗: {hilo_part}/{ml_name}: {e}")

    n_entries = len([k for k in cache if k != "demo"])
    logger.info(f"Matching cache 建構完成: {n_entries} entries")
    return cache


# ====================================================================
# Eval chain — uses pre-built matching cache
# ====================================================================

def run_matched_eval_chain(
    oof_scores: pd.DataFrame,
    matching_cache: dict,
    output_dir: Path,
    partitions: List[str] = None,
    eval_strategies: List[str] = None,
    match_levels: List[str] = None,
    eval_units: List[str] = None,
    match_strategies: List[str] = None,
    caliper: float = 1.0,
    seed: int = 42,
    meta_info: Dict[str, Any] = None,
):
    """Run eval chain using pre-built matching cache.

    Parameters
    ----------
    oof_scores : DataFrame
        Subject-level meta OOF. Columns: base_id, y_score, y_true, subject_id.
    matching_cache : dict
        From build_matching_cache(). Contains matched cohorts + demographics.
    """
    if partitions is None:
        partitions = list(PARTITIONS)
    if eval_strategies is None:
        eval_strategies = list(EVAL_STRATEGIES)
    if match_levels is None:
        match_levels = list(MATCH_LEVELS)
    if eval_units is None:
        eval_units = list(EVAL_UNITS)
    if match_strategies is None:
        match_strategies = list(MATCH_STRATEGIES)

    demo = matching_cache.get("demo")
    if demo is None:
        logger.error("matching_cache 缺少 demo")
        return

    oof = oof_scores.copy()
    if "base_id" not in oof.columns and "subject_id" in oof.columns:
        import re
        oof["base_id"] = oof["subject_id"].apply(
            lambda s: re.match(r"^([A-Za-z]+\d+)", s).group(1)
            if re.match(r"^([A-Za-z]+\d+)", s) else s
        )

    oof_subj = oof[["base_id", "y_score"]].drop_duplicates("base_id")

    results_all = []
    saved_cohorts = set()

    for partition in partitions:
        logger.info(f"  Partition: {partition}")
        keep_groups = PARTITION_KEEP_GROUPS.get(partition)

        for match_strat in match_strategies:
            for ml_name in match_levels:
                if partition in AD_VS_PARTITIONS:
                    entry = matching_cache.get(("ad", match_strat, ml_name))
                elif partition in HILO_PARTITIONS:
                    entry = matching_cache.get((partition, ml_name))
                else:
                    continue
                if entry is None:
                    continue

                full, matched_raw, pairs = entry

                if keep_groups is not None:
                    target_groups = set(keep_groups) - {"P"}
                    target_pair_ids = matched_raw[
                        matched_raw["group"].isin(target_groups)
                    ]["pair_id"].unique()
                    matched_eval = matched_raw[matched_raw["pair_id"].isin(target_pair_ids)]
                else:
                    matched_eval = matched_raw

                matched_bids = set(matched_eval["base_id"].unique())
                if len(matched_bids) < 5:
                    continue

                cohort_key = (match_strat, ml_name, partition)
                cohort_saved = cohort_key in saved_cohorts

                for eval_strat in eval_strategies:
                    for eval_unit in eval_units:
                        cell_dir = (
                            output_dir / eval_strat / ml_name
                            / eval_unit / match_strat / partition
                        )
                        cell_dir.mkdir(parents=True, exist_ok=True)

                        try:
                            cell_result = _eval_cell(
                                oof_subj, matched_eval, pairs,
                                full, keep_groups, demo,
                                eval_strat, ml_name, eval_unit,
                                caliper, seed,
                            )
                        except Exception as e:
                            logger.warning(f"    {cell_dir.name}: {e}")
                            continue

                        cell_result["partition"] = partition
                        cell_result["eval_strategy"] = eval_strat
                        cell_result["match_level"] = ml_name
                        cell_result["eval_unit"] = eval_unit
                        cell_result["match_strategy"] = match_strat
                        if meta_info:
                            cell_result.update(meta_info)

                        _write_json(cell_dir / "metrics.json", cell_result)

                        if not cohort_saved:
                            matched_eval.to_csv(
                                cell_dir / "matched_cohort.csv",
                                index=False, encoding="utf-8-sig",
                            )
                            saved_cohorts.add(cohort_key)

                        results_all.append(cell_result)

    return results_all


def _build_ad_partition(demo, caliper, seed, priority_groups,
                        match_level="subject"):
    cols = ["ID", "Age", "base_id", "group", "label"]
    for extra in ["MMSE", "CASI"]:
        if extra in demo.columns:
            cols.append(extra)
    cohort = demo[[c for c in cols if c in demo.columns]].copy()
    if "base_id" not in cohort.columns:
        cohort["base_id"] = cohort["ID"].str.extract(r"^([A-Za-z]+\d+)")[0]
    if "group" not in cohort.columns:
        cohort["group"] = cohort["base_id"].apply(
            lambda b: "P" if b.startswith("P")
            else ("ACS" if b.startswith("ACS") else "NAD")
        )
    if "label" not in cohort.columns:
        cohort["label"] = (cohort["group"] == "P").astype(int)
    cohort["hc_group"] = np.where(cohort["group"] == "P", "low", "high")

    matched, pairs, _ = match_1to1(
        cohort, caliper=caliper, seed=seed,
        group_col="hc_group", metric=None, match_level=match_level,
        priority_groups=priority_groups,
    )

    keep_cols = ["base_id", "group", "label"]
    for c in keep_cols:
        if c not in matched.columns and c in cohort.columns:
            matched = matched.merge(
                cohort[["ID", c]].drop_duplicates("ID"),
                on="ID", how="left",
            )

    return cohort, matched, pairs


def _build_hilo_partition(demo, caliper, seed, metric,
                          match_level="subject"):
    metric_col = metric.upper() if metric.upper() in demo.columns else metric
    if metric_col not in demo.columns:
        raise FileNotFoundError(f"Demographics 缺少 {metric} 欄位")

    cols = ["ID", "Age", metric_col]
    if "base_id" in demo.columns:
        cols.append("base_id")
    cohort = demo[cols].copy()
    if "base_id" not in cohort.columns:
        cohort["base_id"] = cohort["ID"].str.extract(r"^([A-Za-z]+\d+)")[0]

    p_only = cohort[cohort["base_id"].str.startswith("P")].copy()
    p_only = p_only.dropna(subset=[metric_col])
    if len(p_only) < 10:
        raise ValueError(f"{metric} 資料不足 ({len(p_only)} 筆)")
    median_val = p_only[metric_col].median()
    p_only["label"] = (p_only[metric_col] >= median_val).astype(int)
    p_only["hilo_group"] = p_only["label"].map({1: "high", 0: "low"})

    matched, pairs, _ = match_1to1(
        p_only, caliper=caliper, seed=seed,
        group_col="hilo_group", metric=None, match_level=match_level,
    )

    for c in ["base_id", "label"]:
        if c not in matched.columns and c in p_only.columns:
            matched = matched.merge(
                p_only[["ID", c]].drop_duplicates("ID"),
                on="ID", how="left",
            )

    return p_only, matched, pairs


def _eval_cell(oof_subj, matched_eval, pairs, full_cohort, keep_groups,
               demo, eval_strat, match_level, eval_unit, caliper, seed):
    if eval_unit == "eval_by_visit":
        matched_bids = set(matched_eval["base_id"].unique())
        demo_subset = demo[["ID", "Age", "base_id"]].copy() if "base_id" in demo.columns \
            else demo[["ID", "Age"]].assign(base_id=demo["ID"].str.extract(r"^([A-Za-z]+\d+)")[0])
        demo_subset = demo_subset[demo_subset["base_id"].isin(matched_bids)]
        scores = demo_subset.merge(oof_subj, on="base_id", how="inner")
        scores = scores.merge(
            matched_eval[["base_id", "pair_id", "label"]].drop_duplicates("base_id"),
            on="base_id", how="inner",
        )
    else:
        scores = matched_eval[["base_id", "pair_id", "label"]].drop_duplicates("base_id")
        scores = scores.merge(oof_subj, on="base_id", how="inner")

    if len(scores) < 5 or scores["label"].nunique() < 2:
        return {"metrics_matched": None, "paired_wilcoxon": None}

    if eval_strat == "caliper_group" and "group" in full_cohort.columns:
        try:
            cal_cohort, age_balance = build_caliper_group(
                full_cohort, matched_eval,
                keep_groups=keep_groups, caliper=caliper,
            )
            if eval_unit == "eval_by_visit":
                cal_bids = set(cal_cohort["base_id"].unique())
                demo_cal = demo[["ID", "Age", "base_id"]].copy() if "base_id" in demo.columns \
                    else demo[["ID", "Age"]].assign(base_id=demo["ID"].str.extract(r"^([A-Za-z]+\d+)")[0])
                demo_cal = demo_cal[demo_cal["base_id"].isin(cal_bids)]
                cal_scores = demo_cal.merge(oof_subj, on="base_id", how="inner")
                cal_scores = cal_scores.merge(
                    cal_cohort[["base_id", "label"]].drop_duplicates("base_id"),
                    on="base_id", how="inner",
                )
            else:
                cal_scores = cal_cohort[["base_id", "label"]].merge(
                    oof_subj, on="base_id", how="inner",
                )
            metrics = compute_clf_metrics(
                cal_scores["label"].to_numpy(int),
                cal_scores["y_score"].to_numpy(float),
                seed=seed,
            )
            return {
                "metrics_matched": metrics,
                "age_balance": age_balance,
                "eval_strategy": "caliper_group",
            }
        except Exception:
            pass

    metrics = compute_clf_metrics(
        scores["label"].to_numpy(int),
        scores["y_score"].to_numpy(float),
        seed=seed,
    )
    paired = paired_wilcoxon_by_pair(scores)

    return {
        "metrics_matched": metrics,
        "paired_wilcoxon": paired,
    }
