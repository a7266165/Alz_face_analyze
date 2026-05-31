"""1:1 / caliper 年齡配對 — 全專案 case-control 配對的 canonical 實作。

cohort（src.common.cohort）只負責「人口學族群挑選」，回傳未配對 roster；
配對屬於下游 eval chain（1by1matched / caliper_group / priority_*），由本模組
統一負責。embedding / meta / age / stat 等所有 consumer 共用此模組（gold
standard，已從 src.common.legacy 提升至 src.common）。

公開 API：
    match_cohort_ad_vs_hc — roster → 1:1 年齡配對的 AD-vs-HC（封裝層，caliper 預設 1.0）
    match_1to1            — scipy 最佳指派 1:1 配對（核心，caliper 預設 2.0）
    build_caliper_group   — 在 1:1 之上做 1:N 平衡擴充（Welch t-test 守門）

scipy import 留在函式內，避免 import 此模組就強制載入 scipy。
"""
import numpy as np
import pandas as pd


def match_cohort_ad_vs_hc(cohort, caliper=1.0, seed=42,
                          priority_groups=None, match_level="subject"):
    """從 AD-vs-HC 的未配對 roster 做 1:1 年齡配對，回傳 (matched_cohort, pairs)。

    這是 ``src.common.cohort.cohort_list`` 之後的配對步驟——cohort 只回傳未配對
    roster（6 欄），配對由此處負責。base_id / group / label 由 ID 以正則拆出
    （EACS roster 已帶 ``group`` 欄則沿用，因 EACS ID 無法用 regex 拆 group）。
    """
    prep = cohort.copy()
    prep["base_id"] = prep["ID"].str.extract(r"^(.+)-\d+$")[0]
    if "group" not in prep.columns:
        prep["group"] = prep["ID"].str.extract(r"^([A-Za-z]+)\d")[0]
    prep["label"] = (prep["group"] == "P").astype(int)
    prep["mmse_group"] = np.where(prep["label"] == 1, "high", "low")
    prep["MMSE"] = prep["MMSE"].fillna(999)
    matched, pairs, _ = match_1to1(
        prep, caliper=caliper, seed=seed, metric="MMSE",
        group_col="mmse_group", priority_groups=priority_groups,
        match_level=match_level,
    )
    out = matched.merge(
        prep[["ID", "base_id", "group", "Age", "MMSE", "Global_CDR",
              "label"]].drop_duplicates("ID"),
        on="ID", how="left", suffixes=("", "_p"),
    )
    out = out.drop(columns=[c for c in out.columns if c.endswith("_p")])
    return out, pairs


def match_1to1(cohort, caliper=2.0, seed=42, metric="MMSE", group_col=None,
               match_mode="visit", priority_groups=None, id_col="ID",
               match_level="subject"):
    """1:1 age-optimal match; cohort must have *group_col* 'high'/'low'.

    Uses ``scipy.optimize.linear_sum_assignment`` to maximise the number
    of matched pairs within *caliper* while minimising total age difference.

    When *priority_groups* is set (e.g. ``["ACS"]``), the first group
    is matched in a dedicated optimal-assignment round; remaining
    subjects are matched in a second round against the leftover pool.

    match_level="subject": dedup to one row per base_id before matching.
    match_level="visit": each visit is an independent candidate; same
        subject can appear in multiple pairs (different visits).

    Returns (matched_df, pairs_df, (minor_label, major_label)).
    """
    from scipy.optimize import linear_sum_assignment

    if group_col is None:
        group_col = f"{metric.lower()}_group"
    metric_low = metric.lower() if metric else None
    rng = np.random.RandomState(seed)

    high = cohort[cohort[group_col] == "high"].copy()
    low = cohort[cohort[group_col] == "low"].copy()
    if len(low) <= len(high):
        minor, major = low, high
        minor_label, major_label = "low", "high"
    else:
        minor, major = high, low
        minor_label, major_label = "high", "low"

    subject_level = (match_level == "subject") and ("base_id" in cohort.columns)
    bid_col = "base_id" if subject_level else id_col

    def _to_subject_df(df):
        return (df.sort_values([bid_col, "Age"])
                .drop_duplicates(bid_col, keep="first")
                .reset_index(drop=True))

    if subject_level:
        minor_subj = _to_subject_df(minor)
        major_subj = _to_subject_df(major)
    else:
        minor_subj = minor.reset_index(drop=True)
        major_subj = major.reset_index(drop=True)

    def _optimal_assign(mi, ma):
        if len(mi) == 0 or len(ma) == 0:
            return []
        age_mi = mi["Age"].to_numpy(float)
        age_ma = ma["Age"].to_numpy(float)
        cost = np.abs(age_mi[:, None] - age_ma[None, :])
        cost[cost > caliper] = 1e9
        ri, ci = linear_sum_assignment(cost)
        valid = cost[ri, ci] <= caliper
        return list(zip(ri[valid], ci[valid]))

    pairs_records = []

    def _record_pairs(mi_df, ma_df, assignments):
        for ri, ci in assignments:
            mi_row = mi_df.iloc[ri]
            ma_row = ma_df.iloc[ci]
            rec = {
                "pair_id": None,
                "minor_id": mi_row[id_col], "minor_age": mi_row["Age"],
                "major_id": ma_row[id_col], "major_age": ma_row["Age"],
                "age_diff": ma_row["Age"] - mi_row["Age"],
            }
            if metric_low:
                rec[f"minor_{metric_low}"] = mi_row[metric]
                rec[f"major_{metric_low}"] = ma_row[metric]
            pairs_records.append(rec)

    if priority_groups and "group" in cohort.columns:
        prio = priority_groups[0]
        grp_on_minor = prio in minor_subj["group"].unique()
        if grp_on_minor:
            mi_prio = minor_subj[
                minor_subj["group"] == prio
            ].reset_index(drop=True)
            ma_prio = major_subj.reset_index(drop=True)
        else:
            mi_prio = minor_subj.reset_index(drop=True)
            ma_prio = major_subj[
                major_subj["group"] == prio
            ].reset_index(drop=True)
        assignments = _optimal_assign(mi_prio, ma_prio)
        _record_pairs(mi_prio, ma_prio, assignments)
        used_minor_bids = {mi_prio.iloc[ri][bid_col]
                           for ri, _ in assignments}
        used_major_bids = {ma_prio.iloc[ci][bid_col]
                           for _, ci in assignments}

        mi_rest = minor_subj[
            ~minor_subj[bid_col].isin(used_minor_bids)
        ].reset_index(drop=True)
        ma_rest = major_subj[
            ~major_subj[bid_col].isin(used_major_bids)
        ].reset_index(drop=True)
        assignments = _optimal_assign(mi_rest, ma_rest)
        _record_pairs(mi_rest, ma_rest, assignments)
    else:
        assignments = _optimal_assign(minor_subj, major_subj)
        _record_pairs(minor_subj, major_subj, assignments)

    matched_records = []
    for i, rec in enumerate(pairs_records):
        rec["pair_id"] = i
        minor_rec = {
            "pair_id": i, id_col: rec["minor_id"],
            "Age": rec["minor_age"], group_col: minor_label,
        }
        major_rec = {
            "pair_id": i, id_col: rec["major_id"],
            "Age": rec["major_age"], group_col: major_label,
        }
        if metric_low:
            minor_rec[metric] = rec[f"minor_{metric_low}"]
            major_rec[metric] = rec[f"major_{metric_low}"]
        matched_records.append(minor_rec)
        matched_records.append(major_rec)
    pairs_df = pd.DataFrame(pairs_records)
    matched = pd.DataFrame(matched_records)
    return matched, pairs_df, (minor_label, major_label)


def build_caliper_group(full_cohort, matched_cohort,
                        keep_groups=None, caliper=1.0,
                        ttest_threshold=0.05):
    """1:N balanced matching built on top of a 1:1 matched cohort.

    Starting from the 1:1 pairs, round-robin adds P subjects to each HC
    subject (one per HC per round, within *caliper*).  After each round a
    Welch t-test checks age balance; the process stops (and rolls back the
    last round) when the t-test drops below *ttest_threshold*.

    Parameters
    ----------
    full_cohort : DataFrame  — full (unmatched) cohort with base_id, Age,
        label, group.
    matched_cohort : DataFrame — 1:1 matched cohort with pair_id, base_id,
        Age, label, group.
    keep_groups : optional set  — e.g. ``{"P", "ACS"}``.
    caliper : float — max |age_diff| for added P subjects.
    ttest_threshold : float — stop adding when p drops below this.

    Returns
    -------
    expanded_cohort : DataFrame (base_id, Age, label, group)
    age_balance : dict
    """
    from scipy import stats as _st

    cols = ["base_id", "Age", "label", "group"]
    mc = matched_cohort.drop_duplicates("base_id")

    # --- HC side (fixed) ---
    if keep_groups is not None:
        hc_groups = set(keep_groups) - {"P"}
        hc_subj = mc[mc["group"].isin(hc_groups)][cols].copy()
    else:
        hc_subj = mc[mc["label"] == 0][cols].copy()

    # --- P already in 1:1 ---
    matched_p_bids = set(mc[mc["label"] == 1]["base_id"])
    if keep_groups is not None:
        hc_pair_ids = set(
            matched_cohort[matched_cohort["group"].isin(hc_groups)]["pair_id"])
        matched_p_in_scope = set(
            matched_cohort[
                (matched_cohort["pair_id"].isin(hc_pair_ids))
                & (matched_cohort["label"] == 1)
            ]["base_id"])
    else:
        matched_p_in_scope = matched_p_bids
    p_matched = (full_cohort[full_cohort["base_id"].isin(matched_p_in_scope)]
                 .drop_duplicates("base_id")[cols].copy())

    # --- candidate P pool (not yet matched) ---
    all_p = (full_cohort[full_cohort["label"] == 1]
             .drop_duplicates("base_id"))
    p_candidates = all_p[~all_p["base_id"].isin(matched_p_bids)].copy()

    # --- build global (HC, P_candidate) pairs sorted by |age_diff| ---
    hc_records = hc_subj[["base_id", "Age"]].to_dict("records")
    cand_records = p_candidates[["base_id", "Age"]].to_dict("records")
    cand_age_map = {r["base_id"]: r["Age"] for r in cand_records}

    all_pairs = []
    for hc in hc_records:
        for c in cand_records:
            d = abs(c["Age"] - hc["Age"])
            if d <= caliper:
                all_pairs.append((d, c["base_id"], c["Age"]))
    all_pairs.sort()

    # --- greedy addition: add one P at a time, check t-test ---
    added_p_bids = []
    added_p_ages = []
    used_p = set()

    hc_ages = hc_subj["Age"].to_numpy(float)
    p_ages_base = p_matched["Age"].to_numpy(float)
    n_base = len(p_ages_base)

    for _, p_bid, p_age in all_pairs:
        if p_bid in used_p:
            continue

        trial_ages = np.concatenate([
            p_ages_base,
            np.array(added_p_ages + [p_age], dtype=float),
        ])
        _, pval = _st.ttest_ind(trial_ages, hc_ages, equal_var=False)

        if pval < ttest_threshold:
            continue

        used_p.add(p_bid)
        added_p_bids.append(p_bid)
        added_p_ages.append(p_age)

    # --- assemble final cohort ---
    added_p_df = (all_p[all_p["base_id"].isin(set(added_p_bids))]
                  [cols].copy())
    expanded = pd.concat([hc_subj, p_matched, added_p_df],
                         ignore_index=True)

    all_p_in = expanded[expanded["label"] == 1]
    p_arr_final = all_p_in["Age"].to_numpy(float)
    if len(p_arr_final) >= 2 and len(hc_ages) >= 2:
        t_stat, t_pval = _st.ttest_ind(p_arr_final, hc_ages, equal_var=False)
    else:
        t_stat, t_pval = float("nan"), float("nan")

    age_balance = {
        "caliper": caliper,
        "ttest_threshold": ttest_threshold,
        "n_hc": len(hc_subj),
        "n_p_matched_1to1": len(p_matched),
        "n_p_added": len(added_p_bids),
        "n_p_total": len(all_p_in),
        "n_p_pool": len(all_p),
        "hc_age_mean": float(hc_ages.mean()),
        "p_age_mean": float(p_arr_final.mean()) if len(p_arr_final) else None,
        "ttest_t": float(t_stat),
        "ttest_p": float(t_pval),
    }
    return expanded, age_balance
