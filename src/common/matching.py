"""配對模組：match_by_age（按組別、以年齡配對）和 match_by_score（按問卷分數配對）
兩個主入口，皆吃 cohort spec、回 (case_ids, control_ids) 兩個 ID list。"""
import numpy as np
import pandas as pd

from src.common.cohort import cohort_list

__all__ = ["match_by_age", "match_by_score"]


def _prep_cohort(p_visit, p_score, hc_visit, hc_score):
    """cohort_list(spec) → 補 group(=Group) / base_id(由 ID 去尾)，供配對引擎用。"""
    prep = cohort_list(p_visit, p_score, hc_visit, hc_score).copy()
    prep["group"] = prep["Group"]
    prep["base_id"] = prep["ID"].str.extract(r"^(.+)-\d+$")[0]
    return prep


# ── 主入口 1：按組別、以年齡配對 ───────────────────────────────────────────────

def match_by_age(p_visit, p_score, hc_visit, hc_score, *,
                 controls=None, caliper=1.0, priority=None, level="subject",
                 mode="1to1", keep_groups=None, ttest_threshold=0.05):
    """載入 cohort(spec) 做 AD(P) vs 對照組年齡配對，回 (p_ids, hc_ids)。

    controls 選對照臂（None=全部非 P / ["NAD"] / ["ACS"]）；mode="1to1" 成對
    （兩 list 等長、index 對齊），"1toN" 做 caliper 平衡擴充（兩 list 不等長；
    keep_groups / ttest_threshold 為其參數）。
    """
    prep = _prep_cohort(p_visit, p_score, hc_visit, hc_score)
    if controls is not None:
        prep = prep[prep["group"].isin(["P", *controls])].copy()
    prep["label"] = (prep["group"] == "P").astype(int)
    prep["mmse_group"] = np.where(prep["label"] == 1, "high", "low")
    prep["MMSE"] = prep["MMSE"].fillna(999)
    matched, _, _ = _age_match_1_by_1(
        prep, caliper=caliper, metric="MMSE",
        group_col="mmse_group", priority_groups=priority,
        match_level=level,
    )

    if mode == "1toN":
        out = matched.merge(
            prep[["ID", "base_id", "group", "Age", "MMSE", "Global_CDR",
                  "label"]].drop_duplicates("ID"),
            on="ID", how="left", suffixes=("", "_p"),
        )
        out = out.drop(columns=[c for c in out.columns if c.endswith("_p")])
        expanded, _ = _age_match_1_by_n(
            prep, out, keep_groups=keep_groups, caliper=caliper,
            ttest_threshold=ttest_threshold)
        rep = (prep.sort_values(["base_id", "Age"])
               .drop_duplicates("base_id")[["base_id", "ID"]])
        exp = expanded.merge(rep, on="base_id", how="left")
        p_ids = exp[exp["label"] == 1]["ID"].tolist()
        hc_ids = exp[exp["label"] == 0]["ID"].tolist()
    else:
        lab = prep[["ID", "label"]].drop_duplicates("ID")
        m = matched.merge(lab, on="ID", how="left")
        p_ids = m[m["label"] == 1].sort_values("pair_id")["ID"].tolist()
        hc_ids = m[m["label"] == 0].sort_values("pair_id")["ID"].tolist()

    return p_ids, hc_ids


# ── 主入口 2：按問卷分數配對 ───────────────────────────────────────────────────

def match_by_score(p_visit, p_score, hc_visit, hc_score,
                   within, questionnaire, threshold,
                   *, caliper=1.0, level="subject"):
    """載入 cohort(spec)、篩到 within 組，按 questionnaire 的 threshold 切 high/low
    再做年齡配對，回 (high_ids, low_ids)。

    within ∈ {P, NAD, ACS}；questionnaire ∈ {MMSE, CASI, Global_CDR …}；
    threshold="median"（組內中位數）或給數值。
    """
    prep = _prep_cohort(p_visit, p_score, hc_visit, hc_score)
    prep = prep[prep["group"] == within].copy()
    s = pd.to_numeric(prep[questionnaire], errors="coerce")
    prep = prep[s.notna()].copy()
    s = pd.to_numeric(prep[questionnaire], errors="coerce")
    thr = float(s.median()) if threshold == "median" else float(threshold)
    prep["score_group"] = np.where(s >= thr, "high", "low")
    matched, _, _ = _age_match_1_by_1(
        prep, caliper=caliper, metric=questionnaire,
        group_col="score_group", match_level=level,
    )
    # matched 的 score_group 欄（high/low）由核心輸出，據以分組、按 pair_id 對齊。
    high_ids = matched[matched["score_group"] == "high"].sort_values("pair_id")["ID"].tolist()
    low_ids = matched[matched["score_group"] == "low"].sort_values("pair_id")["ID"].tolist()
    return high_ids, low_ids


# ── 私有核心 ───────────────────────────────────────────────────────────────────

def _age_match_1_by_1(cohort, caliper=2.0, metric="MMSE", group_col=None,
                      priority_groups=None, id_col="ID", match_level="subject"):
    """核心：scipy 最佳指派的 1:1 年齡配對（cohort 需含 group_col 'high'/'low'），回 (matched, pairs, (minor_label, major_label))。

    priority_groups 設定時稀少組先配一輪；match_level="subject" 每人去重、
    "visit" 每次拜訪獨立。
    """
    from scipy.optimize import linear_sum_assignment

    if group_col is None:
        group_col = f"{metric.lower()}_group"
    metric_low = metric.lower() if metric else None

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


def _age_match_1_by_n(full_cohort, matched_cohort,
                      keep_groups=None, caliper=1.0,
                      ttest_threshold=0.05):
    """核心：在 1:1 配對之上做 1:N caliper 平衡擴充（每輪加 P + Welch t-test 守門），回 (expanded, age_balance)。

    keep_groups 限定對照組（如 {"P", "ACS"}）；ttest_threshold 為年齡平衡 p 值下限。
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
