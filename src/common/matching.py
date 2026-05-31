"""年齡配對 — 全專案 case-control 配對的 canonical 實作。

cohort（src.common.cohort）只負責「挑族群」，回傳未配對 roster；配對由本模組
負責。對外只有兩個主入口，都回 MatchedLists(case, control)：

    match_cohort(table, controls, caliper, *, mode)  — 按 Group 標籤分臂（AD vs HC/NAD/ACS）
    match_by_score(table, score, cut, caliper)       — 按問卷分數中位數/閾值分臂（MMSE/CASI/CDR）

MatchedLists.case / .control 各是 DataFrame[ID, Age, MMSE, CASI, Global_CDR]：
    match_cohort   → case = 患者(P) 組,    control = 對照(HC/NAD/ACS) 組
    match_by_score → case = high 組,        control = low 組
mode="1to1"（預設）兩組等長、index 對齊（case[i] 配 control[i]）；
mode="1toN" 做 caliper 平衡擴充（Welch t-test 守門），case(P) 比 control 長。

內部私有引擎 _age_match_1to1（1:1 最佳指派）/ _caliper_group_match（1:N 擴充）
不對外；所有外部 caller 一律走上面兩個主入口。

scipy import 留在函式內，避免 import 此模組就強制載入 scipy。
"""
from collections import namedtuple

import numpy as np
import pandas as pd

__all__ = ["match_cohort", "match_by_score", "MatchedLists"]

MatchedLists = namedtuple("MatchedLists", ["case", "control"])

_LIST_COLS = ["ID", "Age", "MMSE", "CASI", "Global_CDR"]


def _make_lists(case_ids, control_ids, table):
    """依 ID 從原 table 取 _LIST_COLS（原值，不含配對時的 MMSE fillna），組 MatchedLists。

    依 case_ids / control_ids 的順序排列 → 1:1 時 index 對齊即 pair 對應。
    """
    src = table.drop_duplicates("ID").set_index("ID")

    def take(ids):
        ids = list(ids)
        df = pd.DataFrame({"ID": ids})
        for c in _LIST_COLS[1:]:
            df[c] = src[c].reindex(ids).to_numpy() if c in src.columns else np.nan
        return df[_LIST_COLS]

    return MatchedLists(take(case_ids), take(control_ids))


# ── 主入口 1：按組別配對 ───────────────────────────────────────────────────────

def match_cohort(table, controls=None, caliper=1.0, *,
                 priority=None, level="subject",
                 mode="1to1", keep_groups=None, ttest_threshold=0.05):
    """AD(P) vs 對照組的年齡配對封裝（組別配對的主入口）。

    這是 ``src.common.cohort.cohort_list`` 之後的配對步驟——cohort 只回傳未配對
    roster，配對由此處負責。兩臂依 Group 標籤定義：

        controls=None       → 所有非 P 都當對照（標準 AD-vs-HC）
        controls=["NAD"] 等 → 只取指定組當對照（P vs NAD / P vs ACS）

    配對比例由 mode 決定（兩者都回 MatchedLists(case=P, control=HC)）：
        mode="1to1"（預設）→ 1:1 成對，case/control 等長、index 對齊
        mode="1toN"        → 在 1:1 之上做 caliper 平衡擴充（Welch t-test 守門，
                             keep_groups / ttest_threshold 為其參數），case(P) 較長

    group / label 由 ID 以正則拆出（roster 已帶 ``group`` 欄則沿用，因 EACS ID
    無法用 regex 拆 group）。內部把 P 標 high、對照標 low，餵給私有核心
    _age_match_1to1。
    """
    prep = table.copy()
    prep["base_id"] = prep["ID"].str.extract(r"^(.+)-\d+$")[0]
    if "group" not in prep.columns:
        prep["group"] = prep["ID"].str.extract(r"^([A-Za-z]+)\d")[0]
    if controls is not None:
        prep = prep[prep["group"].isin(["P", *controls])].copy()
    prep["label"] = (prep["group"] == "P").astype(int)
    prep["mmse_group"] = np.where(prep["label"] == 1, "high", "low")
    prep["MMSE"] = prep["MMSE"].fillna(999)
    matched, _, _ = _age_match_1to1(
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
        expanded, _ = _caliper_group_match(
            prep, out, keep_groups=keep_groups, caliper=caliper,
            ttest_threshold=ttest_threshold)
        rep = (prep.sort_values(["base_id", "Age"])
               .drop_duplicates("base_id")[["base_id", "ID"]])
        exp = expanded.merge(rep, on="base_id", how="left")
        case_ids = exp[exp["label"] == 1]["ID"].tolist()
        control_ids = exp[exp["label"] == 0]["ID"].tolist()
    else:
        lab = prep[["ID", "label"]].drop_duplicates("ID")
        m = matched.merge(lab, on="ID", how="left")
        case_ids = m[m["label"] == 1].sort_values("pair_id")["ID"].tolist()
        control_ids = m[m["label"] == 0].sort_values("pair_id")["ID"].tolist()

    return _make_lists(case_ids, control_ids, table)


# ── 主入口 2：按問卷分數配對 ───────────────────────────────────────────────────

def match_by_score(table, score, cut="median", caliper=1.0, *,
                   level="subject"):
    """依問卷分數切 high/low 兩臂，做年齡 1:1 配對，回 MatchedLists(case=high, control=low)。

    score ∈ {MMSE, CASI, Global_CDR …}（任何 table 內的數值欄）。
        cut="median"   → 以中位數切（high: score>=median, low: score<median）
        cut=<數值>     → 以該值切（high: score>=cut,    low: score<cut）

    兩臂都來自 table；若只想在某一組內切（例如僅患者 P 內比 high/low），先自行
    篩好子集再傳入。case/control 兩組各為 DataFrame[ID, Age, MMSE, CASI, Global_CDR]，
    index 對齊（case[i] 配 control[i]）。
    """
    prep = table.copy()
    if "base_id" not in prep.columns:
        prep["base_id"] = prep["ID"].str.extract(r"^(.+)-\d+$")[0]
    s = pd.to_numeric(prep[score], errors="coerce")
    prep = prep[s.notna()].copy()
    s = pd.to_numeric(prep[score], errors="coerce")
    thr = float(s.median()) if cut == "median" else float(cut)
    prep["score_group"] = np.where(s >= thr, "high", "low")
    matched, _, _ = _age_match_1to1(
        prep, caliper=caliper, metric=score,
        group_col="score_group", match_level=level,
    )
    # matched 的 score_group 欄（high/low）由核心輸出，據以分組、按 pair_id 對齊。
    case_ids = matched[matched["score_group"] == "high"].sort_values("pair_id")["ID"].tolist()
    control_ids = matched[matched["score_group"] == "low"].sort_values("pair_id")["ID"].tolist()
    return _make_lists(case_ids, control_ids, table)


# ── 核心引擎：年齡 1:1 最佳指派（低階；進階 caller 直接用）──────────────────────

def _age_match_1to1(cohort, caliper=2.0, metric="MMSE", group_col=None,
                    priority_groups=None, id_col="ID", match_level="subject"):
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


# ── 1:N caliper 平衡擴充（低階；接在 1:1 之上）─────────────────────────────────

def _caliper_group_match(full_cohort, matched_cohort,
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
