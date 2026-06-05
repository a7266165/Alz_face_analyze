"""共用評估器。
"""
from pathlib import Path

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

from src.common.cohort import base_id_of, group_of
from src.common.matching import match_by_age

__all__ = [
    "bootstrap_auc_ci", "compute_clf_metrics", "paired_wilcoxon",
    "evaluate",
    "AD_CONTRASTS", "MATCHING_PRIORITIES", "MATCHED_UNITS", "EVAL_UNITS",
    "CONTRAST_KEEP_GROUPS",
]


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def bootstrap_auc_ci(y_true, y_score, *, n=100, seed=42):
    """AUC 的 95% bootstrap CI(percentile 2.5 / 97.5)。重抽到單一類別的樣本跳過;
    全部退化則回 (nan, nan)。"""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt, yp = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_clf_metrics(y_true, y_score, *, threshold=0.5, n_bootstrap=100,
                        seed=42, y_pred=None):
    """單一 cohort 的分類指標包。AUC / CI 永遠用連續 y_score;y_pred 給定時 threshold
    失效(沿用預先算好的 label)。回攤平 dict(CSV-friendly):
      n / n_pos / n_neg / auc / auc_ci_low / auc_ci_high /
      balacc / mcc / f1 / sens / spec / tn / fp / fn / tp。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    multi = len(np.unique(y_true)) > 1
    if multi:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    else:
        tn = fp = fn = tp = 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = float(roc_auc_score(y_true, y_score)) if multi else float("nan")
    ci_low, ci_high = (bootstrap_auc_ci(y_true, y_score, n=n_bootstrap, seed=seed)
                       if multi else (float("nan"), float("nan")))
    return {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "auc": auc,
        "auc_ci_low": ci_low if not np.isnan(ci_low) else None,
        "auc_ci_high": ci_high if not np.isnan(ci_high) else None,
        "balacc": float(balanced_accuracy_score(y_true, y_pred)) if multi else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if multi else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sens": float(sens) if not np.isnan(sens) else None,
        "spec": float(spec) if not np.isnan(spec) else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def paired_wilcoxon(scores, *, pair_col="pair_id", label_col="label",
                    score_col="y_score"):
    """配對 Wilcoxon signed-rank:同 pair_id 的 label==1 vs label==0 分數。
    回 dict: W / p / n_pairs / mean_diff(label1 − label0)。pivot 後 dropna 不成對者。"""
    wide = scores.pivot_table(index=pair_col, columns=label_col,
                              values=score_col, aggfunc="first").dropna()
    n = len(wide)
    if n < 2 or (1 not in wide.columns) or (0 not in wide.columns):
        return {"W": float("nan"), "p": float("nan"),
                "n_pairs": int(n), "mean_diff": float("nan")}
    pos = wide[1].to_numpy(float)
    neg = wide[0].to_numpy(float)
    if np.allclose(pos, neg):
        return {"W": float("nan"), "p": float("nan"),
                "n_pairs": int(n), "mean_diff": float("nan")}
    res = stats.wilcoxon(pos, neg, zero_method="wilcox", alternative="two-sided")
    return {"W": float(res.statistic), "p": float(res.pvalue),
            "n_pairs": int(n), "mean_diff": float(np.mean(pos - neg))}


# ----------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------

AD_CONTRASTS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
MATCHING_PRIORITIES = ["no_priority", "priority_acs", "priority_nad"]
MATCHED_UNITS = ["subject", "visit"]
EVAL_UNITS = ["eval_by_subject", "eval_by_visit"]
CONTRAST_KEEP_GROUPS = {
    "ad_vs_hc": None, "ad_vs_nad": {"P", "NAD"}, "ad_vs_acs": {"P", "ACS"}}
_PRIORITY_GROUPS = {"no_priority": None, "priority_acs": ["ACS"], "priority_nad": ["NAD"]}


def _prep_oof(oof, eval_unit):
    """補 base_id / group;eval_by_subject 聚到 base_id(y_score mean、fold max——
    同 subject 各 visit 必同折故 max 即代表值),eval_by_visit 保留 session 列。"""
    d = oof.copy()
    d["base_id"] = d["ID"].map(base_id_of)
    d["group"] = d["base_id"].map(group_of)
    if eval_unit == "eval_by_subject":
        d = d.groupby("base_id", as_index=False).agg(
            y_true=("y_true", "first"), y_score=("y_score", "mean"),
            fold=("fold", "max"), group=("group", "first"))
    return d


def _matched_df(p_ids, hc_ids):
    """兩個 1:1 對齊的 ID list → [match_id, base_id, group, pair_id, label]
    (case=1 / control=0)。match_id = match_by_age 給的原始 ID(subject-level 為 base_id、
    visit-level 為 session),供 join;base_id / group 供 contrast 過濾。"""
    rows = []
    for i, (p, h) in enumerate(zip(p_ids, hc_ids)):
        for sid, lab in ((p, 1), (h, 0)):
            bid = base_id_of(sid)
            rows.append({"match_id": str(sid), "base_id": bid,
                         "group": group_of(bid), "pair_id": i, "label": lab})
    return pd.DataFrame(rows)


def _build_ad_matched(cohort, matching_priority, matched_unit):
    """P-vs-全HC 的 1:1 年齡配對(優先序由 matching_priority 決定、粒度由 matched_unit 決定)。
    contrast 過濾在 _filter_contrast。"""
    p_ids, hc_ids = match_by_age(*cohort, priority=_PRIORITY_GROUPS[matching_priority],
                                 level=matched_unit, caliper=1.0)
    return _matched_df(p_ids, hc_ids)


def _filter_contrast(matched, contrast):
    kg = CONTRAST_KEEP_GROUPS[contrast]
    if kg is None:
        return matched
    target = set(kg) - {"P"}
    keep = matched[(matched["label"] == 0)
                   & (matched["group"].isin(target))]["pair_id"].unique()
    return matched[matched["pair_id"].isin(keep)]


def _join_matched(scope, matched, *, matched_unit, eval_unit):
    """把 matched cohort 的 label / pair_id 貼到 scope,回貼好的子集 sub(__lab / pair_id)。
    matched_unit='visit' 且 eval_by_visit → 用 session ID 接(視次對視次);其餘 → 用
    base_id 接(subject 對 subject;visit-match + subject-eval 即在此塌回 subject)。"""
    visit_join = (matched_unit == "visit" and eval_unit == "eval_by_visit")
    key_col = "match_id" if visit_join else "base_id"
    lab = matched.drop_duplicates(key_col).set_index(key_col)
    scope_key = scope["ID"] if visit_join else scope["base_id"]
    sub = scope[scope_key.isin(lab.index)].copy()
    k = sub["ID"] if visit_join else sub["base_id"]
    sub["__lab"] = k.map(lab["label"])
    sub["pair_id"] = k.map(lab["pair_id"])
    return sub


def _emit(out, df, labels, *, direction, matched_unit, matching_priority,
          eval_unit, contrast, domain, seed, paired=False):
    """清 NaN label 後算一格指標並 append 到 out;n<5 或單一類別則跳過(paired=True 時加配對 Wilcoxon)。"""
    df = df.reset_index(drop=True)
    labels = pd.Series(np.asarray(labels)).reset_index(drop=True)
    keep = labels.notna()
    df, labels = df[keep.values], labels[keep]
    if len(df) < 5 or labels.nunique() < 2:
        return
    y = labels.astype(int).to_numpy()
    met = compute_clf_metrics(y, df["y_score"].to_numpy(float), seed=seed)
    row = dict(direction=direction, matched_unit=matched_unit,
               matching_priority=matching_priority, eval_unit=eval_unit,
               contrast=contrast, domain=domain, **met)
    if paired and "pair_id" in df.columns:
        pw = paired_wilcoxon(pd.DataFrame({
            "pair_id": df["pair_id"].to_numpy(),
            "label": y, "y_score": df["y_score"].to_numpy(float)}))
        row.update({f"wilcoxon_{k}": v for k, v in pw.items()})
    out.append(row)


def _emit_forward(out, scope, *, eval_unit, contrast, matched_by,
                  matched_units, matching_priorities, seed):
    """forward:先發 'all'(整 scope),再對每個 (matched_unit, priority) 的 1:1 配對發 'paired' 指標。"""
    _emit(out, scope, scope["y_true"], direction="forward",
          matched_unit=None, matching_priority=None,
          eval_unit=eval_unit, contrast=contrast, domain="all", seed=seed)
    # visit-match × eval_by_subject 亦保留：visit 層配到的對塌回 subject 層評估
    # （_join_matched 走 base_id join，每 subject 取其首個 visit-pair 的 label/pair_id）。
    for matched_unit in matched_units:
        for mp in matching_priorities:
            matched = _filter_contrast(matched_by[(matched_unit, mp)], contrast)
            sub = _join_matched(scope, matched,
                                matched_unit=matched_unit, eval_unit=eval_unit)
            _emit(out, sub, sub["__lab"], direction="forward",
                  matched_unit=matched_unit, matching_priority=mp,
                  eval_unit=eval_unit, contrast=contrast,
                  domain="1by1", seed=seed, paired=True)


def _emit_reverse(out, scope, *, eval_unit, contrast, matching_priority_train, seed):
    """reverse:in-fold(fold>=0)發 '1by1'、out-of-fold(fold==-1)發 'other' 指標。"""
    m1 = scope[scope["fold"] >= 0]
    _emit(out, m1, m1["y_true"], direction="reverse",
          matched_unit=None, matching_priority=matching_priority_train,
          eval_unit=eval_unit, contrast=contrast, domain="1by1", seed=seed)
    mo = scope[scope["fold"] == -1]
    _emit(out, mo, mo["y_true"], direction="reverse",
          matched_unit=None, matching_priority=matching_priority_train,
          eval_unit=eval_unit, contrast=contrast, domain="other", seed=seed)


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------

def evaluate(oof_path, cohort, *, direction,
             matched_units=MATCHED_UNITS, matching_priorities=MATCHING_PRIORITIES,
             eval_units=EVAL_UNITS, contrasts=AD_CONTRASTS,
             write=True, out_name="metrics.csv", seed=42):
    """讀 oof_path 的 OOF 預測，逐 eval_unit × contrast 算分類指標；回 metrics DataFrame。

    Args:
        oof_path: OOF 預測 csv 路徑（含 ID / y_true / y_score / fold）。
        cohort: 4-token cohort spec (p_visit, p_score, hc_visit, hc_score)。
        direction: forward | reverse
        write: 為真時把結果寫到 oof_path 同目錄的 out_name。

    Returns:
        指標 DataFrame（每列一個 direction × matched_unit × priority × eval_unit ×
        contrast × domain 組合）。
    """
    if direction not in ("forward", "reverse"):
        raise ValueError(f"direction must be 'forward'|'reverse', got {direction!r}")

    # 讀檔
    oof_path = Path(oof_path)
    matching_priority_train = (
        oof_path.parent.name
        if direction == "reverse" and oof_path.parent.name in MATCHING_PRIORITIES
        else None)
    oof = pd.read_csv(oof_path)

    # 計算
    matched_by = ({(mu, mp): _build_ad_matched(cohort, mp, mu)
                   for mu in matched_units for mp in matching_priorities}
                  if direction == "forward" else {})
    out = []
    for eval_unit in eval_units:
        d = _prep_oof(oof, eval_unit)
        for contrast in contrasts:
            kg = CONTRAST_KEEP_GROUPS[contrast]
            scope = d if kg is None else d[d["group"].isin(set(kg))]

            if direction == "forward":
                _emit_forward(out, scope, eval_unit=eval_unit, contrast=contrast,
                              matched_by=matched_by, matched_units=matched_units,
                              matching_priorities=matching_priorities, seed=seed)
            else:
                _emit_reverse(out, scope, eval_unit=eval_unit, contrast=contrast,
                              matching_priority_train=matching_priority_train, seed=seed)
    res = pd.DataFrame(out)

    # 寫檔
    if write and len(res):
        res.to_csv(oof_path.parent / out_name, index=False, encoding="utf-8")
    return res
