"""
評估的「配對鏈」—— 吃 oof_scores.csv + cohort 4-token,按 direction 提取對的人、算指標。

**核心 = 依方向取「對的域」**(對齊 overview 圖的灰框 Fwd: all+1by1 / Rev: 1by1+other):
  - forward:OOF 是全 cohort 10 折,域 = {all(全體), 1by1(配對子集)}。1by1 需用
            match_by_age 當場重建配對(OOF 自己分不出誰是 matched,全 fold>=0)。
  - reverse:OOF 已按 ms_train 分檔,域 = {1by1(fold>=0, 訓練池 held-out),
            other(fold==-1, external ensemble)}。**只看 fold 欄就能切,不用重建配對。**

軸:partition(ad_vs_hc/nad/acs)× eval_unit(subject/visit)×(forward 再 × ms_eval)。
forward 的 ms 是「評估時挑哪個 matched cohort」;reverse 的 ms 是「訓練池」(已在路徑),
故 reverse 不在此迭代 ms——它由檔案位置決定,記在 match_strategy 欄(ms_train)。

純算、不畫、不碰 embedding 路徑。回 long-format DataFrame(一列 = 一個 cell × domain);
evaluate_oof_file 可把它寫成同資料夾的 metrics.csv。

註:mmse_hilo / casi_hilo 暫不在此——AD-vs-HC 的 OOF 與 reverse 的 fold 結構對 hi/lo
不成立(label 不在 y_true),需各自的 forward-only 處理,之後另開。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.matching import match_by_age
from .metrics import compute_clf_metrics, paired_wilcoxon

__all__ = [
    "evaluate", "evaluate_oof_file",
    "AD_PARTITIONS", "MATCH_STRATEGIES", "EVAL_UNITS", "PARTITION_KEEP_GROUPS",
]

AD_PARTITIONS = ["ad_vs_hc", "ad_vs_nad", "ad_vs_acs"]
MATCH_STRATEGIES = ["no_priority", "priority_acs", "priority_nad"]
EVAL_UNITS = ["eval_by_subject", "eval_by_visit"]
PARTITION_KEEP_GROUPS = {
    "ad_vs_hc": None, "ad_vs_nad": {"P", "NAD"}, "ad_vs_acs": {"P", "ACS"}}
_MS_PRIORITY = {"no_priority": None, "priority_acs": ["ACS"], "priority_nad": ["NAD"]}
_BASE_ID_RE = re.compile(r"^([A-Za-z]+\d+)")  # ID(session) → subject base_id,同 train


# ── 小工具 ─────────────────────────────────────────────────────────────────────

def _base_id(sid):
    m = _BASE_ID_RE.match(str(sid))
    return m.group(1) if m else str(sid)


def _group_of(base_id):
    """base_id(如 NAD1)→ group(NAD)。"""
    return str(base_id).rstrip("0123456789")


def _prep_oof(oof, eval_unit):
    """補 base_id / group;eval_by_subject 聚到 base_id(y_score mean、fold max——
    同 subject 各 visit 必同折故 max 即代表值),eval_by_visit 保留 session 列。"""
    d = oof.copy()
    d["base_id"] = d["ID"].map(_base_id)
    d["group"] = d["base_id"].map(_group_of)
    if eval_unit == "eval_by_subject":
        d = d.groupby("base_id", as_index=False).agg(
            y_true=("y_true", "first"), y_score=("y_score", "mean"),
            fold=("fold", "max"), group=("group", "first"))
    return d


def _matched_df(p_ids, hc_ids):
    """兩個 1:1 對齊的 ID list → [base_id, group, pair_id, label](case=1 / control=0)。"""
    rows = []
    for i, (p, h) in enumerate(zip(p_ids, hc_ids)):
        for sid, lab in ((p, 1), (h, 0)):
            bid = str(sid).rsplit("-", 1)[0]
            rows.append({"base_id": bid, "group": _group_of(bid),
                         "pair_id": i, "label": lab})
    return pd.DataFrame(rows)


def _build_ad_matched(cohort, ms, match_level):
    """P-vs-全HC 的 1:1 年齡配對(priority 由 ms 決定)。partition 過濾在 _filter_partition。"""
    p_ids, hc_ids = match_by_age(*cohort, priority=_MS_PRIORITY[ms],
                                 level=match_level, caliper=1.0)
    return _matched_df(p_ids, hc_ids)


def _filter_partition(matched, partition):
    """ad_vs_nad/acs:只留 control 屬目標組的 pair;ad_vs_hc:全留。"""
    kg = PARTITION_KEEP_GROUPS[partition]
    if kg is None:
        return matched
    target = set(kg) - {"P"}
    keep = matched[(matched["label"] == 0)
                   & (matched["group"].isin(target))]["pair_id"].unique()
    return matched[matched["pair_id"].isin(keep)]


def _emit(out, df, labels, *, partition, eval_unit, direction,
          match_strategy, domain, seed, paired=False):
    """算一個 (cell × domain) 的指標,append 一列 dict 到 out。樣本 <5 或單一類別則略過。"""
    df = df.reset_index(drop=True)
    labels = pd.Series(np.asarray(labels)).reset_index(drop=True)
    keep = labels.notna()
    df, labels = df[keep.values], labels[keep]
    if len(df) < 5 or labels.nunique() < 2:
        return
    y = labels.astype(int).to_numpy()
    met = compute_clf_metrics(y, df["y_score"].to_numpy(float), seed=seed)
    row = dict(partition=partition, eval_unit=eval_unit, direction=direction,
               match_strategy=match_strategy, domain=domain, **met)
    if paired and "pair_id" in df.columns:
        pw = paired_wilcoxon(pd.DataFrame({
            "pair_id": df["pair_id"].to_numpy(),
            "label": y, "y_score": df["y_score"].to_numpy(float)}))
        row.update({f"wilcoxon_{k}": v for k, v in pw.items()})
    out.append(row)


# ── 對外入口 ───────────────────────────────────────────────────────────────────

def evaluate(oof, cohort, *, direction, ms_train=None,
             partitions=AD_PARTITIONS, match_strategies=MATCH_STRATEGIES,
             eval_units=EVAL_UNITS, match_level="subject", seed=42):
    """吃一份 OOF DataFrame([ID, y_true, y_score, fold]),回 long-format 指標表。

    direction='forward':域 = all + 1by1(per ms_eval);需 cohort 重建配對。
    direction='reverse':域 = 1by1(fold>=0)+ other(fold==-1);ms 為訓練池(ms_train)。
    回欄:partition / eval_unit / direction / match_strategy / domain + compute_clf_metrics
         攤平欄(+ forward/1by1 的 wilcoxon_*)。
    """
    if direction not in ("forward", "reverse"):
        raise ValueError(f"direction must be 'forward'|'reverse', got {direction!r}")

    matched_by_ms = ({ms: _build_ad_matched(cohort, ms, match_level)
                      for ms in match_strategies}
                     if direction == "forward" else {})
    out = []
    for eval_unit in eval_units:
        d = _prep_oof(oof, eval_unit)
        for partition in partitions:
            kg = PARTITION_KEEP_GROUPS[partition]
            scope = d if kg is None else d[d["group"].isin(set(kg))]

            if direction == "forward":
                # all:全體(partition scope 內),label = AD y_true
                _emit(out, scope, scope["y_true"], partition=partition,
                      eval_unit=eval_unit, direction="forward",
                      match_strategy=None, domain="all", seed=seed)
                # 1by1:配對子集(per ms),label / pair_id 來自 matched cohort
                for ms in match_strategies:
                    matched = _filter_partition(
                        matched_by_ms[ms], partition).drop_duplicates("base_id")
                    lab = matched.set_index("base_id")
                    sub = scope[scope["base_id"].isin(lab.index)].copy()
                    sub["__lab"] = sub["base_id"].map(lab["label"])
                    sub["pair_id"] = sub["base_id"].map(lab["pair_id"])
                    _emit(out, sub, sub["__lab"], partition=partition,
                          eval_unit=eval_unit, direction="forward",
                          match_strategy=ms, domain="1by1", seed=seed, paired=True)
            else:  # reverse —— 域由 fold 切,ms_train 來自路徑
                m1 = scope[scope["fold"] >= 0]
                _emit(out, m1, m1["y_true"], partition=partition,
                      eval_unit=eval_unit, direction="reverse",
                      match_strategy=ms_train, domain="1by1", seed=seed)
                mo = scope[scope["fold"] == -1]
                _emit(out, mo, mo["y_true"], partition=partition,
                      eval_unit=eval_unit, direction="reverse",
                      match_strategy=ms_train, domain="other", seed=seed)
    return pd.DataFrame(out)


def evaluate_oof_file(oof_path, cohort, *, direction, ms_train=None, write=True,
                      out_name="metrics.csv", seed=42, **kw):
    """讀 oof_path 的 oof_scores.csv → evaluate → (可選)寫同資料夾的 metrics.csv。回 DataFrame。

    reverse 時 ms_train 未給則由路徑推(``.../rev/<ms_train>/oof_scores.csv`` 的父目錄名)。
    """
    oof_path = Path(oof_path)
    if direction == "reverse" and ms_train is None:
        if oof_path.parent.name in MATCH_STRATEGIES:
            ms_train = oof_path.parent.name
    oof = pd.read_csv(oof_path)
    res = evaluate(oof, cohort, direction=direction, ms_train=ms_train, seed=seed, **kw)
    if write and len(res):
        res.to_csv(oof_path.parent / out_name, index=False, encoding="utf-8")
    return res
