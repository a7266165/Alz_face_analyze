"""data/q6ds/*.xlsx → 建模用 CSV(每份 xlsx 一張)+ 對帳報告。

三份原始檔的工作表1 欄位幾乎相同,差別:
  - dementia_20220818 多了 CASI 與 MMSE總分,列數也最多(535)。
  - 另外兩份的 coding 表把連減題寫成「序列-7」,與量表定義(questionaire.jpg
    Table 1:serial 100-3)矛盾;只有 20220818 那份寫「序列-3」。
    對建模無影響(數值就是 0/1 對錯),但對帳報告會把這件事記下來。

q1~q10 對應 questionaire.jpg:
  q1  = D01  是否有憂鬱          0=no, 1=yes
  q2  = M01  是否重複問問題      0=no, 1=sometimes, 2=yes
  q3  = M02  是否記憶力下降      0=no, 1=sometimes, 2=yes
  q4  = TO01 今年是幾年          0=incorrect, 1=correct
  q5  = TO02 現在是幾月份        0=incorrect, 1=correct
  q6~q10 = C01~C05 serial 100-3  0=incorrect, 1=correct
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Q6DS_RAW_DIR, q6ds_path

# dataset_id → 原始檔名。id 取「族群_檔案日期」,不用 A/B/C 這種沒有意義的代號。
DATASETS: Dict[str, str] = {
    "dementia_20220723": "20220722-新篩檢量表for AI 11107 23(dementia).xlsx",
    "very_mild_20220723": "20220722-新篩檢量表for AI 1110723(very mild dementia).xlsx",
    "dementia_20220818": "2024022-新篩檢量表for AI  (dementia) 20220818.xlsx",
}

Q_COLS: List[str] = [f"q{i}" for i in range(1, 11)]
FEATURE_COLS: List[str] = ["age", "sex_M"] + Q_COLS
FEATURE_COLS_NOAGE: List[str] = ["sex_M"] + Q_COLS
LABEL_COL = "Dx"

# 後七題是「答對得 1」,分數要轉成「損傷程度」才與前三題同向(高=差)。
Q_HIGHER_IS_WORSE = ["q1", "q2", "q3"]
Q_HIGHER_IS_BETTER = [f"q{i}" for i in range(4, 11)]

# 原始欄名 → 輸出欄名。ref_ 前綴 = 只供對帳/分層,訓練程式不讀。
_RENAME = {
    "個案編號": "subject_id",
    "年紀": "age",
    "教育程度yr": "ref_edu_years",
    "CASI": "ref_casi",
    "MMSE總分": "ref_mmse",
    "CDR overall": "ref_cdr",
    "AD8總分": "ref_ad8",
    "診斷1": "ref_dx_text",
    "Dx": "Dx",
}

_OUT_ORDER = (
    ["subject_id"] + FEATURE_COLS + [LABEL_COL]
    + ["ref_q6ds_impairment", "ref_edu_years", "ref_cdr", "ref_casi", "ref_mmse",
       "ref_ad8", "ref_dx_text", "ref_source_file"]
)


def raw_path(dataset_id: str):
    if dataset_id not in DATASETS:
        raise ValueError(f"unknown dataset_id: {dataset_id!r} (expected one of {tuple(DATASETS)})")
    p = Q6DS_RAW_DIR / DATASETS[dataset_id]
    if not p.exists():
        raise FileNotFoundError(f"找不到原始 xlsx: {p}")
    return p


def load_raw(dataset_id: str) -> pd.DataFrame:
    """讀第一個工作表,不做任何轉換。"""
    return pd.read_excel(raw_path(dataset_id), sheet_name=0)


def coding_sheet(dataset_id: str) -> Optional[pd.DataFrame]:
    """讀 coding 表(工作表名尾端有空白,故用位置);沒有就回 None。"""
    try:
        return pd.read_excel(raw_path(dataset_id), sheet_name="coding ", header=None)
    except Exception:
        return None


def _as_str(s: pd.Series) -> pd.Series:
    """pandas 3 的 astype(str) 會讓 NaN 維持 float,對帳用 key 需要純字串。"""
    return s.map(lambda x: "" if pd.isna(x) else str(x))


def build_dataset(dataset_id: str) -> pd.DataFrame:
    """原始表 → 建模用 tidy 表(欄位順序固定,缺值保留為 NaN)。"""
    raw = load_raw(dataset_id)
    df = raw.rename(columns=_RENAME)

    # 性別:資料只有 F/M 兩種,仍先去空白再比對,避免尾端空白混進來。
    sex = df["性別"].map(lambda x: "" if pd.isna(x) else str(x).strip().upper())
    if not set(sex.unique()) <= {"F", "M"}:
        raise ValueError(f"{dataset_id}: 性別出現非 F/M 值 {sorted(set(sex.unique()))}")
    df["sex_M"] = (sex == "M").astype("int64")

    for col in ("ref_casi", "ref_mmse"):        # 只有 20220818 那份有
        if col not in df.columns:
            df[col] = np.nan

    df["ref_dx_text"] = df["ref_dx_text"].map(
        lambda x: "" if pd.isna(x) else " ".join(str(x).split()))
    df["ref_source_file"] = DATASETS[dataset_id]

    # 6Q-DS 損傷分:前三題原始值 + 後七題答錯數。純粹由 q1~q10 導出,不含額外資訊,
    # 作為 rule-based 對照組的分數(NaN 由 arm 自己在訓練折上補中位數,這裡照留)。
    df["ref_q6ds_impairment"] = (
        df[Q_HIGHER_IS_WORSE].sum(axis=1, skipna=True)
        + (1 - df[Q_HIGHER_IS_BETTER]).sum(axis=1, skipna=True)
    )

    out = df[_OUT_ORDER].copy()
    if out[LABEL_COL].isna().any():
        raise ValueError(f"{dataset_id}: Dx 有缺值")
    out[LABEL_COL] = out[LABEL_COL].astype("int64")
    return out


def _jsonable(obj):
    """把 numpy 純量與非字串 key 轉成 json 能吃的型別(crosstab().to_dict() 會帶 np.int64 key)。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _feature_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    return df[cols].fillna(-99).map(lambda x: f"{x:g}" if isinstance(x, float) else str(x)) \
                   .agg(lambda r: "|".join(r), axis=1)


def _dup_stats(df: pd.DataFrame, cols: List[str]) -> dict:
    """同特徵向量的重複與標籤矛盾統計 —— 準確率的理論上限來源。"""
    g = pd.DataFrame({"k": _feature_key(df, cols), "y": df[LABEL_COL]}) \
        .groupby("k")["y"].agg(["count", "nunique"])
    return {
        "n_rows": int(len(df)),
        "n_unique_patterns": int(len(g)),
        "n_dup_groups": int((g["count"] > 1).sum()),
        "n_rows_in_dup_groups": int(g.loc[g["count"] > 1, "count"].sum()),
        "n_contradictory_groups": int((g["nunique"] > 1).sum()),
        "n_rows_in_contradictory_groups": int(g.loc[g["nunique"] > 1, "count"].sum()),
    }


def dataset_report(dataset_id: str, df: pd.DataFrame) -> dict:
    """單一 dataset 的體檢報告(缺值 / 重複 / 類別分佈 / 逐題分佈 / 洩漏檢查)。"""
    y = df[LABEL_COL]
    rep = {
        "dataset_id": dataset_id,
        "source_file": DATASETS[dataset_id],
        "n_rows": int(len(df)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "prevalence": round(float((y == 1).mean()), 4),
        "feature_cols": FEATURE_COLS,
        "label_col": LABEL_COL,
        "missing_per_feature": {c: int(df[c].isna().sum()) for c in FEATURE_COLS},
        "duplicates_full_features": _dup_stats(df, FEATURE_COLS),
        "duplicates_without_age": _dup_stats(df, FEATURE_COLS_NOAGE),
        "sex_by_label": pd.crosstab(df["sex_M"], y).to_dict(),
        "age_by_label": df.groupby(LABEL_COL)["age"].describe().round(2).to_dict(),
        "items_by_label": {
            c: pd.crosstab(df[c], y).to_dict() for c in Q_COLS
        },
        "ref_dx_text_by_label": pd.crosstab(df["ref_dx_text"], y).to_dict(),
        "ref_cdr_by_label": pd.crosstab(df["ref_cdr"], y).to_dict(),
    }

    # AD8 是標籤洩漏:它的缺值幾乎完全對應 Dx=0。留在 ref_ 區,這裡明確記一筆。
    ad8_na = df["ref_ad8"].isna()
    rep["ad8_missing_vs_label"] = {
        "note": "AD8 只對失智個案登錄,缺值≈Dx=0;絕不可作為特徵",
        "missing_and_neg": int((ad8_na & (y == 0)).sum()),
        "missing_and_pos": int((ad8_na & (y == 1)).sum()),
        "present_and_neg": int((~ad8_na & (y == 0)).sum()),
        "present_and_pos": int((~ad8_na & (y == 1)).sum()),
    }

    cs = coding_sheet(dataset_id)
    serial = ""
    if cs is not None:
        hits = cs.astype(object).map(lambda x: str(x)).stack()
        serial = next((v for v in hits if v.startswith("序列-")), "")
    rep["coding_serial_label"] = serial
    rep["coding_matches_questionnaire"] = serial.startswith("序列-3")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    rep["dataset_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    return _jsonable(rep)


def reconcile_sources() -> dict:
    """三份 xlsx 的重疊對帳 —— 它們不是獨立資料集,結果不可當作三次獨立驗證。"""
    common = ["age", "sex_M", "ref_cdr"] + Q_COLS
    frames = {k: build_dataset(k) for k in DATASETS}
    keys = {k: set(_feature_key(v, common)) for k, v in frames.items()}
    out = {"key_cols": common, "per_dataset": {}, "pairwise": {}}
    for k, v in frames.items():
        out["per_dataset"][k] = {"n_rows": int(len(v)), "n_unique_keys": len(keys[k])}
    ids = list(DATASETS)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            out["pairwise"][f"{a}|{b}"] = {
                "intersection": len(keys[a] & keys[b]),
                f"only_in_{a}": len(keys[a] - keys[b]),
                f"only_in_{b}": len(keys[b] - keys[a]),
            }
    out["note"] = ("三份檔案高度重疊(very_mild 是 dementia 的 CDR<=1 子集、"
                   "20220723 幾乎完全包含於 20220818),陰性對照組更是同一批人。"
                   "三組指標之間有強相關,不能視為獨立重複驗證。")
    return out


def dataset_csv_path(dataset_id: str):
    return q6ds_path(dataset_id, "dataset") / "q6ds_dataset.csv"


def write_dataset(dataset_id: str) -> dict:
    """產出 CSV + report(json/md),回傳 report dict。"""
    df = build_dataset(dataset_id)
    rep = dataset_report(dataset_id, df)

    out_dir = q6ds_path(dataset_id, "dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_csv_path(dataset_id), index=False, encoding="utf-8-sig")
    (out_dir / "dataset_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "dataset_report.md").write_text(_report_md(rep), encoding="utf-8")
    return rep


def load_dataset(dataset_id: str) -> pd.DataFrame:
    """讀已產出的 CSV;沒有就先建。"""
    p = dataset_csv_path(dataset_id)
    if not p.exists():
        write_dataset(dataset_id)
    return pd.read_csv(p, encoding="utf-8-sig")


def _report_md(rep: dict) -> str:
    d_all = rep["duplicates_full_features"]
    d_noage = rep["duplicates_without_age"]
    ad8 = rep["ad8_missing_vs_label"]
    miss = ", ".join(f"{k}={v}" for k, v in rep["missing_per_feature"].items() if v)
    lines = [
        f"# q6ds dataset report — {rep['dataset_id']}",
        "",
        f"- 原始檔:`{rep['source_file']}`",
        f"- 列數:{rep['n_rows']}(Dx=1 {rep['n_pos']} / Dx=0 {rep['n_neg']},盛行率 {rep['prevalence']:.3f})",
        f"- 特徵({len(rep['feature_cols'])}):{', '.join(rep['feature_cols'])}",
        f"- 特徵缺值:{miss or '無'}",
        f"- coding 表連減題標示:`{rep['coding_serial_label'] or 'N/A'}`"
        f"({'與 questionaire.jpg 的 serial 100-3 一致' if rep['coding_matches_questionnaire'] else '**與 questionaire.jpg 的 serial 100-3 不一致**'})",
        f"- CSV SHA256:`{rep['dataset_sha256']}`",
        "",
        "## 重複與矛盾列(準確率上限來源)",
        "",
        "| 特徵集 | 唯一組合 | 重複組 | 重複組涵蓋列 | 同特徵不同標籤組 | 涵蓋列 |",
        "|---|---|---|---|---|---|",
        f"| 12 特徵(含年齡) | {d_all['n_unique_patterns']} | {d_all['n_dup_groups']} | "
        f"{d_all['n_rows_in_dup_groups']} | {d_all['n_contradictory_groups']} | {d_all['n_rows_in_contradictory_groups']} |",
        f"| 11 特徵(去年齡) | {d_noage['n_unique_patterns']} | {d_noage['n_dup_groups']} | "
        f"{d_noage['n_rows_in_dup_groups']} | {d_noage['n_contradictory_groups']} | {d_noage['n_rows_in_contradictory_groups']} |",
        "",
        "## 洩漏檢查:AD8總分",
        "",
        f"{ad8['note']}。缺值×Dx=0:{ad8['missing_and_neg']}、缺值×Dx=1:{ad8['missing_and_pos']}、"
        f"有值×Dx=0:{ad8['present_and_neg']}、有值×Dx=1:{ad8['present_and_pos']}。",
        "",
        "## 年齡分佈(依標籤)",
        "",
        "```",
        json.dumps(rep["age_by_label"], ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    return "\n".join(lines) + "\n"
