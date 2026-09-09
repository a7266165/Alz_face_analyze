"""把 lr_noage 匯出成零依賴的部署包。

為什麼是線性模型而不是 XGBoost:逐折配對比較下兩者無法區分(Nadeau-Bengio
修正 t 檢定 p=0.80~0.96),但 LR 可以攤平成「11 個權重 + 截距」,收案端用純
Python 就能算完;XGBoost 得帶整個 runtime。AlzheimerResearch6Q.spec 的
excludes 明確排除 sklearn,lite 版 exe 裡本來就沒有任何 ML 套件。

Pipeline 是 SimpleImputer(median) → StandardScaler → LogisticRegression,
三段都是仿射轉換,可以摺成原始特徵空間的單一線性式:
    z_j     = (x_j - mean_j) / scale_j
    logit   = b + Σ w_j·z_j
            = (b - Σ w_j·mean_j/scale_j) + Σ (w_j/scale_j)·x_j
故 raw_intercept = b - Σ w_j·mean_j/scale_j、raw_weight_j = w_j/scale_j。
匯出後會逐列比對 sklearn 的 predict_proba,不合就直接報錯。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import Q6DS_DIR

from .dataset import LABEL_COL, load_dataset
from .model import arm_spec
from .train import load_model

# app 端中文標籤 → 整數。刻意用明確字典而不是選項索引:
# personal_panel.py 的 radio 顯示順序是 ["是","有時","否"](是在前),
# 用 options.index() 會得到完全相反的編碼。
LABEL_MAP: Dict[str, Dict[str, int]] = {
    "性別": {"男": 1, "女": 0},
    "D01": {"否": 0, "是": 1},
    "M01": {"否": 0, "有時": 1, "是": 2},
    "M02": {"否": 0, "有時": 1, "是": 2},
    "TO01": {"錯誤": 0, "正確": 1},
    "TO02": {"錯誤": 0, "正確": 1},
    "C01": {"錯誤": 0, "正確": 1},
    "C02": {"錯誤": 0, "正確": 1},
    "C03": {"錯誤": 0, "正確": 1},
    "C04": {"錯誤": 0, "正確": 1},
    "C05": {"錯誤": 0, "正確": 1},
}

# 模型特徵順序 → app 端欄位名(順序必須與 FEATURE_COLS_NOAGE 完全一致)
FEATURE_SOURCE: List[str] = [
    "性別", "D01", "M01", "M02", "TO01", "TO02", "C01", "C02", "C03", "C04", "C05",
]


def deploy_dir():
    return Q6DS_DIR / "deploy"


def flatten_lr(pipe) -> tuple:
    """Pipeline(imputer→scaler→LR) → (raw_weights, raw_intercept, medians)。"""
    imp, sc, clf = pipe.named_steps["imp"], pipe.named_steps["sc"], pipe.named_steps["clf"]
    w = clf.coef_.ravel().astype(float)
    b = float(clf.intercept_[0])
    mean = sc.mean_.astype(float)
    scale = sc.scale_.astype(float)
    raw_w = w / scale
    raw_b = b - float(np.sum(w * mean / scale))
    return raw_w, raw_b, imp.statistics_.astype(float)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def export_deploy_bundle(dataset_id: str = "dementia_20220818",
                         arm: str = "lr_noage") -> dict:
    """匯出部署 JSON + 零依賴參考實作 + README,並驗證與 sklearn 逐列一致。"""
    spec = arm_spec(arm)
    if spec["kind"] != "lr":
        raise ValueError(f"export 只支援線性 arm,收到 {arm!r}(kind={spec['kind']})")
    feats = spec["features"]
    if feats != FEATURE_SOURCE_ORDER():
        raise ValueError(f"特徵順序不符:模型是 {feats},對照表是 {FEATURE_SOURCE_ORDER()}")

    pipe = load_model(dataset_id, arm)
    raw_w, raw_b, medians = flatten_lr(pipe)

    df = load_dataset(dataset_id)
    X = df[feats]
    p_sklearn = pipe.predict_proba(X)[:, 1]
    p_flat = _sigmoid(X.fillna(pd.Series(medians, index=feats)).to_numpy(float) @ raw_w + raw_b)
    max_err = float(np.max(np.abs(p_sklearn - p_flat)))
    if max_err > 1e-9:
        raise AssertionError(f"攤平後與 sklearn 不一致,max|Δp|={max_err:.3e}")

    card = json.loads((Q6DS_DIR / dataset_id / "model" / arm / "model_card.json")
                      .read_text(encoding="utf-8"))
    bundle = {
        "name": f"q6ds_{arm}_{dataset_id}",
        "task": "6Q-DS 問卷 → 失智症機率(不使用年齡與教育)",
        "model": "logistic regression(已攤平為原始特徵空間的線性式)",
        "feature_order": feats,
        "feature_source_columns": FEATURE_SOURCE,
        "label_map": LABEL_MAP,
        "weights": {f: float(w) for f, w in zip(feats, raw_w)},
        "intercept": float(raw_b),
        "missing_fill": {f: float(m) for f, m in zip(feats, medians)},
        "formula": "p = 1 / (1 + exp(-(intercept + sum(weights[f] * x[f]))))",
        "threshold": float(card["recommended_threshold"]),
        "threshold_note": card["threshold_note"],
        "training": {
            "dataset_id": dataset_id,
            "source_file": df["ref_source_file"].iloc[0],
            "n_rows": int(len(df)),
            "n_pos": int((df[LABEL_COL] == 1).sum()),
            "n_neg": int((df[LABEL_COL] == 0).sum()),
            "cv": "nested 5-fold x 20 repeats(內層選超參與閾值,外層評估)",
            "cv_auroc_mean": None,   # 由呼叫端填(讀 summary)
            "best_params": card["best_params"],
        },
        "verification": {
            "max_abs_prob_diff_vs_sklearn": max_err,
            "n_rows_checked": int(len(df)),
        },
        "versions": card["versions"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    s = pd.read_csv(Q6DS_DIR / dataset_id / "cv" / arm / "summary.csv", encoding="utf-8-sig")
    row = s[s["thr_kind"] == "youden_inner"].iloc[0]
    bundle["training"]["cv_auroc_mean"] = float(row["auroc_mean"])
    bundle["training"]["cv_auroc_sd"] = float(row["auroc_sd"])
    bundle["training"]["cv_sensitivity_mean"] = float(row["sensitivity_mean"])
    bundle["training"]["cv_specificity_mean"] = float(row["specificity_mean"])

    d = deploy_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{bundle['name']}.json"
    out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "predict_q6ds.py").write_text(_reference_impl(out.name), encoding="utf-8")
    (d / "README.md").write_text(_readme(bundle, out.name), encoding="utf-8")
    return bundle


def export_joblib_bundle(dataset_id: str = "dementia_20220818",
                         arm: str = "lr_noage") -> dict:
    """交付 sklearn pipeline 本體:model.joblib + 特徵清單 + 訓練環境。

    joblib 是 pickle,跨版本沒有相容性保證 —— 因此環境檔不是附錄而是契約的一部分,
    載入端的 scikit-learn / numpy / scipy 版本必須對得上,否則輕則 warning、
    重則靜默給出錯的數值。
    """
    import shutil
    import subprocess
    import sys

    spec = arm_spec(arm)
    feats = spec["features"]
    pipe = load_model(dataset_id, arm)

    card = json.loads((Q6DS_DIR / dataset_id / "model" / arm / "model_card.json")
                      .read_text(encoding="utf-8"))
    s = pd.read_csv(Q6DS_DIR / dataset_id / "cv" / arm / "summary.csv", encoding="utf-8-sig")
    row = s[s["thr_kind"] == "youden_inner"].iloc[0]

    d = deploy_dir()
    (d / "environment").mkdir(parents=True, exist_ok=True)
    name = f"q6ds_{arm}_{dataset_id}"

    # 訓練時餵的是 DataFrame,pipeline 因此記住了 feature_names_in_;部署端若傳
    # numpy array,sklearn 每次呼叫都會噴 UserWarning(數值正確,但 GUI log 會被洗)。
    # 拿掉這個屬性,numpy array 就成為乾淨的正式輸入,推論端也不必裝 pandas。
    df = load_dataset(dataset_id)
    p_before = pipe.predict_proba(df[feats])[:, 1]
    _strip_feature_names(pipe)
    p = pipe.predict_proba(df[feats].to_numpy(float))[:, 1]
    if not np.array_equal(p_before, p):
        raise AssertionError("移除 feature_names_in_ 後預測值改變,不應發生")
    import joblib as _joblib
    _joblib.dump(pipe, d / f"{name}.joblib")

    features = {
        "model_file": f"{name}.joblib",
        "estimator": "sklearn.pipeline.Pipeline(SimpleImputer(median) → StandardScaler → LogisticRegression)",
        "n_features": len(feats),
        "feature_order": feats,
        "input": ("shape (n, 11) 的 2D numpy array / list of lists,欄位順序即 feature_order。"
                  "匯出時已移除 feature_names_in_,所以不需要 pandas、也不會有欄名 warning。"),
        "features": [
            {"name": "sex_M", "source_column": "性別",
             "values": {"男": 1, "女": 0}, "dtype": "int"},
            {"name": "q1", "source_column": "D01", "item": "D01 是否感覺憂鬱",
             "values": {"否": 0, "是": 1}, "dtype": "int"},
            {"name": "q2", "source_column": "M01", "item": "M01 是否重複回憶相同事情",
             "values": {"否": 0, "有時": 1, "是": 2}, "dtype": "int"},
            {"name": "q3", "source_column": "M02", "item": "M02 是否覺得記憶或思考有問題",
             "values": {"否": 0, "有時": 1, "是": 2}, "dtype": "int"},
            {"name": "q4", "source_column": "TO01", "item": "TO01 現在是幾年",
             "values": {"錯誤": 0, "正確": 1}, "dtype": "int"},
            {"name": "q5", "source_column": "TO02", "item": "TO02 現在是幾月",
             "values": {"錯誤": 0, "正確": 1}, "dtype": "int"},
            *[{"name": f"q{i + 5}", "source_column": f"C0{i}",
               "item": f"C0{i} 100 連續減 3 第 {i} 次",
               "values": {"錯誤": 0, "正確": 1}, "dtype": "int"}
              for i in range(1, 6)],
        ],
        "warning": ("personal_panel.py 的 radio 顯示順序是 ['是','有時','否'],"
                    "用 options.index() 取值會得到與訓練完全相反的編碼。"
                    "務必以上表的明確對照轉換。"),
        "missing_policy": {
            "handled_by": "pipeline 內的 SimpleImputer(strategy='median')",
            "train_medians": {f: float(m) for f, m in
                              zip(feats, pipe.named_steps["imp"].statistics_)},
        },
        "output": "predict_proba(X)[:, 1] = 失智機率 (0~1)",
        "threshold": float(card["recommended_threshold"]),
        "threshold_note": card["threshold_note"],
        "sanity_check": {
            "n_rows": int(len(df)),
            "prob_min": float(p.min()), "prob_max": float(p.max()),
            "prob_mean": float(p.mean()),
            "all_correct_healthy_male": float(pipe.predict_proba(
                np.array([[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=float))[0, 1]),
            "all_wrong_female": float(pipe.predict_proba(
                np.array([[0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0]], dtype=float))[0, 1]),
        },
        "training": {
            "dataset_id": dataset_id,
            "source_file": df["ref_source_file"].iloc[0],
            "n_rows": int(len(df)),
            "n_pos": int((df[LABEL_COL] == 1).sum()),
            "n_neg": int((df[LABEL_COL] == 0).sum()),
            "cv": "nested 5-fold × 20 repeats",
            "cv_auroc_mean": float(row["auroc_mean"]),
            "cv_auroc_sd": float(row["auroc_sd"]),
            "cv_sensitivity_mean": float(row["sensitivity_mean"]),
            "cv_specificity_mean": float(row["specificity_mean"]),
            "best_params": card["best_params"],
            "seed": card["seed"],
        },
        "runtime_versions": card["versions"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (d / "features.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 環境:載入契約(最小集)+ 訓練當下的完整 freeze ---
    v = card["versions"]
    (d / "environment" / "requirements-minimal.txt").write_text(
        "# 載入本 joblib 所需的最小集合。joblib=pickle,版本不合會 warning 或靜默失準,\n"
        "# 這三個套件請釘死版本(pandas 只在用 DataFrame 餵入時需要)。\n"
        f"# python {v['python']}\n"
        f"numpy=={v['numpy']}\n"
        f"scipy=={_pkg_version('scipy')}\n"
        f"scikit-learn=={v['scikit-learn']}\n"
        f"joblib=={_pkg_version('joblib')}\n",
        encoding="utf-8")
    (d / "environment" / "environment.yml").write_text(
        "# conda env create -f environment.yml\n"
        "name: q6ds_infer\n"
        "channels: [conda-forge]\n"
        "dependencies:\n"
        f"  - python={'.'.join(v['python'].split('.')[:2])}\n"
        "  - pip\n"
        "  - pip:\n"
        f"      - numpy=={v['numpy']}\n"
        f"      - scipy=={_pkg_version('scipy')}\n"
        f"      - scikit-learn=={v['scikit-learn']}\n"
        f"      - joblib=={_pkg_version('joblib')}\n",
        encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, encoding="utf-8")
    (d / "environment" / "pip-freeze-full.txt").write_text(
        f"# 訓練當下 conda env `Alz_face_main_analysis` 的完整 freeze\n"
        f"# python {v['python']} — 僅供重現訓練;推論只需要 requirements-minimal.txt\n"
        + (freeze.stdout or ""), encoding="utf-8")
    return features


def _strip_feature_names(pipe) -> None:
    """就地移除各 step 的 feature_names_in_(只影響欄名檢查,不動數值)。

    只動 step:Pipeline.feature_names_in_ 是唯讀 property,轉發自第一個 step,
    第一個 step 清掉後它自然跟著消失。
    """
    for _, step in pipe.steps:
        if hasattr(step, "feature_names_in_"):
            del step.feature_names_in_


def _pkg_version(name: str) -> str:
    from importlib.metadata import version
    return version(name)


def FEATURE_SOURCE_ORDER() -> List[str]:
    """FEATURE_SOURCE 對應到的模型特徵名(sex_M, q1..q10)。"""
    return ["sex_M"] + [f"q{i}" for i in range(1, 11)]


def _reference_impl(json_name: str) -> str:
    return f'''"""6Q-DS 失智風險評分 —— 零依賴參考實作(只用 Python 標準庫)。

    from predict_q6ds import load_model, predict
    m = load_model("{json_name}")
    p = predict(m, {{"性別": "男", "D01": "否", "M01": "有時", "M02": "是",
                    "TO01": "正確", "TO02": "錯誤", "C01": "正確", "C02": "正確",
                    "C03": "錯誤", "C04": "正確", "C05": "正確"}})

回傳 0~1 的失智機率;p >= model["threshold"] 視為篩檢陽性。

注意:answers 的值是 UI 上的中文標籤,不要用選項索引轉換 —— personal_panel.py
的顯示順序是 ["是","有時","否"],用 index() 會得到完全相反的編碼。
"""
import json
import math


def load_model(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def encode(model, answers):
    """{{中文欄位: 中文標籤}} → 依 feature_order 排好的整數 list。"""
    label_map = model["label_map"]
    out = []
    for src, feat in zip(model["feature_source_columns"], model["feature_order"]):
        raw = answers.get(src)
        if raw is None or raw == "":
            out.append(model["missing_fill"][feat])   # 未作答 → 訓練集中位數
            continue
        raw = str(raw).strip()
        if raw not in label_map[src]:
            raise ValueError(
                f"{{src}} 的值 {{raw!r}} 不在對照表 {{sorted(label_map[src])}} 內")
        out.append(label_map[src][raw])
    return out


def predict(model, answers):
    """回傳失智機率 (0~1)。"""
    x = encode(model, answers)
    z = model["intercept"]
    for feat, v in zip(model["feature_order"], x):
        z += model["weights"][feat] * v
    return 1.0 / (1.0 + math.exp(-z))


def predict_label(model, answers):
    """回傳 (機率, 是否篩檢陽性)。"""
    p = predict(model, answers)
    return p, p >= model["threshold"]
'''


def _readme(b: dict, json_name: str) -> str:
    t = b["training"]
    rows = "\n".join(
        f"| {src} | `{feat}` | {' / '.join(f'{k}→{v}' for k, v in b['label_map'][src].items())} "
        f"| {b['weights'][feat]:+.4f} |"
        for src, feat in zip(b["feature_source_columns"], b["feature_order"]))
    return f"""# 6Q-DS 部署模型

- 檔案:`{json_name}`(權重 + 對照表 + 閾值)、`predict_q6ds.py`(零依賴參考實作)
- 模型:logistic regression,已攤平成原始特徵空間的線性式 —— 收案端**不需要
  sklearn / xgboost / numpy**,純標準庫即可推論。
- 特徵:11 個(性別 + q1~q10),**不含年齡、不含教育**。
- 訓練資料:`{t['source_file']}`({t['n_rows']} 列,Dx=1 {t['n_pos']} / Dx=0 {t['n_neg']})
- 泛化估計({t['cv']}):
  AUROC **{t['cv_auroc_mean']:.4f} ± {t['cv_auroc_sd']:.4f}**、
  sensitivity {t['cv_sensitivity_mean']:.4f}、specificity {t['cv_specificity_mean']:.4f}
- 建議閾值:**{b['threshold']:.4f}**({b['threshold_note']})
- 攤平驗證:{b['verification']['n_rows_checked']} 列逐列比對 sklearn,
  max|Δp| = {b['verification']['max_abs_prob_diff_vs_sklearn']:.2e}

## 特徵契約

app 端只要把中文標籤照下表轉成整數,依 `feature_order` 排好即可。

| CSV 欄位 | 模型特徵 | 標籤→整數 | 權重 |
|---|---|---|---|
{rows}

截距 `{b['intercept']:+.4f}`,`{b['formula']}`。

⚠️ **不要用選項索引轉換。** `personal_panel.py` 的 radio 顯示順序是
`["是","有時","否"]`(是在前),`options.index(answer)` 會得到 是=0 / 否=2,
與訓練編碼完全相反。一定要走 `label_map` 這張明確對照表。

## 使用

```python
from predict_q6ds import load_model, predict_label

m = load_model("{json_name}")
p, positive = predict_label(m, {{
    "性別": "男", "D01": "否", "M01": "否", "M02": "否",
    "TO01": "正確", "TO02": "正確",
    "C01": "正確", "C02": "正確", "C03": "正確", "C04": "正確", "C05": "正確",
}})
```

未作答的題目會以訓練集中位數填補;若要求全部作答,呼叫前自行擋掉即可
(`PersonalViewModel.unanswered_qds()` 已有這個檢查)。
"""
