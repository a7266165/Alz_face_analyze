"""每份資料 × 每個 arm:跑 nested CV 留指標,再用全量資料重擬合並存模型。

nested CV 的數字是「這套流程」的泛化估計;最終模型是同一套流程吃完全部資料
的產物。兩者的超參可以不同,這是設計而非 bug —— 若硬要讓最終模型沿用某個
外層折的超參,反而是拿部分資料的選擇去代表全體。
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import Q6DS_SUMMARY_FILE, q6ds_path

from .dataset import LABEL_COL, dataset_csv_path, load_dataset
from .evaluate import (
    fit_select,
    ranking_metrics,
    run_nested_cv,
    summarize_folds,
    threshold_metrics,
    youden_threshold,
)
from .model import ARMS, arm_spec


def _versions() -> dict:
    import sklearn
    import xgboost
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def fit_final_model(
    df: pd.DataFrame,
    dataset_id: str,
    arm: str,
    *,
    n_splits: int = 5,
    seed: int = 0,
    n_jobs: int = 16,
) -> dict:
    """全量資料選超參 → 重擬合 → 存模型 + model_card,回傳 model_card。"""
    spec = arm_spec(arm, seed=seed)
    X, y = df[spec["features"]].copy(), df[LABEL_COL].to_numpy(dtype=int)
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    est, best, inner_auc = fit_select(
        spec["estimator"], spec["param_dist"], spec["n_iter"],
        X, y, inner_cv, n_jobs, seed)

    # 部署閾值同樣取自 OOF(不是訓練集預測),與 nested CV 的取法一致。
    oof = cross_val_predict(
        clone(est), X, y,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1),
        method="predict_proba", n_jobs=n_jobs)[:, 1]
    thr = youden_threshold(y, oof)

    out_dir = q6ds_path(dataset_id, "model", arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    if spec["saver"] == "xgb":
        model_file = out_dir / "model.json"
        est.save_model(model_file)
    else:
        import joblib
        model_file = out_dir / "model.joblib"
        joblib.dump(est, model_file)

    card = {
        "dataset_id": dataset_id,
        "arm": arm,
        "estimator": type(est).__name__,
        "feature_cols": spec["features"],
        "label_col": LABEL_COL,
        "n_rows": int(len(df)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "best_params": best,
        "inner_search_best_auroc": inner_auc,
        "oof_on_full_data": {
            **ranking_metrics(y, oof),
            "youden_threshold": float(thr),
            **threshold_metrics(y, oof, thr),
        },
        "recommended_threshold": float(thr),
        "threshold_note": ("取自全量資料 5-fold OOF 的 Youden J;此值是選在同一批"
                           "資料上的,實際部署前應以獨立資料重新校準。"),
        "model_file": model_file.name,
        "source_csv": str(dataset_csv_path(dataset_id)),
        "seed": seed,
        "versions": _versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return card


def train_dataset(
    dataset_id: str,
    *,
    arms: Optional[Iterable[str]] = None,
    n_splits: int = 5,
    n_repeats: int = 20,
    seed: int = 0,
    n_jobs: int = 16,
    log: Optional[callable] = None,
) -> pd.DataFrame:
    """一份資料的完整流程,回傳該資料所有 arm 的 summary。"""
    say = log or (lambda m: None)
    df = load_dataset(dataset_id)
    arms = tuple(arms) if arms else ARMS

    for arm in arms:
        say(f"[{dataset_id}] arm={arm} nested CV "
            f"({n_splits}-fold x {n_repeats} repeats = {n_splits * n_repeats} outer folds)")
        folds, oof, params = run_nested_cv(
            df, dataset_id, arm, n_splits=n_splits, n_repeats=n_repeats,
            seed=seed, n_jobs=n_jobs)

        cv_dir = q6ds_path(dataset_id, "cv", arm)
        cv_dir.mkdir(parents=True, exist_ok=True)
        folds.to_csv(cv_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        oof.to_csv(cv_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
        params.to_csv(cv_dir / "best_params.csv", index=False, encoding="utf-8-sig")

        s = summarize_folds(folds)
        s.to_csv(cv_dir / "summary.csv", index=False, encoding="utf-8-sig")
        auroc = s.loc[s["thr_kind"] == "youden_inner", "auroc_mean"].iloc[0]
        auroc_sd = s.loc[s["thr_kind"] == "youden_inner", "auroc_sd"].iloc[0]
        bacc = s.loc[s["thr_kind"] == "youden_inner", "balanced_accuracy_mean"].iloc[0]
        say(f"[{dataset_id}] arm={arm} → AUROC {auroc:.4f} ± {auroc_sd:.4f}, "
            f"balanced acc(youden) {bacc:.4f}")

        card = fit_final_model(df, dataset_id, arm, n_splits=n_splits, seed=seed, n_jobs=n_jobs)
        say(f"[{dataset_id}] arm={arm} 最終模型已存 → {card['model_file']} "
            f"(params={card['best_params'] or 'n/a'})")

    # 同 write_global_summary:由磁碟重建,補跑單一 arm 不會洗掉其他 arm。
    folds_all, _ = load_cv_artifacts(dataset_id)
    summary = summarize_folds(folds_all)
    ds_dir = q6ds_path(dataset_id, "cv")
    ds_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ds_dir / "summary_all_arms.csv", index=False, encoding="utf-8-sig")
    return summary


def load_cv_artifacts(dataset_id: str, arms: Optional[Iterable[str]] = None):
    """讀回已存的 fold_metrics / oof_predictions —— 重畫圖不必重訓。"""
    folds, oofs = [], []
    for arm in (tuple(arms) if arms else ARMS):
        d = q6ds_path(dataset_id, "cv", arm)
        if not (d / "fold_metrics.csv").exists():
            continue
        folds.append(pd.read_csv(d / "fold_metrics.csv", encoding="utf-8-sig"))
        oofs.append(pd.read_csv(d / "oof_predictions.csv", encoding="utf-8-sig"))
    if not folds:
        raise FileNotFoundError(f"{dataset_id}: 找不到任何 cv 產出,請先跑 train")
    return pd.concat(folds, ignore_index=True), pd.concat(oofs, ignore_index=True)


def load_model(dataset_id: str, arm: str):
    """讀回最終模型(xgb 走原生 JSON,其餘走 joblib)。"""
    spec = arm_spec(arm)
    d = q6ds_path(dataset_id, "model", arm)
    if spec["saver"] == "xgb":
        from .model import make_xgb
        est = make_xgb()
        est.load_model(d / "model.json")
        return est
    import joblib
    return joblib.load(d / "model.joblib")


def write_global_summary() -> pd.DataFrame:
    """由磁碟上「所有」已存在的 cv 產出重建總表。

    刻意不吃本次執行的結果:單獨補跑一個 arm 時,總表必須保留其他 arm,
    否則 all_metrics.csv 會被洗成只剩這次跑的那一格。
    """
    from .dataset import DATASETS
    folds = []
    for ds in DATASETS:
        try:
            f, _ = load_cv_artifacts(ds)
        except FileNotFoundError:
            continue
        folds.append(f)
    if not folds:
        raise FileNotFoundError("workspace 裡沒有任何 cv 產出")
    out = summarize_folds(pd.concat(folds, ignore_index=True))
    Q6DS_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(Q6DS_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    return out
