"""Nested repeated stratified CV + 指標。

外層折只用來評估、內層折只用來選超參與選閾值,兩者不共用資料 —— 這是小樣本
(260~535 列)唯一拿得到誠實泛化估計的做法。閾值取自「內層 OOF」而非重擬合後
的訓練集預測,後者過擬合會把閾值系統性推偏。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)

from .dataset import LABEL_COL
from .model import arm_spec

THRESHOLD_KINDS = ("fixed_0.5", "youden_inner")

_RATE_COLS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
              "ppv", "npv", "f1", "mcc"]


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else np.nan


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens, spec = _safe_div(tp, tp + fn), _safe_div(tn, tn + fp)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else np.nan,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }


def ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    single_class = len(np.unique(y_true)) < 2
    return {
        "auroc": np.nan if single_class else float(roc_auc_score(y_true, y_prob)),
        "auprc": np.nan if single_class else float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden J = sensitivity + specificity - 1 的最大點。"""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    return 0.5 if not np.isfinite(t) else t


def fit_select(estimator, param_dist, n_iter, X_tr, y_tr, inner_cv, n_jobs, seed):
    """內層搜尋 → (已在訓練折上重擬合的模型, 選到的超參, 內層最佳 AUROC)。"""
    if not param_dist:
        return clone(estimator).fit(X_tr, y_tr), {}, np.nan
    search = RandomizedSearchCV(
        clone(estimator), param_distributions=param_dist, n_iter=n_iter,
        scoring="roc_auc", cv=inner_cv, n_jobs=n_jobs, random_state=seed,
        refit=True, error_score="raise",
    )
    search.fit(X_tr, y_tr)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


def run_nested_cv(
    df: pd.DataFrame,
    dataset_id: str,
    arm: str,
    *,
    n_splits: int = 5,
    n_repeats: int = 20,
    seed: int = 0,
    n_jobs: int = 16,
    progress: Optional[callable] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """回傳 (fold_metrics, oof_predictions, best_params)。"""
    spec = arm_spec(arm, seed=seed)
    X = df[spec["features"]].copy()
    y = df[LABEL_COL].to_numpy(dtype=int)
    sid = df["subject_id"].to_numpy()

    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows, oof, params = [], [], []

    for i, (tr, te) in enumerate(outer.split(X, y)):
        rep, fold = divmod(i, n_splits)
        fold_seed = seed * 1000 + i
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        est, best, inner_auc = fit_select(
            spec["estimator"], spec["param_dist"], spec["n_iter"],
            X_tr, y_tr, inner_cv, n_jobs, fold_seed)

        # 閾值只能看內層 OOF。用同一組超參 clone 一份,避免動到已重擬合的 est。
        inner_oof = cross_val_predict(
            clone(est), X_tr, y_tr,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed + 1),
            method="predict_proba", n_jobs=n_jobs)[:, 1]
        thr_youden = youden_threshold(y_tr, inner_oof)

        p_te = est.predict_proba(X_te)[:, 1]
        rank = ranking_metrics(y_te, p_te)

        for kind, thr in (("fixed_0.5", 0.5), ("youden_inner", thr_youden)):
            rows.append({
                "dataset": dataset_id, "arm": arm, "repeat": rep, "fold": fold,
                "n_train": len(tr), "n_test": len(te),
                "n_pos_test": int(y_te.sum()), "inner_best_auroc": inner_auc,
                **rank, "thr_kind": kind, "threshold": float(thr),
                **threshold_metrics(y_te, p_te, thr),
            })

        oof.append(pd.DataFrame({
            "dataset": dataset_id, "arm": arm, "repeat": rep, "fold": fold,
            "subject_id": sid[te], "y_true": y_te, "y_prob": p_te,
        }))
        params.append({"dataset": dataset_id, "arm": arm, "repeat": rep, "fold": fold,
                       "inner_best_auroc": inner_auc, **best})

        if progress is not None:
            progress(i + 1, n_splits * n_repeats, rank["auroc"])

    return (pd.DataFrame(rows), pd.concat(oof, ignore_index=True), pd.DataFrame(params))


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """每個 (dataset, arm, thr_kind) 一列:各指標的 mean / sd / 折間 2.5–97.5 百分位。

    刻意不叫 CI —— repeated CV 的折彼此不獨立,常態 CI 會過窄。這裡報的是折間
    分佈的離散程度,不是母體平均的信賴區間。
    """
    metric_cols = ["auroc", "auprc", "brier"] + _RATE_COLS + ["threshold"]
    g = fold_metrics.groupby(["dataset", "arm", "thr_kind"], sort=False)
    out = []
    for keys, sub in g:
        row = dict(zip(["dataset", "arm", "thr_kind"], keys))
        row["n_folds"] = len(sub)
        for m in metric_cols:
            # PPV 在「整折都預測為陰性」時無定義(tp+fp=0),固定 0.5 閾值 + 低盛行率
            # 的組合很容易整折退化,故所有統計都只在非 NaN 的折上算。
            v = sub[m].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            row[f"{m}_n_valid_folds"] = int(len(v))
            row[f"{m}_mean"] = float(np.mean(v)) if len(v) else np.nan
            row[f"{m}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            row[f"{m}_p2.5"] = float(np.percentile(v, 2.5)) if len(v) else np.nan
            row[f"{m}_p97.5"] = float(np.percentile(v, 97.5)) if len(v) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def pooled_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """把每個 repeat 內的 OOF 併回完整樣本(每 repeat 每人恰好一次預測)。"""
    return oof.groupby(["repeat", "subject_id"], as_index=False).agg(
        y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
