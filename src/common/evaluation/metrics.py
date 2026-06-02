"""
評估的「純指標 kernel」—— 只吃 (y_true, y_score)，回 dict / 純統計。

零 cohort / matching / 檔案依賴,只用 numpy / scipy / sklearn。matched_chain 與任何
consumer 都靠這支算分。指標定義 lift 自 src/meta/evaluation/matched_eval.py(行為對齊),
但輸出攤平成 CSV-friendly 欄(confusion matrix 拆成 tn/fp/fn/tp,不放巢狀 list)。
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

__all__ = ["bootstrap_auc_ci", "compute_clf_metrics", "paired_wilcoxon"]


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
