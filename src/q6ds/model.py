"""五個評估 arm:兩個 XGBoost、一個 LR、兩個 rule-based 對照。

對照組存在的理由:年齡單獨的 AUROC 就有 0.8 以上,6Q-DS 損傷分也有 0.86;
沒有這兩條基準線,XGBoost 的數字讀不出任何意義。xgb_noage 則回答審稿最常
問的問題 —— 模型學到的是量表還是年齡。

rule-based 對照不是裸分數排序,而是把分數接一個 1-D logistic 做校準:單調轉換
不動 AUROC / AUPRC,但讓 0.5 閾值與 Brier 對所有 arm 同樣可讀。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import (
    FEATURE_COLS,
    FEATURE_COLS_NOAGE,
    Q_COLS,
    Q_HIGHER_IS_BETTER,
    Q_HIGHER_IS_WORSE,
)

ARMS = ("xgb_full", "xgb_noage", "lr_full", "lr_noage",
        "score_6qds", "age_only", "edu_only")

# 部署候選:只吃 q1~q10 + 性別 的 11 特徵集(不放年齡、不放教育)。
DEPLOY_ARMS = ("xgb_noage", "lr_noage")

# rule-based 單變項對照:kind → 資料表欄名。教育年數平時掛在 ref_ 區(不進主模型),
# 這裡當獨立對照組用 —— 它在 score ~ age + Dx + edu 的迴歸裡係數比年齡強 5 倍。
SINGLE_COL = {"age": "age", "edu": "ref_edu_years"}

# 主 arm(會存模型檔、畫 SHAP);其餘是對照組,只留指標。
PRIMARY_ARM = "xgb_full"

XGB_PARAM_DIST: Dict[str, list] = {
    # 535 列 × 12 特徵:樹要淺、正則要強,否則內層搜尋會挑到純記憶的設定。
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [0.5, 1.0, 5.0, 10.0],
    "gamma": [0.0, 0.5, 2.0],
}

LR_PARAM_DIST: Dict[str, list] = {
    "clf__C": list(np.logspace(-3, 2, 12)),
}


class ScoreLogit(ClassifierMixin, BaseEstimator):
    """rule-based 對照:把固定公式算出的單一分數接 1-D logistic。

    kind="q6ds" → 6Q-DS 損傷分(q1+q2+q3 + 後七題答錯數,高=差)
    kind="age"  → 年齡
    kind="edu"  → 教育年數(方向相反,由 logistic 自己吃掉符號)

    缺值以「訓練折」的中位數補,fit 之外不看任何測試折資訊。
    """

    def __init__(self, kind: str = "q6ds"):
        self.kind = kind

    def _cols(self):
        return [SINGLE_COL[self.kind]] if self.kind in SINGLE_COL else Q_COLS

    def _raw_score(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        if self.kind in SINGLE_COL:
            c = SINGLE_COL[self.kind]
            v = X[c].to_numpy(dtype=float)
            return np.where(np.isnan(v), self.medians_[c], v)
        f = X[Q_COLS].astype(float).copy()
        for c in Q_COLS:
            f[c] = f[c].fillna(self.medians_[c])
        return (f[Q_HIGHER_IS_WORSE].sum(axis=1)
                + (1 - f[Q_HIGHER_IS_BETTER]).sum(axis=1)).to_numpy(dtype=float)

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.medians_ = {c: float(np.nanmedian(X[c].astype(float))) for c in self._cols()}
        s = self._raw_score(X).reshape(-1, 1)
        self.classes_ = np.unique(np.asarray(y))
        self.logit_ = LogisticRegression(max_iter=1000).fit(s, y)
        return self

    def predict_proba(self, X):
        return self.logit_.predict_proba(self._raw_score(X).reshape(-1, 1))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def make_xgb(seed: int = 42, n_jobs: int = 1):
    from xgboost import XGBClassifier
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", device="cpu",       # 535×12,GPU 只會被搬運成本吃掉
        random_state=seed, verbosity=0, n_jobs=n_jobs,
    )


def make_lr(seed: int = 42):
    # LR 不吃 NaN 也需要標準化;q2/q3 各只有 1~2 個缺值,中位數補影響可忽略。
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
    ])


def arm_spec(arm: str, *, seed: int = 42, n_jobs: int = 1) -> dict:
    """arm → {features, estimator, param_dist, n_iter, kind, saver}。"""
    if arm == "xgb_full":
        return {"features": list(FEATURE_COLS), "estimator": make_xgb(seed, n_jobs),
                "param_dist": XGB_PARAM_DIST, "n_iter": 40, "kind": "xgb", "saver": "xgb"}
    if arm == "xgb_noage":
        return {"features": list(FEATURE_COLS_NOAGE), "estimator": make_xgb(seed, n_jobs),
                "param_dist": XGB_PARAM_DIST, "n_iter": 40, "kind": "xgb", "saver": "xgb"}
    if arm == "lr_full":
        return {"features": list(FEATURE_COLS), "estimator": make_lr(seed),
                "param_dist": LR_PARAM_DIST, "n_iter": 12, "kind": "lr", "saver": "joblib"}
    if arm == "lr_noage":
        return {"features": list(FEATURE_COLS_NOAGE), "estimator": make_lr(seed),
                "param_dist": LR_PARAM_DIST, "n_iter": 12, "kind": "lr", "saver": "joblib"}
    if arm == "score_6qds":
        # 只吃 q1~q10:這條對照要回答「量表本身能做到多少」,放進年齡就不是對照了。
        return {"features": list(Q_COLS), "estimator": ScoreLogit("q6ds"),
                "param_dist": None, "n_iter": 0, "kind": "rule", "saver": "joblib"}
    if arm == "age_only":
        return {"features": ["age"], "estimator": ScoreLogit("age"),
                "param_dist": None, "n_iter": 0, "kind": "rule", "saver": "joblib"}
    if arm == "edu_only":
        return {"features": ["ref_edu_years"], "estimator": ScoreLogit("edu"),
                "param_dist": None, "n_iter": 0, "kind": "rule", "saver": "joblib"}
    raise ValueError(f"unknown arm: {arm!r} (expected one of {ARMS})")


def arm_features(arm: str) -> List[str]:
    return arm_spec(arm)["features"]
