"""
下游 Stage-1 的「不對稱分數」分支 —— l2 / centroid / lda。

這支對應概念模型裡的 **scorer**:把(asymmetry)特徵映成一個不對稱程度的分數。注意三者
機制不同(見 build_scorer):
  - centroid / lda **會 fit**(從 labeled train fold 學質心 / 判別方向)→ 是 estimator、
    需進 10-fold,行為跟 trainer 一樣(否則用到 test 樣本算 → leakage)。
  - l2_norm **不 fit**:``score = ‖x‖``,純函數、不看 label、不需 CV → 不進迴圈
    (引擎見 needs_cv=False 直接在全體算 norm)。

build_scorer 與 trainer.py 的 trainer() 同協定,回傳 ``(estimator, score_method, needs_cv)``,
讓兩支能丟進 eval 裡同一個迴圈。scorer 固定 no_drop(不接 reducer,對齊 legacy)。
"""
from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

ASYMMETRY_METHODS = ("l2_norm", "centroid_dist", "lda_projection")


class CentroidEstimator(ClassifierMixin, BaseEstimator):
    """非對稱質心距離評分:score(x) = cosdist(x, μ_HC) − cosdist(x, μ_AD)(越大越像 AD)。

    在 train fold 上 fit 兩個 class 質心,對 test 用 decision_function 算分數。
    做成 ClassifierMixin 子類並設 classes_,以滿足 cross_val_predict(method='decision_function')。
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.c_ad_ = X[y == 1].mean(axis=0)
        self.c_hc_ = X[y == 0].mean(axis=0)
        return self

    def decision_function(self, X):
        from scipy.spatial.distance import cosine
        X = np.asarray(X, dtype=np.float64)
        return np.array([cosine(x, self.c_hc_) - cosine(x, self.c_ad_) for x in X])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def build_scorer(method: str) -> Tuple[Optional[object], Optional[str], bool]:
    """回傳 (estimator, score_method, needs_cv)。method ∈ ASYMMETRY_METHODS。

    l2_norm 無 estimator、不需 CV(引擎直接算 norm);centroid / lda 會 fit、進 CV。
    """
    if method == "l2_norm":
        return None, None, False
    if method == "centroid_dist":
        return CentroidEstimator(), "decision_function", True
    if method == "lda_projection":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(n_components=1), "decision_function", True

    raise ValueError(f"unknown scorer method: {method!r} (expected one of {ASYMMETRY_METHODS})")
