"""把 asymmetry 特徵映成不對稱程度的評分器(l2_norm | centroid_dist | lda_projection)。"""
from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

ASYMMETRY_METHODS = ("l2_norm", "centroid_dist", "lda_projection")


class CentroidEstimator(ClassifierMixin, BaseEstimator):
    """以「離 AD 質心 vs 離 HC 質心的 cosine 距離差」當分數的 sklearn 評分器。

    fit 時對每群取平均向量當質心:μ_AD = AD(y=1) 樣本均值、μ_HC = HC(y=0) 樣本均值。
    分數 score(x) = cosdist(x, μ_HC) − cosdist(x, μ_AD);越大代表 x 在 cosine 空間裡
    越靠近 AD 質心、遠離 HC 質心,越可能是 AD。predict 以 0 為門檻。
    """

    def fit(self, X, y):
        """估出 AD / HC 兩群的質心。

        Args:
            X: 特徵矩陣 (n_samples, n_features)。
            y: 標籤,1=AD、0=HC。

        Returns:
            self(已存好 c_ad_、c_hc_、classes_)。
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.c_ad_ = X[y == 1].mean(axis=0)
        self.c_hc_ = X[y == 0].mean(axis=0)
        return self

    def decision_function(self, X):
        """回傳每筆樣本的不對稱分數(越大越像 AD)。

        Args:
            X: 特徵矩陣 (n_samples, n_features)。

        Returns:
            分數陣列 (n_samples,)。
        """
        X = np.asarray(X, dtype=np.float64)
        # 因 cosdist = 1 − cossim,兩項相減後常數 1 對消,故
        #   score = cosdist(x,μ_HC) − cosdist(x,μ_AD) = cossim(x,μ_AD) − cossim(x,μ_HC)。
        # 把 x 與兩質心都先做 L2 normalize,cossim 即化為內積,可一次矩陣乘算完全部樣本。
        eps = 1e-12
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
        ad_n = self.c_ad_ / (np.linalg.norm(self.c_ad_) + eps)
        hc_n = self.c_hc_ / (np.linalg.norm(self.c_hc_) + eps)
        return (Xn @ ad_n) - (Xn @ hc_n)

    def predict(self, X):
        """分數 >= 0 判為 AD(1),否則 HC(0)。

        Args:
            X: 特徵矩陣 (n_samples, n_features)。

        Returns:
            預測標籤陣列 (n_samples,),值為 0 或 1。
        """
        return (self.decision_function(X) >= 0).astype(int)


def build_scorer(method: str) -> Tuple[Optional[object], Optional[str], bool]:
    """依 method 建立不對稱評分器。

    Args:
        method: l2_norm | centroid_dist | lda_projection

    Returns:
        (estimator, score_method, needs_cv)。l2_norm 無 estimator、不需 CV
        (引擎直接算 norm);centroid_dist / lda_projection 會 fit、需進 CV。
    """
    if method == "l2_norm":
        return None, None, False
    if method == "centroid_dist":
        return CentroidEstimator(), "decision_function", True
    if method == "lda_projection":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(n_components=1), "decision_function", True

    raise ValueError(f"unknown scorer method: {method!r} (expected one of {ASYMMETRY_METHODS})")
