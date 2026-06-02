"""
下游 Stage-1 的「訓練」分支 —— 降維 + 分類器(logistic / xgb)。

這支對應概念模型裡的 **trainer**:從 labeled train fold「學」一個分類器、對 test
吐機率分數。降維(no_drop / pca / drop_corr)是分類 Pipeline 的「一步」,**只有
trainer 用得到**,故 reducer factory 住這裡(不與 scorer 並排)。reducer 與 clf 必在
同一次 ``fit(X_train)`` 內 → leakage 結構上不可能。

``build_trainer(classifier_type, ...)`` 回傳 ``(estimator, score_method, needs_cv)``,與
scorer.build_scorer 同協定,讓兩者能丟進 eval 裡同一個 10-fold 迴圈。
"""
from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

DEFAULT_XGB_PARAMS = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}

CLASSIFY_METHODS = ("logistic", "xgb")


# ----------------------------------------------------------------------------
# Reducers(降維)—— transformer,分類 Pipeline 的一步。做成 BaseEstimator 子類,
# cross_val_predict 每折才能 clone。
# ----------------------------------------------------------------------------
class DropCorrReducer(BaseEstimator, TransformerMixin):
    """DropCorrelatedFeatures(feature_engine)的 ndarray 包裝。

    包這層純粹是 ndarray↔DataFrame 轉接頭:feature_engine 吃/吐 DataFrame,而本 pipeline
    走 ndarray。丟相關特徵的演算法 100% 是 feature_engine,不自刻(lift 自 legacy 的
    drop_corr 分支,保 behavior parity)。
    """

    def __init__(self, threshold=0.9, method="pearson"):
        self.threshold = threshold
        self.method = method

    def fit(self, X, y=None):
        import pandas as pd
        from feature_engine.selection import DropCorrelatedFeatures
        self.sel_ = DropCorrelatedFeatures(threshold=self.threshold, method=self.method)
        self.sel_.fit(pd.DataFrame(np.asarray(X)))
        return self

    def transform(self, X):
        import pandas as pd
        return self.sel_.transform(pd.DataFrame(np.asarray(X))).to_numpy()


class TorchGPUPCA(BaseEstimator, TransformerMixin):
    """GPU PCA(torch.pca_lowrank),固定 n_components。lift 自 legacy _TorchGPUPCA。

    fit 內 seed torch RNG —— randomized SVD 否則跨 subprocess 不可重現(會讓 y_score /
    AUC 漂移)。做成 BaseEstimator 子類,n_components/niter 為 __init__ 參數(無尾底線),
    故 sklearn.clone 可正確複製(cross_val_predict 每折會 clone)。
    """

    def __init__(self, n_components, niter=4):
        self.n_components = n_components
        self.niter = niter

    def fit(self, X, y=None):
        import torch
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
        self.mean_ = Xt.mean(0, keepdim=True)
        q = min(int(self.n_components), Xt.shape[0], Xt.shape[1])
        _, _, V = torch.pca_lowrank(Xt - self.mean_, q=q, niter=self.niter)
        self.V_ = V[:, :q].contiguous()
        self.n_components_ = q
        return self

    def transform(self, X):
        import torch
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
        return ((Xt - self.mean_) @ self.V_).cpu().numpy()


def _build_reducer(reducer_type="no_drop", *, pca_components=None,
                   drop_corr_threshold=None, drop_corr_method="pearson"):
    """(內部)回傳 sklearn transformer 或 sentinel 'passthrough'(no_drop)。只有 build_trainer 用。

    pca_components 的兩種語意分派到兩個後端:
      float ∈ (0,1) → sklearn PCA(保留解釋該比例變異的維度;torch 無法表達此模式)
      int ≥ 1       → GPU PCA(固定維度,對齊 legacy)
    """
    if reducer_type == "no_drop":
        return "passthrough"
    if reducer_type == "drop_corr":
        if drop_corr_threshold is None:
            raise ValueError("drop_corr 需要 drop_corr_threshold")
        return DropCorrReducer(threshold=drop_corr_threshold, method=drop_corr_method)
    if reducer_type == "pca":
        if pca_components is None:
            raise ValueError("pca 需要 pca_components")
        if isinstance(pca_components, float) and 0 < pca_components < 1:
            from sklearn.decomposition import PCA
            return PCA(n_components=pca_components, random_state=0)
        return TorchGPUPCA(n_components=int(pca_components))
    raise ValueError(f"unknown reducer_type: {reducer_type!r}")


def reducer_label(reducer_type="no_drop", *, pca_components=None,
                  drop_corr_threshold=None, drop_corr_method="pearson") -> str:
    """輸出路徑用的 reducer 標籤(可含 '/' → 變多層子目錄;對齊 legacy)。"""
    if reducer_type == "no_drop":
        return "no_drop"
    if reducer_type == "drop_corr":
        return f"drop_feats/{drop_corr_method}_r_{drop_corr_threshold}"
    if reducer_type == "pca":
        return f"pca/n_components_{pca_components}"
    raise ValueError(f"unknown reducer_type: {reducer_type!r}")


# ----------------------------------------------------------------------------
# Trainer factory
# ----------------------------------------------------------------------------
def build_trainer(
    classifier_type: str,
    *,
    reducer_type: str = "no_drop",
    pca_components=None,
    drop_corr_threshold=None,
    drop_corr_method: str = "pearson",
    lr_C: float = 1.0,
    xgb_params: Optional[dict] = None,
    xgb_device: str = "cuda",
    seed: int = 42,
) -> Tuple[object, str, bool]:
    """回傳 (estimator, score_method, needs_cv)。classifier_type ∈ CLASSIFY_METHODS。

    estimator 是 Pipeline(scaler → [reducer] → classifier),reducer+clf 同次 fit 防 leakage。
    """
    rd = _build_reducer(reducer_type, pca_components=pca_components,
                        drop_corr_threshold=drop_corr_threshold,
                        drop_corr_method=drop_corr_method)
    if classifier_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        steps = [("scaler", StandardScaler())]   # 線性模型 + L2,需要 scale
        if rd != "passthrough":
            steps.append(("reducer", rd))
        steps.append(("classifier", LogisticRegression(
            C=lr_C, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=seed)))
        return Pipeline(steps), "predict_proba", True
    if classifier_type == "xgb":
        from xgboost import XGBClassifier
        p = xgb_params or DEFAULT_XGB_PARAMS
        steps = []                                # xgb 是 tree-based、scale-invariant → 不放 scaler
        if rd != "passthrough":
            steps.append(("reducer", rd))
        steps.append(("classifier", XGBClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], objective="binary:logistic",
            eval_metric="logloss", random_state=seed, verbosity=0,
            tree_method="hist", device=xgb_device)))
        return Pipeline(steps), "predict_proba", True

    raise ValueError(f"unknown classifier_type: {classifier_type!r} "
                     f"(expected one of {CLASSIFY_METHODS})")
