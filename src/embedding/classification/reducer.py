"""
Classification 下游的「降維」leaf —— no_drop / PCA / DropCorr。

對應概念模型裡 reduce 列(original 路徑用)。reducer 是分類 Pipeline 的「一步」,
**串接(scaler/reducer/clf 組成 Pipeline)由 producer 的 _build_estimator 做**,這裡只負責
「給一個 reducer 物件」。純 leaf:不知道 classifier、不碰 Pipeline。

做成 BaseEstimator 子類,cross-val / 折迴圈每折才能 clone(主成分、相關欄位只能從
train fold 學 → leakage 結構上不可能,前提是 reducer 在 Pipeline 內隨整支一起 fit)。
"""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DIM_REDUCERS = ("no_drop", "pca", "drop_corr")


# ----------------------------------------------------------------------------
# Reducers(降維)—— transformer,分類 Pipeline 的一步
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
    故 sklearn.clone 可正確複製(折迴圈每折會 clone)。
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
        # clamp 為強制:torch.pca_lowrank 在 q > min(m, n) 時直接報錯,故 q 必須 ≤ fold rank。
        # 但 clamp 會悄悄降維,故發 warning 讓使用者可發現(實際維度見 n_components_)。
        q = min(int(self.n_components), Xt.shape[0], Xt.shape[1])
        if q < int(self.n_components):
            warnings.warn(
                f"TorchGPUPCA: n_components {self.n_components} exceeds fold rank; "
                f"clamped to {q} (see n_components_)."
            )
        _, _, V = torch.pca_lowrank(Xt - self.mean_, q=q, niter=self.niter)
        self.V_ = V[:, :q].contiguous()
        self.n_components_ = q
        return self

    def transform(self, X):
        import torch
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
        return ((Xt - self.mean_) @ self.V_).cpu().numpy()


# ----------------------------------------------------------------------------
# Reducer factory(inline 分支 —— 函數本身即 dispatcher)
# ----------------------------------------------------------------------------
def build_reducer(reducer_type="no_drop", *, pca_components=None,
                  drop_corr_threshold=None, drop_corr_method="pearson"):
    """回傳 sklearn transformer 或 sentinel 'passthrough'(no_drop)。

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
