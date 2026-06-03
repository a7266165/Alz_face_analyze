"""
降維器(reducer)，使用時，後面需串聯分類器(classifier)。
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DIM_REDUCERS = ("no_drop", "pca", "drop_corr")


# ----------------------------------------------------------------------------
# Reducers(降維)
# ----------------------------------------------------------------------------
class DropCorrReducer(BaseEstimator, TransformerMixin):
    """feature_engine的DropCorrelatedFeatures再包裝。"""

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
    """GPU PCA"""

    def __init__(self, n_components, niter=4):
        self.n_components = n_components
        self.niter = niter

    def fit(self, X, y=None):
        import torch

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
        self.mean_ = Xt.mean(0, keepdim=True)
        # Xt.shape[0] 是 fold 樣本數, Xt.shape[1] 是 feature 數,
        # PCA降維維度 q 需 ≤ min(樣本數, feature 數)。
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
# Reducer factory
# ----------------------------------------------------------------------------
def build_reducer(
    reducer_type="no_drop",
    *,
    pca_components=None,
    drop_corr_threshold=None,
    drop_corr_method="pearson",
):
    """回傳 sklearn transformer 或 sentinel 'passthrough'(no_drop)。

    根據pca_components輸入值，採用不同方法：
      float ∈ (0,1) → PCA後保留足以解釋該變異比例的維度
      int ≥ 1       → PCA後保留固定維度
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


def reducer_label(
    reducer_type="no_drop",
    *,
    pca_components=None,
    drop_corr_threshold=None,
    drop_corr_method="pearson",
) -> str:
    """輸出 reducer 儲存路徑"""
    if reducer_type == "no_drop":
        return "no_drop"
    if reducer_type == "drop_corr":
        return f"drop_feats/{drop_corr_method}_r_{drop_corr_threshold}"
    if reducer_type == "pca":
        return f"pca/n_components_{pca_components}"
    raise ValueError(f"unknown reducer_type: {reducer_type!r}")
