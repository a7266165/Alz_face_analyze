"""Embedding 下游 Stage-1:**OOF 分數生產**(重構版)。對 src/meta 零依賴,只用 common。

契約見 memory ``project_embedding_downstream_refactor``。核心觀念:
  - reducer / classifier / scorer 是三個 leaf 建構器;reducer+classifier 串成單一
    Pipeline 的組裝由 producer(run_classification 的 _build_estimator)做。
  - 訓練(產 OOF)與落地切成兩支 dispatcher:``train(..., direction)`` → oof;
    ``report(oof, ..., direction)`` → 寫 oof_scores.csv。各自內部分 fwd/rev 私有 worker。
  - **不做評估**:配對評估(年齡配對 × partition × 指標)是獨立下游步驟,吃 oof_scores.csv。
  - OOF schema = [ID, y_true, y_score, fold],1 列/ID;Subject 由 ID 推得(inline regex)、不持久化。
  - 中間落地的 oof_scores.csv 才是下游(評估 / Stage-2 meta)的入口。
"""
from .reducer import (
    DIM_REDUCERS,
    DropCorrReducer,
    TorchGPUPCA,
    build_reducer,
    reducer_label,
)
from .classifier import (
    CLASSIFIERS,
    DEFAULT_XGB_PARAMS,
    build_classifier,
)
from .scorer import (
    ASYMMETRY_METHODS,
    CentroidEstimator,
    build_scorer,
)
from .train import train
from .report import report
from .paths import clf_param_label, oof_dir, oof_paths

ALL_METHODS = CLASSIFIERS + ASYMMETRY_METHODS

__all__ = [
    "ALL_METHODS",
    "CLASSIFIERS",
    "ASYMMETRY_METHODS",
    "DIM_REDUCERS",
    "DEFAULT_XGB_PARAMS",
    "build_reducer",
    "reducer_label",
    "build_classifier",
    "build_scorer",
    "train",
    "report",
    "clf_param_label",
    "oof_dir",
    "oof_paths",
]
