"""
Classification 下游的「分類器」leaf —— logistic / xgb。

對應概念模型裡 classify 列。**只回 classifier 本體 + needs_scaler 旗標**;scaler/reducer/
clf 串成 Pipeline 由 producer 的 _build_estimator 做。純 leaf:不知道 reducer、不碰 Pipeline。

needs_scaler:LR 是線性 + L2,需要 scale → True;XGB 是 tree-based、scale-invariant → False。
"""
from typing import Optional, Tuple

DEFAULT_XGB_PARAMS = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}

CLASSIFIERS = ("logistic", "xgb")


def build_classifier(
    classifier_type: str,
    *,
    lr_C: float = 1.0,
    xgb_params: Optional[dict] = None,
    xgb_device: str = "cuda",
    seed: int = 42,
) -> Tuple[object, bool]:
    """回傳 (estimator, needs_scaler)。classifier_type ∈ CLASSIFIERS。

    inline 分支 —— 函數本身即 dispatcher(每條分支只是建物件回傳,不抽私有 worker)。
    """
    if classifier_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=lr_C, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=seed), True
    if classifier_type == "xgb":
        from xgboost import XGBClassifier
        p = xgb_params or DEFAULT_XGB_PARAMS
        return XGBClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], objective="binary:logistic",
            eval_metric="logloss", random_state=seed, verbosity=0,
            tree_method="hist", device=xgb_device), False

    raise ValueError(f"unknown classifier_type: {classifier_type!r} "
                     f"(expected one of {CLASSIFIERS})")
