"""
分類器(classifier)，使用時，前面需串聯降維器(reducer)。
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
    """依 classifier_type 建立分類器。

    Args:
        classifier_type: logistic | xgb
        lr_C: LogisticRegression 正則化強度倒數。
        xgb_params: XGBClassifier 參數。
        xgb_device: cuda | cpu
        seed: 隨機種子。

    Returns:
        (estimator, needs_scaler)，needs_scaler 表示該模型是否需要先標準化特徵
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
