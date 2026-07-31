"""Meta 端 stacker:TabPFN v3(指向 v3 ckpt,找不到退回套件預設)、XGB 與 LR(皆重用 embedding
build_classifier),外加不需訓練的 mean(PDF Step 12 的對照組)。"""
from pathlib import Path

import numpy as np

_V3_CKPT = (Path.home() / "AppData" / "Roaming" / "tabpfn"
            / "tabpfn-v3-classifier-v3_default.ckpt")

META_CLASSIFIERS = ("tabpfn_v3", "xgb", "lr", "mean")


class MeanScorer:
    """把特徵欄逐列平均當分數的「stacker」,fit 是 no-op。

    PDF Step 12 的對照組用這個表達,就不必為它另開一條管線:
      feature_set=embedding_only  → 單獨的 embedding 分數
      feature_set=asymmetry_only  → 單獨的 asymmetry 分數
      feature_set=bases           → 兩者的簡單平均
    因為不 fit,nested 與非 nested 兩條路徑對它必然給出相同結果。
    """

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        p = np.asarray(X, dtype=float).mean(axis=1)
        return np.column_stack([1.0 - p, p])


def make_tabpfn_v3(seed=42, device="auto"):
    """建立指向 v3 ckpt 的 TabPFNClassifier(找不到 ckpt 則退回套件預設權重)。"""
    from tabpfn import TabPFNClassifier
    if _V3_CKPT.exists():
        return TabPFNClassifier(model_path=str(_V3_CKPT), device=device,
                                random_state=seed, ignore_pretraining_limits=True)
    return TabPFNClassifier(device=device, random_state=seed,
                            ignore_pretraining_limits=True)


def make_meta_clf(name, *, seed=42, device="auto"):
    """依名稱建 meta stacker estimator(predict_proba 介面);name ∈ META_CLASSIFIERS。

    tabpfn_v3 → make_tabpfn_v3;xgb → build_classifier("xgb", 固定 DEFAULT_XGB_PARAMS);
    lr → build_classifier("logistic") 串 imputer+scaler(LR 需標準化、不能吃 NaN;complete-case 下 imputer 為 no-op)。
    """
    if name == "tabpfn_v3":
        return make_tabpfn_v3(seed=seed, device=device)
    if name == "xgb":
        from src.embedding.classification import build_classifier
        xgb_device = "cpu" if device == "cpu" else "cuda"
        est, _ = build_classifier("xgb", xgb_device=xgb_device, seed=seed)
        return est
    if name == "mean":
        return MeanScorer()
    if name == "lr":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from src.embedding.classification import build_classifier
        est, _ = build_classifier("logistic", seed=seed)   # LR C=1.0, class_weight=balanced
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()), ("clf", est)])
    raise ValueError(f"unknown meta classifier: {name!r} (expected one of {META_CLASSIFIERS})")
