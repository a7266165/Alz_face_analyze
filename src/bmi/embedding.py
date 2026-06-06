"""凍結 embedding → sklearn 迴歸（Ridge / SVR / XGBoost）。

build_embedding_dataset 把 {ID: 向量} 對上 BMI 標籤，cross_validate / train_final 跑
10-fold GroupKFold。ArcFace 與 MeFEm 兩路線共用本檔，只差 embedding 來源。
"""

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .core import encode_groups, regression_metrics

logger = logging.getLogger(__name__)

REGRESSORS = ("ridge", "svr", "xgb")


def load_arcface_features(ids: Sequence[str], features_dir: Path) -> Dict[str, np.ndarray]:
    """讀 arcface/original/{id}.npy，每 visit 對 ~10 張取平均 → {ID: (512,) float32}。"""
    emb_dir = features_dir / "arcface" / "original"
    out = {}
    for sid in ids:
        npy = emb_dir / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy)
        if a.ndim == 2:
            a = a.mean(axis=0)
        out[sid] = a.astype(np.float32)
    return out


def build_embedding_dataset(
    ids: Sequence[str],
    bmi: np.ndarray,
    base_ids: Sequence[str],
    features: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """把 (ids, bmi, base_ids) 篩到 features 有的 ID，疊成 (X, y, groups, kept_ids)。

    ArcFace 與 MeFEm 路線的共用組裝點：上游只需把各自的 embedding 整成 {ID: 向量}。
    """
    keep = [(sid, b, g) for sid, b, g in zip(ids, bmi, base_ids) if sid in features]
    if not keep:
        raise RuntimeError("沒有任何 ID 同時有 BMI 與 embedding")
    kept_ids = [sid for sid, _, _ in keep]
    X = np.stack([features[sid] for sid in kept_ids], axis=0)
    y = np.array([b for _, b, _ in keep], dtype=np.float64)
    groups = encode_groups([g for _, _, g in keep])
    logger.info(f"Dataset: {X.shape[0]} visits, {len(set(groups))} subjects, "
                f"dim={X.shape[1]}, BMI [{y.min():.1f}, {y.max():.1f}]")
    return X, y, groups, kept_ids


def make_regressor(name: str):
    """ridge | svr | xgb → sklearn 迴歸器。"""
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-2, 6, 50), scoring="neg_mean_absolute_error")
    if name == "svr":
        return SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, n_jobs=-1, tree_method="hist", device="cuda",
        )
    raise ValueError(f"unknown regressor: {name!r} (expected one of {REGRESSORS})")


def cross_validate(X, y, groups, model_name, n_splits=10, seed=42) -> Dict:
    """GroupKFold CV：每折 StandardScaler + 迴歸器。

    Returns:
        fold_metrics（逐折）、oof_true/oof_pred/oof_fold（折外預測）、aggregate（全 OOF）。
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics, oof_true, oof_pred, oof_fold = [], [], [], []

    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        model = make_regressor(model_name)
        model.fit(X_tr, y[tr])
        pred = model.predict(X_te)

        m = regression_metrics(y[te], pred)
        m["fold"] = fold_i
        fold_metrics.append(m)
        oof_true.append(y[te])
        oof_pred.append(pred)
        oof_fold.append(np.full(len(te), fold_i))
        logger.info(f"  Fold {fold_i}: MAE={m['mae']:.2f}  R2={m['r2']:.3f}  r={m['pearson_r']:.3f}")

    all_true = np.concatenate(oof_true)
    all_pred = np.concatenate(oof_pred)
    aggregate = regression_metrics(all_true, all_pred)
    logger.info(f"  Aggregate: MAE={aggregate['mae']:.2f}  R2={aggregate['r2']:.3f}  "
                f"r={aggregate['pearson_r']:.3f}")

    return {
        "fold_metrics": fold_metrics,
        "oof_true": all_true,
        "oof_pred": all_pred,
        "oof_fold": np.concatenate(oof_fold),
        "aggregate": aggregate,
    }


def train_final(X, y, model_name) -> Tuple:
    """全資料 fit，回 (model, scaler)。"""
    scaler = StandardScaler()
    model = make_regressor(model_name)
    model.fit(scaler.fit_transform(X), y)
    logger.info(f"Final {model_name} trained on {X.shape[0]} samples")
    return model, scaler
