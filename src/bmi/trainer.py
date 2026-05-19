"""
ArcFace embedding → BMI regression.

Core training / evaluation logic for Ridge and XGBoost regressors.
Uses GroupKFold (by base_id) to prevent data leakage across visits of
the same subject.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

ModelName = Literal["ridge", "svr", "xgb"]

EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_arcface_features(
    ids: List[str],
    features_dir: Path,
    photo_mode: str = "mean",
) -> Dict[str, np.ndarray]:
    """Load ArcFace original embeddings for *ids*, mean-pooling per visit.

    Returns dict[ID → 1-D array of shape (512,)].
    """
    emb_dir = features_dir / "arcface" / "original"
    out = {}
    for sid in ids:
        npy = emb_dir / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy, allow_pickle=True)
        if a.dtype == object:
            a = list(a.item().values())[0]
        if a.ndim == 2:
            if photo_mode == "mean":
                a = a.mean(axis=0)
            else:
                a = a.mean(axis=0)
        out[sid] = a.astype(np.float32)
    return out


def build_dataset(
    features_dir: Path,
    demographics_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, groups, ids) from ArcFace embeddings + demographics BMI.

    Loads all groups (P, NAD, ACS) and keeps only visits with both
    a valid ArcFace embedding and a non-NaN BMI.

    Returns:
        X:      (N, 512) float32 feature matrix
        y:      (N,) float64 BMI target
        groups: (N,) int — base_id encoded as integer for GroupKFold
        ids:    list of N ID strings
    """
    dfs = []
    for csv_name, id_prefix_re in [
        ("P.csv", r"^(P\d+)"),
        ("NAD.csv", r"^(NAD\d+)"),
        ("ACS.csv", r"^(ACS\d+)"),
    ]:
        path = demographics_dir / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "BMI" not in df.columns:
            logger.warning(f"{csv_name} has no BMI column, skipping")
            continue
        df = df[["ID", "BMI"]].dropna(subset=["BMI"])
        df["base_id"] = df["ID"].str.extract(id_prefix_re)
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No demographics with BMI found")

    demo = pd.concat(dfs, ignore_index=True)

    all_ids = demo["ID"].tolist()
    feats = load_arcface_features(all_ids, features_dir, photo_mode="mean")

    rows_x, rows_y, rows_g, rows_id = [], [], [], []
    for _, row in demo.iterrows():
        sid = row["ID"]
        if sid not in feats:
            continue
        rows_x.append(feats[sid])
        rows_y.append(float(row["BMI"]))
        rows_g.append(row["base_id"])
        rows_id.append(sid)

    X = np.stack(rows_x, axis=0)
    y = np.array(rows_y, dtype=np.float64)

    base_id_to_int = {b: i for i, b in enumerate(sorted(set(rows_g)))}
    groups = np.array([base_id_to_int[g] for g in rows_g])

    logger.info(f"Dataset: {X.shape[0]} visits, {len(base_id_to_int)} subjects, "
                f"BMI range [{y.min():.1f}, {y.max():.1f}]")
    return X, y, groups, rows_id


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_ridge():
    return RidgeCV(alphas=np.logspace(-2, 6, 50), scoring="neg_mean_absolute_error")


def _make_svr():
    return SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")


def _make_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        device="cuda",
    )


def make_model(name: ModelName):
    if name == "ridge":
        return _make_ridge()
    if name == "svr":
        return _make_svr()
    if name == "xgb":
        return _make_xgb()
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R^2, Pearson r."""
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "n": len(y_true),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: ModelName,
    n_splits: int = 10,
    seed: int = 42,
) -> Dict:
    """GroupKFold cross-validation for BMI regression.

    Returns dict with:
        fold_metrics: list[dict] — per-fold metrics
        oof_predictions: DataFrame(ID, y_true, y_pred, fold)
        aggregate: dict — aggregate metrics across all OOF predictions
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    oof_true, oof_pred, oof_fold = [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = make_model(model_name)
        if model_name == "ridge":
            model.fit(X_train_s, y_train)
        else:
            model.fit(X_train_s, y_train)

        y_pred_test = model.predict(X_test_s)
        metrics = regression_metrics(y_test, y_pred_test)
        metrics["fold"] = fold_i
        fold_metrics.append(metrics)

        oof_true.append(y_test)
        oof_pred.append(y_pred_test)
        oof_fold.append(np.full(len(y_test), fold_i))

        logger.info(f"  Fold {fold_i}: MAE={metrics['mae']:.2f}  "
                    f"R2={metrics['r2']:.3f}  r={metrics['pearson_r']:.3f}")

    all_true = np.concatenate(oof_true)
    all_pred = np.concatenate(oof_pred)
    aggregate = regression_metrics(all_true, all_pred)

    logger.info(f"  Aggregate: MAE={aggregate['mae']:.2f}  "
                f"R2={aggregate['r2']:.3f}  r={aggregate['pearson_r']:.3f}")

    return {
        "fold_metrics": fold_metrics,
        "oof_true": all_true,
        "oof_pred": all_pred,
        "oof_fold": np.concatenate(oof_fold),
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Full training (for inference model)
# ---------------------------------------------------------------------------

def train_final(
    X: np.ndarray,
    y: np.ndarray,
    model_name: ModelName,
) -> Tuple:
    """Train on the entire dataset. Returns (model, scaler)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = make_model(model_name)
    model.fit(X_s, y)
    logger.info(f"Final {model_name} trained on {X.shape[0]} samples")
    return model, scaler
