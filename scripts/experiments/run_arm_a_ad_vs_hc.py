"""
Arm A: Naive AD vs HC cross-sectional classification.

No age balance — intentionally so, to demonstrate the age-confound baseline.
Each feature modality gets 5-fold GroupKFold XGBoost (AUC/BalAcc/MCC/F1/Sens/Spec
+ bootstrap AUC 95% CI); age-only logistic baseline is also reported so that
per-modality contribution can be measured as (modality AUC − age-only AUC).

Also reports per-feature Cohen's d / Hedges' g (AD mean − HC mean), and
per-modality ROC + PR curves (PNG).

Usage:
    conda run -n Alz_face_test_2 python scripts/experiments/run_arm_a_ad_vs_hc.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_auc_score, roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
AGES_FILE = PROJECT_ROOT / "workspace" / "age" / "age_prediction" / "predicted_ages.json"
EMOTION_DIR = PROJECT_ROOT / "workspace" / "emotion" / "au_features" / "aggregated"
LANDMARK_FEATURES_CSV = PROJECT_ROOT / "workspace" / "asymmetry" / "features.csv"
EMBEDDING_DIR = PROJECT_ROOT / "workspace" / "embedding" / "features"
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "arms_analysis" / "per_arm" / "arm_a"

EMOTION_METHODS = ["openface", "libreface", "pyfeat", "dan",
                   "hsemotion", "vit", "poster_pp", "fer"]
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
STATS = ["mean", "std", "range", "entropy"]
LANDMARK_REGIONS = ["eye", "nose", "mouth", "face_oval"]
EMBEDDING_MODELS = ["arcface", "topofr", "dlib"]

# 5 modality directions for per-direction subfolder layout
DIRECTIONS = ["age", "embedding_mean", "embedding_asymmetry",
              "landmark_asymmetry", "emotion"]
MODALITY_DIRECTION = {
    "age_only": "age",
    "age_error": "age",
    "embedding_arcface_mean": "embedding_mean",
    "embedding_dlib_mean": "embedding_mean",
    "embedding_topofr_mean": "embedding_mean",
    "embedding_asymmetry": "embedding_asymmetry",
    "landmark_asymmetry": "landmark_asymmetry",
    "emotion_8methods": "emotion",
}

N_BOOTSTRAP = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Cohort
# ============================================================

def load_cohort(cdr_thresh=0.5, healthy_strict=True):
    frames = []
    for grp in ["P", "NAD", "ACS"]:
        df = pd.read_csv(DEMOGRAPHICS_DIR / f"{grp}.csv")
        df["group"] = grp
        if "ID" not in df.columns:
            for col in df.columns:
                if col in ("ACS", "NAD"):
                    df = df.rename(columns={col: "ID"})
                    break
        frames.append(df)
    demo = pd.concat(frames, ignore_index=True)
    demo["Global_CDR"] = pd.to_numeric(demo.get("Global_CDR"), errors="coerce")
    demo["MMSE"] = pd.to_numeric(demo.get("MMSE"), errors="coerce")
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["base_id"] = demo["ID"].str.extract(r"^([A-Za-z]+\d+)")
    demo["visit"] = demo["ID"].str.extract(r"-(\d+)$").astype(float)

    demo["label"] = np.nan
    p_mask = demo["group"] == "P"
    demo.loc[p_mask & (demo["Global_CDR"] >= cdr_thresh), "label"] = 1

    hc_mask = demo["group"].isin(["NAD", "ACS"])
    if healthy_strict:
        has_cog = hc_mask & (demo["Global_CDR"].notna() | demo["MMSE"].notna())
        healthy = has_cog & (
            (demo["Global_CDR"] == 0) |
            (demo["Global_CDR"].isna() & (demo["MMSE"] >= 26))
        )
        demo.loc[healthy, "label"] = 0
    else:
        cdr0 = hc_mask & (demo["Global_CDR"] == 0)
        mmse26 = hc_mask & demo["Global_CDR"].isna() & (demo["MMSE"] >= 26)
        demo.loc[cdr0 | mmse26, "label"] = 0

    cohort = demo[demo["label"].notna() & demo["Age"].notna()].copy()
    cohort = cohort.sort_values(["base_id", "visit"])
    cohort = cohort.groupby("base_id", as_index=False).first()
    return cohort


# ============================================================
# Feature loading (unchanged from prior version)
# ============================================================

def load_age_error(ids):
    with open(AGES_FILE) as f:
        pred = json.load(f)
    rows = [(sid, pred.get(sid)) for sid in ids]
    return pd.DataFrame(rows, columns=["ID", "predicted_age"])


def load_emotion_matrix(ids):
    emo_cols = [f"{emo}_{stat}" for emo in EMOTIONS for stat in STATS]
    frames = []
    for method in EMOTION_METHODS:
        path = EMOTION_DIR / f"{method}_harmonized.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        keep = ["subject_id"] + [c for c in emo_cols if c in df.columns]
        df = df[keep].copy()
        rename = {c: f"{method}__{c}" for c in keep if c != "subject_id"}
        df = df.rename(columns=rename)
        frames.append(df)
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="subject_id", how="outer")
    return out[out["subject_id"].isin(ids)]


def load_landmark_matrix(ids):
    df = pd.read_csv(LANDMARK_FEATURES_CSV)
    df = df[df["subject_id"].isin(ids)].copy()
    rows = []
    for _, r in df.iterrows():
        out = {"subject_id": r["subject_id"]}
        total_sq = 0.0
        for region in LANDMARK_REGIONS:
            xy_cols = [c for c in df.columns
                       if c.startswith(f"{region}_") and "area_diff" not in c]
            vec = r[xy_cols].to_numpy(dtype=float)
            l2 = float(np.sqrt(np.nansum(vec * vec)))
            out[f"landmark__{region}_l2"] = l2
            out[f"landmark__{region}_area_diff"] = float(r[f"{region}_area_diff"])
            total_sq += np.nansum(vec * vec)
        out["landmark__total_l2"] = float(np.sqrt(total_sq))
        rows.append(out)
    return pd.DataFrame(rows)


def load_embedding_mean(ids, model):
    emb_dir = EMBEDDING_DIR / model / "original"
    rows = []
    for sid in ids:
        npy = emb_dir / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy, allow_pickle=True)
        if a.dtype == object:
            a = list(a.item().values())[0]
        vec = a.mean(axis=0) if a.ndim == 2 else a
        rows.append({"subject_id": sid,
                     **{f"{model}__dim_{i}": float(vec[i]) for i in range(vec.shape[0])}})
    return pd.DataFrame(rows) if rows else None


def load_embedding_asymmetry(ids):
    rows = []
    for sid in ids:
        out = {"subject_id": sid}
        for model in EMBEDDING_MODELS:
            npy = EMBEDDING_DIR / model / "difference" / f"{sid}.npy"
            if not npy.exists():
                out[f"embasym__{model}_l2"] = np.nan
                continue
            a = np.load(npy, allow_pickle=True)
            if a.dtype == object:
                a = list(a.item().values())[0]
            md = a.mean(axis=0) if a.ndim == 2 else a
            out[f"embasym__{model}_l2"] = float(np.linalg.norm(md))
        rows.append(out)
    return pd.DataFrame(rows)


# ============================================================
# Stats
# ============================================================

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def hedges_g(a, b):
    d = cohens_d(a, b)
    if np.isnan(d):
        return np.nan
    na = len([x for x in a if not np.isnan(x)])
    nb = len([x for x in b if not np.isnan(x)])
    df = na + nb - 2
    if df < 1:
        return d
    J = 1.0 - (3.0 / (4.0 * df - 1.0))
    return d * J


# ============================================================
# Classification + bootstrap CI
# ============================================================

def cv_eval(X, y, base_ids, model_cls="xgb", n_folds=5, seed=42, return_preds=False):
    """GroupKFold CV returning pooled metrics (+ optional y_true/y_prob)."""
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(base_ids))))
    y_true, y_pred, y_prob = [], [], []
    for tri, tei in gkf.split(X, y, groups=base_ids):
        if model_cls == "xgb":
            clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 objective="binary:logistic", eval_metric="logloss",
                                 random_state=seed, n_jobs=-1, verbosity=0)
            clf.fit(X[tri], y[tri])
            y_prob_fold = clf.predict_proba(X[tei])[:, 1]
        else:
            scaler = StandardScaler().fit(X[tri])
            Xt, Xv = scaler.transform(X[tri]), scaler.transform(X[tei])
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(Xt, y[tri])
            y_prob_fold = clf.predict_proba(Xv)[:, 1]
        y_pred_fold = (y_prob_fold >= 0.5).astype(int)
        y_true.extend(y[tei]); y_pred.extend(y_pred_fold); y_prob.extend(y_prob_fold)
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    metrics = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "balacc": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "sens": float(sens) if not np.isnan(sens) else np.nan,
        "spec": float(spec) if not np.isnan(spec) else np.nan,
    }
    if return_preds:
        metrics["y_true"] = y_true
        metrics["y_prob"] = y_prob
    return metrics


def bootstrap_auc_ci(y_true, y_prob, n=N_BOOTSTRAP, seed=42):
    """Stratified bootstrap of pooled predictions for AUC 95% CI."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n):
        pb = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        nb = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pb, nb])
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def run_modality(name, X_df, cohort, model_cls="xgb", n_folds=5, seed=42,
                  compute_ci=True, save_curve_dir=None):
    """Align features with cohort, CV evaluate, return result dict incl. CI."""
    merged = cohort[["ID", "base_id", "label"]].merge(
        X_df, left_on="ID", right_on="subject_id", how="inner"
    )
    feat_cols = [c for c in merged.columns if c not in
                 ["ID", "base_id", "label", "subject_id"]]
    merged = merged.dropna(subset=feat_cols)
    if len(merged) < 50:
        return {"modality": name, "n": len(merged), "auc": np.nan,
                "n_features": len(feat_cols)}
    X = merged[feat_cols].to_numpy(dtype=float)
    y = merged["label"].to_numpy(dtype=int)
    g = merged["base_id"].to_numpy()
    m = cv_eval(X, y, g, model_cls=model_cls, n_folds=n_folds, seed=seed,
                return_preds=True)
    y_true = m.pop("y_true"); y_prob = m.pop("y_prob")

    ci_low, ci_high = (bootstrap_auc_ci(y_true, y_prob, seed=seed)
                        if compute_ci else (np.nan, np.nan))

    # Save ROC + PR curves
    if save_curve_dir is not None:
        save_curve_dir.mkdir(parents=True, exist_ok=True)
        _plot_roc_pr(y_true, y_prob, name, save_curve_dir)

    return {
        "modality": name, "n": len(merged),
        "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "n_features": len(feat_cols),
        **m,
        "auc_ci_low": ci_low, "auc_ci_high": ci_high,
    }


def _plot_roc_pr(y_true, y_prob, modality, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color="#4C72B0", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"{modality}\nROC AUC = {auc_val:.3f}")
    axes[1].plot(rec, prec, color="#C44E52", linewidth=2)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{modality}\nPR curve")
    fig.tight_layout()
    fig.savefig(out_dir / f"{modality}.png", dpi=120)
    plt.close(fig)


# ============================================================
# Per-feature effect sizes
# ============================================================

def per_feature_effect_sizes(modality, feat_df, cohort, out_dir, top_n=10):
    """For each feature in a modality, compute AD vs HC Cohen's d & Welch p.
    Save full table + top-N by |d|."""
    merged = cohort[["ID", "label"]].merge(
        feat_df, left_on="ID", right_on="subject_id", how="inner"
    )
    feat_cols = [c for c in merged.columns if c not in
                 ["ID", "label", "subject_id"]]
    rows = []
    for col in feat_cols:
        ad = merged.loc[merged["label"] == 1, col].dropna().values
        hc = merged.loc[merged["label"] == 0, col].dropna().values
        if len(ad) < 5 or len(hc) < 5:
            continue
        d = cohens_d(ad, hc)
        g = hedges_g(ad, hc)
        t, p = stats.ttest_ind(ad, hc, equal_var=False)
        rows.append({
            "modality": modality, "feature": col,
            "n_ad": len(ad), "n_hc": len(hc),
            "mean_ad": float(ad.mean()), "mean_hc": float(hc.mean()),
            "cohen_d": d, "hedges_g": g,
            "welch_t": float(t), "welch_p": float(p),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"per_feature_{modality}.csv", index=False)
    return df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdr-thresh", type=float, default=0.5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--healthy-strict", action="store_true", default=True)
    parser.add_argument("--skip-ci", action="store_true",
                         help="skip bootstrap AUC CI (for quick smoke test)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_dir = OUTPUT_DIR / "ad_vs_hc"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for d in DIRECTIONS:
        (comparison_dir / d).mkdir(parents=True, exist_ok=True)

    def dir_for(modality):
        return comparison_dir / MODALITY_DIRECTION[modality]

    cohort = load_cohort(cdr_thresh=args.cdr_thresh, healthy_strict=args.healthy_strict)
    n_ad = int((cohort["label"] == 1).sum())
    n_hc = int((cohort["label"] == 0).sum())
    logger.info(f"Cohort: AD={n_ad}, HC={n_hc}")
    cohort.to_csv(OUTPUT_DIR / "cohort.csv", index=False)

    age_mean_diff = (cohort[cohort["label"] == 1]["Age"].mean() -
                     cohort[cohort["label"] == 0]["Age"].mean())
    logger.info(f"Age gap AD − HC: {age_mean_diff:+.2f} years")

    ids = cohort["ID"].tolist()
    results = []
    compute_ci = not args.skip_ci

    # Age-only baseline
    X_age = cohort[["Age"]].to_numpy(dtype=float)
    y_all = cohort["label"].to_numpy(dtype=int)
    g_all = cohort["base_id"].to_numpy()
    m = cv_eval(X_age, y_all, g_all, model_cls="logistic",
                 n_folds=args.n_folds, seed=args.seed, return_preds=True)
    y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
    ci = bootstrap_auc_ci(y_true, y_prob, seed=args.seed) if compute_ci else (np.nan, np.nan)
    results.append({"modality": "age_only", "n": len(cohort),
                    "n_pos": n_ad, "n_neg": n_hc, "n_features": 1,
                    **m, "auc_ci_low": ci[0], "auc_ci_high": ci[1]})
    logger.info(f"age_only AUC={m['auc']:.3f}  CI=[{ci[0]:.3f}, {ci[1]:.3f}]")

    # age_error
    age_err = load_age_error(ids)
    age_err_df = cohort[["ID"]].merge(age_err, on="ID", how="left")
    age_err_df["age_error"] = cohort["Age"].to_numpy() - age_err_df["predicted_age"]
    age_err_df = age_err_df.dropna(subset=["age_error"]).rename(columns={"ID": "subject_id"})
    results.append(run_modality("age_error",
                                 age_err_df[["subject_id", "age_error"]],
                                 cohort, model_cls="logistic",
                                 n_folds=args.n_folds, seed=args.seed,
                                 compute_ci=compute_ci,
                                 save_curve_dir=dir_for("age_error")))

    # Emotion
    emo = load_emotion_matrix(ids)
    if emo is not None:
        results.append(run_modality("emotion_8methods", emo, cohort,
                                     n_folds=args.n_folds, seed=args.seed,
                                     compute_ci=compute_ci,
                                     save_curve_dir=dir_for("emotion_8methods")))
        per_feature_effect_sizes("emotion_8methods", emo, cohort,
                                  dir_for("emotion_8methods"))

    # Landmark
    lmk = load_landmark_matrix(ids)
    results.append(run_modality("landmark_asymmetry", lmk, cohort,
                                 n_folds=args.n_folds, seed=args.seed,
                                 compute_ci=compute_ci,
                                 save_curve_dir=dir_for("landmark_asymmetry")))
    per_feature_effect_sizes("landmark_asymmetry", lmk, cohort,
                              dir_for("landmark_asymmetry"))

    # Embedding asymmetry
    emb_asym = load_embedding_asymmetry(ids)
    results.append(run_modality("embedding_asymmetry", emb_asym, cohort,
                                 n_folds=args.n_folds, seed=args.seed,
                                 compute_ci=compute_ci,
                                 save_curve_dir=dir_for("embedding_asymmetry")))
    per_feature_effect_sizes("embedding_asymmetry", emb_asym, cohort,
                              dir_for("embedding_asymmetry"))

    # Embedding mean per model
    for model in EMBEDDING_MODELS:
        emb = load_embedding_mean(ids, model)
        if emb is not None:
            modality = f"embedding_{model}_mean"
            results.append(run_modality(modality, emb, cohort,
                                         n_folds=args.n_folds, seed=args.seed,
                                         compute_ci=compute_ci,
                                         save_curve_dir=dir_for(modality)))

    df = pd.DataFrame(results)
    df["age_gap_years"] = age_mean_diff
    age_auc = df.loc[df["modality"] == "age_only", "auc"].iloc[0]
    df["delta_vs_age_only"] = df["auc"] - age_auc
    df.to_csv(comparison_dir / "summary_per_modality.csv", index=False)

    logger.info(f"Done. Outputs at {OUTPUT_DIR}")
    for r in results:
        logger.info(
            f"  {r['modality']:<25} AUC={r.get('auc', np.nan):.3f} "
            f"[{r.get('auc_ci_low', np.nan):.3f}, {r.get('auc_ci_high', np.nan):.3f}]  "
            f"BalAcc={r.get('balacc', np.nan):.3f}  MCC={r.get('mcc', np.nan):.3f}  "
            f"F1={r.get('f1', np.nan):.3f}  n={r['n']}"
        )


if __name__ == "__main__":
    main()
