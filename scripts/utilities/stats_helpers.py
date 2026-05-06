"""
Shared statistical helpers: classifier CV, bootstrap CI, effect sizes,
multivariate tests (Welch, Hotelling T², PERMANOVA), and BH-FDR.

Consumed by:
  - scripts/overview/run_cross_naive.py
  - scripts/overview/run_cross_matched.py
  - scripts/overview/run_stat_grid.py
  - scripts/age/run_classifiers.py
  - scripts/embedding/run_fwd_rev.py (only bootstrap_auc_ci)
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_N_PERMS = 1000
DEFAULT_SEED = 42


# ============================================================
# Effect sizes
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
    """Hedges' g = Cohen's d × small-sample bias-correction J."""
    d = cohens_d(a, b)
    if np.isnan(d):
        return np.nan
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = int((~np.isnan(a)).sum())
    nb = int((~np.isnan(b)).sum())
    df = na + nb - 2
    if df < 1:
        return d
    J = 1.0 - (3.0 / (4.0 * df - 1.0))
    return d * J


# ============================================================
# CV classifier evaluation
# ============================================================

def cv_eval(X, y, base_ids, model_cls="xgb", n_folds=5, seed=DEFAULT_SEED,
            return_preds=False):
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

    tn, fp, fn, tp = (confusion_matrix(y_true, y_pred).ravel()
                      if len(np.unique(y_true)) > 1 else (0, 0, 0, 0))
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


def bootstrap_auc_ci(y_true, y_prob, n=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_SEED):
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


def auc_with_ci(X, y, g=None, model_cls="xgb", seed=DEFAULT_SEED):
    """AUC + 95% CI via cv_eval + bootstrap_auc_ci. Tolerates NaN in X."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]
    if g is not None:
        g = np.asarray(g)[mask]
    if len(X) < 20 or len(np.unique(y)) < 2:
        return {"auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan,
                "n": int(mask.sum())}
    if g is None:
        g = np.arange(len(X))
    try:
        m = cv_eval(X, y, g, model_cls=model_cls, n_folds=5, seed=seed,
                    return_preds=True)
        y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
        lo, hi = bootstrap_auc_ci(y_true, y_prob, seed=seed)
        return {"auc": float(m["auc"]), "auc_ci_low": float(lo),
                "auc_ci_high": float(hi), "n": int(mask.sum())}
    except Exception as e:
        logger.warning(f"AUC calc failed: {e}")
        return {"auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan,
                "n": int(mask.sum())}


# ============================================================
# BH-FDR
# ============================================================

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


# ============================================================
# Univariate Welch t (with Mann-Whitney + Cohen's d)
# ============================================================

def welch_t_test(x, y):
    """Return dict: t, p_welch, p_mw, d, n1, n2, mean1, mean2."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return {"t": np.nan, "p_welch": np.nan, "p_mw": np.nan, "d": np.nan,
                "n1": len(x), "n2": len(y),
                "mean1": float(x.mean()) if len(x) else np.nan,
                "mean2": float(y.mean()) if len(y) else np.nan}
    t, p = stats.ttest_ind(x, y, equal_var=False)
    try:
        u, pu = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        u, pu = np.nan, np.nan
    return {"t": float(t), "p_welch": float(p), "p_mw": float(pu),
            "d": cohens_d(x, y),
            "n1": len(x), "n2": len(y),
            "mean1": float(x.mean()), "mean2": float(y.mean())}


# ============================================================
# Two-sample Hotelling's T²
# ============================================================

def hotelling_t2(X1, X2, n_perms=DEFAULT_N_PERMS, seed=DEFAULT_SEED):
    """Two-sample Hotelling's T². Returns T², F, p_F, p_perm, D².
    Assumes n1+n2 > p; for p ≈ n use PERMANOVA."""
    X1 = np.asarray(X1, dtype=float); X2 = np.asarray(X2, dtype=float)
    mask1 = ~np.isnan(X1).any(axis=1); mask2 = ~np.isnan(X2).any(axis=1)
    X1, X2 = X1[mask1], X2[mask2]
    n1, n2 = len(X1), len(X2)
    p = X1.shape[1]
    if n1 < 2 or n2 < 2 or n1 + n2 - 2 <= p:
        return {"T2": np.nan, "F": np.nan, "p_F": np.nan, "p_perm": np.nan,
                "D2": np.nan, "n1": n1, "n2": n2, "p": p}
    m1, m2 = X1.mean(axis=0), X2.mean(axis=0)
    S1 = np.cov(X1, rowvar=False, ddof=1)
    S2 = np.cov(X2, rowvar=False, ddof=1)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    try:
        Sinv = np.linalg.pinv(Sp)
    except np.linalg.LinAlgError:
        return {"T2": np.nan, "F": np.nan, "p_F": np.nan, "p_perm": np.nan,
                "D2": np.nan, "n1": n1, "n2": n2, "p": p}
    diff = m1 - m2
    T2 = float(diff @ Sinv @ diff * (n1 * n2) / (n1 + n2))
    D2 = T2 * (n1 + n2) / (n1 * n2)
    F = T2 * (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)
    p_F = float(1 - stats.f.cdf(F, p, n1 + n2 - p - 1)) if F > 0 else 1.0

    X = np.vstack([X1, X2])
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(n1 + n2)
        Xp1, Xp2 = X[perm[:n1]], X[perm[n1:]]
        mp1, mp2 = Xp1.mean(axis=0), Xp2.mean(axis=0)
        Sp1 = np.cov(Xp1, rowvar=False, ddof=1)
        Sp2 = np.cov(Xp2, rowvar=False, ddof=1)
        Spp = ((n1 - 1) * Sp1 + (n2 - 1) * Sp2) / (n1 + n2 - 2)
        try:
            Sppinv = np.linalg.pinv(Spp)
        except np.linalg.LinAlgError:
            continue
        dp = mp1 - mp2
        T2p = dp @ Sppinv @ dp * (n1 * n2) / (n1 + n2)
        if T2p >= T2:
            count += 1
    p_perm = (count + 1) / (n_perms + 1)
    return {"T2": T2, "F": float(F), "p_F": p_F, "p_perm": float(p_perm),
            "D2": float(D2), "n1": n1, "n2": n2, "p": p}


# ============================================================
# PERMANOVA (Anderson 2001)
# ============================================================

def permanova(X1, X2, metric="euclidean", n_perms=DEFAULT_N_PERMS, seed=DEFAULT_SEED):
    """Permutational MANOVA. pseudo-F + permutation p + R² + ω²."""
    X1 = np.asarray(X1, dtype=float); X2 = np.asarray(X2, dtype=float)
    mask1 = ~np.isnan(X1).any(axis=1); mask2 = ~np.isnan(X2).any(axis=1)
    X1, X2 = X1[mask1], X2[mask2]
    n1, n2 = len(X1), len(X2)
    N = n1 + n2
    if n1 < 2 or n2 < 2:
        return {"pseudo_F": np.nan, "p_perm": np.nan, "R2": np.nan,
                "omega2": np.nan, "n1": n1, "n2": n2, "metric": metric}

    X = np.vstack([X1, X2])
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        D = squareform(pdist(Xn, metric="euclidean"))
    else:
        D = squareform(pdist(X, metric=metric))
    D2 = D ** 2
    SS_total = D2.sum() / (2 * N)

    def _pseudo_F(labels):
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        n0, n1_ = len(idx0), len(idx1)
        if n0 < 1 or n1_ < 1:
            return 0.0
        ss_w = (D2[np.ix_(idx0, idx0)].sum() / (2 * n0) +
                D2[np.ix_(idx1, idx1)].sum() / (2 * n1_))
        ss_b = SS_total - ss_w
        if ss_w <= 0:
            return 0.0
        return (ss_b / 1.0) / (ss_w / (N - 2))

    labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    F_obs = _pseudo_F(labels)
    ss_w_obs = (D2[np.ix_(np.arange(n1), np.arange(n1))].sum() / (2 * n1) +
                D2[np.ix_(np.arange(n1, N), np.arange(n1, N))].sum() / (2 * n2))
    ss_b_obs = SS_total - ss_w_obs
    R2 = float(ss_b_obs / SS_total) if SS_total > 0 else np.nan
    ms_w = ss_w_obs / (N - 2)
    omega2 = ((ss_b_obs - 1 * ms_w) / (SS_total + ms_w)
              if SS_total + ms_w > 0 else np.nan)

    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(labels)
        if _pseudo_F(perm) >= F_obs:
            count += 1
    p_perm = (count + 1) / (n_perms + 1)
    return {"pseudo_F": float(F_obs), "p_perm": float(p_perm),
            "R2": R2, "omega2": float(omega2) if not np.isnan(omega2) else np.nan,
            "n1": n1, "n2": n2, "metric": metric}


# ============================================================
# Fisher's method for combining p-values
# ============================================================

def fishers_method(pvals):
    pvals = np.asarray([p for p in pvals if not np.isnan(p)], dtype=float)
    pvals = np.clip(pvals, 1e-300, 1.0)
    if len(pvals) == 0:
        return {"chi2": np.nan, "df": 0, "p": np.nan, "k": 0}
    chi2 = -2.0 * np.log(pvals).sum()
    df = 2 * len(pvals)
    p = float(1 - stats.chi2.cdf(chi2, df))
    return {"chi2": float(chi2), "df": df, "p": p, "k": len(pvals)}


# ============================================================
# High vs low compare (used by run_cross_matched)
# ============================================================

def compare_groups(values_high, values_low, feature_name, stat_mode="auto",
                   paired_values=None):
    """Two-group compare with auto-fallback to MWU at small n.

    paired_values=True: also compute paired t-test (caller must pass aligned arrays).
    """
    vh = np.asarray(values_high, dtype=float)
    vl = np.asarray(values_low, dtype=float)
    vh_valid = vh[~np.isnan(vh)]
    vl_valid = vl[~np.isnan(vl)]
    n_h, n_l = len(vh_valid), len(vl_valid)

    base = {
        "feature": feature_name,
        "n_high": n_h, "n_low": n_l,
        "mean_high": vh_valid.mean() if n_h else np.nan,
        "std_high": vh_valid.std(ddof=1) if n_h > 1 else np.nan,
        "mean_low": vl_valid.mean() if n_l else np.nan,
        "std_low": vl_valid.std(ddof=1) if n_l > 1 else np.nan,
    }

    if n_h < 2 or n_l < 2:
        return {**base, "test": "skip", "stat": np.nan, "pvalue": np.nan,
                "cohen_d": np.nan, "hedges_g": np.nan,
                "paired_t_stat": np.nan, "paired_t_p": np.nan, "n_paired": 0}

    use_mwu = stat_mode == "mannwhitney" or (stat_mode == "auto" and min(n_h, n_l) < 20)
    if use_mwu:
        stat_val, pval = stats.mannwhitneyu(vh_valid, vl_valid,
                                            alternative="two-sided")
        test_name = "mannwhitney_u"
    else:
        stat_val, pval = stats.ttest_ind(vh_valid, vl_valid, equal_var=False)
        test_name = "welch_t"

    paired_stat, paired_p, n_paired = np.nan, np.nan, 0
    if paired_values is not None and len(vh) == len(vl):
        valid_mask = ~np.isnan(vh) & ~np.isnan(vl)
        if valid_mask.sum() >= 2:
            paired_stat, paired_p = stats.ttest_rel(vh[valid_mask], vl[valid_mask])
            n_paired = int(valid_mask.sum())

    return {
        **base,
        "test": test_name, "stat": float(stat_val), "pvalue": float(pval),
        "cohen_d": cohens_d(vh_valid, vl_valid),
        "hedges_g": hedges_g(vh_valid, vl_valid),
        "paired_t_stat": float(paired_stat) if not np.isnan(paired_stat) else np.nan,
        "paired_t_p": float(paired_p) if not np.isnan(paired_p) else np.nan,
        "n_paired": n_paired,
    }
