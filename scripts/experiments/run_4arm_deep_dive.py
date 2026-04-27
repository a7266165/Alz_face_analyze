"""
Section 3.5 Deep-dive: 4-arm × 4-comparison × 10-modality statistical grid.

Replaces the old v6.1 3.5 master table (AUC-only, 7×3 grid) with a richer
statistical characterization:
  - 10 modality parents (3 embedding mean, 3 embedding asymmetry — each split
    into L2-scalar + full-vector sub-rows, landmark — split into 4-d per-region
    L2 + 134-d raw pair diff, emotion, age_error, age_only) = 14 data rows
  - 12 comparison cells = 3 arm × 4 comparison
        Arm A (naive)                  : HC | NAD | ACS | hi-lo
        Arm B (cross-sec age-matched)  : HC | NAD | ACS | hi-lo
        Arm C (longitudinal Δ matched) : HC | NAD | ACS | hi-lo
  - Cell primary = per-modality statistical test (Welch t / Hotelling T² /
    PERMANOVA / per-method-Hotelling-Fisher) + effect size + q
  - Cell secondary (inline annotation) = AUC + 95% CI (reuse run_arm_a utilities)

Usage:
    "C:/Users/4080/anaconda3/envs/Alz_face_test_2/python.exe" \
        scripts/experiments/run_4arm_deep_dive.py
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_arm_a = _load_module("arm_a_ad_vs_hc",
                       PROJECT_ROOT / "scripts" / "experiments" / "run_arm_a_ad_vs_hc.py")
_arm_b = _load_module("mmse_hilo_standalone",
                       PROJECT_ROOT / "scripts" / "experiments" / "run_mmse_hilo_standalone.py")

cv_eval = _arm_a.cv_eval
bootstrap_auc_ci = _arm_a.bootstrap_auc_ci
load_cohort_a = _arm_a.load_cohort
load_emotion_matrix = _arm_a.load_emotion_matrix
load_landmark_matrix = _arm_a.load_landmark_matrix  # produces 4-region L2 + total_l2
load_embedding_mean = _arm_a.load_embedding_mean
load_embedding_asymmetry_l2 = _arm_a.load_embedding_asymmetry  # produces L2 scalars
cohens_d = _arm_a.cohens_d
match_1to1 = _arm_b.match_1to1
bh_fdr = _arm_b.bh_fdr
load_p_demographics = _arm_b.load_p_demographics

# === Paths ===
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
AGES_FILE = PROJECT_ROOT / "workspace" / "age" / "age_prediction" / "predicted_ages.json"
EMOTION_DIR = PROJECT_ROOT / "workspace" / "emotion" / "au_features" / "aggregated"
LANDMARK_FEATURES_CSV = PROJECT_ROOT / "workspace" / "asymmetry" / "features.csv"
EMBEDDING_DIR = PROJECT_ROOT / "workspace" / "embedding" / "features"
LONGITUDINAL_CSV = PROJECT_ROOT / "workspace" / "longitudinal" / "patient_deltas.csv"
LANDMARK_LONG_CSV = (PROJECT_ROOT / "workspace" / "asymmetry" / "analysis" /
                     "longitudinal_landmark_deltas.csv")
# New unified-schema files from build_longitudinal_hc_and_vectors.py
AD_DELTAS_CSV = (PROJECT_ROOT / "workspace" / "longitudinal" /
                  "ad_patient_deltas.csv")
HC_LONGITUDINAL_CSV = (PROJECT_ROOT / "workspace" / "longitudinal" /
                        "hc_patient_deltas.csv")
VECTOR_DELTAS_NPZ = (PROJECT_ROOT / "workspace" / "longitudinal" /
                      "vector_deltas.npz")
HC_SOURCE_MODE = os.environ.get("HC_SOURCE_MODE", "ACS")
# 允許由環境變數或 CLI 設定；CLI 在 __main__ 區塊解析並覆寫。
# 值：
#   "ACS"     → 只讀 data/demographics/ACS.csv（原行為）
#   "ACS_ext" → 讀 ACS.csv + EACS.csv，ACS 群體 = internal + external
#   "EACS"    → 只讀 EACS.csv，ACS 群體 = external only（strict HC 全 bypass）

GRID_ROOT = PROJECT_ROOT / "workspace" / "arms_analysis" / "grid"
# Baseline output：grid/<hc_source.lower()>/，CLI subset 覆寫見 __main__。
# mkdir 延後到 run_all() / __main__，避免 import 即建立非預期目錄。
OUTPUT_DIR = GRID_ROOT / HC_SOURCE_MODE.lower()

EMBEDDING_MODELS = ["arcface", "topofr", "dlib"]

# Modality direction grouping for per-direction grid subfolders.
# 5 directions: age / embedding_mean / embedding_asymmetry / landmark_asymmetry / emotion
DIRECTION_MAP = {
    "age_only": "age",
    "age_error": "age",
    "embedding_arcface_mean": "embedding_mean",
    "embedding_dlib_mean": "embedding_mean",
    "embedding_topofr_mean": "embedding_mean",
    "embedding_arcface_asymmetry": "embedding_asymmetry",
    "embedding_dlib_asymmetry": "embedding_asymmetry",
    "embedding_topofr_asymmetry": "embedding_asymmetry",
    "landmark_asymmetry": "landmark_asymmetry",
    "emotion_8methods": "emotion",
}
EMOTION_METHODS = ["openface", "libreface", "pyfeat", "dan",
                   "hsemotion", "vit", "poster_pp", "fer"]
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
STATS = ["mean", "std", "range", "entropy"]
LANDMARK_REGIONS = ["eye", "nose", "mouth", "face_oval"]

N_PERMS = int(os.environ.get("DEEP_DIVE_N_PERMS", 1000))
SEED = 42
MIN_CELL_N = 20  # gate: cells with either group < 20 marked n/a (relaxed from
                 # 25 to allow D:ACS n=21 pairs through; power is very weak at
                 # n=20-25 — treat those results as exploratory)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Multivariate test helpers
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


def hotelling_t2(X1, X2, n_perms=N_PERMS, seed=SEED):
    """Two-sample Hotelling's T². Returns T², F, p_F, p_perm, D² (Mahalanobis).

    Assumes n1+n2 > p. For p ≈ n, use PERMANOVA instead.
    """
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

    # Permutation p
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


def permanova(X1, X2, metric="euclidean", n_perms=N_PERMS, seed=SEED):
    """Permutational MANOVA (Anderson 2001).

    pseudo-F and permutation p-value; effect size R² = SS_b / SS_t.
    """
    X1 = np.asarray(X1, dtype=float); X2 = np.asarray(X2, dtype=float)
    mask1 = ~np.isnan(X1).any(axis=1); mask2 = ~np.isnan(X2).any(axis=1)
    X1, X2 = X1[mask1], X2[mask2]
    n1, n2 = len(X1), len(X2)
    N = n1 + n2
    if n1 < 2 or n2 < 2:
        return {"pseudo_F": np.nan, "p_perm": np.nan, "R2": np.nan,
                "omega2": np.nan, "n1": n1, "n2": n2, "metric": metric}

    X = np.vstack([X1, X2])
    # Precompute distance matrix
    if metric == "cosine":
        # Normalize rows then Euclidean on unit sphere == 2(1-cos)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        D = squareform(pdist(Xn, metric="euclidean"))
    else:
        D = squareform(pdist(X, metric=metric))
    D2 = D ** 2
    SS_total = D2.sum() / (2 * N)

    def _pseudo_F(labels):
        # labels in {0,1}
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        n0, n1_ = len(idx0), len(idx1)
        if n0 < 1 or n1_ < 1:
            return 0.0
        ss_w = (D2[np.ix_(idx0, idx0)].sum() / (2 * n0) +
                D2[np.ix_(idx1, idx1)].sum() / (2 * n1_))
        ss_b = SS_total - ss_w
        # pseudo-F per Anderson 2001: SS_b / (SS_w / (N-2))
        if ss_w <= 0:
            return 0.0
        return (ss_b / 1.0) / (ss_w / (N - 2))

    labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    F_obs = _pseudo_F(labels)
    ss_w_obs = (D2[np.ix_(np.arange(n1), np.arange(n1))].sum() / (2 * n1) +
                D2[np.ix_(np.arange(n1, N), np.arange(n1, N))].sum() / (2 * n2))
    ss_b_obs = SS_total - ss_w_obs
    R2 = float(ss_b_obs / SS_total) if SS_total > 0 else np.nan
    # Omega-squared (bias-corrected): 1 - (N-1) * MS_w / (SS_t + MS_w)
    ms_w = ss_w_obs / (N - 2)
    omega2 = (ss_b_obs - 1 * ms_w) / (SS_total + ms_w) if SS_total + ms_w > 0 else np.nan

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


def fishers_method(pvals):
    """Combine p-values via Fisher's method. Returns chi2, df, combined_p."""
    pvals = np.asarray([p for p in pvals if not np.isnan(p)], dtype=float)
    pvals = np.clip(pvals, 1e-300, 1.0)
    if len(pvals) == 0:
        return {"chi2": np.nan, "df": 0, "p": np.nan, "k": 0}
    chi2 = -2.0 * np.log(pvals).sum()
    df = 2 * len(pvals)
    p = float(1 - stats.chi2.cdf(chi2, df))
    return {"chi2": float(chi2), "df": df, "p": p, "k": len(pvals)}


def auc_with_ci(X, y, g=None, model_cls="xgb"):
    """Return AUC + 95% CI using run_arm_a's cv_eval + bootstrap."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    mask = ~np.isnan(X).any(axis=1) if X.ndim > 1 else ~np.isnan(X.ravel())
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X, y = X[mask], y[mask]
    if g is not None:
        g = np.asarray(g)[mask]
    if len(X) < 20 or len(np.unique(y)) < 2:
        return {"auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan,
                "n": int(mask.sum())}
    if g is None:
        g = np.arange(len(X))
    try:
        m = cv_eval(X, y, g, model_cls=model_cls, n_folds=5, seed=SEED,
                    return_preds=True)
        y_true = m.pop("y_true"); y_prob = m.pop("y_prob")
        lo, hi = bootstrap_auc_ci(y_true, y_prob, seed=SEED)
        return {"auc": float(m["auc"]), "auc_ci_low": float(lo),
                "auc_ci_high": float(hi), "n": int(mask.sum())}
    except Exception as e:
        logger.warning(f"AUC calc failed: {e}")
        return {"auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan,
                "n": int(mask.sum())}


# ============================================================
# Extended feature loaders
# ============================================================

def load_embedding_asymmetry_vec(ids, model):
    """Per-subject mean-pooled difference vector (512 or 128-d)."""
    rows = []
    for sid in ids:
        npy = EMBEDDING_DIR / model / "difference" / f"{sid}.npy"
        if not npy.exists():
            continue
        a = np.load(npy, allow_pickle=True)
        if a.dtype == object:
            a = list(a.item().values())[0]
        vec = a.mean(axis=0) if a.ndim == 2 else a
        rows.append({"subject_id": sid,
                     **{f"embasym_vec_{model}_{i}": float(vec[i])
                        for i in range(vec.shape[0])}})
    return pd.DataFrame(rows) if rows else None


def load_landmark_raw(ids):
    """Per-subject 130-d raw xy-pair diff vector (area_diff excluded).

    area_diff 由 4-d per-region L2 子列聚合觀察；此列只放 xy-only raw pairs
    避免 pixel vs pixel² scale 差 30× 導致 Euclidean PERMANOVA 被 area 主導。
    """
    df = pd.read_csv(LANDMARK_FEATURES_CSV)
    df = df[df["subject_id"].isin(ids)].copy()
    feat_cols = []
    for region in LANDMARK_REGIONS:
        feat_cols += [c for c in df.columns
                       if c.startswith(f"{region}_") and "area_diff" not in c]
    keep = ["subject_id"] + feat_cols
    return df[keep].copy()


def load_landmark_region_l2(ids):
    """Per-subject 4-d per-region L2 (no global summary)."""
    df = load_landmark_matrix(ids)
    if df is None or df.empty:
        return df
    cols = ["subject_id"] + [f"landmark__{r}_l2" for r in LANDMARK_REGIONS]
    return df[[c for c in cols if c in df.columns]].copy()


# ============================================================
# Cohort builders (arm × comparison = 12 combinations)
# ============================================================

_FEATURE_IDS_CACHE = None


def _ids_with_features():
    """Return set of visit-level IDs that have BOTH arcface embedding and
    landmark features (the two that drive Arm A/B modality n)."""
    global _FEATURE_IDS_CACHE
    if _FEATURE_IDS_CACHE is None:
        emb_dir = EMBEDDING_DIR / "arcface" / "original"
        emb_ids = {p.stem for p in emb_dir.glob("*.npy")}
        lmk = pd.read_csv(LANDMARK_FEATURES_CSV)
        lmk_ids = set(lmk["subject_id"].astype(str))
        _FEATURE_IDS_CACHE = emb_ids & lmk_ids
    return _FEATURE_IDS_CACHE


def _pick_first_visit_with_features(df_visits):
    """Per base_id, pick earliest visit that has both embedding + landmark
    features; fall back to earliest visit if none does."""
    good_ids = _ids_with_features()
    picked = []
    for bid, g in df_visits.groupby("base_id", as_index=False, sort=False):
        g = g.sort_values("visit")
        with_feat = g[g["ID"].astype(str).isin(good_ids)]
        if len(with_feat) > 0:
            picked.append(with_feat.iloc[0])
        else:
            picked.append(g.iloc[0])
    return pd.DataFrame(picked).reset_index(drop=True)


def _strict_hc_filter_all_visits(demo, hc_source):
    """Strict HC criteria per-visit (same rule as _strict_hc_filter) but keep
    ALL qualifying visits (not just first). Allows downstream
    _pick_first_visit_with_features to pick the earliest strict-qualifying
    visit that has embedding+landmark features.

    Source != 'internal' (外部公開資料集) 自動過 strict HC（age-only control）。
    """
    if hc_source == "HC":
        mask = demo["group"].isin(["NAD", "ACS"])
    else:
        mask = demo["group"] == hc_source
    sub = demo[mask].copy()
    sub["Global_CDR"] = pd.to_numeric(sub.get("Global_CDR"), errors="coerce")
    sub["MMSE"] = pd.to_numeric(sub.get("MMSE"), errors="coerce")
    has_cog = sub["Global_CDR"].notna() | sub["MMSE"].notna()
    ok_strict = has_cog & (
        (sub["Global_CDR"] == 0) |
        (sub["Global_CDR"].isna() & (sub["MMSE"] >= 26))
    )
    is_external = sub.get("Source", "internal") != "internal"
    ok = ok_strict | is_external
    sub = sub[ok].copy()
    sub = sub.sort_values(["base_id", "visit"])
    sub["label"] = 0
    return sub


def _strict_hc_filter(demo, hc_source):
    """Apply strict HC filter (CDR=0 or MMSE>=26) on NAD/ACS subset.

    hc_source in {'HC', 'NAD', 'ACS'}; 'HC' = NAD ∪ ACS.
    Source != 'internal' 外部公開資料集自動過 strict HC。
    Returns DataFrame with one visit per base_id (first) and label=0.
    """
    if hc_source == "HC":
        mask = demo["group"].isin(["NAD", "ACS"])
    else:
        mask = demo["group"] == hc_source
    sub = demo[mask].copy()
    sub["Global_CDR"] = pd.to_numeric(sub.get("Global_CDR"), errors="coerce")
    sub["MMSE"] = pd.to_numeric(sub.get("MMSE"), errors="coerce")
    has_cog = sub["Global_CDR"].notna() | sub["MMSE"].notna()
    ok_strict = has_cog & (
        (sub["Global_CDR"] == 0) |
        (sub["Global_CDR"].isna() & (sub["MMSE"] >= 26))
    )
    is_external = sub.get("Source", "internal") != "internal"
    ok = ok_strict | is_external
    sub = sub[ok].copy()
    sub = sub.sort_values(["base_id", "visit"]).groupby("base_id", as_index=False).first()
    sub["label"] = 0
    return sub


def build_cohort_ad_vs_HCgroup(hc_source, arm="A", caliper=2.0, seed=SEED):
    """Build cohort for AD vs {HC, NAD, ACS}. Returns (cohort_df, pairs_df).

    arm='A': no matching.
    arm='B': 1:1 age NN on first visit (caliper 2y).
    arm='C': longitudinal Δ — requires multi-visit on BOTH sides.
    """
    # Load full demographics across groups. HC_SOURCE_MODE 控制 ACS 群體的組成：
    #   ACS     → 只讀 ACS.csv
    #   ACS_ext → 讀 ACS.csv + EACS.csv（都歸入 ACS 群）
    #   EACS    → 只讀 EACS.csv（取代內部 ACS）
    frames = []
    groups_to_load = ["P", "NAD"]
    if HC_SOURCE_MODE != "EACS":
        groups_to_load.append("ACS")
    for grp in groups_to_load:
        df = pd.read_csv(DEMOGRAPHICS_DIR / f"{grp}.csv")
        if "ID" not in df.columns:
            for col in df.columns:
                if col in ("ACS", "NAD"):
                    df = df.rename(columns={col: "ID"})
                    break
        df["group"] = grp
        df["Source"] = "internal"
        frames.append(df)
    if HC_SOURCE_MODE in ("ACS_ext", "EACS"):
        df_e = pd.read_csv(DEMOGRAPHICS_DIR / "EACS.csv")
        # 可選擇性 filter EACS source 子集（環境變數 EACS_SOURCES="UTKFace,MegaAge"）
        eacs_sources_env = os.environ.get("EACS_SOURCES", "").strip()
        if eacs_sources_env:
            wanted = {s.strip() for s in eacs_sources_env.split(",") if s.strip()}
            if "Source" in df_e.columns:
                df_e = df_e[df_e["Source"].isin(wanted)].copy()
                logger.info(f"filtered EACS to sources {wanted}: {len(df_e)} rows")
        df_e["group"] = "ACS"  # external 也歸入 ACS 群
        # Source 欄已存在於 EACS.csv
        frames.append(df_e)
    demo = pd.concat(frames, ignore_index=True)
    if "Source" not in demo.columns:
        demo["Source"] = "internal"
    demo["Source"] = demo["Source"].fillna("internal")
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["Global_CDR"] = pd.to_numeric(demo.get("Global_CDR"), errors="coerce")
    demo["MMSE"] = pd.to_numeric(demo.get("MMSE"), errors="coerce")
    # base_id 用泛型 regex 支援 EACS_{SRC}_xxx-N 格式
    demo["base_id"] = demo["ID"].str.extract(r"^(.+)-\d+$")
    demo["visit"] = demo["ID"].str.extract(r"-(\d+)$").astype(float)

    # AD — pick first visit, but if the first visit lacks embedding/landmark
    # features, fall back to the earliest visit that has both. Recovers 3 of 7
    # AD subjects whose visit-1 is missing from the raw dataset.
    ad_all = demo[(demo["group"] == "P") & (demo["Global_CDR"] >= 0.5) &
                    demo["Age"].notna()].copy()
    ad_all = ad_all.sort_values(["base_id", "visit"])
    ad = _pick_first_visit_with_features(ad_all)
    ad["label"] = 1

    hc_all = _strict_hc_filter_all_visits(demo, hc_source)
    hc_all = hc_all[hc_all["Age"].notna()].copy()
    hc = _pick_first_visit_with_features(hc_all)

    if arm == "A":
        cohort = pd.concat([ad, hc], ignore_index=True)
        cohort["mmse_group"] = np.nan  # not applicable for AD vs HC
        return cohort, None

    if arm == "B":
        # 1:1 age NN match; cohort fed to match_1to1 must have ID, Age,
        # mmse_group. We use label (1=AD, 0=HC) as the matching group key
        # -- adapt match_1to1.
        prep = pd.concat([ad, hc], ignore_index=True)
        prep["mmse_group"] = np.where(prep["label"] == 1, "high", "low")
        prep["MMSE"] = prep["MMSE"].fillna(999)  # placeholder (match_1to1 reads Age)
        matched, pairs, _ = match_1to1(prep, caliper=caliper, seed=seed)
        cohort = matched.merge(prep[["ID", "base_id", "group", "Age", "MMSE",
                                       "Global_CDR", "label"]].drop_duplicates("ID"),
                                on="ID", how="left", suffixes=("", "_p"))
        cohort = cohort.drop(columns=[c for c in cohort.columns if c.endswith("_p")])
        return cohort, pairs

    if arm in ("C", "D"):
        # C = longitudinal naive (no age matching)
        # D = longitudinal matched (baseline age 1:1 NN)
        do_match = (arm == "D")
        return _build_longitudinal_cohort_hc(hc_source, demo, caliper, seed,
                                               do_match=do_match)

    raise ValueError(f"unknown arm: {arm}")


def _build_longitudinal_cohort_hc(hc_source, demo, caliper, seed, do_match=True):
    """Longitudinal AD vs {HC/NAD/ACS} cohort. do_match=True → Arm D (1:1 NN
    on baseline_age); do_match=False → Arm C (naive, no matching).

    Uses:
      - workspace/longitudinal/ad_patient_deltas.csv (AD side)
      - workspace/longitudinal/hc_patient_deltas.csv (HC side)
    Both have ann_* annualized Δ cols.
    """
    # AD side — prefer the unified-schema ad_patient_deltas.csv (has lmk_*
    # + ann_* already); fall back to annualizing legacy patient_deltas.csv.
    if AD_DELTAS_CSV.exists():
        ad_delta = pd.read_csv(AD_DELTAS_CSV)
        ad_delta = ad_delta[(ad_delta["follow_up_days"] >= 180) &
                             ad_delta["first_age"].notna()].copy()
    else:
        ad_delta = pd.read_csv(LONGITUDINAL_CSV)
        ad_delta = ad_delta[(ad_delta["follow_up_days"] >= 180) &
                             ad_delta["first_age"].notna()].copy()
        ad_delta = _annualize_patient_deltas(ad_delta)
    ad_delta["label"] = 1
    if "group" not in ad_delta.columns:
        ad_delta["group"] = "P"

    # HC side — already annualized during pipeline build
    if not HC_LONGITUDINAL_CSV.exists():
        raise FileNotFoundError(
            f"Missing {HC_LONGITUDINAL_CSV}; run "
            f"build_longitudinal_hc_and_vectors.py first")
    hc_delta_all = pd.read_csv(HC_LONGITUDINAL_CSV)

    # External EACS delta（當 HC_SOURCE_MODE 指定 ACS_ext / EACS）
    eacs_csv = (PROJECT_ROOT / "workspace" / "longitudinal" /
                 "eacs_patient_deltas.csv")
    if HC_SOURCE_MODE in ("ACS_ext", "EACS") and eacs_csv.exists():
        eacs_delta_all = pd.read_csv(eacs_csv)
        eacs_delta_all["group"] = "ACS"  # 併入 ACS 群
        if HC_SOURCE_MODE == "EACS":
            hc_delta_all = eacs_delta_all
        else:
            hc_delta_all = pd.concat([hc_delta_all, eacs_delta_all],
                                       ignore_index=True, sort=False)

    # Filter by requested HC subgroup + strict HC criteria (CDR=0 or MMSE>=26
    # in first visit)；Source != "internal" 在 _strict_hc_filter 已自動 bypass
    hc_strict = _strict_hc_filter(demo, hc_source)
    allowed_base = set(hc_strict["base_id"])
    hc_delta = hc_delta_all[hc_delta_all["base_id"].isin(allowed_base)].copy()
    if hc_source in ("NAD", "ACS"):
        hc_delta = hc_delta[hc_delta["group"] == hc_source]
    hc_delta["label"] = 0

    if hc_delta.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_delta = pd.concat([ad_delta, hc_delta], ignore_index=True, sort=False)

    if not do_match:
        # Arm C: naive — return all multi-visit subjects, no matching
        cohort = all_delta.copy()
        cohort["mmse_group"] = np.where(cohort["label"] == 1, "high", "low")
        return cohort, None

    # Arm D: 1:1 baseline-age match
    ad_delta["mmse_group"] = "high"  # drives match_1to1 split
    hc_delta["mmse_group"] = "low"
    prep = pd.concat([
        ad_delta[["base_id", "first_age", "mmse_group",
                  "follow_up_years", "follow_up_days", "label"]].rename(
                      columns={"first_age": "Age"}),
        hc_delta[["base_id", "first_age", "mmse_group",
                  "follow_up_years", "follow_up_days", "label"]].rename(
                      columns={"first_age": "Age"}),
    ], ignore_index=True)
    matched, pairs_df = _match_longitudinal_1to1(prep, caliper, seed)
    cohort = matched.merge(all_delta, on="base_id", how="left",
                             suffixes=("", "_p"))
    return cohort, pairs_df


def _annualize_patient_deltas(df):
    """Add ann_* columns to AD patient_deltas-style DataFrame, mirroring the
    logic in build_longitudinal_hc_and_vectors so Arm C × {HC/NAD/ACS} and
    Arm C × hi-lo use identical schemas."""
    df = df.copy()
    y = df["follow_up_years"].replace(0, np.nan)
    for col in list(df.columns):
        if (col.startswith("delta_") or col.startswith("emb_cosine_dist_")
                or col == "emb_cosine_dist"):
            df[f"ann_{col}"] = df[col] / y
    if "ann_emb_cosine_dist_arcface" in df.columns:
        df["ann_emb_cosine_dist"] = df["ann_emb_cosine_dist_arcface"]
    return df


def _match_longitudinal_1to1(prep, caliper, seed):
    """Local 1:1 age NN matcher using 'Age' column."""
    rng = np.random.RandomState(seed)
    g_hi = prep[prep["mmse_group"] == "high"].copy()
    g_lo = prep[prep["mmse_group"] == "low"].copy()
    if len(g_lo) <= len(g_hi):
        minor, major = g_lo, g_hi
        minor_lbl, major_lbl = "low", "high"
    else:
        minor, major = g_hi, g_lo
        minor_lbl, major_lbl = "high", "low"
    minor_order = minor.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    available = major.copy().reset_index(drop=True)

    pairs = []
    for _, row in minor_order.iterrows():
        if len(available) == 0:
            break
        age = row["Age"]
        diffs = (available["Age"] - age).abs()
        md = diffs.min()
        if md > caliper:
            continue
        cands = available[diffs == md].sort_values("base_id")
        pk = cands.iloc[0]
        pairs.append({
            "pair_id": len(pairs),
            "minor_id": row["base_id"], "minor_age": age,
            "major_id": pk["base_id"], "major_age": pk["Age"],
            "age_diff": pk["Age"] - age,
        })
        available = available.drop(pk.name).reset_index(drop=True)
    pairs_df = pd.DataFrame(pairs)

    records = []
    for _, p in pairs_df.iterrows():
        records.append({"pair_id": p["pair_id"], "base_id": p["minor_id"],
                         "mmse_group": minor_lbl})
        records.append({"pair_id": p["pair_id"], "base_id": p["major_id"],
                         "mmse_group": major_lbl})
    return pd.DataFrame(records), pairs_df


def build_cohort_ad_hi_lo(arm="A", caliper=2.0, seed=SEED, metric="MMSE"):
    """AD-internal hi-lo cohort, per arm. metric ∈ {"MMSE","CASI"}.

    For B (cross-sec matched): reads pre-built matched cohort from
    arm_b/<metric.lower()>_high_vs_low/matched_features.csv.
    """
    metric_low = metric.lower()
    group_col = f"{metric_low}_group"
    first_metric = f"first_{metric}"
    if arm == "A":
        demo = load_p_demographics()
        cohort = demo.dropna(subset=[metric, "Age"]).copy()
        cohort = cohort.sort_values(["base_id", "visit"]).groupby(
            "base_id", as_index=False).first()
        # Filter by AD: CDR>=0.5
        cohort["Global_CDR"] = pd.to_numeric(cohort.get("Global_CDR"),
                                                errors="coerce")
        cohort = cohort[cohort["Global_CDR"] >= 0.5].copy()
        med = cohort[metric].median()
        cohort[group_col] = np.where(cohort[metric] >= med, "high", "low")
        cohort["label"] = (cohort[group_col] == "high").astype(int)
        return cohort, None
    if arm == "B":
        arm_b_csv = (PROJECT_ROOT / "workspace" / "arms_analysis" / "per_arm" /
                     "arm_b" / f"{metric_low}_high_vs_low" /
                     "matched_features.csv")
        if not arm_b_csv.exists():
            raise FileNotFoundError(
                f"Run run_mmse_hilo_standalone.py with HILO_METRIC={metric} "
                f"first")
        cohort = pd.read_csv(arm_b_csv)
        cohort["label"] = (cohort[group_col] == "high").astype(int)
        p_df = pd.read_csv(DEMOGRAPHICS_DIR / "P.csv")
        keep_cog = [c for c in ["MMSE", "CASI", "Global_CDR", "CDR_SB"]
                    if c in p_df.columns and c not in cohort.columns]
        if keep_cog:
            cohort = cohort.merge(p_df[["ID"] + keep_cog].drop_duplicates("ID"),
                                    on="ID", how="left")
            for c in keep_cog:
                cohort[c] = pd.to_numeric(cohort[c], errors="coerce")
        return cohort, None
    if arm == "C":
        # Longitudinal naive hi-lo: all multi-visit AD, metric median split, no matching.
        if not AD_DELTAS_CSV.exists():
            raise FileNotFoundError(
                f"Missing {AD_DELTAS_CSV}; run "
                f"build_longitudinal_hc_and_vectors.py first")
        ad_d = pd.read_csv(AD_DELTAS_CSV)
        ad_d = ad_d[ad_d["follow_up_days"] >= 180].copy()
        ad_d = ad_d[ad_d[first_metric].notna() & ad_d["first_age"].notna()]
        med = ad_d[first_metric].median()
        ad_d[group_col] = np.where(ad_d[first_metric] >= med, "high", "low")
        ad_d["label"] = (ad_d[group_col] == "high").astype(int)
        return ad_d, None
    if arm == "D":
        # Longitudinal matched hi-lo: same as C but +1:1 baseline-age matching.
        if not AD_DELTAS_CSV.exists():
            raise FileNotFoundError(
                f"Missing {AD_DELTAS_CSV}; run "
                f"build_longitudinal_hc_and_vectors.py first")
        ad_d = pd.read_csv(AD_DELTAS_CSV)
        ad_d = ad_d[ad_d["follow_up_days"] >= 180].copy()
        ad_d = ad_d[ad_d[first_metric].notna() & ad_d["first_age"].notna()]
        med = ad_d[first_metric].median()
        ad_d[group_col] = np.where(ad_d[first_metric] >= med, "high", "low")
        # _match_longitudinal_1to1 looks for "mmse_group" hardcoded — pass
        # alias column (kept under both names so downstream is happy).
        prep = ad_d.copy()
        prep["Age"] = prep["first_age"]
        if group_col != "mmse_group":
            prep["mmse_group"] = prep[group_col]
        matched, pairs_df = _match_longitudinal_1to1(prep, caliper, seed)
        if "mmse_group" in matched.columns and group_col != "mmse_group":
            matched = matched.rename(columns={"mmse_group": group_col})
        cohort = matched.merge(ad_d, on="base_id", how="left",
                                suffixes=("", "_p"))
        cohort["label"] = (cohort[group_col] == "high").astype(int)
        return cohort, pairs_df
    raise ValueError(f"unknown arm: {arm}")


# ============================================================
# Modality dispatch
# ============================================================

def _group_split(X_df, cohort, feat_cols, label_col="label"):
    """Align features with cohort labels; returns (X1, X2, ids1, ids2)."""
    m = cohort[[label_col] + (["ID"] if "ID" in cohort.columns else ["base_id"])]
    id_col = "ID" if "ID" in cohort.columns else "base_id"
    feat_id_col = "subject_id" if "subject_id" in X_df.columns else "base_id"
    merged = m.merge(X_df, left_on=id_col, right_on=feat_id_col, how="inner")
    feat_cols_present = [c for c in feat_cols if c in merged.columns]
    merged = merged.dropna(subset=feat_cols_present)
    X = merged[feat_cols_present].to_numpy(dtype=float)
    y = merged[label_col].to_numpy(dtype=int)
    ids = merged[id_col].to_numpy()
    X1 = X[y == 1]; X2 = X[y == 0]
    return X1, X2, merged, feat_cols_present


def run_scalar_modality(name, X_df, cohort):
    feat_cols = [c for c in X_df.columns if c not in ("subject_id", "base_id", "ID")]
    if len(feat_cols) != 1:
        raise ValueError(f"scalar modality {name} must be 1 feature")
    X1, X2, merged, feat_cols = _group_split(X_df, cohort, feat_cols)
    if len(X1) < MIN_CELL_N or len(X2) < MIN_CELL_N:
        return {"modality": name, "n": len(merged), "test": "welch_t",
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "effect_type": "cohens_d"}
    res = welch_t_test(X1[:, 0], X2[:, 0])
    auc_row = auc_with_ci(merged[feat_cols].values, merged["label"].values,
                           model_cls="logistic")
    return {
        "modality": name, "n": res["n1"] + res["n2"],
        "n1": res["n1"], "n2": res["n2"],
        "test": "welch_t", "statistic": res["t"],
        "p": res["p_welch"], "p_secondary": res["p_mw"],
        "effect": res["d"], "effect_type": "cohens_d",
        "mean1": res["mean1"], "mean2": res["mean2"],
        **{f"auc_{k}": v for k, v in auc_row.items()},
    }


def run_hotelling_modality(name, X_df, cohort, n_perms=N_PERMS):
    feat_cols = [c for c in X_df.columns if c not in ("subject_id", "base_id", "ID")]
    X1, X2, merged, feat_cols = _group_split(X_df, cohort, feat_cols)
    if len(X1) < MIN_CELL_N or len(X2) < MIN_CELL_N:
        return {"modality": name, "n": len(merged), "test": "hotelling_t2",
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "effect_type": "mahalanobis_D2"}
    res = hotelling_t2(X1, X2, n_perms=n_perms)
    auc_row = auc_with_ci(merged[feat_cols].values, merged["label"].values,
                           model_cls="xgb")
    return {
        "modality": name, "n": res["n1"] + res["n2"],
        "n1": res["n1"], "n2": res["n2"], "dim": res["p"],
        "test": "hotelling_t2", "statistic": res["T2"],
        "p": res["p_perm"], "p_secondary": res["p_F"],
        "effect": res["D2"], "effect_type": "mahalanobis_D2",
        **{f"auc_{k}": v for k, v in auc_row.items()},
    }


def run_permanova_modality(name, X_df, cohort, metric="euclidean",
                             n_perms=N_PERMS):
    feat_cols = [c for c in X_df.columns if c not in ("subject_id", "base_id", "ID")]
    X1, X2, merged, feat_cols = _group_split(X_df, cohort, feat_cols)
    if len(X1) < MIN_CELL_N or len(X2) < MIN_CELL_N:
        return {"modality": name, "n": len(merged), "test": "permanova",
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "effect_type": "R2"}
    res = permanova(X1, X2, metric=metric, n_perms=n_perms)
    auc_row = auc_with_ci(merged[feat_cols].values, merged["label"].values,
                           model_cls="xgb")
    return {
        "modality": name, "n": res["n1"] + res["n2"],
        "n1": res["n1"], "n2": res["n2"], "dim": len(feat_cols),
        "test": f"permanova_{metric}", "statistic": res["pseudo_F"],
        "p": res["p_perm"],
        "effect": res["R2"], "effect_type": "R2",
        "omega2": res["omega2"],
        **{f"auc_{k}": v for k, v in auc_row.items()},
    }


def run_emotion_modality(name, X_df, cohort, n_perms=N_PERMS):
    """Emotion: 8-method per-method Hotelling + Fisher combine."""
    all_feat_cols = [c for c in X_df.columns
                     if c not in ("subject_id", "base_id", "ID")]
    # Per method
    per_method_p = []
    per_method_stat = []
    per_method_n = []
    for method in EMOTION_METHODS:
        mcols = [c for c in all_feat_cols if c.startswith(f"{method}__")]
        if not mcols:
            continue
        sub_df = X_df[["subject_id"] + mcols]
        X1, X2, merged, mcols_p = _group_split(sub_df, cohort, mcols)
        if len(X1) < MIN_CELL_N or len(X2) < MIN_CELL_N or len(mcols_p) < 2:
            continue
        res = hotelling_t2(X1, X2, n_perms=n_perms)
        per_method_p.append(res["p_perm"])
        per_method_stat.append(res["T2"])
        per_method_n.append(res["n1"] + res["n2"])

    fish = fishers_method(per_method_p)
    # AUC still on the full 224-d
    X1_full, X2_full, merged_full, feat_cols = _group_split(
        X_df, cohort, all_feat_cols)
    auc_row = auc_with_ci(merged_full[feat_cols].values,
                           merged_full["label"].values, model_cls="xgb")
    return {
        "modality": name,
        "n": int(np.mean(per_method_n)) if per_method_n else len(merged_full),
        "dim": len(all_feat_cols),
        "test": "hotelling_per_method_fisher",
        "statistic": float(np.mean(per_method_stat)) if per_method_stat else np.nan,
        "p": fish["p"], "p_secondary": fish["chi2"],
        "effect": float(np.mean(per_method_stat)) if per_method_stat else np.nan,
        "effect_type": "mean_T2",
        "k_methods_used": fish["k"],
        **{f"auc_{k}": v for k, v in auc_row.items()},
    }


# ============================================================
# Orchestrator
# ============================================================

MODALITY_SPECS = [
    # (parent, sub_label, test_kind, extra)
    ("age_only", None, "scalar_age", {}),
    ("embedding_arcface_mean", None, "permanova_cosine", {"model": "arcface"}),
    ("embedding_dlib_mean", None, "permanova_euclidean", {"model": "dlib"}),
    ("embedding_topofr_mean", None, "permanova_euclidean", {"model": "topofr"}),
    ("embedding_arcface_asymmetry", "L2 scalar", "scalar_embasym_l2",
     {"model": "arcface"}),
    ("embedding_arcface_asymmetry", "full vector", "permanova_cosine_embasym",
     {"model": "arcface"}),
    ("embedding_dlib_asymmetry", "L2 scalar", "scalar_embasym_l2",
     {"model": "dlib"}),
    ("embedding_dlib_asymmetry", "full vector", "permanova_euclidean_embasym",
     {"model": "dlib"}),
    ("embedding_topofr_asymmetry", "L2 scalar", "scalar_embasym_l2",
     {"model": "topofr"}),
    ("embedding_topofr_asymmetry", "full vector", "permanova_euclidean_embasym",
     {"model": "topofr"}),
    ("landmark_asymmetry", "4-d per-region L2", "hotelling_landmark4", {}),
    ("landmark_asymmetry", "130-d raw xy", "permanova_euclidean_landmark130", {}),
    ("emotion_8methods", None, "emotion_fisher", {}),
    ("age_error", None, "scalar_age_error", {}),
]


def _get_subject_ids(cohort):
    if "ID" in cohort.columns:
        return cohort["ID"].dropna().astype(str).tolist()
    return cohort["base_id"].dropna().astype(str).tolist()


def _load_age_error(ids):
    """Compute age_error = Age - predicted_age per subject_id."""
    with open(AGES_FILE) as f:
        pred = json.load(f)
    demo_all = []
    for grp in ["P", "NAD", "ACS"]:
        df = pd.read_csv(DEMOGRAPHICS_DIR / f"{grp}.csv")
        if "ID" not in df.columns:
            for col in df.columns:
                if col in ("ACS", "NAD"):
                    df = df.rename(columns={col: "ID"}); break
        demo_all.append(df[["ID", "Age"]])
    demo_all = pd.concat(demo_all, ignore_index=True).drop_duplicates("ID")
    demo_all["Age"] = pd.to_numeric(demo_all["Age"], errors="coerce")
    rows = []
    for sid in ids:
        row = demo_all[demo_all["ID"] == sid]
        if len(row) == 0:
            continue
        a = float(row.iloc[0]["Age"])
        pa = pred.get(sid)
        if pa is None or np.isnan(a):
            continue
        rows.append({"subject_id": sid, "age_error": a - float(pa)})
    return pd.DataFrame(rows)


def _feature_df_for_modality(test_kind, extra, cohort, arm):
    """Load the feature DF for a given modality spec and cohort.

    For arm 'C' (longitudinal naive) and 'D' (longitudinal matched), features
    come from the cohort's pre-computed ann_* Δ columns. For arm 'A'/'B',
    features come from raw npy/csv loaders keyed by visit-level ID.
    """
    if arm in ("C", "D"):
        return _arm_c_feature_df(test_kind, extra, cohort)

    # Arm A/B: visit-level ID
    ids = _get_subject_ids(cohort)
    if test_kind == "scalar_age":
        if "Age" in cohort.columns:
            idc = "ID" if "ID" in cohort.columns else "base_id"
            return cohort[[idc, "Age"]].rename(
                columns={idc: "subject_id", "Age": "value_age"})
        return None
    if test_kind == "scalar_age_error":
        return _load_age_error(ids)
    if test_kind == "scalar_embasym_l2":
        df = load_embedding_asymmetry_l2(ids)
        if df is None or df.empty:
            return None
        col = f"embasym__{extra['model']}_l2"
        if col not in df.columns:
            return None
        return df[["subject_id", col]].copy()
    if test_kind.endswith("embasym"):
        return load_embedding_asymmetry_vec(ids, extra["model"])
    if test_kind == "hotelling_landmark4":
        return load_landmark_region_l2(ids)
    if test_kind == "permanova_euclidean_landmark130":
        return load_landmark_raw(ids)
    if test_kind in ("permanova_cosine", "permanova_euclidean"):
        return load_embedding_mean(ids, extra["model"])
    if test_kind == "emotion_fisher":
        # TODO (1): filter to _mean only → 8 methods × 7 emotions = 56 features
        # (was 8 × 7 × 4 = 224; std/range/entropy dropped).
        emo_df = load_emotion_matrix(ids)
        if emo_df is None:
            return None
        keep_cols = ["subject_id"] + [c for c in emo_df.columns
                                         if c != "subject_id" and c.endswith("_mean")]
        return emo_df[keep_cols].copy()
    return None


_VECTOR_DELTAS_CACHE = None


def _get_vector_deltas():
    """Lazy-load vector_deltas.npz once; returns (base_ids ndarray, dict key→matrix)."""
    global _VECTOR_DELTAS_CACHE
    if _VECTOR_DELTAS_CACHE is None:
        if not VECTOR_DELTAS_NPZ.exists():
            _VECTOR_DELTAS_CACHE = (None, {})
            return _VECTOR_DELTAS_CACHE
        z = np.load(VECTOR_DELTAS_NPZ, allow_pickle=False)
        base_ids = z["base_ids"]
        mats = {k: z[k] for k in z.files if k != "base_ids"}
        _VECTOR_DELTAS_CACHE = (base_ids, mats)
    return _VECTOR_DELTAS_CACHE


def _load_vector_delta(cohort, vec_key):
    """Return feature DF with subject_id + full-vector Δ cols from NPZ."""
    base_ids, mats = _get_vector_deltas()
    if base_ids is None or vec_key not in mats:
        return None
    mat = mats[vec_key]
    # Build index from base_id -> row index
    bid_to_i = {str(b): i for i, b in enumerate(base_ids)}
    idc = "base_id" if "base_id" in cohort.columns else "ID"
    out_rows = []
    cohort_bids = cohort[idc].astype(str).values
    D = mat.shape[1]
    for bid in cohort_bids:
        i = bid_to_i.get(bid)
        if i is None:
            continue
        vec = mat[i]
        if np.any(np.isnan(vec)):
            continue
        row = {"subject_id": bid}
        for d in range(D):
            row[f"{vec_key}_{d}"] = float(vec[d])
        out_rows.append(row)
    if not out_rows:
        return None
    return pd.DataFrame(out_rows)


def _arm_c_feature_df(test_kind, extra, cohort):
    """Build feature DF for Arm C using pre-computed ann_* Δ columns.

    Cohort is matched_features_longitudinal.csv (for hi-lo). For AD vs
    {HC,NAD,ACS}, HC Δ aren't pre-computed → return None → cell becomes n/a.
    """
    has_ann = any(c.startswith("ann_") for c in cohort.columns)
    if not has_ann:
        return None
    idc = "base_id"
    if test_kind == "scalar_age":
        # In Arm C, age_only = follow_up_years (rate-level proxy) or first_age
        return cohort[[idc, "first_age"]].rename(
            columns={idc: "subject_id", "first_age": "value_age"})
    if test_kind == "scalar_age_error":
        col = "ann_delta_age_error"
        if col not in cohort.columns:
            return None
        return cohort[[idc, col]].rename(columns={idc: "subject_id"})
    if test_kind == "scalar_embasym_l2":
        col = f"ann_delta_embasym_{extra['model']}"
        if col not in cohort.columns:
            return None
        return cohort[[idc, col]].rename(columns={idc: "subject_id"})
    if test_kind.endswith("embasym"):
        # Full vector Δ: load from vector_deltas.npz
        return _load_vector_delta(cohort, f"emb_asym_delta_vec_{extra['model']}")
    if test_kind == "hotelling_landmark4":
        cols = [f"ann_lmk_delta_{r}_l2" for r in LANDMARK_REGIONS]
        avail = [c for c in cols if c in cohort.columns]
        if len(avail) < 4:
            return None
        return cohort[[idc] + avail].rename(columns={idc: "subject_id"})
    if test_kind == "permanova_euclidean_landmark130":
        # 130-d raw xy pair Δ from vector_deltas.npz
        return _load_vector_delta(cohort, "lmk_raw_xy_delta")
    if test_kind in ("permanova_cosine", "permanova_euclidean"):
        # Mean embedding Δ as full vector (last_mean - first_mean), loaded
        # from vector_deltas.npz. Metric selection (cosine for arcface,
        # Euclidean for dlib/topofr) handled by _dispatch_test via test_kind.
        return _load_vector_delta(cohort, f"emb_drift_vec_{extra['model']}")
    if test_kind == "emotion_fisher":
        # 56 ann_delta_{method}__{emo} columns — rename to {method}__{emo}_mean
        # so run_emotion_modality's per-method filter (startswith "{method}__")
        # works the same as A/B.
        cols = [f"ann_delta_{m}__{e}" for m in EMOTION_METHODS for e in EMOTIONS]
        avail = [c for c in cols if c in cohort.columns]
        if len(avail) < 7:
            return None
        sub = cohort[[idc] + avail].rename(columns={idc: "subject_id"})
        rename = {f"ann_delta_{m}__{e}": f"{m}__{e}_mean"
                  for m in EMOTION_METHODS for e in EMOTIONS
                  if f"ann_delta_{m}__{e}" in sub.columns}
        sub = sub.rename(columns=rename)
        return sub
    return None


def _dispatch_test(test_kind, extra, name, X_df, cohort, arm):
    if X_df is None or (hasattr(X_df, "empty") and X_df.empty):
        return {"modality": name, "n": 0, "test": test_kind,
                "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "skip_reason": "feature unavailable"}

    # For Arm C/D (longitudinal) — respect test_kind so metric choice
    # (cosine for ArcFace, Euclidean for Dlib/TopoFR) matches A/B.
    if arm in ("C", "D"):
        n_feat = len([c for c in X_df.columns if c not in ("subject_id",)])
        if n_feat == 1:
            return run_scalar_modality(name, X_df, cohort)
        # emotion_fisher may have n_feat > 10 when we expand to 56-d (TODO 2)
        if test_kind == "emotion_fisher":
            return run_emotion_modality(name, X_df, cohort)
        if n_feat <= 10:
            return run_hotelling_modality(name, X_df, cohort)
        if test_kind == "permanova_cosine" or test_kind.endswith("cosine_embasym"):
            return run_permanova_modality(name, X_df, cohort, metric="cosine")
        return run_permanova_modality(name, X_df, cohort, metric="euclidean")

    # Arm A / B: use declared test kind
    if test_kind in ("scalar_age", "scalar_age_error", "scalar_embasym_l2"):
        return run_scalar_modality(name, X_df, cohort)
    if test_kind == "hotelling_landmark4":
        return run_hotelling_modality(name, X_df, cohort)
    if test_kind == "permanova_cosine" or test_kind.endswith("cosine_embasym"):
        return run_permanova_modality(name, X_df, cohort, metric="cosine")
    if test_kind.startswith("permanova_euclidean"):
        return run_permanova_modality(name, X_df, cohort, metric="euclidean")
    if test_kind == "emotion_fisher":
        return run_emotion_modality(name, X_df, cohort)
    return {"modality": name, "n": 0, "test": test_kind,
            "statistic": np.nan, "p": np.nan, "effect": np.nan}


_DEMO_VISIT_CACHE = None


def _all_visits_per_base_id():
    """Cached: DataFrame with base_id column for every visit across P/NAD/ACS.
    Used to compute n_all (total raw visits) for a cohort's base_ids."""
    global _DEMO_VISIT_CACHE
    if _DEMO_VISIT_CACHE is None:
        frames = []
        for grp in ["P", "NAD", "ACS"]:
            df = pd.read_csv(DEMOGRAPHICS_DIR / f"{grp}.csv")
            if "ID" not in df.columns:
                for col in df.columns:
                    if col in ("ACS", "NAD"):
                        df = df.rename(columns={col: "ID"})
                        break
            df["base_id"] = df["ID"].astype(str).str.extract(r"^([A-Za-z]+\d+)")
            frames.append(df[["ID", "base_id"]])
        _DEMO_VISIT_CACHE = pd.concat(frames, ignore_index=True)
    return _DEMO_VISIT_CACHE


def _compute_cell_header_stats(arm, compare, cohort):
    """Per-cell header stats for multi-row thead:
      - n_all (total raw visits across cohort base_ids), n_unique
      - Age mean±SD per label group + Welch t p
      - For hi-lo: MMSE/CASI/CDR_SB mean±SD per group + Welch t p
    Group label convention: 1 = AD (or high-MMSE in hi-lo), 0 = control (or low-MMSE).
    """
    row = {"arm": arm, "comparison": compare}
    if cohort is None or len(cohort) == 0:
        return row
    # n_unique — split per label group
    row["n_unique_1"] = int((cohort["label"] == 1).sum())
    row["n_unique_0"] = int((cohort["label"] == 0).sum())
    row["n_unique"] = row["n_unique_1"] + row["n_unique_0"]
    # n_all (raw visits) — split per label group
    visit_df = _all_visits_per_base_id()
    if "base_id" in cohort.columns:
        bids_1 = cohort.loc[cohort["label"] == 1, "base_id"].astype(str).unique()
        bids_0 = cohort.loc[cohort["label"] == 0, "base_id"].astype(str).unique()
    else:
        _b = cohort["ID"].astype(str).str.extract(r"^([A-Za-z]+\d+)")[0]
        bids_1 = _b[cohort["label"] == 1].unique()
        bids_0 = _b[cohort["label"] == 0].unique()
    row["n_all_1"] = int(visit_df[visit_df["base_id"].isin(bids_1)].shape[0])
    row["n_all_0"] = int(visit_df[visit_df["base_id"].isin(bids_0)].shape[0])
    row["n_all"] = row["n_all_1"] + row["n_all_0"]

    # Age stats
    age_col = "first_age" if "first_age" in cohort.columns else "Age"
    if age_col in cohort.columns:
        g1 = cohort.loc[cohort["label"] == 1, age_col].dropna().astype(float)
        g2 = cohort.loc[cohort["label"] == 0, age_col].dropna().astype(float)
        row["age_mean_1"] = float(g1.mean()) if len(g1) else np.nan
        row["age_sd_1"] = float(g1.std(ddof=1)) if len(g1) > 1 else np.nan
        row["age_mean_2"] = float(g2.mean()) if len(g2) else np.nan
        row["age_sd_2"] = float(g2.std(ddof=1)) if len(g2) > 1 else np.nan
        if len(g1) >= 2 and len(g2) >= 2:
            _, p = stats.ttest_ind(g1, g2, equal_var=False)
            row["age_p"] = float(p)
        else:
            row["age_p"] = np.nan
    # Cog stats only for hi-lo cells (AD internal severity split)
    if compare in ("mmse-hi-lo", "casi-hi-lo"):
        cog_triples = [
            ("mmse", "MMSE", "first_MMSE"),
            ("casi", "CASI", "first_CASI"),
            ("cdr",  "Global_CDR", "first_Global_CDR"),
        ]
        for name, ab_col, cd_col in cog_triples:
            col = ab_col if ab_col in cohort.columns else (
                cd_col if cd_col in cohort.columns else None)
            if col is None:
                for k in ("mean_1", "sd_1", "mean_2", "sd_2", "p"):
                    row[f"{name}_{k}"] = np.nan
                continue
            g1 = cohort.loc[cohort["label"] == 1, col].dropna().astype(float)
            g2 = cohort.loc[cohort["label"] == 0, col].dropna().astype(float)
            row[f"{name}_mean_1"] = float(g1.mean()) if len(g1) else np.nan
            row[f"{name}_sd_1"] = float(g1.std(ddof=1)) if len(g1) > 1 else np.nan
            row[f"{name}_mean_2"] = float(g2.mean()) if len(g2) else np.nan
            row[f"{name}_sd_2"] = float(g2.std(ddof=1)) if len(g2) > 1 else np.nan
            if len(g1) >= 2 and len(g2) >= 2:
                _, p = stats.ttest_ind(g1, g2, equal_var=False)
                row[f"{name}_p"] = float(p)
            else:
                row[f"{name}_p"] = np.nan
    return row


def run_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Env-driven subset controls（optional；預設跑全部）
    arms_env = os.environ.get("ARMS", "").strip()
    arms_to_run = ([s.strip() for s in arms_env.split(",") if s.strip()]
                    if arms_env else ["A", "B", "C", "D"])
    modalities_env = os.environ.get("MODALITIES", "").strip()
    if modalities_env:
        wanted_mods = {s.strip() for s in modalities_env.split(",") if s.strip()}
        active_specs = [s for s in MODALITY_SPECS if s[0] in wanted_mods]
    else:
        active_specs = list(MODALITY_SPECS)
    logger.info(f"arms={arms_to_run}  modalities={[s[0] for s in active_specs]}")

    # Define cells: arm × comparison (5 columns: HC, NAD, ACS, mmse-hi-lo, casi-hi-lo)
    cells = []
    for arm in ["A", "B", "C", "D"]:
        for compare in ["HC", "NAD", "ACS", "mmse-hi-lo", "casi-hi-lo"]:
            cells.append((arm, compare))

    # Feasibility check first
    feasibility_rows = []
    header_stat_rows = []
    cohorts = {}
    for arm, compare in cells:
        if arm not in arms_to_run:
            # Arm 不在 whitelist → 標 n/a 但留在 grid
            feasibility_rows.append({
                "arm": arm, "comparison": compare,
                "n_total": 0, "n_pos": 0, "n_neg": 0,
                "status": "n/a (arm skipped)", "note": "arm not in ARMS env",
            })
            header_stat_rows.append(_compute_cell_header_stats(arm, compare, None))
            cohorts[(arm, compare)] = (None, None)
            continue
        try:
            if compare == "mmse-hi-lo":
                cohort, pairs = build_cohort_ad_hi_lo(arm=arm, metric="MMSE")
            elif compare == "casi-hi-lo":
                cohort, pairs = build_cohort_ad_hi_lo(arm=arm, metric="CASI")
            else:
                cohort, pairs = build_cohort_ad_vs_HCgroup(compare, arm=arm)
            n = len(cohort) if cohort is not None else 0
            n_pos = int((cohort["label"] == 1).sum()) if n > 0 else 0
            n_neg = int((cohort["label"] == 0).sum()) if n > 0 else 0
            note = ""
            status = "ok"
            if min(n_pos, n_neg) < MIN_CELL_N:
                status = f"n/a (min-cell-n {min(n_pos, n_neg)} < {MIN_CELL_N})"
                note = "pair count below threshold"
        except Exception as e:
            cohort, pairs = None, None
            n, n_pos, n_neg = 0, 0, 0
            status = f"error: {e}"; note = str(e)
        feasibility_rows.append({
            "arm": arm, "comparison": compare,
            "n_total": n, "n_pos": n_pos, "n_neg": n_neg,
            "status": status, "note": note,
        })
        # Header stats per cell (n_all, n_unique, age, cog for hi-lo)
        header_stat_rows.append(_compute_cell_header_stats(arm, compare, cohort))
        cohorts[(arm, compare)] = (cohort, pairs)
    feas_df = pd.DataFrame(feasibility_rows)
    feas_df.to_csv(OUTPUT_DIR / "feasibility_report.csv", index=False)
    hdr_df = pd.DataFrame(header_stat_rows)
    hdr_df.to_csv(OUTPUT_DIR / "cell_header_stats.csv", index=False)
    logger.info(f"Feasibility + header stats saved. Active cells: "
                 f"{(feas_df['status']=='ok').sum()}/16")

    # Run stats per modality × active cell
    long_rows = []
    for arm, compare in cells:
        cohort, pairs = cohorts[(arm, compare)]
        fstat = feas_df[(feas_df.arm == arm) & (feas_df.comparison == compare)].iloc[0]
        if fstat["status"] != "ok":
            for (parent, sub, test_kind, extra) in MODALITY_SPECS:
                long_rows.append({
                    "arm": arm, "comparison": compare,
                    "modality_parent": parent, "modality_sub": sub,
                    "test": test_kind, "n": 0, "p": np.nan, "q": np.nan,
                    "statistic": np.nan, "effect": np.nan,
                    "skip_reason": fstat["status"],
                })
            continue
        logger.info(f"=== {arm} × {compare} : n_pos={fstat['n_pos']} "
                     f"n_neg={fstat['n_neg']} ===")
        cell_results = []
        # 先把所有 MODALITY_SPECS 都建立 placeholder，被選中的才覆寫
        active_keys = {s[0] for s in active_specs}
        for (parent, sub, test_kind, extra) in MODALITY_SPECS:
            name = parent if sub is None else f"{parent}::{sub}"
            if parent not in active_keys:
                res = {"modality": name, "n": 0, "p": np.nan,
                        "statistic": np.nan, "effect": np.nan,
                        "skip_reason": "modality not in MODALITIES env"}
                res["arm"] = arm; res["comparison"] = compare
                res["modality_parent"] = parent; res["modality_sub"] = sub
                cell_results.append(res)
                continue
            try:
                X_df = _feature_df_for_modality(test_kind, extra, cohort, arm)
                res = _dispatch_test(test_kind, extra, name, X_df, cohort, arm)
            except Exception as e:
                logger.warning(f"{arm}×{compare} {name}: {e}")
                res = {"modality": name, "n": 0, "p": np.nan,
                        "statistic": np.nan, "effect": np.nan,
                        "error": str(e)}
            res["arm"] = arm; res["comparison"] = compare
            res["modality_parent"] = parent; res["modality_sub"] = sub
            cell_results.append(res)
            logger.info(f"  {name}: p={res.get('p')} eff={res.get('effect')}")

        # FDR within cell
        pvals = np.array([r.get("p", np.nan) for r in cell_results], dtype=float)
        valid = ~np.isnan(pvals)
        qvals = np.full(len(pvals), np.nan)
        if valid.sum() > 0:
            qvals[valid] = bh_fdr(pvals[valid])
        for r, q in zip(cell_results, qvals):
            r["q"] = float(q) if not np.isnan(q) else np.nan
            long_rows.append(r)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUTPUT_DIR / "stat_grid_long.csv", index=False)
    logger.info(f"Long-form saved: {len(long_df)} rows")

    # Pivot wide
    wide = long_df.copy()
    wide["cell"] = wide["arm"] + "-" + wide["comparison"]
    wide["label"] = wide["modality_parent"] + wide["modality_sub"].apply(
        lambda s: "" if pd.isna(s) else f" [{s}]")
    pivot = wide.pivot_table(
        index="label", columns="cell",
        values=["statistic", "p", "q", "effect", "auc_auc",
                 "auc_auc_ci_low", "auc_auc_ci_high", "n"],
        aggfunc="first"
    )
    pivot.to_csv(OUTPUT_DIR / "stat_grid_wide.csv")
    logger.info(f"Wide grid saved")

    # Per-direction subfolder splits (5 directions: age / embedding_mean /
    # embedding_asymmetry / landmark_asymmetry / emotion)
    long_df_dir = long_df.copy()
    long_df_dir["direction"] = long_df_dir["modality_parent"].map(DIRECTION_MAP)
    for direction, sub in long_df_dir.groupby("direction"):
        d_dir = OUTPUT_DIR / direction
        d_dir.mkdir(parents=True, exist_ok=True)
        sub.drop(columns=["direction"]).to_csv(
            d_dir / "stat_grid_long.csv", index=False)
        sub_wide = sub.copy()
        sub_wide["cell"] = sub_wide["arm"] + "-" + sub_wide["comparison"]
        sub_wide["label"] = sub_wide["modality_parent"] + sub_wide["modality_sub"].apply(
            lambda s: "" if pd.isna(s) else f" [{s}]")
        sub_pivot = sub_wide.pivot_table(
            index="label", columns="cell",
            values=["statistic", "p", "q", "effect", "auc_auc",
                    "auc_auc_ci_low", "auc_auc_ci_high", "n"],
            aggfunc="first"
        )
        sub_pivot.to_csv(d_dir / "stat_grid_wide.csv")
    logger.info(f"Per-direction subfolders saved: "
                f"{sorted(long_df_dir['direction'].dropna().unique())}")

    return long_df, feas_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hc-source", choices=["ACS", "ACS_ext", "EACS"], default=None,
        help="ACS 群體組成：ACS=內部 91 人（預設）；ACS_ext=內部+EACS；EACS=僅 EACS",
    )
    ap.add_argument(
        "--eacs-sources", nargs="+", default=None,
        help="EACS source 子集（例：UTKFace），只在 ACS_ext/EACS mode 有效；"
             "預設用全部 EACS sources",
    )
    ap.add_argument(
        "--arms", nargs="+", choices=["A", "B", "C", "D"], default=None,
        help="只跑指定 arms；其他 arms 在 grid 上標 n/a",
    )
    ap.add_argument(
        "--modalities", nargs="+", default=None,
        help="只算指定 modality_parent；其他 modality 在 grid 上標 skip_reason",
    )
    ap.add_argument(
        "--output-suffix", default=None,
        help="覆寫 subset 資料夾名，輸出到 grid/subsets/<suffix>/",
    )
    args = ap.parse_args()
    if args.hc_source is not None:
        HC_SOURCE_MODE = args.hc_source
    if args.eacs_sources:
        os.environ["EACS_SOURCES"] = ",".join(args.eacs_sources)
    if args.arms:
        os.environ["ARMS"] = ",".join(args.arms)
    if args.modalities:
        os.environ["MODALITIES"] = ",".join(args.modalities)

    is_subset = bool(args.eacs_sources or args.arms or args.modalities or
                     args.output_suffix)
    if is_subset:
        if args.output_suffix is not None:
            suffix = args.output_suffix
        else:
            parts = []
            if HC_SOURCE_MODE != "ACS":
                parts.append(HC_SOURCE_MODE.lower())
            if args.eacs_sources:
                parts.append("+".join(s.lower() for s in args.eacs_sources))
            if args.arms:
                parts.append("".join(sorted(args.arms)))
            if args.modalities:
                parts.append("_".join(args.modalities)[:40])
            suffix = "_".join(parts)
        OUTPUT_DIR = GRID_ROOT / "subsets" / suffix
    else:
        OUTPUT_DIR = GRID_ROOT / HC_SOURCE_MODE.lower()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"HC_SOURCE_MODE={HC_SOURCE_MODE}  OUTPUT_DIR={OUTPUT_DIR}")
    run_all()
