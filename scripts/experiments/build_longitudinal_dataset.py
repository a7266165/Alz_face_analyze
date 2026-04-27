"""
建立縱向分析資料集

以 P.csv 為主表，join:
- Emotion/AU features (openface_harmonized)
- Age prediction (predicted_ages.json)
- Embedding cosine distance (跨 visit 計算)

產出:
- workspace/longitudinal/longitudinal_dataset.csv (per-visit 完整資料)
- workspace/longitudinal/patient_deltas.csv (per-patient 首末差值)
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Paths ===
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
EMOTION_DIR = PROJECT_ROOT / "workspace" / "emotion" / "au_features" / "aggregated"
AGE_FILE = PROJECT_ROOT / "workspace" / "age" / "age_prediction" / "predicted_ages.json"
EMBEDDING_DIR = PROJECT_ROOT / "workspace" / "embedding" / "features"
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "longitudinal"

# Use openface as default emotion model (most complete coverage)
EMOTION_MODEL = "openface"
# Embedding models for cosine drift + asymmetry delta
EMBEDDING_MODELS = ["arcface", "topofr", "dlib"]
EMBEDDING_TYPE = "original"  # used for cosine drift vs first visit
EMBEDDING_DIFF_TYPE = "difference"  # used for asymmetry magnitude per visit


def load_base_table():
    """Load P.csv as base table, filter to CDR >= 0.5 multi-visit patients."""
    p = pd.read_csv(DEMOGRAPHICS_DIR / "P.csv")
    p["Global_CDR"] = pd.to_numeric(p["Global_CDR"], errors="coerce")

    # Extract base_id and visit number
    p["base_id"] = p["ID"].str.extract(r"^([A-Z]+\d+)", expand=False)
    p["visit_num"] = p["ID"].str.extract(r"-(\d+)$", expand=False).astype(int)

    # Patient-level filter: keep any patient with at least one visit at CDR>=0.5
    ad_patients = p.groupby("base_id")["Global_CDR"].max()
    ad_patients = ad_patients[ad_patients >= 0.5].index
    p = p[p["base_id"].isin(ad_patients)].copy()

    # Keep only multi-visit patients
    visit_counts = p.groupby("base_id")["visit_num"].count()
    multi_visit_ids = visit_counts[visit_counts >= 2].index
    p = p[p["base_id"].isin(multi_visit_ids)].copy()

    p = p.sort_values(["base_id", "visit_num"])
    logger.info(f"Base table: {len(p)} visits from {p['base_id'].nunique()} patients")
    return p


def join_emotion_features(base_df):
    """Join emotion/AU features."""
    csv_path = EMOTION_DIR / f"{EMOTION_MODEL}_harmonized.csv"
    emo = pd.read_csv(csv_path)

    # Keep only mean columns for longitudinal simplicity
    mean_cols = [c for c in emo.columns if c.endswith("_mean")]
    emo_subset = emo[["subject_id"] + mean_cols].copy()

    # Rename for clarity
    emo_subset.columns = ["subject_id"] + [c.replace("_mean", "") for c in mean_cols]

    merged = base_df.merge(emo_subset, left_on="ID", right_on="subject_id", how="left")
    merged = merged.drop(columns=["subject_id"])

    n_matched = merged[mean_cols[0].replace("_mean", "")].notna().sum()
    logger.info(f"Emotion features joined: {n_matched}/{len(merged)} matched")
    return merged


def join_age_predictions(base_df):
    """Join predicted ages."""
    with open(AGE_FILE) as f:
        ages = json.load(f)

    base_df = base_df.copy()
    base_df["predicted_age"] = base_df["ID"].map(ages)
    # Sign convention aligned with L3 (run_ad_mmse_age_balanced.py):
    # positive age_error = face looks younger than real age (predicted < real)
    base_df["age_error"] = base_df["Age"] - base_df["predicted_age"]

    n_matched = base_df["predicted_age"].notna().sum()
    logger.info(f"Age predictions joined: {n_matched}/{len(base_df)} matched")
    return base_df


def _load_embedding_vec(npy_path, pool="mean"):
    """Load a per-visit embedding stack (10, D) and return a pooled 1-D vector.

    pool='mean' returns the mean-pooled embedding; pool='l2_mean' returns the
    L2 norm of the mean-pooled vector (used for asymmetry magnitude).
    """
    if not npy_path.exists():
        return None
    a = np.load(npy_path, allow_pickle=True)
    if a.dtype == object:
        a = list(a.item().values())[0]
    vec = a.mean(axis=0) if a.ndim == 2 else a
    if pool == "l2_mean":
        return float(np.linalg.norm(vec))
    return vec


def compute_embedding_cosine_distances(base_df):
    """Compute cosine drift vs first visit for each of EMBEDDING_MODELS.

    Adds per-visit columns `emb_cosine_dist_from_first_{model}` for each model,
    and keeps `emb_cosine_dist_from_first` (= arcface) for backward compat.
    """
    base_df = base_df.copy()
    for model in EMBEDDING_MODELS:
        base_df[f"emb_cosine_dist_from_first_{model}"] = np.nan

    for model in EMBEDDING_MODELS:
        emb_dir = EMBEDDING_DIR / model / EMBEDDING_TYPE
        col = f"emb_cosine_dist_from_first_{model}"

        for patient_id in base_df["base_id"].unique():
            patient_visits = (base_df[base_df["base_id"] == patient_id]
                              .sort_values("visit_num"))
            visit_ids = patient_visits["ID"].tolist()

            first_vec = _load_embedding_vec(emb_dir / f"{visit_ids[0]}.npy")
            if first_vec is None:
                continue
            base_df.loc[base_df["ID"] == visit_ids[0], col] = 0.0

            for vid in visit_ids[1:]:
                vec = _load_embedding_vec(emb_dir / f"{vid}.npy")
                if vec is None:
                    continue
                base_df.loc[base_df["ID"] == vid, col] = float(
                    cosine_distance(first_vec, vec)
                )

        n_computed = base_df[col].notna().sum()
        logger.info(f"[{model}] cosine drift computed: {n_computed}/{len(base_df)}")

    # Back-compat alias: default `emb_cosine_dist_from_first` = arcface
    base_df["emb_cosine_dist_from_first"] = base_df[
        "emb_cosine_dist_from_first_arcface"
    ]
    return base_df


def compute_embasym_delta(first_visit_id, last_visit_id, model):
    """Return |last_L2| - |first_L2| of the per-visit mean-pooled difference
    vector, for one embedding model. NaN if either visit's .npy is missing."""
    diff_dir = EMBEDDING_DIR / model / EMBEDDING_DIFF_TYPE
    first_l2 = _load_embedding_vec(diff_dir / f"{first_visit_id}.npy", pool="l2_mean")
    last_l2 = _load_embedding_vec(diff_dir / f"{last_visit_id}.npy", pool="l2_mean")
    if first_l2 is None or last_l2 is None:
        return np.nan
    return float(last_l2 - first_l2)


def compute_patient_deltas(longitudinal_df):
    """Compute per-patient first-to-last deltas."""
    delta_rows = []

    cognitive_cols = ["MMSE", "CASI", "CDR_SB", "Global_CDR"]
    emotion_cols = [c for c in longitudinal_df.columns
                    if c in ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26",
                             "anger", "disgust", "fear", "happiness", "sadness",
                             "surprise", "neutral"]]
    age_cols = ["predicted_age", "age_error"]

    for patient_id in longitudinal_df["base_id"].unique():
        visits = longitudinal_df[longitudinal_df["base_id"] == patient_id].sort_values("visit_num")
        if len(visits) < 2:
            continue

        first = visits.iloc[0]
        last = visits.iloc[-1]

        follow_up_days = (
            (pd.to_datetime(last["Photo_Date"]) - pd.to_datetime(first["Photo_Date"])).days
            if pd.notna(first.get("Photo_Date")) and pd.notna(last.get("Photo_Date"))
            else None
        )
        row = {
            "base_id": patient_id,
            "n_visits": len(visits),
            "first_visit_id": first["ID"],
            "last_visit_id": last["ID"],
            "first_age": first["Age"],
            "last_age": last["Age"],
            "follow_up_days": follow_up_days,
            "follow_up_years": (follow_up_days / 365.25) if follow_up_days is not None else None,
            "first_CDR": first["Global_CDR"],
            "last_CDR": last["Global_CDR"],
            "cdr_worsened": int(last["Global_CDR"] > first["Global_CDR"]),
        }

        # Cognitive baselines + deltas
        for col in cognitive_cols:
            if col in visits.columns:
                first_val = first.get(col)
                last_val = last.get(col)
                row[f"first_{col}"] = float(first_val) if pd.notna(first_val) else np.nan
                row[f"last_{col}"] = float(last_val) if pd.notna(last_val) else np.nan
                if pd.notna(first_val) and pd.notna(last_val):
                    row[f"delta_{col}"] = float(last_val) - float(first_val)
                else:
                    row[f"delta_{col}"] = np.nan

        # Emotion deltas
        for col in emotion_cols:
            first_val = first.get(col)
            last_val = last.get(col)
            if pd.notna(first_val) and pd.notna(last_val):
                row[f"delta_{col}"] = float(last_val) - float(first_val)
            else:
                row[f"delta_{col}"] = np.nan

        # Age deltas
        for col in age_cols:
            first_val = first.get(col)
            last_val = last.get(col)
            if pd.notna(first_val) and pd.notna(last_val):
                row[f"delta_{col}"] = float(last_val) - float(first_val)
            else:
                row[f"delta_{col}"] = np.nan

        # Embedding cosine drift (last visit vs first), per model
        for model in EMBEDDING_MODELS:
            row[f"emb_cosine_dist_{model}"] = last.get(
                f"emb_cosine_dist_from_first_{model}"
            )
        # Back-compat: emb_cosine_dist = arcface
        row["emb_cosine_dist"] = row["emb_cosine_dist_arcface"]

        # Embedding asymmetry delta (last_L2 - first_L2), per model
        for model in EMBEDDING_MODELS:
            row[f"delta_embasym_{model}"] = compute_embasym_delta(
                first["ID"], last["ID"], model
            )

        # Baseline features for prediction
        for col in emotion_cols + age_cols:
            row[f"baseline_{col}"] = first.get(col)

        delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)
    logger.info(f"Patient deltas: {len(delta_df)} patients")
    return delta_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base table
    logger.info("Step 1: Loading base table (P.csv, CDR>=0.5, multi-visit)...")
    base = load_base_table()

    # Step 2: Join emotion features
    logger.info("Step 2: Joining emotion features...")
    base = join_emotion_features(base)

    # Step 3: Join age predictions
    logger.info("Step 3: Joining age predictions...")
    base = join_age_predictions(base)

    # Step 4: Compute embedding cosine distances
    logger.info("Step 4: Computing embedding cosine distances...")
    base = compute_embedding_cosine_distances(base)

    # Step 5: Save longitudinal dataset
    longitudinal_path = OUTPUT_DIR / "longitudinal_dataset.csv"
    base.to_csv(longitudinal_path, index=False)
    logger.info(f"Saved: {longitudinal_path}")

    # Step 6: Compute and save patient deltas
    logger.info("Step 6: Computing patient deltas...")
    deltas = compute_patient_deltas(base)
    deltas_path = OUTPUT_DIR / "patient_deltas.csv"
    deltas.to_csv(deltas_path, index=False)
    logger.info(f"Saved: {deltas_path}")

    # Summary stats
    print("\n" + "=" * 60)
    print("LONGITUDINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Total visits: {len(base)}")
    print(f"Unique patients: {base['base_id'].nunique()}")
    print(f"CDR distribution: {base['Global_CDR'].value_counts().sort_index().to_dict()}")

    print(f"\nDELTA SUMMARY (n={len(deltas)}):")
    print(f"CDR worsened: {deltas['cdr_worsened'].sum()} ({100*deltas['cdr_worsened'].mean():.1f}%)")
    if "delta_MMSE" in deltas.columns:
        valid = deltas["delta_MMSE"].dropna()
        print(f"MMSE change: {valid.mean():.2f} ± {valid.std():.2f} (n={len(valid)})")
    if "delta_CASI" in deltas.columns:
        valid = deltas["delta_CASI"].dropna()
        print(f"CASI change: {valid.mean():.2f} ± {valid.std():.2f} (n={len(valid)})")
    for model in EMBEDDING_MODELS:
        col = f"emb_cosine_dist_{model}"
        if col in deltas.columns:
            valid = deltas[col].dropna()
            print(f"Embedding cosine drift [{model}]: "
                  f"{valid.mean():.4f} ± {valid.std():.4f} (n={len(valid)})")
        col2 = f"delta_embasym_{model}"
        if col2 in deltas.columns:
            valid = deltas[col2].dropna()
            print(f"Δ embedding asymmetry L2 [{model}]: "
                  f"{valid.mean():+.4f} ± {valid.std():.4f} (n={len(valid)})")


if __name__ == "__main__":
    main()
