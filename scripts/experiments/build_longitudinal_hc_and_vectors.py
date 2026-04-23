"""
Build Arm-C longitudinal Δ supplements:
  (A) HC/NAD/ACS scalar Δ (hc_patient_deltas.csv) — enables Arm C × {HC/NAD/ACS}
  (B) AD + HC/NAD/ACS full-vector Δ (vector_deltas.npz) — enables Arm C × hi-lo
      full-vector embedding-asymmetry and 130-d landmark raw-xy modalities

Outputs:
  workspace/longitudinal/hc_patient_deltas.csv
  workspace/longitudinal/vector_deltas.npz

Usage:
  "C:/Users/4080/anaconda3/envs/Alz_face_test_2/python.exe" \
      scripts/experiments/build_longitudinal_hc_and_vectors.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
EMOTION_DIR = PROJECT_ROOT / "workspace" / "emotion" / "features" / "aggregated"
AGE_FILE = PROJECT_ROOT / "workspace" / "age" / "age_prediction" / "predicted_ages.json"
EMBEDDING_DIR = PROJECT_ROOT / "workspace" / "embedding" / "features"
LANDMARK_FEATURES_CSV = PROJECT_ROOT / "workspace" / "asymmetry" / "features.csv"
LONGITUDINAL_DIR = PROJECT_ROOT / "workspace" / "longitudinal"

EMBEDDING_MODELS = ["arcface", "topofr", "dlib"]
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness",
            "surprise", "neutral"]
EMOTION_METHODS = ["openface", "libreface", "pyfeat", "dan",
                   "hsemotion", "vit", "poster_pp", "fer"]
LANDMARK_REGIONS = ["eye", "nose", "mouth", "face_oval"]
MIN_FOLLOW_UP_DAYS = 180

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_emb_vec(npy_path):
    if not npy_path.exists():
        return None
    a = np.load(npy_path, allow_pickle=True)
    if a.dtype == object:
        a = list(a.item().values())[0]
    return a.mean(axis=0) if a.ndim == 2 else a


def _visits_with_features():
    """Return set of visit_ids that have ArcFace original .npy extracted.

    Demographics contains rows for visits whose photo folder is absent
    (e.g. P62-1, P75-1 — recorded but no photo taken). We filter those
    out before selecting first/last visit for Δ, otherwise the first
    visit picked is one without embeddings and the subject is dropped
    from vector_deltas.npz entirely.
    """
    arcface_dir = EMBEDDING_DIR / "arcface" / "original"
    if not arcface_dir.exists():
        return set()
    return {p.stem for p in arcface_dir.glob("*.npy")}


def _load_all_demographics(include_eacs=False):
    """Load internal ACS/NAD/P demographics; optionally append external EACS.

    EACS rows get group='EACS'; base_id uses the whole ID prefix before '-visit'
    since EACS_IMDB_nm0000002-03 has no trailing digits after 'EACS_IMDB_nm0000002'.
    """
    frames = []
    for grp in ["P", "NAD", "ACS"]:
        df = pd.read_csv(DEMOGRAPHICS_DIR / f"{grp}.csv")
        if "ID" not in df.columns:
            for col in df.columns:
                if col in ("ACS", "NAD"):
                    df = df.rename(columns={col: "ID"})
                    break
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["MMSE"] = pd.to_numeric(df.get("MMSE"), errors="coerce")
        df["CASI"] = pd.to_numeric(df.get("CASI"), errors="coerce")
        df["Global_CDR"] = pd.to_numeric(df.get("Global_CDR"), errors="coerce")
        df["CDR_SB"] = pd.to_numeric(df.get("CDR_SB"), errors="coerce")
        # 通用 base_id regex：match everything before trailing -\d+
        df["base_id"] = df["ID"].str.extract(r"^(.+)-\d+$")
        df["visit"] = df["ID"].str.extract(r"-(\d+)$").astype(float)
        df["group"] = grp
        frames.append(df)
    if include_eacs:
        eacs_path = DEMOGRAPHICS_DIR / "EACS.csv"
        if eacs_path.exists():
            df = pd.read_csv(eacs_path)
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
            df["MMSE"] = pd.to_numeric(df.get("MMSE"), errors="coerce")
            df["CASI"] = pd.to_numeric(df.get("CASI"), errors="coerce")
            df["Global_CDR"] = np.nan
            df["CDR_SB"] = np.nan
            df["base_id"] = df["ID"].str.extract(r"^(.+)-\d+$")
            df["visit"] = df["ID"].str.extract(r"-(\d+)$").astype(float)
            df["group"] = "EACS"
            frames.append(df)
            logger.info(f"Loaded {len(df)} EACS rows")
        else:
            logger.warning(f"EACS.csv not found at {eacs_path}")
    return pd.concat(frames, ignore_index=True)


def _load_emotion_dict():
    """Map subject_id -> {f"{method}__{emo}": mean_value} across all 8 methods.

    Expanded from openface-only to 8 methods × 7 emotions = 56 entries per subject,
    enabling Arm C/D emotion_fisher Δ tests to parallel A/B.
    """
    out = {}
    for method in EMOTION_METHODS:
        path = EMOTION_DIR / f"{method}_harmonized.csv"
        if not path.exists():
            logger.warning(f"Emotion CSV missing: {path}")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            sid = r["subject_id"]
            if sid not in out:
                out[sid] = {}
            for e in EMOTIONS:
                col = f"{e}_mean"
                if col in df.columns and pd.notna(r[col]):
                    out[sid][f"{method}__{e}"] = float(r[col])
    return out


def _load_landmark_dict():
    df = pd.read_csv(LANDMARK_FEATURES_CSV)
    return {r["subject_id"]: r.to_dict() for _, r in df.iterrows()}


def _compute_delta(first, last, emo_dict, pred_ages, lmk_dict,
                   save_vector=True):
    """Returns (scalar_row, vec_dict) for one subject pair of visits."""
    fid, lid = first["ID"], last["ID"]
    try:
        fd = (pd.to_datetime(last["Photo_Date"]) -
              pd.to_datetime(first["Photo_Date"])).days
    except Exception:
        fd = None
    if fd is None or fd < MIN_FOLLOW_UP_DAYS:
        return None, None
    if pd.isna(first["Age"]) or pd.isna(last["Age"]):
        return None, None
    fy = fd / 365.25

    def _num(x):
        return float(x) if pd.notna(x) else np.nan

    row = {
        "base_id": first["base_id"], "group": first["group"],
        "first_visit_id": fid, "last_visit_id": lid,
        "first_age": float(first["Age"]), "last_age": float(last["Age"]),
        "first_MMSE": _num(first.get("MMSE")),
        "last_MMSE": _num(last.get("MMSE")),
        "first_CASI": _num(first.get("CASI")),
        "last_CASI": _num(last.get("CASI")),
        "first_Global_CDR": _num(first.get("Global_CDR")),
        "last_Global_CDR": _num(last.get("Global_CDR")),
        "first_CDR_SB": _num(first.get("CDR_SB")),
        "last_CDR_SB": _num(last.get("CDR_SB")),
        "follow_up_days": fd, "follow_up_years": fy,
    }
    vec_dict = {}

    # Embedding drift + asymmetry Δ per model
    for model in EMBEDDING_MODELS:
        orig = EMBEDDING_DIR / model / "original"
        diff = EMBEDDING_DIR / model / "difference"
        fv, lv = _load_emb_vec(orig / f"{fid}.npy"), _load_emb_vec(orig / f"{lid}.npy")
        if fv is not None and lv is not None:
            row[f"emb_cosine_dist_{model}"] = float(cosine_distance(fv, lv))
            if save_vector:
                vec_dict[f"emb_drift_vec_{model}"] = (lv - fv).astype(np.float32)
        else:
            row[f"emb_cosine_dist_{model}"] = np.nan

        fvd = _load_emb_vec(diff / f"{fid}.npy")
        lvd = _load_emb_vec(diff / f"{lid}.npy")
        if fvd is not None and lvd is not None:
            row[f"delta_embasym_{model}"] = float(
                np.linalg.norm(lvd) - np.linalg.norm(fvd))
            if save_vector:
                vec_dict[f"emb_asym_delta_vec_{model}"] = (lvd - fvd).astype(np.float32)
        else:
            row[f"delta_embasym_{model}"] = np.nan

    # Landmark per-region L2 Δ + 130-d raw xy Δ
    fl, ll = lmk_dict.get(fid), lmk_dict.get(lid)
    if fl is not None and ll is not None:
        all_xy_cols = []
        for region in LANDMARK_REGIONS:
            xy_cols = [c for c in fl.keys()
                       if c.startswith(f"{region}_") and "area_diff" not in c]
            all_xy_cols.extend(xy_cols)
            fvec = np.asarray([fl[c] for c in xy_cols], dtype=float)
            lvec = np.asarray([ll[c] for c in xy_cols], dtype=float)
            row[f"lmk_delta_{region}_l2"] = float(
                np.sqrt(np.nansum((lvec - fvec) ** 2)))
        if save_vector:
            fvec_all = np.asarray([fl[c] for c in all_xy_cols], dtype=float)
            lvec_all = np.asarray([ll[c] for c in all_xy_cols], dtype=float)
            vec_dict["lmk_raw_xy_delta"] = (lvec_all - fvec_all).astype(np.float32)
    else:
        for region in LANDMARK_REGIONS:
            row[f"lmk_delta_{region}_l2"] = np.nan

    # Emotion Δ — 8 methods × 7 emotions = 56 delta columns
    # (key format: delta_{method}__{emotion} to match A/B's emotion column naming)
    fe = emo_dict.get(fid, {}); le = emo_dict.get(lid, {})
    for method in EMOTION_METHODS:
        for emo in EMOTIONS:
            key = f"{method}__{emo}"
            if key in fe and key in le:
                row[f"delta_{key}"] = le[key] - fe[key]
            else:
                row[f"delta_{key}"] = np.nan
    # Back-compat alias: delta_{emotion} = delta_openface__{emotion}
    for emo in EMOTIONS:
        row[f"delta_{emo}"] = row.get(f"delta_openface__{emo}", np.nan)

    # age_error Δ
    fa_pred = pred_ages.get(fid); la_pred = pred_ages.get(lid)
    if fa_pred is not None and la_pred is not None:
        row["delta_age_error"] = float(
            (last["Age"] - la_pred) - (first["Age"] - fa_pred))
    else:
        row["delta_age_error"] = np.nan

    # Annualize
    for k in list(row.keys()):
        if (k.startswith("delta_") or k.startswith("lmk_delta_")
                or k.startswith("emb_cosine_dist_")):
            v = row[k]
            row[f"ann_{k}"] = v / fy if pd.notna(v) else np.nan
    if "ann_emb_cosine_dist_arcface" in row:
        row["ann_emb_cosine_dist"] = row["ann_emb_cosine_dist_arcface"]

    return row, vec_dict


def build_group(demo_all, groups, label, emo_dict, pred_ages, lmk_dict):
    """Iterate multi-visit subjects in given groups and compute Δ."""
    sel = demo_all[demo_all["group"].isin(groups)].copy()
    # Drop demographics rows whose visit has no extracted features
    # (recorded but no photo taken / missing upload). Without this filter
    # the first-visit pick can land on a feature-less visit and cascade
    # the subject out of vector_deltas.npz.
    feat_vids = _visits_with_features()
    n_before = len(sel)
    sel = sel[sel["ID"].isin(feat_vids)]
    logger.info(f"{label}: filtered {n_before - len(sel)} demo rows without "
                f"feature npy ({len(sel)} visits remain)")
    sel = sel.sort_values(["base_id", "visit"])
    # Multi-visit filter
    vcnt = sel.groupby("base_id")["visit"].count()
    sel = sel[sel["base_id"].isin(vcnt[vcnt >= 2].index)]

    scalar_rows, vec_store = [], {}
    for bid, g in sel.groupby("base_id"):
        g = g.sort_values("visit")
        first, last = g.iloc[0], g.iloc[-1]
        row, vec = _compute_delta(first, last, emo_dict, pred_ages, lmk_dict)
        if row is None:
            continue
        scalar_rows.append(row)
        if vec:
            vec_store[bid] = vec

    df = pd.DataFrame(scalar_rows)
    logger.info(f"{label}: {len(df)} subjects with Δ (follow-up ≥ 180d)")
    return df, vec_store


def save_vector_npz(vec_store, out_path):
    if not vec_store:
        logger.info("No vector Δ to save.")
        return
    bids = sorted(vec_store.keys())
    all_keys = set().union(*[v.keys() for v in vec_store.values()])
    arrs = {"base_ids": np.array(bids, dtype=str)}
    for k in sorted(all_keys):
        dim = None
        for bid in bids:
            if k in vec_store[bid]:
                dim = vec_store[bid][k].shape[0]
                break
        if dim is None:
            continue
        stack = np.full((len(bids), dim), np.nan, dtype=np.float32)
        for i, bid in enumerate(bids):
            if k in vec_store[bid]:
                stack[i] = vec_store[bid][k]
        arrs[k] = stack
    np.savez_compressed(out_path, **arrs)
    logger.info(f"Saved {out_path}  [{len(bids)} subjects × {len(all_keys)} vector keys]")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-eacs", action="store_true",
                    help="額外產 eacs_patient_deltas.csv（僅 IMDB multi-visit 有效）")
    args = ap.parse_args()

    LONGITUDINAL_DIR.mkdir(parents=True, exist_ok=True)
    demo_all = _load_all_demographics(include_eacs=args.include_eacs)
    emo_dict = _load_emotion_dict()
    with open(AGE_FILE) as f:
        pred_ages = json.load(f)
    lmk_dict = _load_landmark_dict()

    # HC/NAD/ACS side
    df_hc, vec_hc = build_group(demo_all, ["NAD", "ACS"], "HC(NAD+ACS)",
                                 emo_dict, pred_ages, lmk_dict)
    df_hc.to_csv(LONGITUDINAL_DIR / "hc_patient_deltas.csv", index=False)
    logger.info(f"Saved hc_patient_deltas.csv")

    # AD side — CDR>=0.5 multi-visit (same filter as patient_deltas.csv)
    ad_patients = demo_all[demo_all["group"] == "P"].groupby(
        "base_id")["Global_CDR"].max()
    ad_patients = ad_patients[ad_patients >= 0.5].index
    demo_ad = demo_all[(demo_all["group"] == "P") &
                        (demo_all["base_id"].isin(ad_patients))].copy()
    df_ad, vec_ad = build_group(demo_ad, ["P"], "AD",
                                 emo_dict, pred_ages, lmk_dict)
    df_ad.to_csv(LONGITUDINAL_DIR / "ad_patient_deltas.csv", index=False)
    logger.info(f"Saved ad_patient_deltas.csv "
                 f"(AD mirror with lmk_delta_* + ann_* for deep-dive)")

    # External ACS side — IMDB multi-visit（optional）
    vec_eacs = {}
    df_eacs = None
    if args.include_eacs:
        df_eacs, vec_eacs = build_group(demo_all, ["EACS"], "EACS(IMDB)",
                                         emo_dict, pred_ages, lmk_dict)
        df_eacs.to_csv(LONGITUDINAL_DIR / "eacs_patient_deltas.csv",
                        index=False)
        logger.info(f"Saved eacs_patient_deltas.csv")

    # Combine vector stores (AD + HC/NAD/ACS [+ EACS])
    vec_all = {**vec_ad, **vec_hc, **vec_eacs}
    save_vector_npz(vec_all, LONGITUDINAL_DIR / "vector_deltas.npz")

    # Summary
    print("\n=== Summary ===")
    print(f"HC+NAD+ACS scalar Δ: {len(df_hc)} subjects "
          f"({(df_hc['group']=='NAD').sum()} NAD, "
          f"{(df_hc['group']=='ACS').sum()} ACS)")
    print(f"AD scalar Δ (vector-only mirror): {len(df_ad)} subjects")
    if df_eacs is not None:
        print(f"EACS scalar Δ (IMDB multi-visit): {len(df_eacs)} subjects")
    print(f"Total vector Δ: {len(vec_all)} subjects")


if __name__ == "__main__":
    main()
