"""把部署用的「10 折 base LR 集成」fit 並落地到 workspace/deploy/(範圍 A:只動 base 兩條 LR)。

動機:API 端要把 base 分數(embedding_LR_score / asymmetry_LR_score)改成「10 折模型取平均」,
且要能回頭稽核。本腳本用與 scripts/embedding/classification 的 forward OOF **完全相同**的切法
(GroupKFold, n_splits=10, group=base_id, fold_seed=0 → 確定性折)重切、逐折 fit 一條 LR,並:

  1. 存下 10 個 fitted Pipeline(給 API 現場對新上傳做集成打分)。
  2. 存下 10×N 的 session 級分數矩陣(每個 session 被 10 折各打一分,標註它自己的 test fold)。
  3. 逐 session 驗證「test-fold 那一格的分數」== 落地 oof_scores.csv 的 y_score / fold,
     確保這 10 折就是產生現有 OOF 的同一套(可複現)。

TabPFN 不動(範圍 A)。輸出全部寫到 workspace/deploy/。

執行(在分析 conda env,需 sklearn/pandas/joblib + 能載 embedding 特徵):
  conda activate Alz_face_main_analysis
  python scripts/deploy/export_deploy_folds.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 讓 `import src.*` 解析得到(scripts/deploy/ → 上兩層 = repo 根)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.cohort import cohort_list, base_id_of
from src.common.features import load_feature_matrix
from src.embedding.classification import build_classifier, oof_paths
from src.config import EMBEDDING_CLASSIFICATION_REFACTOR_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_deploy_folds")

# ============================================================================
# CONFIG（與 export_deploy_models.py 同一組 combo;改任一項都會改變產出/破壞可複現）
# ============================================================================
COHORT = ("p_first", "p_cdrall", "hc_all", "hc_cdrall_or_mmseall")
EMB = "arcface"
BG = "background"
PHOTO = "all"
REDUCER = "no_drop"
LR_C = 0.001
SEED_FOLD = 0            # 0 → GroupKFold(shuffle=False),與 train.py._kfold 一致
N_SPLITS = 10

# variant → API 端要取代的那條 base LR 名稱(僅供輸出命名/對照)
VARIANTS = {"original": "lr_embedding", "differences": "lr_asymmetry"}

OUT = REPO_ROOT / "workspace" / "deploy"
TOL = 1e-9              # test-fold 分數與落地 OOF 的容許誤差


def build_pipe() -> Pipeline:
    """複製 run.py:_build_estimator(logistic, reducer=no_drop):
    Pipeline([StandardScaler, LogisticRegression(C, max_iter=2000, lbfgs, balanced, rs=42)])。"""
    clf, needs_scaler = build_classifier("logistic", lr_C=LR_C)
    steps = []
    if needs_scaler:                       # logistic → True
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", clf))      # reducer=no_drop → 不插 reducer step
    return Pipeline(steps)


def load_landed_oof(variant: str) -> pd.DataFrame:
    """讀這格已落地的 forward oof_scores.csv(單一真相,供逐格對照)。"""
    path = oof_paths(COHORT, BG, EMB, variant, PHOTO, REDUCER, "logistic",
                     "forward", lr_C=LR_C, seed=SEED_FOLD,
                     root=EMBEDDING_CLASSIFICATION_REFACTOR_DIR)[0]
    if not path.exists():
        raise FileNotFoundError(f"找不到落地 OOF(無法驗證可複現):{path}")
    logger.info(f"  對照 OOF:{path}")
    return pd.read_csv(path)


def process_variant(variant: str, out_name: str) -> dict:
    logger.info("=" * 78)
    logger.info(f"variant={variant}  ({out_name})")

    full = cohort_list(*COHORT)
    full["label"] = (full["Group"] == "P").astype(int)     # 同 build_ad_full_cohort
    label_map = dict(zip(full["ID"], full["label"]))

    # 與 run.py 完全相同的載入(順序決定 GroupKFold 切法,務必一致)
    X, ids = load_feature_matrix(full["ID"].tolist(), EMB, variant, BG, PHOTO)
    if len(X) == 0:
        raise RuntimeError(f"load_feature_matrix 回空:{EMB}/{BG}/{variant}/{PHOTO}")
    y = np.array([label_map[i] for i in ids], dtype=int)
    g = np.array([base_id_of(i) for i in ids], dtype=object)

    k = min(N_SPLITS, len(np.unique(g)))
    gkf = GroupKFold(n_splits=k)           # SEED_FOLD=0 → shuffle=False,對齊 _kfold
    logger.info(f"  X={X.shape}  photos={len(X)}  subjects={len(np.unique(g))}  folds={k}")

    folds_dir = OUT / "folds" / variant
    folds_dir.mkdir(parents=True, exist_ok=True)

    # per-photo 分數矩陣:每一折模型對「全部」照片打分(train+test 都打,供稽核)
    photo_scores = np.full((len(X), k), np.nan, dtype=np.float64)
    test_fold = np.full(len(X), -1, dtype=int)

    for f, (tri, tei) in enumerate(gkf.split(X, y, groups=g)):
        pipe = build_pipe()
        pipe.fit(X[tri], y[tri])
        photo_scores[:, f] = pipe.predict_proba(X)[:, 1]   # 全部照片(含 in-train)
        test_fold[tei] = f
        joblib.dump(pipe, folds_dir / f"fold_{f}.joblib")
    logger.info(f"  ✓ dump 了 {k} 個 fold 模型 → {folds_dir}")

    # 聚合到 session 級(每 ID 對每折取「照片平均」,對齊 _pool_to_id 的 y_score mean)
    cols = [f"fold_{i}" for i in range(k)]
    df = pd.DataFrame(photo_scores, columns=cols)
    df["ID"] = ids
    df["y_true"] = y
    df["test_fold"] = test_fold
    agg = {c: "mean" for c in cols}
    agg["y_true"] = "first"
    agg["test_fold"] = "first"
    sess = df.groupby("ID", as_index=False).agg(agg)

    # 部署用:10 折平均(API 對新受測者就是這個)
    sess["ensemble_mean"] = sess[cols].mean(axis=1)
    # 稽核用:每個 session 取「它自己那一折(test_fold)」的分數 = 應等於落地 OOF
    sess["oof_repro"] = [sess.loc[i, f"fold_{int(sess.loc[i, 'test_fold'])}"]
                         for i in range(len(sess))]

    sess.to_csv(OUT / f"fold_scores_{variant}.csv", index=False, encoding="utf-8")

    # ── 逐 session 驗證可複現:my(test-fold 分數/fold) vs 落地 oof_scores.csv ──
    oof = load_landed_oof(variant)[["ID", "y_score", "fold"]].rename(
        columns={"y_score": "oof_y_score", "fold": "oof_fold"})
    chk = sess.merge(oof, on="ID", how="inner")
    chk["score_diff"] = (chk["oof_repro"] - chk["oof_y_score"]).abs()
    chk["fold_match"] = (chk["test_fold"] == chk["oof_fold"])
    chk[["ID", "test_fold", "oof_fold", "fold_match",
         "oof_repro", "oof_y_score", "score_diff"]].to_csv(
        OUT / f"reproduce_check_{variant}.csv", index=False, encoding="utf-8")

    n_common = len(chk)
    fold_mismatch = int((~chk["fold_match"]).sum())
    max_diff = float(chk["score_diff"].max()) if n_common else float("nan")
    score_mismatch = int((chk["score_diff"] > TOL).sum())
    passed = (n_common == len(sess) == len(oof)
              and fold_mismatch == 0 and score_mismatch == 0)
    logger.info(f"  驗證:sessions(mine={len(sess)}, oof={len(oof)}, common={n_common})  "
                f"fold_mismatch={fold_mismatch}  max_score_diff={max_diff:.3e}  "
                f"score_mismatch(>{TOL:g})={score_mismatch}  → "
                f"{'PASS' if passed else 'FAIL'}")

    return {"variant": variant, "out_name": out_name, "n_folds": k,
            "n_sessions_mine": int(len(sess)), "n_sessions_oof": int(len(oof)),
            "n_common": n_common, "fold_mismatch": fold_mismatch,
            "max_score_diff": max_diff, "score_mismatch": score_mismatch,
            "reproduced": bool(passed)}


def main() -> None:
    logger.info(f"cohort={COHORT}  emb={EMB}  bg={BG}  photo={PHOTO}  "
                f"C={LR_C}  folds={N_SPLITS}  seed={SEED_FOLD}")
    logger.info(f"輸出目錄:{OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    summary = [process_variant(v, name) for v, name in VARIANTS.items()]

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    all_ok = all(s["reproduced"] for s in summary)
    logger.info("=" * 78)
    logger.info(f"完成。可複現驗證:{'全部 PASS' if all_ok else '有 FAIL(見上)'}")
    logger.info(f"  10 折模型:{OUT / 'folds'}/<variant>/fold_<k>.joblib")
    logger.info(f"  分數矩陣:{OUT}/fold_scores_<variant>.csv")
    logger.info(f"  對照檔:{OUT}/reproduce_check_<variant>.csv、summary.json")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
