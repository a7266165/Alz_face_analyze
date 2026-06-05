"""精簡 meta：把 [age, MMSE, CASI, embedding-original OOF, asymmetry OOF] 餵 TabPFN v3。

asymmetry 特徵窮舉 variant × scorer;兩個 base OOF 以共用模組(src.embedding.classification)
forward 重新產生(GroupKFold by base_id),meta 沿用 original-OOF 的 fold 做 fold-aligned 評估。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.cohort import base_id_of, cohort_list
from src.common.evaluate import compute_clf_metrics
from src.common.features import load_feature_matrix
from src.embedding.classification import (
    CLASSIFIERS, build_classifier, build_reducer, build_scorer, train,
)

__all__ = [
    "ASYM_VARIANTS", "ASYM_METHODS", "FEATURE_COLS",
    "base_oof", "to_subject", "subject_demographics",
    "build_feature_table", "make_tabpfn_v3", "tabpfn_oof", "sweep",
]

ASYM_VARIANTS = ("differences", "absolute_differences",
                 "relative_differences", "absolute_relative_differences")
ASYM_METHODS = ("l2_norm", "centroid_dist", "lda_projection")
FEATURE_COLS = ["age", "mmse", "casi", "oof_original", "oof_asym"]

_V3_CKPT = (Path.home() / "AppData" / "Roaming" / "tabpfn"
            / "tabpfn-v3-classifier-v3_default.ckpt")


def _build_estimator(model, *, reducer="no_drop", lr_C=1.0):
    """model → (build_estimator_thunk, score_method, needs_cv);比照 producer 的 router。

    scorer(l2_norm/centroid_dist/lda_projection)直接回 build_scorer;classifier
    (logistic/xgb)串成 scaler?/reducer?/clf 單一 Pipeline。回 0-arg thunk 供每折建新 estimator。
    """
    if model not in CLASSIFIERS:
        est, score_method, needs_cv = build_scorer(model)
        return (lambda: build_scorer(model)[0]), score_method, needs_cv

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def _make():
        rd = build_reducer(reducer)
        clf, needs_scaler = build_classifier(model, lr_C=lr_C)
        steps = []
        if needs_scaler:
            steps.append(("scaler", StandardScaler()))
        if rd != "passthrough":
            steps.append(("reducer", rd))
        steps.append(("classifier", clf))
        return Pipeline(steps)

    return _make, "predict_proba", True


def base_oof(cohort, emb, variant, bg_mode, photo_mode, model, *,
             reducer="no_drop", lr_C=1.0):
    """單一 base 模型的 forward OOF,回 session 層級 DataFrame[ID, y_true, y_score, fold]。

    Args:
        cohort: (p_visit, p_score, hc_visit, hc_score) 4-token。
        emb: embedding backbone(arcface/…)。
        variant: original | differences | absolute_differences | … (feature type)。
        bg_mode: background | no_background
        photo_mode: mean | all
        model: logistic/xgb(classifier)| l2_norm/centroid_dist/lda_projection(scorer)。
    """
    full = cohort_list(*cohort)
    label_map = dict(zip(full["ID"], (full["Group"] == "P").astype(int)))
    X, ids = load_feature_matrix(full["ID"].tolist(), emb, variant, bg_mode, photo_mode)
    y = np.array([label_map[i] for i in ids], dtype=int)
    build_estimator, score_method, needs_cv = _build_estimator(
        model, reducer=reducer, lr_C=lr_C)
    return train(X, ids, y, build_estimator, score_method, needs_cv, "forward")


def to_subject(oof):
    """session 層級 OOF → subject 層級[base_id, y_true, y_score, fold]。

    y_score 取受試者各 visit 平均、fold 取 max(同 subject 各 visit 必同折);
    比照 src.common.evaluate._prep_oof 的 eval_by_subject。
    """
    d = oof.copy()
    d["base_id"] = d["ID"].map(base_id_of)
    return d.groupby("base_id", as_index=False).agg(
        y_true=("y_true", "first"), y_score=("y_score", "mean"), fold=("fold", "max"))


def subject_demographics(cohort):
    """cohort → subject 層級[base_id, age, mmse, casi, label]，每 base_id 取首訪。"""
    demo = cohort_list(*cohort).copy()
    demo["base_id"] = demo["ID"].map(base_id_of)
    demo = demo.sort_values("ID").groupby("base_id", as_index=False).first()
    demo["label"] = (demo["Group"] == "P").astype(int)
    return demo[["base_id", "Age", "MMSE", "CASI", "label"]].rename(
        columns={"Age": "age", "MMSE": "mmse", "CASI": "casi"})


def build_feature_table(demo, sub_original, sub_asym):
    """以 base_id inner-join 三來源 → (table, X, y, fold)，5 欄見 FEATURE_COLS。

    fold 取自 original-OOF(real 0..k-1,當 meta 切分);assert label == original 的 y_true。
    """
    t = (demo.merge(
            sub_original[["base_id", "y_score", "fold", "y_true"]]
            .rename(columns={"y_score": "oof_original"}), on="base_id", how="inner")
         .merge(sub_asym[["base_id", "y_score"]]
                .rename(columns={"y_score": "oof_asym"}), on="base_id", how="inner"))
    assert (t["label"].to_numpy() == t["y_true"].to_numpy()).all(), "label 與 OOF y_true 不一致"
    X = t[FEATURE_COLS].to_numpy(dtype=float)
    y = t["label"].to_numpy(dtype=int)
    fold = t["fold"].to_numpy(dtype=int)
    return t, X, y, fold


def make_tabpfn_v3(seed=42, device="auto"):
    """建立指向 v3 ckpt 的 TabPFNClassifier(找不到 ckpt 則退回套件預設權重)。"""
    from tabpfn import TabPFNClassifier
    if _V3_CKPT.exists():
        return TabPFNClassifier(model_path=str(_V3_CKPT), device=device,
                                random_state=seed, ignore_pretraining_limits=True)
    return TabPFNClassifier(device=device, random_state=seed,
                            ignore_pretraining_limits=True)


def tabpfn_oof(X, y, fold, *, seed=42, device="auto"):
    """fold-aligned OOF：對每個 fold k 在 fold≠k 上 fit、預測 k，回正類機率陣列。

    同一個 classifier 跨折重 fit(TabPFN 的 fit 只換 in-context 訓練集),避免每折重載 ckpt。
    """
    clf = make_tabpfn_v3(seed=seed, device=device)
    oof = np.full(len(y), np.nan)
    for k in np.unique(fold):
        te = fold == k
        clf.fit(X[~te], y[~te])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def sweep(cohort, *, emb="arcface", bg_mode="background", photo_mode="mean",
          reducer="no_drop", variants=ASYM_VARIANTS, methods=ASYM_METHODS,
          seed=42, device="auto"):
    """窮舉 asymmetry variant × scorer,各訓練 TabPFN v3,回 (leaderboard, {(variant,method): oof_df})。

    embedding-original OOF 只算一次、跨組合共用;每組合的 5 特徵 = demographics +
    original-OOF + 該組合 asymmetry-OOF,指標走 src.common.evaluate.compute_clf_metrics。
    """
    demo = subject_demographics(cohort)
    sub_o = to_subject(base_oof(cohort, emb, "original", bg_mode, photo_mode,
                                "logistic", reducer=reducer, lr_C=1.0))
    rows, oof_dump = [], {}
    for variant in variants:
        for method in methods:
            sub_a = to_subject(base_oof(cohort, emb, variant, bg_mode, photo_mode, method))
            t, X, y, fold = build_feature_table(demo, sub_o, sub_a)
            meta = tabpfn_oof(X, y, fold, seed=seed, device=device)
            metrics = compute_clf_metrics(y, meta, n_bootstrap=100, seed=seed)
            rows.append({"variant": variant, "method": method, **metrics})
            oof_dump[(variant, method)] = pd.DataFrame(
                {"base_id": t["base_id"].to_numpy(), "y_true": y,
                 "y_score": meta, "fold": fold})
    leaderboard = pd.DataFrame(rows).sort_values(
        "auc", ascending=False, ignore_index=True)
    return leaderboard, oof_dump
