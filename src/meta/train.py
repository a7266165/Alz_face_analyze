"""Meta 訓練流程:讀 base OOF → subject 表組裝 → TabPFN v3 fold-aligned OOF → sweep。"""
import numpy as np
import pandas as pd

from src.common.cohort import base_id_of, cohort_list
from src.common.evaluate import compute_clf_metrics
from src.embedding.classification import CLASSIFIERS, oof_paths
from src.meta.classifier import make_tabpfn_v3

ASYM_VARIANTS = ("differences", "absolute_differences",
                 "relative_differences", "absolute_relative_differences")
ASYM_METHODS = ("l2_norm", "centroid_dist", "lda_projection")
# full = age/MMSE/CASI/original-OOF/asym-OOF;其餘三個為只用 MMSE/CASI 的對照(±original-OOF、±asym-OOF)。
FEATURE_SETS = {
    "full":           ["age", "mmse", "casi", "oof_original", "oof_asym"],
    "mmse_casi":      ["mmse", "casi"],
    "mmse_casi_orig": ["mmse", "casi", "oof_original"],
    "mmse_casi_asym": ["mmse", "casi", "oof_asym"],
}


def base_oof(cohort, emb, variant, bg_mode, photo_mode, model, *,
             reducer="no_drop", lr_C=1.0, root=None):
    """讀 workspace 落地的 forward OOF,回 session 層級 DataFrame[ID, y_true, y_score, fold]。

    base 模型不在此重訓——OOF 由 embedding 分類流程(scripts/embedding/classification)
    產出並落地,此處只按 embedding 用的同一組參數定位、讀檔;對應檔不存在即報錯。

    Args:
        cohort: (p_visit, p_score, hc_visit, hc_score) 4-token。
        emb: embedding backbone(arcface/…)。
        variant: original | differences | absolute_differences | … (feature type)。
        bg_mode: background | no_background
        photo_mode: mean | all
        model: logistic/xgb(classifier)| l2_norm/centroid_dist/lda_projection(scorer)。
        reducer / lr_C: 定位 classifier 落地格用(scorer 忽略);須與 embedding 產出時一致。
        root: embedding OOF 根目錄,預設 EMBEDDING_CLASSIFICATION_DIR。
    """
    path = oof_paths(cohort, bg_mode, emb, variant, photo_mode, reducer, model,
                     "forward", lr_C=lr_C, root=root)[0]
    if not path.exists():
        param = f" lr_C={lr_C}" if model in CLASSIFIERS else ""
        raise FileNotFoundError(
            f"找不到 base OOF:{path}\n"
            f"  請先跑 embedding forward 分類產生此格(emb={emb} variant={variant} "
            f"model={model} reducer={reducer}{param})。")
    return pd.read_csv(path)


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


def build_base_table(demo, sub_original):
    """demo + original-OOF 以 base_id inner-join → subject 表(age, mmse, casi, oof_original, label, fold)。

    fold/y_true 取自 original-OOF:fold 為所有 feature set 共用的 meta 切分;assert label == y_true。
    """
    t = demo.merge(
        sub_original[["base_id", "y_score", "fold", "y_true"]]
        .rename(columns={"y_score": "oof_original"}), on="base_id", how="inner")
    assert (t["label"].to_numpy() == t["y_true"].to_numpy()).all(), "label 與 OOF y_true 不一致"
    return t


def add_asym(base_table, sub_asym):
    """base 表再 inner-join 某 (variant, scorer) 的 asymmetry-OOF → 多一欄 oof_asym。"""
    return base_table.merge(
        sub_asym[["base_id", "y_score"]].rename(columns={"y_score": "oof_asym"}),
        on="base_id", how="inner")


def slice_xy(table, feature_cols):
    """從 subject 表取 (X, y, fold):X 欄位由 feature_cols 指定、y=label、fold 為 meta 切分。"""
    X = table[feature_cols].to_numpy(dtype=float)
    y = table["label"].to_numpy(dtype=int)
    fold = table["fold"].to_numpy(dtype=int)
    return X, y, fold


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


def _require_base_oof(cohort, emb, bg_mode, photo_mode, reducer, base_clf, base_lr_C,
                      variants, methods, need_asym, root):
    """預檢:把所有需要的 base OOF 路徑掃一遍,缺的一次列齊再 raise(避免跑到一半才爆)。"""
    needed = [("original", base_clf, base_lr_C)]
    if need_asym:
        needed += [(v, m, 1.0) for v in variants for m in methods]
    missing = [str(oof_paths(cohort, bg_mode, emb, v, photo_mode, reducer, m,
                             "forward", lr_C=c, root=root)[0])
               for v, m, c in needed
               if not oof_paths(cohort, bg_mode, emb, v, photo_mode, reducer, m,
                                "forward", lr_C=c, root=root)[0].exists()]
    if missing:
        raise FileNotFoundError(
            "缺少以下 base OOF(請先跑 embedding forward 分類):\n  " + "\n  ".join(missing))


def sweep(cohort, *, emb="arcface", bg_mode="background", photo_mode="mean",
          reducer="no_drop", base_clf="logistic", base_lr_C=1.0,
          variants=ASYM_VARIANTS, methods=ASYM_METHODS,
          feature_sets=FEATURE_SETS, seed=42, device="auto", root=None):
    """窮舉 feature set × asymmetry variant × scorer,各訓練 TabPFN v3。

    不含 oof_asym 的 feature set 與 variant/scorer 無關,各只跑一次(variant/method 留空);
    含 oof_asym 者對每個 (variant, scorer) 各跑一次。base OOF 一律讀 workspace 既有落地檔
    (original 用 base_clf/base_lr_C 定位、asym 用各 scorer),所有 feature set 共用 original-OOF
    的 fold。開跑前先預檢所有需要的落地檔,缺檔即報錯。指標走 src.common.evaluate.compute_clf_metrics。
    回 (leaderboard_df, {(feature_set, variant, method): oof_df}),variant/method 對 asym-無關
    組合為空字串。
    """
    indep = {k: v for k, v in feature_sets.items() if "oof_asym" not in v}
    asym = {k: v for k, v in feature_sets.items() if "oof_asym" in v}
    _require_base_oof(cohort, emb, bg_mode, photo_mode, reducer, base_clf, base_lr_C,
                      variants, methods, need_asym=bool(asym), root=root)

    demo = subject_demographics(cohort)
    base = build_base_table(demo, to_subject(base_oof(
        cohort, emb, "original", bg_mode, photo_mode, base_clf,
        reducer=reducer, lr_C=base_lr_C, root=root)))

    jobs = [(fs, cols, base, "", "") for fs, cols in indep.items()]
    for variant in variants:
        for method in methods:
            t = add_asym(base, to_subject(
                base_oof(cohort, emb, variant, bg_mode, photo_mode, method, root=root)))
            jobs += [(fs, cols, t, variant, method) for fs, cols in asym.items()]

    rows, oof_dump = [], {}
    for fs, cols, table, variant, method in jobs:
        X, y, fold = slice_xy(table, cols)
        meta = tabpfn_oof(X, y, fold, seed=seed, device=device)
        metrics = compute_clf_metrics(y, meta, n_bootstrap=100, seed=seed)
        rows.append({"feature_set": fs, "variant": variant, "method": method, **metrics})
        oof_dump[(fs, variant, method)] = pd.DataFrame(
            {"base_id": table["base_id"].to_numpy(), "y_true": y,
             "y_score": meta, "fold": fold})

    leaderboard = pd.DataFrame(rows).sort_values(
        "auc", ascending=False, ignore_index=True)
    return leaderboard, oof_dump
