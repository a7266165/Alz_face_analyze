"""Meta 訓練流程(單一 session 層級):讀 base OOF + 人口學 → 組 session 特徵表 →
TabPFN v3 fold-aligned OOF。

統一一套 feature combo(META_FEATURE_SETS),全部以 visit(session)為訓練樣本、評估交給
src.common.evaluate 做 eval_by_subject(GroupKFold-by-base_id,無 leakage)。asymmetry 一律用
logistic(asymmetry_LR_score),不再用 scorer;asym variant 由參數決定。
"""
import numpy as np
import pandas as pd

from src.age import build_cohort_with_age_error
from src.common.cohort import load_demographics
from src.embedding.classification import CLASSIFIERS, oof_paths
from src.meta.classifier import make_meta_clf

ASYM_VARIANTS = ("differences", "absolute_differences",
                 "relative_differences", "absolute_relative_differences")

# session 層級可用的全部欄(canonical 欄名);下列 8 個 combo 為其子集。各 combo 共用同一張 session
# 表(同母體 = 有 embedding 的 sessions ∩ 有年齡預測者),故可直接互比。
ALL_FEATURE_COLS = ["real_age", "age_error", "embedding_LR_score",
                    "asymmetry_LR_score", "bmi", "mmse", "casi"]
# 帶這兩欄之一的 combo 才依賴 embedding OOF(→ 有 variant / C 軸);其餘為純認知 combo(只跑一次)。
OOF_FEATURE_COLS = ("embedding_LR_score", "asymmetry_LR_score")
META_FEATURE_SETS = {
    "mmse":                 ["mmse"],
    "casi":                 ["casi"],
    "mmse_casi":            ["mmse", "casi"],
    "core4":                ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score"],
    "core4_bmi":            ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "bmi"],
    "core4_bmi_mmse":       ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "bmi", "mmse"],
    "core4_bmi_casi":       ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "bmi", "casi"],
    "core4_bmi_mmse_casi":  ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "bmi", "mmse", "casi"],
}


def feature_set_needs_oof(feature_cols):
    """這組特徵是否吃 embedding OOF(→ 需 variant / C 軸;否則純認知,variant/C 無關)。"""
    return any(c in OOF_FEATURE_COLS for c in feature_cols)


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


def meta_oof(X, y, fold, *, meta_clf="tabpfn_v3", seed=42, device="auto"):
    """fold-aligned OOF：對每個 fold k 在 fold≠k 上 fit、預測 k，回正類機率陣列。

    meta_clf ∈ META_CLASSIFIERS(tabpfn_v3 / xgb);同一 estimator 跨折重 fit
    (TabPFN 只換 in-context 訓練集、XGB 每折重訓),兩者皆走 predict_proba[:, 1]。
    """
    clf = make_meta_clf(meta_clf, seed=seed, device=device)
    oof = np.full(len(y), np.nan)
    for k in np.unique(fold):
        te = fold == k
        clf.fit(X[~te], y[~te])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def session_feature_table(cohort, *, variant="relative_differences", emb="arcface",
                          bg_mode="background", photo_mode="mean", reducer="no_drop",
                          base_clf="logistic", lr_C=1.0, root=None):
    """組 per-session 全欄特徵表
    [ID, y_true, fold, embedding_LR_score, asymmetry_LR_score, real_age, age_error, bmi, mmse, casi]。

    visit 層級(不收斂到 subject):
      - embedding_LR_score / asymmetry_LR_score:original / <variant> 兩條 forward OOF 直接讀落地
        (base_oof,不重訓;asym 一律 logistic);
      - real_age / age_error / mmse / casi:取自 src.age.build_cohort_with_age_error(內含 cohort_list 的
        Age/MMSE/CASI + 年齡誤差);
      - bmi:取自 demographics(量測值)。
    全部 by ID join;母體 = 兩條 OOF ∩ 有年齡預測者(inner),bmi 以 left join 併入故不縮母體。
    fold/y_true 取 original-OOF 的(GroupKFold-by-base_id → 同 subject 各 visit 同折,無 leakage)。

    Args:
        cohort: (p_visit, p_score, hc_visit, hc_score) 4-token。
        variant: asymmetry feature 的 variant(差異圖類型,見 ASYM_VARIANTS)。
        emb / bg_mode / photo_mode / reducer: 定位落地 OOF 用(須與 embedding 產出一致)。
        base_clf: original 與 asymmetry 共用的 base 模型(預設 logistic)。
        lr_C: base_clf 為 logistic 時定位 C_<value> 落地格;original/asym 共用同一個 C。
        root: embedding OOF 根目錄,預設 EMBEDDING_CLASSIFICATION_DIR。
    """
    orig = base_oof(cohort, emb, "original", bg_mode, photo_mode, base_clf,
                    reducer=reducer, lr_C=lr_C, root=root)
    asym = base_oof(cohort, emb, variant, bg_mode, photo_mode, base_clf,
                    reducer=reducer, lr_C=lr_C, root=root)
    age = (build_cohort_with_age_error(*cohort)[["ID", "MMSE", "CASI", "real_age", "age_error"]]
           .rename(columns={"MMSE": "mmse", "CASI": "casi"}))
    bmi = load_demographics()[["ID", "BMI"]].drop_duplicates("ID").copy()
    bmi["bmi"] = pd.to_numeric(bmi["BMI"], errors="coerce")

    t = (orig[["ID", "y_true", "fold", "y_score"]]
         .rename(columns={"y_score": "embedding_LR_score"})
         .merge(asym[["ID", "y_true", "y_score"]]
                .rename(columns={"y_score": "asymmetry_LR_score", "y_true": "y_true_a"}),
                on="ID", how="inner")
         .merge(age, on="ID", how="inner")
         .merge(bmi[["ID", "bmi"]], on="ID", how="left"))
    assert (t["y_true"].to_numpy() == t["y_true_a"].to_numpy()).all(), \
        "original 與 asymmetry OOF 的 y_true 不一致"
    return t[["ID", "y_true", "fold"] + ALL_FEATURE_COLS]


def oof_from_table(table, feature_cols, *, meta_clf="tabpfn_v3", seed=42, device="auto"):
    """從 session 特徵表取指定欄 → meta stacker fold-aligned OOF,回標準 OOF[ID, y_true, y_score, fold]。

    meta_clf ∈ META_CLASSIFIERS;fold 取 original-OOF 的 GroupKFold-by-base_id(逐折 fold≠k 訓練、
    預測 k,無 leakage);subject 評估交給 src.common.evaluate.evaluate。多個 feature set / meta_clf 可
    共用同一張 table 直接互比。
    """
    fold = table["fold"].to_numpy(dtype=int)
    if (fold < 0).all():
        raise ValueError(
            "original-OOF fold 全為 -1:base_clf 須為有 CV 折的 classifier(如 logistic),不能用短路 scorer")
    X = table[list(feature_cols)].to_numpy(dtype=float)
    y = table["y_true"].to_numpy(dtype=int)
    meta = meta_oof(X, y, fold, meta_clf=meta_clf, seed=seed, device=device)
    return pd.DataFrame({"ID": table["ID"].to_numpy(), "y_true": y,
                         "y_score": meta, "fold": fold})


def session_oof(cohort, *, feature_cols, variant="relative_differences", emb="arcface",
                bg_mode="background", photo_mode="mean", reducer="no_drop",
                base_clf="logistic", lr_C=1.0, meta_clf="tabpfn_v3",
                root=None, seed=42, device="auto"):
    """便捷一呼:組 session 特徵表(session_feature_table)→ 取 feature_cols → oof_from_table。

    需對同 cohort 跑多組 feature set / meta_clf 時,建議改在外層 session_feature_table 取一次表、
    各自 oof_from_table,避免重複讀 OOF。
    """
    t = session_feature_table(cohort, variant=variant, emb=emb, bg_mode=bg_mode,
                              photo_mode=photo_mode, reducer=reducer,
                              base_clf=base_clf, lr_C=lr_C, root=root)
    return oof_from_table(t, feature_cols, meta_clf=meta_clf, seed=seed, device=device)
