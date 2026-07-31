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
from src.embedding.classification import CLASSIFIERS, inner_path, oof_paths
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
    "core3":                ["embedding_LR_score", "asymmetry_LR_score", "age_error"],  # core4 去 real_age(年齡 confound)
    "core4_bmi":            ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "bmi"],
    "core4_mmse":           ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "mmse"],
    "core4_casi":           ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "casi"],
    "core4_bmi_mmse":       ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "bmi", "mmse"],
    "core4_bmi_casi":       ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "bmi", "casi"],
    "core4_bmi_mmse_casi":  ["real_age", "age_error", "embedding_LR_score", "asymmetry_LR_score", "bmi", "mmse", "casi"],
    # PDF Step 12 對照組:配 meta_clf=mean(不訓練,逐列平均)就是「單一 base 分數」
    # 與「兩個 base 分數的簡單平均」。meta layer 的加值 = core3/core4 減掉這三格。
    "embedding_only":       ["embedding_LR_score"],
    "asymmetry_only":       ["asymmetry_LR_score"],
    "bases":                ["embedding_LR_score", "asymmetry_LR_score"],
}

# meta_clf=mean 只對這三組有意義:它不 fit,直接把特徵當分數用,所以特徵必須本身
# 就是「AD 的機率」。套到 mmse/casi 會得到 1-AUC(認知分數方向相反),套到含 real_age
# 的 combo 會得到落在 [0,1] 外的分數(閾值型指標全毀)。
BASELINE_FEATURE_SETS = ("embedding_only", "asymmetry_only", "bases")


def feature_set_needs_oof(feature_cols):
    """這組特徵是否吃 embedding OOF(→ 需 variant / C 軸;否則純認知,variant/C 無關)。"""
    return any(c in OOF_FEATURE_COLS for c in feature_cols)


def base_oof(cohort, emb, variant, bg_mode, photo_mode, model, *,
             reducer="no_drop", lr_C=1.0, seed=0, root=None):
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
        seed: repeated-CV 折分 seed(路徑 seed_<N>);須與 embedding 產出時一致(預設 0)。
        root: embedding OOF 根目錄,預設 EMBEDDING_CLASSIFICATION_DIR。
    """
    path = oof_paths(cohort, bg_mode, emb, variant, photo_mode, reducer, model,
                     "forward", lr_C=lr_C, seed=seed, root=root)[0]
    if not path.exists():
        param = f" lr_C={lr_C}" if model in CLASSIFIERS else ""
        raise FileNotFoundError(
            f"找不到 base OOF:{path}\n"
            f"  請先跑 embedding forward 分類產生此格(emb={emb} variant={variant} "
            f"model={model} reducer={reducer}{param})。")
    return pd.read_csv(path)


def base_inner(cohort, emb, variant, bg_mode, photo_mode, model, *,
               reducer="no_drop", lr_C=1.0, seed=0, root=None):
    """讀落地的內折 OOF → DataFrame[ID, y_true, y_score, outer_fold, inner_fold]。

    這是 stacking 的 meta 訓練列:對外折 k,這些分數來自「只在 outer-train 的
    4/5 上 fit」的模型,故第 k 折的人完全沒被看過。與 base_oof 讀的
    oof_scores.csv(= 用完整 outer-train 重訓後預測 outer-test)成對使用。
    """
    path = inner_path(cohort, bg_mode, emb, variant, photo_mode, reducer, model,
                      "forward", lr_C=lr_C, seed=seed, root=root)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到內折 OOF:{path}\n"
            f"  請用 --inner-folds 5 重跑該格 embedding forward 分類"
            f"(emb={emb} variant={variant} model={model} reducer={reducer} lr_C={lr_C} "
            f"seed={seed})。")
    return pd.read_csv(path)


def meta_oof(X, y, fold, *, meta_clf="tabpfn_v3", seed=42, device="auto"):
    """fold-aligned OOF：對每個 fold k 在 fold≠k 上 fit、預測 k，回正類機率陣列。

    meta_clf ∈ META_CLASSIFIERS(tabpfn_v3 / xgb);同一 estimator 跨折重 fit
    (TabPFN 只換 in-context 訓練集、XGB 每折重訓),兩者皆走 predict_proba[:, 1]。

    註:這條路徑的訓練列(fold≠k)其 base 分數來自「排除自己那一折」的 base 模型,
    那些模型都看過第 k 折,故 meta 是在看過 outer test 的特徵上被 fit 的。
    要避免這件事請改用 meta_oof_nested(吃內折表)。
    """
    clf = make_meta_clf(meta_clf, seed=seed, device=device)
    oof = np.full(len(y), np.nan)
    for k in np.unique(fold):
        te = fold == k
        clf.fit(X[~te], y[~te])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def meta_oof_nested(train_table, test_table, feature_cols, *, meta_clf="tabpfn_v3",
                    seed=42, device="auto"):
    """nested 版的 fold-aligned OOF:每個外折 k 各自 fit 一個 meta(共 k 個)。

    對外折 k:
      訓練列 = 內折表裡 outer_fold == k 的列(第 k 折的人不在其中,且這些分數是由
               沒看過第 k 折的 base 模型產生的);
      測試列 = 外折表裡 fold == k 的列(分數來自用完整 outer-train 重訓的 base 模型)。

    回正類機率陣列,長度與 test_table 相同。
    """
    cols = list(feature_cols)
    clf = make_meta_clf(meta_clf, seed=seed, device=device)
    fold = test_table["fold"].to_numpy(dtype=int)
    ids = test_table["ID"].to_numpy()
    Xte = test_table[cols].to_numpy(dtype=float)
    oof = np.full(len(test_table), np.nan)
    for k in np.unique(fold):
        tr = train_table[train_table["outer_fold"] == k]
        if tr.empty:
            raise ValueError(f"內折表缺少 outer_fold={k} 的列(內折表與外折表不成對?)")
        te = fold == k
        leak = set(tr["ID"]) & set(ids[te])
        if leak:                          # 保險絲:內折表本來就不該含該折的人
            raise ValueError(f"outer_fold={k} 的 meta 訓練列含有測試折的 ID "
                             f"({len(leak)} 個,如 {sorted(leak)[:3]})")
        clf.fit(tr[cols].to_numpy(dtype=float), tr["y_true"].to_numpy(dtype=int))
        oof[te] = clf.predict_proba(Xte[te])[:, 1]
    return oof


def covariate_table(cohort):
    """ID 層共變數表 [ID, real_age, age_error, bmi, mmse, casi]。

    這些欄都不是在本資料上訓練出來的模型輸出：real_age / mmse / casi 是量測值，
    age_error 來自凍結的 MiVOLO 預訓練權重（scripts/age/predict.py 只做推論，
    全 src/age 沒有任何 fit），bmi 是量測值。故無 fold 依賴，兩層 CV 都可直接 join。

    age/mmse/casi 取自 build_cohort_with_age_error（內含 cohort_list 的 Age/MMSE/CASI
    + 年齡誤差）；bmi 取自 demographics，以 left join 併入故不縮母體。
    """
    age = (build_cohort_with_age_error(*cohort)[["ID", "MMSE", "CASI", "real_age", "age_error"]]
           .rename(columns={"MMSE": "mmse", "CASI": "casi"}))
    bmi = load_demographics()[["ID", "BMI"]].drop_duplicates("ID").copy()
    bmi["bmi"] = pd.to_numeric(bmi["BMI"], errors="coerce")
    return age.merge(bmi[["ID", "bmi"]], on="ID", how="left")


def session_feature_table(cohort, *, variant="relative_differences", emb="arcface",
                          bg_mode="background", photo_mode="mean", reducer="no_drop",
                          base_clf="logistic", lr_C=1.0, seed=0, complete_case=True, root=None):
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
        complete_case: True(預設)→ 丟掉 mmse/casi 缺值的 session(complete-case,所有 combo 同母體公平比較,
            且全表零 NaN);False → 保留(full cohort,缺值交給能吃 NaN 的 stacker)。base/embedding 不受此影響。
        root: embedding OOF 根目錄,預設 EMBEDDING_CLASSIFICATION_DIR。
    """
    orig = base_oof(cohort, emb, "original", bg_mode, photo_mode, base_clf,
                    reducer=reducer, lr_C=lr_C, seed=seed, root=root)
    asym = base_oof(cohort, emb, variant, bg_mode, photo_mode, base_clf,
                    reducer=reducer, lr_C=lr_C, seed=seed, root=root)
    return _merge_bases(orig, asym, covariate_table(cohort), ("fold",),
                        complete_case=complete_case)


def _merge_bases(orig, asym, cov, fold_cols, *, complete_case):
    """兩條 base 分數 + 共變數 → 特徵表。fold_cols 是要保留的折欄。

    外折表 fold_cols=("fold",)、內折表 fold_cols=("outer_fold",)。折欄一起當 join key:
    內折表裡一個 ID 有 9 列(每個它屬於訓練集的外折一列),只用 ID join 會產生
    9x9 的笛卡兒積。
    """
    keys = ["ID", *fold_cols]
    t = (orig[[*keys, "y_true", "y_score"]]
         .rename(columns={"y_score": "embedding_LR_score"})
         .merge(asym[[*keys, "y_true", "y_score"]]
                .rename(columns={"y_score": "asymmetry_LR_score", "y_true": "y_true_a"}),
                on=keys, how="inner")
         .merge(cov[["ID", "mmse", "casi", "real_age", "age_error"]], on="ID", how="inner")
         .merge(cov[["ID", "bmi"]], on="ID", how="left"))
    assert (t["y_true"].to_numpy() == t["y_true_a"].to_numpy()).all(), \
        "original 與 asymmetry OOF 的 y_true 不一致"
    if complete_case:                       # 丟認知缺值 session → 全表零 NaN、各 combo 同母體比較
        t = t[t["mmse"].notna() & t["casi"].notna()].reset_index(drop=True)
    return t[["ID", "y_true", *fold_cols] + ALL_FEATURE_COLS]


def inner_feature_table(cohort, *, variant="relative_differences", emb="arcface",
                        bg_mode="background", photo_mode="mean", reducer="no_drop",
                        base_clf="logistic", lr_C=1.0, seed=0, complete_case=True,
                        root=None):
    """meta 的**訓練**列:[ID, y_true, outer_fold, <ALL_FEATURE_COLS>]。

    與 session_feature_table 完全對稱,差別只在 base 分數讀的是 inner_scores.csv
    而非 oof_scores.csv,且折欄是 outer_fold(這個 ID 屬於訓練集的折)而非 fold
    (這個 ID 被留出的折)。每個 ID 有 k-1 列。

    共變數(real_age / age_error / bmi / mmse / casi)與外折表用同一張 covariate_table
    ——它們不是在本資料上訓練出來的,沒有 fold 依賴。
    """
    orig = base_inner(cohort, emb, "original", bg_mode, photo_mode, base_clf,
                      reducer=reducer, lr_C=lr_C, seed=seed, root=root)
    asym = base_inner(cohort, emb, variant, bg_mode, photo_mode, base_clf,
                      reducer=reducer, lr_C=lr_C, seed=seed, root=root)
    return _merge_bases(orig, asym, covariate_table(cohort), ("outer_fold",),
                        complete_case=complete_case)


def oof_from_table(table, feature_cols, *, inner=None, meta_clf="tabpfn_v3", seed=42,
                   device="auto"):
    """從 session 特徵表取指定欄 → meta stacker fold-aligned OOF[ID, y_true, y_score, fold]。

    inner 給定(inner_feature_table 的產物)→ 走 nested:每個外折各自 fit 一個 meta,
    訓練列是該折訓練集的內折分數。這是正確的做法,scripts/meta/run.py 走這條。

    inner=None → 舊路徑:直接用同一張表的 fold≠k 當訓練列。這些列的 base 分數來自
    看過第 k 折的 base 模型,meta 會吃到 outer test 的資訊。保留只為了讓不做 CV 評估的
    呼叫端(如部署匯出)沿用,不應用來產生要報告的數字。

    fold 取 original-OOF 的 GroupKFold-by-base_id(同 subject 各 visit 同折);
    subject 層級評估交給 src.common.evaluate.evaluate。多個 feature set / meta_clf
    可共用同一張 table 直接互比。
    """
    fold = table["fold"].to_numpy(dtype=int)
    if (fold < 0).all():
        raise ValueError(
            "original-OOF fold 全為 -1:base_clf 須為有 CV 折的 classifier(如 logistic),不能用短路 scorer")
    y = table["y_true"].to_numpy(dtype=int)
    if inner is None:
        X = table[list(feature_cols)].to_numpy(dtype=float)
        meta = meta_oof(X, y, fold, meta_clf=meta_clf, seed=seed, device=device)
    else:
        meta = meta_oof_nested(inner, table, feature_cols, meta_clf=meta_clf,
                               seed=seed, device=device)
    return pd.DataFrame({"ID": table["ID"].to_numpy(), "y_true": y,
                         "y_score": meta, "fold": fold})


def session_oof(cohort, *, feature_cols, variant="relative_differences", emb="arcface",
                bg_mode="background", photo_mode="mean", reducer="no_drop",
                base_clf="logistic", lr_C=1.0, meta_clf="tabpfn_v3",
                complete_case=True, root=None, seed=42, device="auto"):
    """便捷一呼:組 session 特徵表(session_feature_table)→ 取 feature_cols → oof_from_table。

    需對同 cohort 跑多組 feature set / meta_clf 時,建議改在外層 session_feature_table 取一次表、
    各自 oof_from_table,避免重複讀 OOF。
    """
    t = session_feature_table(cohort, variant=variant, emb=emb, bg_mode=bg_mode,
                              photo_mode=photo_mode, reducer=reducer,
                              base_clf=base_clf, lr_C=lr_C, complete_case=complete_case, root=root)
    return oof_from_table(t, feature_cols, meta_clf=meta_clf, seed=seed, device=device)
