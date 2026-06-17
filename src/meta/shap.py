"""Meta core3 的逐個案 SHAP:沿用 meta OOF 的折分,逐折無洩漏地解釋 held-out fold
(fold≠k 上 fit stacker、解釋 fold==k 的 session),回每個 session 的 SHAP 值與全域重要性。

- 解釋目標 = stacker 的 predict_proba[:, 1](正類=AD 機率,機率尺度);
  base_value + Σ shap_<feat> ≈ 該 session 的預測機率,逐特徵貢獻可直接相加解讀。
- explainer = KernelExplainer(model-agnostic),背景取訓練折的 kmeans 摘要;core3 僅 3 維,
  KernelExplainer 會枚舉全部 coalition(近似精確),TabPFN/XGB/LR 共用同一語意故可互比。
- 折分取自 session 特徵表的 `fold`(= base OOF 的 GroupKFold-by-base_id,無 leakage)。
"""
import logging

import numpy as np
import pandas as pd
import shap

from src.meta.classifier import make_meta_clf

logger = logging.getLogger(__name__)


def fold_aligned_shap(table, feature_cols, *, meta_clf="tabpfn_v3", seed=42,
                      device="auto", background=25, nsamples="auto"):
    """逐折 fit stacker + KernelExplainer 解釋 held-out fold → 每個 session 的 SHAP。

    Args:
        table: session 特徵表(含 ID / y_true / fold + feature_cols),見 session_feature_table。
        feature_cols: 要解釋的欄(core3 = embedding_LR_score / asymmetry_LR_score / age_error)。
        meta_clf: stacker(tabpfn_v3 / xgb / lr);與落地 cell 同一個。
        seed / device: 同 meta_oof(估計器 seed、TabPFN 裝置)。
        background: 每折背景樣本數(kmeans 摘要的 centroid 數;< 訓練折樣本數時取訓練折樣本數)。
        nsamples: KernelExplainer 的 coalition 取樣數;"auto" 在 3 維會枚舉全部(近似精確)。

    Returns:
        (per_case, importance):
          per_case  — 每 session 一列:ID, y_true, fold, p_pred, base_value, shap_<feat>...
          importance— 每特徵一列:feature, mean_abs_shap, mean_shap, std_abs_shap,
                      importance_pct(mean_abs 佔比), rank(mean_abs 由大到小)。
    """
    fold = table["fold"].to_numpy(dtype=int)
    if (fold < 0).all():
        raise ValueError(
            "fold 全為 -1:base_clf 須為有 CV 折的 classifier(如 logistic),不能用短路 scorer")
    feats = list(feature_cols)
    X = table[feats].to_numpy(dtype=float)
    y = table["y_true"].to_numpy(dtype=int)
    ids = table["ID"].to_numpy()

    n, d = X.shape
    shap_vals = np.full((n, d), np.nan)
    base_vals = np.full(n, np.nan)
    p_pred = np.full(n, np.nan)

    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        clf = make_meta_clf(meta_clf, seed=seed, device=device)
        clf.fit(X[tr], y[tr])

        def f(Xq, _clf=clf):                       # 機率尺度、正類=AD;_clf 綁本折估計器
            return _clf.predict_proba(Xq)[:, 1]

        bg = shap.kmeans(X[tr], int(min(background, tr.sum())))
        explainer = shap.KernelExplainer(f, bg)
        sv = explainer.shap_values(X[te], nsamples=nsamples, silent=True)
        shap_vals[te] = np.asarray(sv)
        base_vals[te] = float(explainer.expected_value)
        p_pred[te] = f(X[te])
        logger.info(f"  fold {k}: explained {int(te.sum())} sessions "
                    f"(train={int(tr.sum())}, base={float(explainer.expected_value):.3f})")

    per_case = pd.DataFrame({"ID": ids, "y_true": y, "fold": fold,
                             "p_pred": p_pred, "base_value": base_vals})
    for j, fcol in enumerate(feats):
        per_case[f"shap_{fcol}"] = shap_vals[:, j]
    for j, fcol in enumerate(feats):            # 原始特徵值(beeswarm 著色 / 重畫自足用)
        per_case[f"value_{fcol}"] = X[:, j]

    mean_abs = np.nanmean(np.abs(shap_vals), axis=0)
    importance = pd.DataFrame({
        "feature": feats,
        "mean_abs_shap": mean_abs,
        "mean_shap": np.nanmean(shap_vals, axis=0),
        "std_abs_shap": np.nanstd(np.abs(shap_vals), axis=0),
        "importance_pct": mean_abs / mean_abs.sum() if mean_abs.sum() else np.nan,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1
    return per_case, importance


def beeswarm_from_per_case(per_case, feature_cols, out_png, *, title=None):
    """讀 fold_aligned_shap 的 per_case(含 shap_<feat> 與 value_<feat>)→ 畫 SHAP beeswarm 存 out_png。

    每特徵一排,每點=一個 session 的 SHAP 值(x),顏色=該特徵原始值(高/低)→ 看方向 + 分散。
    需 value_<feat> 欄(原始值供著色);舊版 CSV 若缺,請以新版 fold_aligned_shap 重產 shap_per_case.csv。
    """
    import matplotlib.pyplot as plt
    import shap

    feats = list(feature_cols)
    missing = [f"value_{f}" for f in feats if f"value_{f}" not in per_case.columns]
    if missing:
        raise ValueError(
            f"per_case 缺原始特徵欄 {missing};請以新版 fold_aligned_shap 重產 shap_per_case.csv")
    S = per_case[[f"shap_{f}" for f in feats]].to_numpy(dtype=float)
    D = per_case[[f"value_{f}" for f in feats]].to_numpy(dtype=float)
    base = per_case["base_value"].to_numpy(dtype=float)
    expl = shap.Explanation(values=S, base_values=base, data=D, feature_names=feats)

    plt.figure()
    shap.plots.beeswarm(expl, max_display=len(feats), show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 4.2)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
