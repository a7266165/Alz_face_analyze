"""評估圖:ROC / PR / calibration / confusion / arm 比較 / SHAP。

全部圖文用英文標籤,避免 matplotlib 缺 CJK 字型時出現豆腐框。
配色取通過 CVD 驗證的固定 8 槽序列前 5 槽,依 arm 固定指派(不隨圖上出現幾條
線而輪替),另用線型做第二層編碼 —— 三個對照 arm 在淺色槽上對比度不足,故每條
線都直接把 AUROC 標在圖例裡當作 relief。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import q6ds_path

from .dataset import LABEL_COL
from .evaluate import pooled_oof
from .model import ARMS

# 固定槽位:arm → (顏色, 線型)。顏色永遠跟著 arm 走,不跟著排名或出現順序。
ARM_STYLE: Dict[str, tuple] = {
    "xgb_full":   ("#2a78d6", "-"),
    "xgb_noage":  ("#eb6834", "--"),
    "lr_full":    ("#1baf7a", "-"),
    "lr_noage":   ("#4a3aa7", (0, (5, 1))),
    "score_6qds": ("#eda100", "-."),
    "age_only":   ("#e87ba4", ":"),
    "edu_only":   ("#008300", (0, (1, 1, 3, 1))),
}
DATASET_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")   # 前三槽 all-pairs 驗證過

_GRID = dict(color="#d8d7d2", linewidth=0.6, alpha=0.9)
_INK, _INK2 = "#0b0b0b", "#52514e"
_FPR_GRID = np.linspace(0, 1, 201)


def _style_axes(ax, *, xlabel: str, ylabel: str, title: str = ""):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, **_GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b3")
    ax.tick_params(colors=_INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=_INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=_INK2, fontsize=10)
    if title:
        ax.set_title(title, color=_INK, fontsize=12, loc="left", pad=10)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def _mean_roc(oof_arm: pd.DataFrame):
    """每個 repeat 是一份完整樣本的 OOF → 各自一條 ROC,再在 FPR 網格上平均。"""
    from sklearn.metrics import roc_curve
    p = pooled_oof(oof_arm)
    tprs, aucs = [], []
    from sklearn.metrics import roc_auc_score
    for _, sub in p.groupby("repeat"):
        fpr, tpr, _ = roc_curve(sub["y_true"], sub["y_prob"])
        tprs.append(np.interp(_FPR_GRID, fpr, tpr))
        aucs.append(roc_auc_score(sub["y_true"], sub["y_prob"]))
    m = np.mean(tprs, axis=0)
    m[0], m[-1] = 0.0, 1.0
    return m, float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0


def plot_roc(oof: pd.DataFrame, dataset_id: str, out_path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for arm in ARMS:
        sub = oof[oof["arm"] == arm]
        if sub.empty:
            continue
        color, ls = ARM_STYLE[arm]
        tpr, auc, sd = _mean_roc(sub)
        ax.plot(_FPR_GRID, tpr, color=color, linestyle=ls, linewidth=2,
                label=f"{arm}  AUROC {auc:.3f} ± {sd:.3f}")
    ax.plot([0, 1], [0, 1], color="#b9b8b3", linewidth=1, linestyle=(0, (2, 3)), zorder=0)
    _style_axes(ax, xlabel="False positive rate (1 − specificity)",
                ylabel="True positive rate (sensitivity)",
                title=f"ROC — {dataset_id}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(_INK)     # 圖例文字用墨色,不吃系列色
    _save(fig, out_path)


def plot_pr(oof: pd.DataFrame, dataset_id: str, prevalence: float, out_path):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    rec_grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for arm in ARMS:
        sub = oof[oof["arm"] == arm]
        if sub.empty:
            continue
        color, ls = ARM_STYLE[arm]
        p = pooled_oof(sub)
        precs, aps = [], []
        for _, s in p.groupby("repeat"):
            pr, rc, _ = precision_recall_curve(s["y_true"], s["y_prob"])
            precs.append(np.interp(rec_grid, rc[::-1], pr[::-1]))
            aps.append(average_precision_score(s["y_true"], s["y_prob"]))
        ax.plot(rec_grid, np.mean(precs, axis=0), color=color, linestyle=ls, linewidth=2,
                label=f"{arm}  AUPRC {np.mean(aps):.3f}")
    ax.axhline(prevalence, color="#b9b8b3", linewidth=1, linestyle=(0, (2, 3)), zorder=0)
    ax.text(0.01, prevalence + 0.012, f"prevalence {prevalence:.3f}",
            color=_INK2, fontsize=8)
    _style_axes(ax, xlabel="Recall (sensitivity)", ylabel="Precision (PPV)",
                title=f"Precision–Recall — {dataset_id}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(_INK)
    _save(fig, out_path)


def plot_calibration(oof: pd.DataFrame, dataset_id: str, out_path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for arm in ARMS:
        sub = oof[oof["arm"] == arm]
        if sub.empty:
            continue
        color, ls = ARM_STYLE[arm]
        p = pooled_oof(sub)
        q = pd.qcut(p["y_prob"], n_bins, labels=False, duplicates="drop")
        b = p.assign(bin=q).groupby("bin", observed=True).agg(
            pred=("y_prob", "mean"), obs=("y_true", "mean"))
        ax.plot(b["pred"], b["obs"], color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=4.5, markeredgecolor="#fcfcfb",
                markeredgewidth=1.2, label=arm)
    ax.plot([0, 1], [0, 1], color="#b9b8b3", linewidth=1, linestyle=(0, (2, 3)), zorder=0)
    _style_axes(ax, xlabel="Mean predicted probability", ylabel="Observed frequency",
                title=f"Calibration (OOF, {n_bins} quantile bins) — {dataset_id}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(_INK)
    _save(fig, out_path)


def plot_arm_comparison(summary: pd.DataFrame, dataset_id: str, out_path):
    """AUROC 排序長條 —— 比較量值用長條,誤差棒是折間 sd(非信賴區間)。"""
    s = summary[summary["thr_kind"] == "youden_inner"].copy()
    s = s.sort_values("auroc_mean")
    fig, ax = plt.subplots(figsize=(7.0, 0.62 * len(s) + 1.9))
    y = np.arange(len(s))
    ax.barh(y, s["auroc_mean"], height=0.44,
            color=[ARM_STYLE[a][0] for a in s["arm"]], zorder=3)
    ax.errorbar(s["auroc_mean"], y, xerr=s["auroc_sd"], fmt="none",
                ecolor="#52514e", elinewidth=1.2, capsize=3, zorder=4)
    for i, (m, sd) in enumerate(zip(s["auroc_mean"], s["auroc_sd"])):
        ax.text(m + sd + 0.006, i, f"{m:.3f} ± {sd:.3f}", va="center",
                color=_INK, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(s["arm"], color=_INK, fontsize=10)
    _style_axes(ax, xlabel="AUROC (mean over outer folds, ± fold SD)", ylabel="",
                title=f"Arm comparison — {dataset_id}")
    ax.set_xlim(0.4, min(1.0, float(s["auroc_mean"].max() + s["auroc_sd"].max()) + 0.09))
    ax.axvline(0.5, color="#b9b8b3", linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    _save(fig, out_path)


def plot_confusion(fold_metrics: pd.DataFrame, dataset_id: str, arm: str, out_path):
    s = fold_metrics[(fold_metrics["arm"] == arm)
                     & (fold_metrics["thr_kind"] == "youden_inner")]
    cm = np.array([[s["tn"].mean(), s["fp"].mean()],
                   [s["fn"].mean(), s["tp"].mean()]])
    row = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.1f}\n({row[i, j]:.1%})", ha="center", va="center",
                    color="#ffffff" if row[i, j] > 0.55 else _INK, fontsize=11)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"], color=_INK2)
    ax.set_yticks([0, 1], ["true 0", "true 1"], color=_INK2)
    ax.set_title(f"Mean confusion per outer fold — {dataset_id} / {arm}\n"
                 f"(Youden threshold from inner OOF)",
                 color=_INK, fontsize=10.5, loc="left", pad=10)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    _save(fig, out_path)


def plot_xgb_importance(model, feature_names, dataset_id: str, arm: str, out_path):
    gain = model.get_booster().get_score(importance_type="gain")
    vals = pd.Series({f: gain.get(f, 0.0) for f in feature_names}).sort_values()
    fig, ax = plt.subplots(figsize=(6.0, 0.36 * len(vals) + 1.8))
    ax.barh(np.arange(len(vals)), vals.to_numpy(), height=0.6, color="#2a78d6", zorder=3)
    for i, v in enumerate(vals.to_numpy()):
        ax.text(v + vals.max() * 0.012, i, f"{v:.1f}", va="center", color=_INK, fontsize=8.5)
    ax.set_yticks(np.arange(len(vals))); ax.set_yticklabels(vals.index, color=_INK, fontsize=9)
    _style_axes(ax, xlabel="Gain", ylabel="", title=f"XGBoost gain — {dataset_id} / {arm}")
    _save(fig, out_path)


def plot_shap(model, X: pd.DataFrame, dataset_id: str, arm: str, out_path):
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    fig = plt.figure(figsize=(6.4, 0.38 * X.shape[1] + 2.0))
    shap.summary_plot(sv, X, show=False, plot_size=None, color_bar=True)
    plt.title(f"SHAP — {dataset_id} / {arm}", color=_INK, fontsize=11, loc="left")
    _save(fig, out_path)


def plot_cross_dataset(all_summary: pd.DataFrame, out_path):
    """三份資料 × 五個 arm 的 AUROC 群組長條(資料集用前三個色槽)。"""
    s = all_summary[all_summary["thr_kind"] == "youden_inner"]
    datasets = list(dict.fromkeys(s["dataset"]))
    arms = [a for a in ARMS if a in set(s["arm"])]
    fig, ax = plt.subplots(figsize=(8.2, 0.46 * len(arms) * len(datasets) + 2.2))
    h = 0.8 / len(datasets)
    for k, ds in enumerate(datasets):
        d = s[s["dataset"] == ds].set_index("arm").reindex(arms)
        y = np.arange(len(arms)) + (k - (len(datasets) - 1) / 2) * h
        ax.barh(y, d["auroc_mean"], height=h * 0.86,
                color=DATASET_COLORS[k % len(DATASET_COLORS)], label=ds, zorder=3)
        for yy, m in zip(y, d["auroc_mean"]):
            if np.isfinite(m):
                ax.text(m + 0.005, yy, f"{m:.3f}", va="center", color=_INK, fontsize=8)
    ax.set_yticks(np.arange(len(arms))); ax.set_yticklabels(arms, color=_INK, fontsize=10)
    _style_axes(ax, xlabel="AUROC (mean over outer folds)", ylabel="",
                title="AUROC by arm and dataset")
    ax.axvline(0.5, color="#b9b8b3", linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    ax.set_xlim(0.4, 1.02)
    # 圖例放在座標區外的下方:任何 loc 都會壓到某一根長條的數值標籤。
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06 - 0.9 / fig.get_figheight()),
                    ncol=len(datasets), frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(_INK)
    _save(fig, out_path)


def make_dataset_figures(dataset_id: str, df: pd.DataFrame,
                         fold_metrics: pd.DataFrame, oof: pd.DataFrame,
                         summary: pd.DataFrame,
                         xgb_models: Optional[Dict[str, tuple]] = None):
    """單一 dataset 的全部圖。xgb_models: arm → (model, X_used)。"""
    fig_dir = q6ds_path(dataset_id, "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    prevalence = float((df[LABEL_COL] == 1).mean())

    plot_roc(oof, dataset_id, fig_dir / "roc_arms.png")
    plot_pr(oof, dataset_id, prevalence, fig_dir / "pr_arms.png")
    plot_calibration(oof, dataset_id, fig_dir / "calibration_arms.png")
    plot_arm_comparison(summary, dataset_id, fig_dir / "arm_comparison_auroc.png")
    for arm in sorted(set(fold_metrics["arm"])):
        plot_confusion(fold_metrics, dataset_id, arm, fig_dir / f"confusion_{arm}.png")
    for arm, (model, X) in (xgb_models or {}).items():
        plot_xgb_importance(model, list(X.columns), dataset_id, arm,
                            fig_dir / f"importance_gain_{arm}.png")
        plot_shap(model, X, dataset_id, arm, fig_dir / f"shap_beeswarm_{arm}.png")
    return fig_dir
