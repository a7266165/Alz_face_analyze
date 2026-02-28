"""比對五位提取者的 ArcFace 特徵差異視覺化"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import matplotlib
matplotlib.rc("font", family="Microsoft JhengHei")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEMINAR_DIR = PROJECT_ROOT / "workspace" / "features_seminar"
OUTPUT_PATH = SEMINAR_DIR / "feature_comparison_5people.png"

SUBJECTS = ["ACS1-1", "ACS5-1", "ACS10-1", "ACS13-1", "ACS14-1"]
PHOTO_IDX = 0  # 取每位個案的第一張照片

NPY_DIR = PROJECT_ROOT / "workspace" / "features" / "arcface" / "original"
JC_DIR = SEMINAR_DIR / "arc_raw_俊成"

EXTRACTORS = ["馨芳", "以芯", "杰勳", "俊成", "Project"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]


def load_all():
    """載入五位提取者的資料"""

    # 馨芳 (csv): col0=Subject_ID, col3~=features
    df_xf = pd.read_csv(SEMINAR_DIR / "arc_raw_馨芳.csv")
    feat_cols_xf = [c for c in df_xf.columns if c.startswith("feature_")]

    # 以芯 (npy object): col0=group, col1=Subject_ID, col2=filename, col3~=features
    arr_yx = np.load(SEMINAR_DIR / "arc_raw_以芯.npy", allow_pickle=True)

    # 杰勳 (npy object): col0=label, col1=Subject_ID, col2=filename, col3~=features
    arr_jx = np.load(SEMINAR_DIR / "arc_raw_杰勳.npy", allow_pickle=True)

    data = {}
    for sid in SUBJECTS:
        vectors = {}

        # 馨芳
        rows_xf = df_xf[df_xf["Subject_ID"] == sid][feat_cols_xf].values
        if len(rows_xf) > PHOTO_IDX:
            vectors["馨芳"] = rows_xf[PHOTO_IDX].astype(np.float64)

        # 以芯
        mask_yx = arr_yx[:, 1] == sid
        rows_yx = arr_yx[mask_yx]
        if len(rows_yx) > PHOTO_IDX:
            vectors["以芯"] = rows_yx[PHOTO_IDX, 3:].astype(np.float64)

        # 杰勳
        mask_jx = arr_jx[:, 1] == sid
        rows_jx = arr_jx[mask_jx]
        if len(rows_jx) > PHOTO_IDX:
            vectors["杰勳"] = rows_jx[PHOTO_IDX, 3:].astype(np.float64)

        # 俊成 (per-subject npy): shape=(10, 512)
        jc_path = JC_DIR / f"{sid}.npy"
        if jc_path.exists():
            arr_jc = np.load(jc_path)
            if arr_jc.ndim == 2 and len(arr_jc) > PHOTO_IDX:
                vectors["俊成"] = arr_jc[PHOTO_IDX].astype(np.float64)
            elif arr_jc.ndim == 1:
                vectors["俊成"] = arr_jc.astype(np.float64)

        # Project (per-subject npy): shape=(N, 512)
        npy_path = NPY_DIR / f"{sid}.npy"
        if npy_path.exists():
            arr_proj = np.load(npy_path)
            if arr_proj.ndim == 2 and len(arr_proj) > PHOTO_IDX:
                vectors["Project"] = arr_proj[PHOTO_IDX].astype(np.float64)
            elif arr_proj.ndim == 1:
                vectors["Project"] = arr_proj.astype(np.float64)

        data[sid] = vectors

    return data


def compute_pairwise_metrics(vectors):
    """計算所有提取者兩兩之間的指標"""
    names = list(vectors.keys())
    metrics = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v1, v2 = vectors[names[i]], vectors[names[j]]
            cos_sim = 1 - cosine(v1, v2)
            r, _ = pearsonr(v1, v2)
            l2 = np.linalg.norm(v1 - v2)
            metrics.append({
                "pair": f"{names[i]} vs {names[j]}",
                "cosine_sim": cos_sim,
                "pearson_r": r,
                "l2_dist": l2,
            })
    return metrics


def plot(data):
    n_subjects = len(SUBJECTS)
    fig, axes = plt.subplots(n_subjects, 1, figsize=(14, 4 * n_subjects))
    fig.suptitle(
        "Feature Comparison: 5 Sources × 5 Subjects (ArcFace 512 dims, Photo #1)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for i, sid in enumerate(SUBJECTS):
        ax = axes[i]
        vectors = data[sid]
        dims = np.arange(1, 513)

        for name, color in zip(EXTRACTORS, COLORS):
            if name in vectors:
                ax.plot(dims, vectors[name], alpha=0.65, linewidth=0.6,
                        label=name, color=color)

        ax.set_title(f"{sid} — Feature Profile", fontsize=11)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.set_xlim(1, 512)

        # 指標文字
        metrics = compute_pairwise_metrics(vectors)
        if metrics:
            lines = [
                f"{m['pair']}: cos={m['cosine_sim']:.4f}, r={m['pearson_r']:.4f}, L2={m['l2_dist']:.2f}"
                for m in metrics
            ]
            txt = "  |  ".join(lines[:3])  # 只顯示前三對避免太擠
            if len(lines) > 3:
                txt += f"\n{'  |  '.join(lines[3:])}"
            ax.text(
                0.5, -0.22, txt, transform=ax.transAxes,
                fontsize=7, horizontalalignment="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

    # Print summary
    for sid in SUBJECTS:
        print(f"\n{sid}:")
        metrics = compute_pairwise_metrics(data[sid])
        for m in metrics:
            print(f"  {m['pair']:<16} cos={m['cosine_sim']:.4f}  r={m['pearson_r']:.4f}  L2={m['l2_dist']:.2f}")


if __name__ == "__main__":
    data = load_all()
    plot(data)
