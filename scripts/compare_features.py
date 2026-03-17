"""比對五位提取者的 ArcFace 特徵差異視覺化"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import matplotlib
matplotlib.rc("font", family="Microsoft JhengHei")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
OUTPUT_PATH = WORKSPACE_DIR / "feature_comparison_5people.png"

SUBJECTS = ["ACS1-1", "ACS5-1", "ACS10-1", "ACS13-1", "ACS14-1"]
PHOTO_IDX = 0  # 取每位個案的第一張照片

# 各提取者的 arcface/original 目錄
EXTRACTOR_DIRS = {
    "馨方": WORKSPACE_DIR / "features_馨方" / "arcface" / "original",
    "以芯": WORKSPACE_DIR / "features_以芯" / "arcface" / "original",
    "杰勳": WORKSPACE_DIR / "features_杰勳" / "arcface" / "original",
    "俊成": WORKSPACE_DIR / "features_俊成" / "arcface" / "original",
    "Project": WORKSPACE_DIR / "features" / "arcface" / "original",
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]


def _load_npy_vector(npy_path: Path, photo_idx: int = 0):
    """從 per-subject npy 載入指定照片的特徵向量"""
    if not npy_path.exists():
        return None
    arr = np.load(npy_path, allow_pickle=True)
    if arr.ndim == 2 and len(arr) > photo_idx:
        return arr[photo_idx].astype(np.float64)
    elif arr.ndim == 1:
        return arr.astype(np.float64)
    return None


def load_all():
    """載入所有提取者的資料"""
    data = {}
    for sid in SUBJECTS:
        vectors = {}
        for name, feat_dir in EXTRACTOR_DIRS.items():
            vec = _load_npy_vector(feat_dir / f"{sid}.npy", PHOTO_IDX)
            if vec is not None:
                vectors[name] = vec
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

        for name, color in zip(EXTRACTOR_DIRS.keys(), COLORS):
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
