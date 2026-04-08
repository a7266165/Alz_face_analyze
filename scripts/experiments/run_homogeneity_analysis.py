"""
六模型表情同質性分析 (Homogeneity Analysis)

對每張照片，取出 6 個模型各自的 7 維表情機率向量，
計算模型配對的 dot product 及 cosine similarity，評估模型間一致性。

Analysis 1: 整體相片級同質矩陣（所有照片平均）
Analysis 2: Session 層級同質分析（每個 session 的 10 張照片平均，再彙總統計）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

from src.extractor.features.emotion.extractor.au_config import HARMONIZED_EMOTIONS, AU_RAW_DIR, AU_ANALYSIS_DIR

# ── 設定 ──
MODELS = ["pyfeat", "poster_pp", "fer", "dan", "hsemotion", "vit"]
MODEL_NAMES = {
    "pyfeat": "Py-Feat",
    "poster_pp": "POSTER++",
    "fer": "FER",
    "dan": "DAN",
    "hsemotion": "HSEmotion",
    "vit": "ViT",
}
EMOTIONS = HARMONIZED_EMOTIONS  # 7 emotions
OUT_DIR = AU_ANALYSIS_DIR / "homogeneity"


# =====================================================================
# 資料載入
# =====================================================================

def load_raw_emotions(model: str, subject_id: str) -> np.ndarray | None:
    """載入單一 subject 的 7 維表情向量，過濾 _aligned_aligned。

    Returns:
        (N, 7) ndarray，N 通常為 10；若檔案不存在回傳 None
    """
    path = AU_RAW_DIR / model / f"{subject_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 過濾 double-aligned
    if "frame" in df.columns:
        df = df[~df["frame"].astype(str).str.contains("_aligned_aligned")]
    return df[EMOTIONS].values.astype(np.float64)


def load_all_data(models: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """載入所有 subject 在所有模型的表情向量。

    Returns:
        {subject_id: {model: ndarray(N, 7)}}
        只包含所有模型都有資料的 subject
    """
    # 取模型交集的 subject_id
    subject_sets = []
    for m in models:
        model_dir = AU_RAW_DIR / m
        sids = {p.stem for p in model_dir.glob("*.csv")}
        subject_sets.append(sids)
    common_subjects = sorted(set.intersection(*subject_sets))
    print(f"共同 subject 數: {len(common_subjects)}")

    data = {}
    for sid in tqdm(common_subjects, desc="載入資料"):
        sid_data = {}
        valid = True
        for m in models:
            arr = load_raw_emotions(m, sid)
            if arr is None or len(arr) == 0:
                valid = False
                break
            sid_data[m] = arr
        if valid:
            data[sid] = sid_data
    print(f"有效 subject 數: {len(data)}")
    return data


# =====================================================================
# 同質性計算
# =====================================================================

def compute_photo_level_matrix(
    data: dict, models: list[str], metric: str = "dot"
) -> np.ndarray:
    """整體相片級同質矩陣：所有照片的配對相似度平均。"""
    n = len(models)
    pair_sum = np.zeros((n, n))
    pair_count = np.zeros((n, n))

    for sid, model_data in data.items():
        # 取各模型最小幀數對齊
        n_frames = min(len(model_data[m]) for m in models)
        for f in range(n_frames):
            vecs = [model_data[m][f] for m in models]
            for i in range(n):
                for j in range(i, n):
                    a, b = vecs[i], vecs[j]
                    if metric == "dot":
                        val = np.dot(a, b)
                    else:  # cosine
                        denom = np.linalg.norm(a) * np.linalg.norm(b)
                        val = np.dot(a, b) / denom if denom > 1e-12 else 0.0
                    pair_sum[i, j] += val
                    pair_count[i, j] += 1

    # 填充對稱
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = pair_sum[i, j] / max(pair_count[i, j], 1)
            matrix[j, i] = matrix[i, j]
    return matrix


def compute_per_session_matrices(
    data: dict, models: list[str], metric: str = "dot"
) -> dict[str, np.ndarray]:
    """每個 session 一個同質矩陣（該 session 的照片平均）。"""
    n = len(models)
    session_matrices = {}

    for sid, model_data in data.items():
        n_frames = min(len(model_data[m]) for m in models)
        if n_frames == 0:
            continue
        pair_sum = np.zeros((n, n))
        for f in range(n_frames):
            vecs = [model_data[m][f] for m in models]
            for i in range(n):
                for j in range(i, n):
                    a, b = vecs[i], vecs[j]
                    if metric == "dot":
                        val = np.dot(a, b)
                    else:
                        denom = np.linalg.norm(a) * np.linalg.norm(b)
                        val = np.dot(a, b) / denom if denom > 1e-12 else 0.0
                    pair_sum[i, j] += val
        mat = pair_sum / n_frames
        # 對稱
        for i in range(n):
            for j in range(i + 1, n):
                mat[j, i] = mat[i, j]
        session_matrices[sid] = mat

    return session_matrices


# =====================================================================
# 視覺化
# =====================================================================

def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    fmt: str = ".4f",
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  saved: {output_path}")


def plot_session_distribution(
    session_matrices: dict[str, np.ndarray],
    models: list[str],
    metric_label: str,
    output_path: Path,
):
    """下三角 + 對角線的配對分布 violin plot。"""
    n = len(models)
    labels = [MODEL_NAMES[m] for m in models]

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    all_mats = np.array(list(session_matrices.values()))  # (S, n, n)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if j > i:
                # 上三角：顯示均值數字
                mean_val = all_mats[:, i, j].mean()
                ax.text(0.5, 0.5, f"{mean_val:.4f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                # 下三角 + 對角線：histogram
                vals = all_mats[:, i, j]
                val_range = vals.max() - vals.min()
                if val_range < 1e-10:
                    # 所有值相同（如 cosine 對角線 = 1.0）
                    ax.text(0.5, 0.5, f"{vals.mean():.4f}\n(const)", ha="center",
                            va="center", fontsize=10, transform=ax.transAxes)
                else:
                    ax.hist(vals, bins=min(40, max(10, int(len(vals) ** 0.5))),
                            color="#3498DB", alpha=0.7, edgecolor="white")
                    ax.axvline(vals.mean(), color="#E74C3C", lw=1.5, ls="--")

            ax.set_xticks([])
            ax.set_yticks([])
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=10)

    fig.suptitle(f"Per-Session {metric_label} Distribution", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {output_path}")


# =====================================================================
# 儲存
# =====================================================================

def save_matrix_csv(matrix: np.ndarray, labels: list[str], path: Path):
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(path, encoding="utf-8-sig")
    print(f"  saved: {path}")


def save_summary_csv(
    session_matrices: dict[str, np.ndarray],
    models: list[str],
    metric_label: str,
    path: Path,
):
    """每對模型的統計摘要。"""
    n = len(models)
    labels = [MODEL_NAMES[m] for m in models]
    all_mats = np.array(list(session_matrices.values()))

    rows = []
    for i in range(n):
        for j in range(i, n):
            vals = all_mats[:, i, j]
            rows.append({
                "model_i": labels[i],
                "model_j": labels[j],
                "metric": metric_label,
                "mean": vals.mean(),
                "std": vals.std(),
                "median": np.median(vals),
                "q1": np.percentile(vals, 25),
                "q3": np.percentile(vals, 75),
                "n_sessions": len(vals),
            })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  saved: {path}")


# =====================================================================
# Main
# =====================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [MODEL_NAMES[m] for m in MODELS]

    # ── 載入資料 ──
    print("=== 載入原始表情資料 ===")
    data = load_all_data(MODELS)

    # ── Analysis 1: 整體相片級同質矩陣 ──
    print("\n=== Analysis 1: Photo-Level Homogeneity ===")
    for metric, name in [("dot", "Dot Product"), ("cosine", "Cosine Similarity")]:
        matrix = compute_photo_level_matrix(data, MODELS, metric=metric)
        tag = "dot_product" if metric == "dot" else "cosine_similarity"
        save_matrix_csv(matrix, labels, OUT_DIR / f"homogeneity_{tag}.csv")
        plot_heatmap(
            matrix, labels,
            title=f"Photo-Level Homogeneity ({name})",
            output_path=OUT_DIR / f"homogeneity_{tag}.png",
        )

    # ── Analysis 2: Per-Session 同質分析 ──
    print("\n=== Analysis 2: Per-Session Homogeneity ===")
    for metric, name in [("dot", "Dot Product"), ("cosine", "Cosine Similarity")]:
        tag = "dot" if metric == "dot" else "cosine"
        sess_mats = compute_per_session_matrices(data, MODELS, metric=metric)

        all_mats = np.array(list(sess_mats.values()))
        mean_mat = all_mats.mean(axis=0)
        std_mat = all_mats.std(axis=0)

        save_matrix_csv(mean_mat, labels, OUT_DIR / f"per_session_{tag}_mean_matrix.csv")
        save_matrix_csv(std_mat, labels, OUT_DIR / f"per_session_{tag}_std_matrix.csv")
        plot_heatmap(
            mean_mat, labels,
            title=f"Per-Session Mean Homogeneity ({name})",
            output_path=OUT_DIR / f"per_session_{tag}_mean.png",
        )
        plot_heatmap(
            std_mat, labels,
            title=f"Per-Session Std of Homogeneity ({name})",
            output_path=OUT_DIR / f"per_session_{tag}_std.png",
            cmap="Blues",
        )
        save_summary_csv(sess_mats, MODELS, name, OUT_DIR / f"per_session_{tag}_summary.csv")
        plot_session_distribution(sess_mats, MODELS, name, OUT_DIR / f"per_session_{tag}_distribution.png")

    print(f"\nDone! All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
