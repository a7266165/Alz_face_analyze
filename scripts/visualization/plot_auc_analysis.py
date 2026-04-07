"""
六模型 × 七表情 Per-Frame AUC 分析（3 組對比）

每張照片獨立作為一筆樣本，計算各 emotion 的 ROC AUC。
產出：
  (A) 3 張 Heatmap（每組對比一張，6 models × 7 emotions）
  (C) 3 張 Grid ROC（每組對比一張，7 subplots × 6 curves）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

import matplotlib
matplotlib.use("Agg")
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── 路徑 ──
RAW_DIR = PROJECT_ROOT / "workspace" / "au_features" / "raw"
OUT_DIR = PROJECT_ROOT / "workspace" / "au_analysis" / "emotion_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 常數 ──
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

ALL_TOOLS = ["pyfeat", "poster_pp", "fer", "dan", "hsemotion", "vit"]
TOOL_NAMES = {
    "pyfeat": "Py-Feat", "poster_pp": "POSTER++", "fer": "FER",
    "dan": "DAN", "hsemotion": "HSEmotion", "vit": "ViT",
}
MODEL_COLORS = {
    "Py-Feat": "#9B59B6", "POSTER++": "#E67E22", "FER": "#1ABC9C",
    "DAN": "#E74C3C", "HSEmotion": "#3498DB", "ViT": "#2C3E50",
}

PAIRS = [
    ("ACS", "AD",  "ACS vs AD"),
    ("ACS", "NAD", "ACS vs NAD"),
    ("NAD", "AD",  "NAD vs AD"),
]

# POSTER++ 欄名對映（順序不同）
POSTER_PP_RENAME = {
    "surprise": "surprise", "fear": "fear", "disgust": "disgust",
    "happiness": "happiness", "sadness": "sadness", "anger": "anger", "neutral": "neutral",
}


def infer_group(filename: str) -> str:
    """從檔名推斷群組"""
    if filename.startswith("ACS"):
        return "ACS"
    elif filename.startswith("NAD"):
        return "NAD"
    elif filename.startswith("P"):
        return "AD"
    return "Unknown"


def load_perframe(tool: str) -> pd.DataFrame:
    """載入某工具的所有 per-frame 資料，回傳 DataFrame (columns: 7 emotions + group)"""
    tool_dir = RAW_DIR / tool
    csv_files = sorted(tool_dir.glob("*.csv"))

    all_rows = []
    for csv_path in csv_files:
        group = infer_group(csv_path.stem)
        if group == "Unknown":
            continue
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df["group"] = group
            all_rows.append(df)
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()

    big_df = pd.concat(all_rows, ignore_index=True)

    # 確保 7 emotion columns 都存在且名稱統一
    for emo in EMOTIONS:
        if emo not in big_df.columns:
            big_df[emo] = np.nan

    return big_df[EMOTIONS + ["group"]].dropna()


def compute_auc(df: pd.DataFrame, group_neg: str, group_pos: str):
    """計算 7 個 emotion 的 AUC + ROC curve"""
    sub = df[df["group"].isin([group_neg, group_pos])].copy()
    y_true = (sub["group"] == group_pos).astype(int).values

    results = {}
    for emo in EMOTIONS:
        scores = sub[emo].values
        try:
            auc_val = roc_auc_score(y_true, scores)
            fpr, tpr, _ = roc_curve(y_true, scores)

            # 翻轉處理：若 AUC < 0.5，反向區分
            if auc_val < 0.5:
                auc_val = 1 - auc_val
                fpr, tpr, _ = roc_curve(y_true, -scores)

            results[emo] = {"auc": auc_val, "fpr": fpr, "tpr": tpr}
        except Exception:
            results[emo] = {"auc": 0.5, "fpr": np.array([0, 1]), "tpr": np.array([0, 1])}

    return results


def plot_heatmap(auc_matrix: dict, pair_label: str, pair_key: str):
    """方案 A: Heatmap 總覽"""
    # Build matrix: rows=models, cols=emotions
    model_names = [TOOL_NAMES[t] for t in ALL_TOOLS]
    data = np.zeros((len(ALL_TOOLS), len(EMOTIONS)))
    for i, tool in enumerate(ALL_TOOLS):
        for j, emo in enumerate(EMOTIONS):
            data[i, j] = auc_matrix[tool][emo]["auc"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.5, vmax=1.0, linewidths=0.5,
        xticklabels=[e.capitalize() for e in EMOTIONS],
        yticklabels=model_names,
        ax=ax, cbar_kws={"label": "AUC"},
    )
    ax.set_title(f"Per-Frame AUC Heatmap — {pair_label}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / f"auc_heatmap_{pair_key}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info(f"  saved: {out}")


def plot_roc_grid(auc_matrix: dict, pair_label: str, pair_key: str):
    """方案 C: Grid ROC 曲線"""
    fig, axes = plt.subplots(1, 7, figsize=(28, 4))

    for j, emo in enumerate(EMOTIONS):
        ax = axes[j]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Chance")

        for tool in ALL_TOOLS:
            tool_label = TOOL_NAMES[tool]
            r = auc_matrix[tool][emo]
            ax.plot(r["fpr"], r["tpr"], color=MODEL_COLORS[tool_label],
                    lw=1.5, label=f"{tool_label} ({r['auc']:.3f})")

        ax.set_title(emo.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        if j == 0:
            ax.set_ylabel("TPR", fontsize=10)
        ax.set_xlabel("FPR", fontsize=9)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Per-Frame ROC Curves — {pair_label}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / f"auc_roc_grid_{pair_key}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved: {out}")


def main():
    # Step 1: 載入所有工具的 per-frame 資料
    tool_data = {}
    for tool in ALL_TOOLS:
        logger.info(f"Loading {TOOL_NAMES[tool]}...")
        df = load_perframe(tool)
        if len(df) > 0:
            tool_data[tool] = df
            counts = df["group"].value_counts()
            logger.info(f"  {TOOL_NAMES[tool]}: {len(df)} frames "
                        f"(AD={counts.get('AD', 0)}, NAD={counts.get('NAD', 0)}, ACS={counts.get('ACS', 0)})")
        else:
            logger.warning(f"  {TOOL_NAMES[tool]}: no data")

    # Step 2: 對每組對比計算 AUC + 畫圖
    for group_neg, group_pos, pair_label in PAIRS:
        pair_key = f"{group_neg.lower()}_vs_{group_pos.lower()}"
        logger.info(f"\n=== {pair_label} ===")

        auc_matrix = {}
        for tool in ALL_TOOLS:
            if tool not in tool_data:
                continue
            auc_matrix[tool] = compute_auc(tool_data[tool], group_neg, group_pos)
            aucs = [auc_matrix[tool][e]["auc"] for e in EMOTIONS]
            logger.info(f"  {TOOL_NAMES[tool]}: "
                        + ", ".join(f"{e[:3]}={a:.3f}" for e, a in zip(EMOTIONS, aucs)))

        # 畫圖
        plot_heatmap(auc_matrix, pair_label, pair_key)
        plot_roc_grid(auc_matrix, pair_label, pair_key)

    logger.info(f"\nDone! All plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
