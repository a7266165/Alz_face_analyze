"""
六模型七種表情折線圖比較
  (1) 同模型不同族群: 每個模型一張圖，Patient / NAD / ACS 三條線
  (2) 同族群不同模型: 每個族群一張圖 (All / Patient / NAD / ACS)，六條線
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── paths ──
AGG_DIR = PROJECT_ROOT / "workspace" / "au_features" / "aggregated"
OUT_DIR = PROJECT_ROOT / "workspace" / "au_analysis" / "emotion_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
EMO_COLS = [f"{e}_mean" for e in EMOTIONS]
EMO_LABELS = [e.capitalize() for e in EMOTIONS]

# 六個模型
ALL_TOOLS = ["pyfeat", "poster_pp", "fer", "dan", "hsemotion", "vit"]
TOOL_NAMES = {
    "pyfeat": "Py-Feat",
    "poster_pp": "POSTER++",
    "fer": "FER",
    "dan": "DAN",
    "hsemotion": "HSEmotion",
    "vit": "ViT",
}

# 族群顏色/marker
GROUP_COLORS = {"Patient": "#E74C3C", "NAD": "#3498DB", "ACS": "#2ECC71"}
GROUP_MARKERS = {"Patient": "o", "NAD": "s", "ACS": "D"}

# 模型顏色/marker
MODEL_COLORS = {
    "Py-Feat": "#9B59B6",
    "POSTER++": "#E67E22",
    "FER": "#1ABC9C",
    "DAN": "#E74C3C",
    "HSEmotion": "#3498DB",
    "ViT": "#2C3E50",
}
MODEL_MARKERS = {
    "Py-Feat": "o",
    "POSTER++": "s",
    "FER": "D",
    "DAN": "^",
    "HSEmotion": "v",
    "ViT": "P",
}


def infer_group(subject_id: str) -> str:
    if subject_id.startswith("ACS"):
        return "ACS"
    elif subject_id.startswith("NAD"):
        return "NAD"
    elif subject_id.startswith("P"):
        return "Patient"
    return "Unknown"


def load_tool(tool_key: str) -> pd.DataFrame:
    path = AGG_DIR / f"{tool_key}_harmonized.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["group"] = df["subject_id"].apply(infer_group)
    return df


def plot_same_model_diff_group(df: pd.DataFrame, tool_key: str):
    """同模型，Patient / NAD / ACS 三條線 + dashed border ribbon band"""
    tool_label = TOOL_NAMES[tool_key]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(EMOTIONS))
    for grp in ["Patient", "NAD", "ACS"]:
        sub = df[df["group"] == grp]
        if len(sub) == 0:
            continue
        means = sub[EMO_COLS].mean().values
        stds = sub[EMO_COLS].std().values
        color = GROUP_COLORS[grp]
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.10)
        ax.plot(x, means + stds, "--", color=color, lw=0.8, alpha=0.6)
        ax.plot(x, means - stds, "--", color=color, lw=0.8, alpha=0.6)
        ax.plot(x, means, "o-", color=color, label=f"{grp} (n={len(sub)})",
                marker=GROUP_MARKERS[grp], markersize=7, linewidth=2, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(EMO_LABELS, fontsize=12)
    ax.set_ylabel("Mean Probability", fontsize=13)
    ax.set_title(f"{tool_label} — Emotion Profiles by Group", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"{tool_key}_by_group.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_same_group_diff_model(dfs: dict, group_key: str, group_label: str):
    """同族群，六個模型各一條線 + dashed border ribbon band"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(EMOTIONS))
    for tool_key in ALL_TOOLS:
        if tool_key not in dfs:
            continue
        df = dfs[tool_key]
        tool_label = TOOL_NAMES[tool_key]
        if group_key == "All":
            sub = df
        else:
            sub = df[df["group"] == group_key]
        if len(sub) == 0:
            continue
        means = sub[EMO_COLS].mean().values
        stds = sub[EMO_COLS].std().values
        color = MODEL_COLORS[tool_label]
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.06)
        ax.plot(x, means + stds, "--", color=color, lw=0.5, alpha=0.4)
        ax.plot(x, means - stds, "--", color=color, lw=0.5, alpha=0.4)
        ax.plot(x, means, "-", color=color, label=f"{tool_label} (n={len(sub)})",
                marker=MODEL_MARKERS[tool_label], markersize=6, linewidth=1.8, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(EMO_LABELS, fontsize=12)
    ax.set_ylabel("Mean Probability", fontsize=13)
    ax.set_title(f"{group_label} — 6-Model Emotion Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"model_compare_{group_key.lower()}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved: {out}")


def main():
    # load all models
    dfs = {}
    for tool_key in ALL_TOOLS:
        try:
            dfs[tool_key] = load_tool(tool_key)
            print(f"Loaded {TOOL_NAMES[tool_key]}: {len(dfs[tool_key])} subjects")
        except FileNotFoundError:
            print(f"  skip {tool_key}: no aggregated CSV")

    # ── (1) 同模型不同族群 ──
    print("\n=== 同模型不同族群 ===")
    for tool_key in dfs:
        plot_same_model_diff_group(dfs[tool_key], tool_key)

    # ── (2) 同族群不同模型 ──
    print("\n=== 同族群不同模型 ===")
    for group_key, group_label in [
        ("All", "All Subjects"),
        ("Patient", "Patient (AD)"),
        ("NAD", "NAD"),
        ("ACS", "ACS"),
    ]:
        plot_same_group_diff_model(dfs, group_key, group_label)

    print(f"\nDone! All plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
