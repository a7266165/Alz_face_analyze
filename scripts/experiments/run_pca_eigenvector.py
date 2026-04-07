"""
PCA 分析: 累積解釋變異量 + Eigenvector 組間比較

分組方案: CDR>=2 vs Normal / 無分層 (All P vs Normal)
年齡控制: Original / +Age / PSM (+/-2歲)
工具: OpenFace / LibreFace
特徵集: Harmonized / Extended
"""

import re
import sys
import logging
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.modules.emotion.extractor.au_config import AU_AGGREGATED_DIR, AU_ANALYSIS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
OUTPUT_DIR = AU_ANALYSIS_DIR / "pca"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PCS = 8  # 比較前幾個 PC


# ═══════════════════════════════════════════════════════════
# 資料載入與分組
# ═══════════════════════════════════════════════════════════

def load_demographics():
    frames = []
    for g in ["P", "NAD", "ACS"]:
        df = pd.read_csv(DEMOGRAPHICS_DIR / f"{g}.csv")
        df["group"] = g
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def assign_label_cdr2(row):
    g, cdr, mmse = row["group"], row["Global_CDR"], row["MMSE"]
    if g == "P":
        return "AD" if pd.notna(cdr) and cdr >= 2 else None
    if pd.notna(cdr) and cdr == 0:
        return "Normal"
    if pd.isna(cdr) and pd.notna(mmse) and mmse >= 26:
        return "Normal"
    return None


def assign_label_nostrat(row):
    g, cdr, mmse = row["group"], row["Global_CDR"], row["MMSE"]
    if g == "P":
        return "AD"
    if pd.notna(cdr) and cdr == 0:
        return "Normal"
    if pd.isna(cdr) and pd.notna(mmse) and mmse >= 26:
        return "Normal"
    return None


def load_features_with_demo(tool, feature_set, demo, label_fn):
    csv_path = AU_AGGREGATED_DIR / f"{tool}_{feature_set}.csv"
    if not csv_path.exists():
        return None, None

    feat_df = pd.read_csv(csv_path)
    feature_columns = [c for c in feat_df.columns if c != "subject_id"]

    label_map, age_map = {}, {}
    for _, row in demo.iterrows():
        lbl = label_fn(row)
        if lbl is not None:
            label_map[row["ID"]] = lbl
            age_map[row["ID"]] = row["Age"]

    records = []
    for _, row in feat_df.iterrows():
        sid = row["subject_id"]
        if sid not in label_map or pd.isna(age_map.get(sid)):
            continue
        record = {"subject_id": sid, "label": label_map[sid], "age": age_map[sid]}
        for c in feature_columns:
            record[c] = row[c]
        records.append(record)

    return pd.DataFrame(records), feature_columns


def psm_match(df, caliper=2.0):
    ad_df = df[df["label"] == "AD"].copy().reset_index(drop=True)
    nm_df = df[df["label"] == "Normal"].copy().reset_index(drop=True)

    ad_ages = ad_df["age"].values.reshape(-1, 1)
    nm_ages = nm_df["age"].values.reshape(-1, 1)

    tree = KDTree(ad_ages)
    k = min(20, len(ad_ages))
    dists, indices = tree.query(nm_ages, k=k)

    order = np.argsort(dists[:, 0])
    matched_ad, matched_nm = set(), set()

    for nm_i in order:
        for j in range(k):
            ad_i = indices[nm_i, j]
            if dists[nm_i, j] > caliper:
                break
            if ad_i not in matched_ad:
                matched_ad.add(ad_i)
                matched_nm.add(nm_i)
                break

    return pd.concat([
        ad_df.iloc[list(matched_ad)],
        nm_df.iloc[list(matched_nm)],
    ], ignore_index=True)


# ═══════════════════════════════════════════════════════════
# PCA 分析核心
# ═══════════════════════════════════════════════════════════

def run_pca_comparison(ad_X, nm_X, feature_names, n_pcs=N_PCS):
    """對 AD 和 Normal 分別做 PCA，回傳分析結果"""
    n_pcs = min(n_pcs, min(ad_X.shape[1], ad_X.shape[0], nm_X.shape[0]))

    # 各組獨立 StandardScaler + PCA
    scaler_ad = StandardScaler()
    scaler_nm = StandardScaler()

    ad_scaled = scaler_ad.fit_transform(ad_X)
    nm_scaled = scaler_nm.fit_transform(nm_X)

    pca_ad = PCA(n_components=n_pcs)
    pca_nm = PCA(n_components=n_pcs)

    pca_ad.fit(ad_scaled)
    pca_nm.fit(nm_scaled)

    # Cumulative variance
    cum_var_ad = np.cumsum(pca_ad.explained_variance_ratio_)
    cum_var_nm = np.cumsum(pca_nm.explained_variance_ratio_)

    # Eigenvector cosine similarity (取絕對值，因為 PC 方向可能翻轉)
    cosine_sims = []
    for i in range(n_pcs):
        v_ad = pca_ad.components_[i]
        v_nm = pca_nm.components_[i]
        cos_sim = abs(np.dot(v_ad, v_nm) / (np.linalg.norm(v_ad) * np.linalg.norm(v_nm)))
        cosine_sims.append(cos_sim)

    # Loading 差異 (PC1 ~ PC3)
    loading_diffs = {}
    for i in range(min(3, n_pcs)):
        diff = np.abs(pca_ad.components_[i]) - np.abs(pca_nm.components_[i])
        loading_diffs[f"PC{i+1}"] = dict(zip(feature_names, diff))

    return {
        "cum_var_ad": cum_var_ad,
        "cum_var_nm": cum_var_nm,
        "var_ratio_ad": pca_ad.explained_variance_ratio_,
        "var_ratio_nm": pca_nm.explained_variance_ratio_,
        "cosine_sims": cosine_sims,
        "loading_diffs": loading_diffs,
        "components_ad": pca_ad.components_,
        "components_nm": pca_nm.components_,
        "n_pcs": n_pcs,
        "n_ad": ad_X.shape[0],
        "n_nm": nm_X.shape[0],
        "feature_names": feature_names,
    }


# ═══════════════════════════════════════════════════════════
# 圖表生成
# ═══════════════════════════════════════════════════════════

def plot_cumulative_variance(result, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, result["n_pcs"] + 1)

    ax.plot(x, result["cum_var_ad"], "o-", color="red",
            label=f"AD (n={result['n_ad']})")
    ax.plot(x, result["cum_var_nm"], "s-", color="blue",
            label=f"Normal (n={result['n_nm']})")

    for i, (va, vn) in enumerate(zip(result["cum_var_ad"], result["cum_var_nm"])):
        ax.annotate(f"{va:.3f}", (x[i], va), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="red")
        ax.annotate(f"{vn:.3f}", (x[i], vn), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color="blue")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cosine_similarity(result, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, result["n_pcs"] + 1)
    sims = result["cosine_sims"]

    bars = ax.bar(x, sims, color=["#2ecc71" if s > 0.8 else "#e74c3c" if s < 0.5 else "#f39c12"
                                   for s in sims], edgecolor="black", linewidth=0.5)

    for i, s in enumerate(sims):
        ax.text(x[i], s + 0.02, f"{s:.3f}", ha="center", fontsize=8)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("|Cosine Similarity|")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="High similarity (0.8)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loading_comparison(result, title_prefix, save_dir):
    """繪製前 3 個 PC 的 loading 比較圖"""
    feat_names = result["feature_names"]
    n_feat = len(feat_names)

    for pc_idx in range(min(3, result["n_pcs"])):
        fig, ax = plt.subplots(figsize=(max(10, n_feat * 0.25), 6))

        loadings_ad = np.abs(result["components_ad"][pc_idx])
        loadings_nm = np.abs(result["components_nm"][pc_idx])

        # 按 AD loading 排序
        sort_idx = np.argsort(loadings_ad)[::-1][:20]  # Top 20
        sorted_names = [feat_names[i] for i in sort_idx]
        sorted_ad = loadings_ad[sort_idx]
        sorted_nm = loadings_nm[sort_idx]

        x = np.arange(len(sort_idx))
        width = 0.35

        ax.bar(x - width/2, sorted_ad, width, label=f"AD (n={result['n_ad']})",
               color="#e74c3c", alpha=0.7)
        ax.bar(x + width/2, sorted_nm, width, label=f"Normal (n={result['n_nm']})",
               color="#3498db", alpha=0.7)

        ax.set_xlabel("Feature")
        ax.set_ylabel("|Loading|")
        ax.set_title(f"{title_prefix} — PC{pc_idx+1} Loading (Top 20)")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(save_dir / f"loading_PC{pc_idx+1}.png", dpi=150, bbox_inches="tight")
        plt.close()


# ═══════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════

def main():
    demo = load_demographics()

    stratifications = [
        ("cdr2", "CDR>=2 vs Normal", assign_label_cdr2),
        ("nostrat", "All Patient vs Normal", assign_label_nostrat),
    ]
    age_methods = ["original", "age", "psm"]
    tools = ["openface", "libreface"]
    feature_sets = ["harmonized", "extended"]

    all_cosine_results = []

    for (strat_key, strat_name, label_fn), tool, fs in product(
        stratifications, tools, feature_sets
    ):
        merged, feat_cols = load_features_with_demo(tool, fs, demo, label_fn)
        if merged is None:
            continue

        for age_method in age_methods:
            # 準備資料
            if age_method == "psm":
                data = psm_match(merged, caliper=2.0)
                extra_cols = []
                method_label = "PSM"
            elif age_method == "age":
                data = merged.copy()
                extra_cols = ["age"]
                method_label = "+Age"
            else:
                data = merged.copy()
                extra_cols = []
                method_label = "Original"

            all_cols = feat_cols + extra_cols

            ad_data = data[data["label"] == "AD"]
            nm_data = data[data["label"] == "Normal"]

            if len(ad_data) < 10 or len(nm_data) < 10:
                logger.warning(f"跳過 {strat_key}/{tool}/{fs}/{age_method}: 樣本不足")
                continue

            ad_X = ad_data[all_cols].values.astype(np.float32)
            nm_X = nm_data[all_cols].values.astype(np.float32)

            ad_X = np.nan_to_num(ad_X, nan=0.0)
            nm_X = np.nan_to_num(nm_X, nan=0.0)

            config_key = f"{strat_key}_{tool}_{fs}_{age_method}"
            config_title = f"{strat_name} | {tool} {fs} | {method_label}"

            logger.info(f"PCA: {config_title} (AD={len(ad_data)}, Normal={len(nm_data)})")

            # 執行 PCA 比較
            result = run_pca_comparison(ad_X, nm_X, all_cols)

            # 建立輸出目錄
            out_dir = OUTPUT_DIR / config_key
            out_dir.mkdir(parents=True, exist_ok=True)

            # 繪圖
            plot_cumulative_variance(
                result, f"Cumulative Variance — {config_title}",
                out_dir / "cumulative_variance.png"
            )
            plot_cosine_similarity(
                result, f"Eigenvector Cosine Similarity — {config_title}",
                out_dir / "cosine_similarity.png"
            )
            plot_loading_comparison(result, config_title, out_dir)

            # 儲存 cosine similarity
            for i, cs in enumerate(result["cosine_sims"]):
                all_cosine_results.append({
                    "stratification": strat_key,
                    "tool": tool,
                    "feature_set": fs,
                    "age_method": age_method,
                    "PC": f"PC{i+1}",
                    "cosine_similarity": round(cs, 4),
                    "var_ratio_AD": round(result["var_ratio_ad"][i], 4),
                    "var_ratio_Normal": round(result["var_ratio_nm"][i], 4),
                    "n_AD": result["n_ad"],
                    "n_Normal": result["n_nm"],
                })

    # 彙總 CSV
    if all_cosine_results:
        cos_df = pd.DataFrame(all_cosine_results)
        cos_df.to_csv(OUTPUT_DIR / "eigenvector_cosine_similarity.csv",
                      index=False, encoding="utf-8-sig")
        logger.info(f"\nCosine similarity 彙總已儲存: {OUTPUT_DIR / 'eigenvector_cosine_similarity.csv'}")

        # 印出摘要
        logger.info("\n" + "=" * 70)
        logger.info("Eigenvector Cosine Similarity 摘要 (PC1~PC3)")
        logger.info("=" * 70)

        summary = cos_df[cos_df["PC"].isin(["PC1", "PC2", "PC3"])].pivot_table(
            index=["stratification", "tool", "feature_set", "age_method"],
            columns="PC",
            values="cosine_similarity",
        )
        logger.info("\n" + summary.to_string())

    # 繪製 cosine similarity heatmap 總覽
    if all_cosine_results:
        plot_cosine_heatmap(cos_df)

    logger.info("\n全部完成！")


def plot_cosine_heatmap(cos_df):
    """繪製 cosine similarity 總覽 heatmap"""
    pivot = cos_df.pivot_table(
        index=["stratification", "tool", "feature_set", "age_method"],
        columns="PC",
        values="cosine_similarity",
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))

    pc_cols = [c for c in pivot.columns if c.startswith("PC")]
    data = pivot[pc_cols].values

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pc_cols)))
    ax.set_xticklabels(pc_cols, fontsize=9)

    ylabels = [f"{s}/{t}/{f}/{m}" for s, t, f, m in pivot.index]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if data[i, j] > 0.4 else "white")

    ax.set_title("Eigenvector Cosine Similarity (AD vs Normal)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cosine_similarity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap 已儲存: {OUTPUT_DIR / 'cosine_similarity_heatmap.png'}")


if __name__ == "__main__":
    main()
