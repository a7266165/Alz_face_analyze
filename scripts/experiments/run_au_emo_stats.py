"""
AU/Emotion 多面向分析（三組版：AD / NAD / ACS）

(0) 基礎統計: 三組兩兩 t-test + Cohen's d
(1) 跨工具向量差: OpenFace vs Py-Feat per-frame L2 norm (ECDF)
(2) PCA eigenvalue 表 + 累積變異量圖
(3) Eigenvector loading heatmap + cosine similarity + 符號校正
"""

import sys
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import fitz  # PyMuPDF
from feature_engine.selection import DropCorrelatedFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.modules.emotion.extractor.au_config import AU_ANALYSIS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
RAW_DIR = PROJECT_ROOT / "workspace" / "au_features" / "raw"
AGG_DIR = PROJECT_ROOT / "workspace" / "au_features" / "aggregated"
OUTPUT_BASE = AU_ANALYSIS_DIR / "au_emo_analysis"

COMMON_AU = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
COMMON_EMO = ["neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger"]

PYFEAT_AU_RENAME = {"AU01": "AU1", "AU02": "AU2", "AU04": "AU4", "AU06": "AU6",
                    "AU09": "AU9", "AU12": "AU12", "AU25": "AU25", "AU26": "AU26"}

AU_REGION = {
    "AU1": "Inner Brow Raiser", "AU2": "Outer Brow Raiser",
    "AU4": "Brow Lowerer", "AU6": "Cheek Raiser",
    "AU9": "Nose Wrinkler", "AU12": "Lip Corner Puller",
    "AU25": "Lips Part", "AU26": "Jaw Drop",
}

GROUPS = ["AD", "NAD", "ACS"]
GROUP_COLORS = {"AD": "red", "NAD": "blue", "ACS": "green"}
GROUP_PAIRS = [("AD", "NAD"), ("AD", "ACS"), ("NAD", "ACS")]

FIVE_GROUPS = ["All", "AD", "NAD", "ACS", "NAD+ACS"]
FIVE_GROUP_COLORS = {
    "All": "#888888", "AD": "#E74C3C", "NAD": "#3498DB",
    "ACS": "#2ECC71", "NAD+ACS": "#9B59B6",
}


def compile_tex_to_png(tex_path, dpi=200):
    """將 .tex 表格編譯成 PDF 再轉 PNG"""
    tex_path = Path(tex_path)
    tmp_dir = tempfile.mkdtemp()
    try:
        # 產生完整的 standalone LaTeX 文件（置中 + 緊湊裁切）
        tex_content = tex_path.read_text(encoding="utf-8")
        # 移除 table 環境，只保留 tabular（standalone 不支援 float）
        tex_content = tex_content.replace(r"\begin{table}[htbp]", "")
        tex_content = tex_content.replace(r"\end{table}", "")
        tex_content = tex_content.replace(r"\centering", "")
        # 移除 \caption 行
        lines = [l for l in tex_content.split("\n") if not l.strip().startswith(r"\caption")]
        tex_content = "\n".join(lines).strip()

        standalone_tex = (
            r"\documentclass[border=3pt]{standalone}" "\n"
            r"\usepackage{booktabs}" "\n"
            r"\usepackage{amsmath}" "\n"
            r"\begin{document}" "\n"
            + tex_content + "\n"
            r"\end{document}" "\n"
        )
        tmp_tex = Path(tmp_dir) / "table.tex"
        tmp_tex.write_text(standalone_tex, encoding="utf-8")

        # pdflatex
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(tmp_tex)],
            cwd=tmp_dir, capture_output=True, timeout=30,
        )
        tmp_pdf = Path(tmp_dir) / "table.pdf"
        if not tmp_pdf.exists():
            logger.warning(f"  LaTeX 編譯失敗: {tex_path.name}")
            return

        # PDF → PNG via PyMuPDF
        doc = fitz.open(str(tmp_pdf))
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        png_path = tex_path.with_suffix(".table.png")
        pix.save(str(png_path))
        doc.close()

        logger.info(f"  LaTeX → PNG: {png_path.name}")
    except Exception as e:
        logger.warning(f"  LaTeX 編譯/轉檔失敗: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
    """CDR>=2 分析: AD=CDR>=2的P, NAD=CDR=0或MMSE>=26的NAD, ACS=MMSE>=26的ACS"""
    g, cdr, mmse = row["group"], row["Global_CDR"], row["MMSE"]
    if g == "P":
        return "AD" if pd.notna(cdr) and cdr >= 2 else None
    if g == "NAD":
        if pd.notna(cdr) and cdr == 0:
            return "NAD"
        if pd.isna(cdr) and pd.notna(mmse) and mmse >= 26:
            return "NAD"
        return None
    if g == "ACS":
        if pd.notna(mmse) and mmse >= 26:
            return "ACS"
        return None
    return None


def assign_label_nostrat(row):
    """無分層: AD=有認知評估的P, NAD/ACS 篩選同 cdr2"""
    g, cdr, mmse = row["group"], row["Global_CDR"], row["MMSE"]
    if g == "P":
        if pd.notna(mmse) or pd.notna(cdr):
            return "AD"
        return None
    if g == "NAD":
        if pd.notna(cdr) and cdr == 0:
            return "NAD"
        if pd.isna(cdr) and pd.notna(mmse) and mmse >= 26:
            return "NAD"
        return None
    if g == "ACS":
        if pd.notna(mmse) and mmse >= 26:
            return "ACS"
        return None
    return None


def build_label_age_map(demo, label_fn):
    label_map, age_map = {}, {}
    for _, row in demo.iterrows():
        lbl = label_fn(row)
        if lbl is not None:
            label_map[row["ID"]] = lbl
            age_map[row["ID"]] = row["Age"]
    return label_map, age_map


def psm_match(df, caliper=2.0):
    """PSM: AD 與 NAD 配對，ACS 不參與"""
    ad_df = df[df["label"] == "AD"].copy().reset_index(drop=True)
    nad_df = df[df["label"] == "NAD"].copy().reset_index(drop=True)
    acs_df = df[df["label"] == "ACS"].copy()

    if len(ad_df) == 0 or len(nad_df) == 0:
        return df

    ad_ages = ad_df["age"].values.reshape(-1, 1)
    nad_ages = nad_df["age"].values.reshape(-1, 1)
    tree = KDTree(ad_ages)
    k = min(20, len(ad_ages))
    dists, indices = tree.query(nad_ages, k=k)

    order = np.argsort(dists[:, 0])
    matched_ad, matched_nad = set(), set()
    for nad_i in order:
        for j in range(k):
            if dists[nad_i, j] > caliper:
                break
            if indices[nad_i, j] not in matched_ad:
                matched_ad.add(indices[nad_i, j])
                matched_nad.add(nad_i)
                break

    return pd.concat([
        ad_df.iloc[list(matched_ad)],
        nad_df.iloc[list(matched_nad)],
        acs_df,
    ], ignore_index=True)


def load_aggregated_mean(tool, label_map, age_map):
    csv_path = AGG_DIR / f"{tool}_harmonized.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    au_mean_cols = [f"{au}_mean" for au in COMMON_AU]
    emo_mean_cols = [f"{e}_mean" for e in COMMON_EMO]

    records = []
    for _, row in df.iterrows():
        sid = row["subject_id"]
        if sid not in label_map or pd.isna(age_map.get(sid)):
            continue
        record = {"subject_id": sid, "label": label_map[sid], "age": age_map[sid]}
        for c in au_mean_cols + emo_mean_cols:
            if c in df.columns:
                record[c] = row[c]
        records.append(record)

    return pd.DataFrame(records) if records else None


def _cohend(a, b):
    pooled_sd = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0


def _sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ═══════════════════════════════════════════════════════════
# (0) 基礎統計 — 三組兩兩 t-test
# ═══════════════════════════════════════════════════════════

def group_statistics(data, tool, config_key, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    group_data = {g: data[data["label"] == g] for g in GROUPS}

    for feat_type, cols, labels in [
        ("AU", [f"{au}_mean" for au in COMMON_AU], COMMON_AU),
        ("Emotion", [f"{e}_mean" for e in COMMON_EMO], COMMON_EMO),
    ]:
        results = []
        for col, label in zip(cols, labels):
            row = {"Feature": label, "Region": AU_REGION.get(label, "")}

            # 各組 mean/std
            for g in GROUPS:
                vals = group_data[g][col].dropna()
                row[f"{g}_n"] = len(vals)
                row[f"{g}_mean"] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
                row[f"{g}_std"] = round(vals.std(), 4) if len(vals) > 0 else np.nan

            # 兩兩比較
            for g1, g2 in GROUP_PAIRS:
                v1 = group_data[g1][col].dropna()
                v2 = group_data[g2][col].dropna()
                if len(v1) < 2 or len(v2) < 2:
                    row[f"d_{g1}_{g2}"] = np.nan
                    row[f"p_{g1}_{g2}"] = np.nan
                    row[f"sig_{g1}_{g2}"] = ""
                    continue
                _, p_val = stats.ttest_ind(v1, v2, equal_var=False)
                d = _cohend(v1, v2)
                row[f"d_{g1}_{g2}"] = round(d, 3)
                row[f"p_{g1}_{g2}"] = p_val
                row[f"sig_{g1}_{g2}"] = _sig(p_val)

            results.append(row)

        if results:
            res_df = pd.DataFrame(results)
            csv_name = f"{config_key}_{feat_type.lower()}.csv"
            res_df.to_csv(out_dir / csv_name, index=False, encoding="utf-8-sig")

            n_sig = sum(1 for r in results if any(r.get(f"sig_{g1}_{g2}") for g1, g2 in GROUP_PAIRS))
            logger.info(f"  (0) {tool} {feat_type}: {n_sig}/{len(results)} features with >=1 sig pair")


# ═══════════════════════════════════════════════════════════
# (1) 跨工具向量差 (OpenFace vs Py-Feat) — ECDF
# ═══════════════════════════════════════════════════════════

def load_raw_frames(tool, subject_id):
    csv_path = RAW_DIR / tool / f"{subject_id}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if tool == "pyfeat":
        df = df.rename(columns=PYFEAT_AU_RENAME)
    if "frame" not in df.columns:
        return None
    keep = ["frame"] + [c for c in COMMON_AU + COMMON_EMO if c in df.columns]
    return df[keep].copy()


def cross_tool_comparison(label_map, age_map, config_key, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for sid, label in label_map.items():
        if pd.isna(age_map.get(sid)):
            continue

        of_df = load_raw_frames("openface", sid)
        pf_df = load_raw_frames("pyfeat", sid)
        if of_df is None or pf_df is None:
            continue

        merged = of_df.merge(pf_df, on="frame", suffixes=("_of", "_pf"))
        if len(merged) == 0:
            continue

        au_of_cols = [f"{au}_of" for au in COMMON_AU if f"{au}_of" in merged.columns]
        au_pf_cols = [f"{au}_pf" for au in COMMON_AU if f"{au}_pf" in merged.columns]
        if au_of_cols and au_pf_cols:
            au_diff = merged[au_of_cols].values - merged[au_pf_cols].values
            au_l2_mean = np.mean(np.sqrt(np.sum(au_diff**2, axis=1)))
        else:
            au_l2_mean = np.nan

        emo_of_cols = [f"{e}_of" for e in COMMON_EMO if f"{e}_of" in merged.columns]
        emo_pf_cols = [f"{e}_pf" for e in COMMON_EMO if f"{e}_pf" in merged.columns]
        if emo_of_cols and emo_pf_cols:
            emo_diff = merged[emo_of_cols].values - merged[emo_pf_cols].values
            emo_l2_mean = np.mean(np.sqrt(np.sum(emo_diff**2, axis=1)))
        else:
            emo_l2_mean = np.nan

        records.append({
            "subject_id": sid, "label": label, "age": age_map[sid],
            "n_frames": len(merged), "au_l2_mean": au_l2_mean, "emo_l2_mean": emo_l2_mean,
        })

    if not records:
        logger.warning("  (1) 無有效配對資料")
        return

    result_df = pd.DataFrame(records)
    result_df.to_csv(out_dir / f"{config_key}_cross_tool.csv", index=False, encoding="utf-8-sig")

    # 三組兩兩統計
    for feat_type, col in [("AU", "au_l2_mean"), ("Emotion", "emo_l2_mean")]:
        for g1, g2 in GROUP_PAIRS:
            v1 = result_df[result_df["label"] == g1][col].dropna()
            v2 = result_df[result_df["label"] == g2][col].dropna()
            if len(v1) < 2 or len(v2) < 2:
                continue
            _, p_val = stats.ttest_ind(v1, v2, equal_var=False)
            d = _cohend(v1, v2)
            logger.info(
                f"  (1) {feat_type} L2 {g1} vs {g2}: "
                f"{g1}={v1.mean():.4f}+/-{v1.std():.4f} (n={len(v1)}), "
                f"{g2}={v2.mean():.4f}+/-{v2.std():.4f} (n={len(v2)}), "
                f"d={d:.3f}, p={p_val:.2e}"
            )

    # ECDF 圖
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (feat_type, col) in zip(axes, [("AU", "au_l2_mean"), ("Emotion", "emo_l2_mean")]):
        for g in GROUPS:
            vals = result_df[result_df["label"] == g][col].dropna().values
            if len(vals) == 0:
                continue
            sorted_vals = np.sort(vals)
            ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, ecdf, color=GROUP_COLORS[g],
                    label=f"{g} (n={len(vals)})", linewidth=1.5)

        ax.set_xlabel(f"{feat_type} L2 Norm (OpenFace - Py-Feat)")
        ax.set_ylabel("Cumulative Proportion")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{feat_type} Cross-Tool Difference")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cross-Tool Vector Difference (ECDF) — {config_key}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{config_key}_cross_tool.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════
# (2) PCA eigenvalue 表 + 累積變異量圖 — 三組
# ═══════════════════════════════════════════════════════════

def pca_eigenvalues(data, tool, config_key, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    group_data = {g: data[data["label"] == g] for g in GROUPS}

    for feat_type, cols in [
        ("AU", [f"{au}_mean" for au in COMMON_AU]),
        ("Emotion", [f"{e}_mean" for e in COMMON_EMO]),
    ]:
        available_cols = [c for c in cols if c in data.columns]
        if not available_cols:
            continue

        n_components = len(available_cols)
        results = []
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(1, n_components + 1)

        for g in GROUPS:
            gd = group_data[g]
            X = gd[available_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)
            if len(X) < n_components:
                continue

            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)

            cum_var = np.cumsum(pca.explained_variance_ratio_)
            ax.plot(x, cum_var, "o-", color=GROUP_COLORS[g],
                    label=f"{g} (n={len(X)})", linewidth=1.5)

            for i, v in enumerate(cum_var):
                offset = {"AD": 10, "NAD": -14, "ACS": -14}.get(g, 8)
                ax.annotate(f"{v:.3f}", (x[i], v), textcoords="offset points",
                            xytext=(0, offset), ha="center", fontsize=7,
                            color=GROUP_COLORS[g])

            for i in range(n_components):
                results.append({
                    "Group": g, "PC": f"PC{i+1}",
                    "Eigenvalue": round(pca.explained_variance_[i], 4),
                    "Variance_Ratio": round(pca.explained_variance_ratio_[i], 4),
                    "Cumulative": round(cum_var[i], 4),
                })

        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Cumulative Explained Variance Ratio")
        ax.set_title(f"PCA Cumulative Variance — {tool} {feat_type} | {config_key}")
        ax.set_xticks(x)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{config_key}_{feat_type.lower()}_cumvar.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        if results:
            pd.DataFrame(results).to_csv(
                out_dir / f"{config_key}_{feat_type.lower()}_eigenvalues.csv",
                index=False, encoding="utf-8-sig"
            )


# ═══════════════════════════════════════════════════════════
# (3) PCA loadings — 三組 + 符號校正 + cosine similarity
# ═══════════════════════════════════════════════════════════

def _sign_correct(ref_components, other_components):
    """以 ref 為基準，翻轉 other 的 PC 使 cosine similarity >= 0"""
    corrected = other_components.copy()
    for i in range(len(corrected)):
        cos_sim = np.dot(ref_components[i], corrected[i]) / (
            np.linalg.norm(ref_components[i]) * np.linalg.norm(corrected[i]) + 1e-12
        )
        if cos_sim < 0:
            corrected[i] *= -1
    return corrected


def _cosine_sim_matrix(components_dict):
    """計算三組兩兩 PC 的 cosine similarity"""
    records = []
    groups = list(components_dict.keys())
    n_pcs = components_dict[groups[0]].shape[0]

    for i in range(n_pcs):
        row = {"PC": f"PC{i+1}"}
        for g1, g2 in combinations(groups, 2):
            v1 = components_dict[g1][i]
            v2 = components_dict[g2][i]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            row[f"{g1}_vs_{g2}"] = round(cos_sim, 4)
        records.append(row)

    return pd.DataFrame(records)


def pca_loadings(data, tool, config_key, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    group_data = {g: data[data["label"] == g] for g in GROUPS}

    for feat_type, cols in [
        ("AU", [f"{au}_mean" for au in COMMON_AU]),
        ("Emotion", [f"{e}_mean" for e in COMMON_EMO]),
    ]:
        available_cols = [c for c in cols if c in data.columns]
        if not available_cols:
            continue

        n_components = len(available_cols)
        labels_short = [c.replace("_mean", "") for c in available_cols]

        # 各組做 PCA
        components = {}
        pca_models = {}
        n_samples = {}
        for g in GROUPS:
            gd = group_data[g]
            X = gd[available_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)
            if len(X) < n_components:
                continue

            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
            components[g] = pca.components_.copy()  # (n_components, n_features)
            pca_models[g] = pca
            n_samples[g] = len(X)

        if "AD" not in components:
            continue

        # 符號校正: 以 AD 為基準
        ref = components["AD"]
        for g in ["NAD", "ACS"]:
            if g in components:
                components[g] = _sign_correct(ref, components[g])

        # Cosine similarity (校正後)
        cos_df = _cosine_sim_matrix(components)
        cos_df.to_csv(out_dir / f"{config_key}_{feat_type.lower()}_cosine_sim.csv",
                      index=False, encoding="utf-8-sig")

        # 繪圖: 三組 loading heatmap
        n_groups = len(components)
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, max(5, n_components * 0.5)))
        if n_groups == 1:
            axes = [axes]

        for ax, g in zip(axes, [g for g in GROUPS if g in components]):
            loading_matrix = components[g].T  # (n_features, n_components)
            loading_df = pd.DataFrame(
                loading_matrix, index=labels_short,
                columns=[f"PC{i+1}" for i in range(n_components)]
            )
            sns.heatmap(loading_df, annot=True, fmt=".2f", cmap="RdBu_r",
                        center=0, vmin=-1, vmax=1, ax=ax,
                        cbar_kws={"shrink": 0.8})
            ax.set_title(f"{g} (n={n_samples[g]})")
            ax.set_ylabel("Feature" if ax == axes[0] else "")

        # cosine similarity 標注
        cos_text = " | ".join(
            f"{col}: [{', '.join(f'{cos_df.iloc[i][col]:.2f}' for i in range(min(3, len(cos_df))))}]"
            for col in cos_df.columns if col != "PC"
        )
        fig.suptitle(
            f"PCA Loadings — {tool} {feat_type} | {config_key}\n"
            f"Cosine Sim (PC1-3): {cos_text}",
            fontsize=10
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{config_key}_{feat_type.lower()}_loadings.png",
                    dpi=150, bbox_inches="tight")
        plt.close()


# ═══════════════════════════════════════════════════════════
# (4) 五組描述統計 — grouped bar chart + LaTeX table
# ═══════════════════════════════════════════════════════════

def descriptive_bar_plot(data, tool, config_key, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 建立五組子集
    group_subsets = {
        "All": data,
        "AD": data[data["label"] == "AD"],
        "NAD": data[data["label"] == "NAD"],
        "ACS": data[data["label"] == "ACS"],
        "NAD+ACS": data[data["label"].isin(["NAD", "ACS"])],
    }

    for feat_type, cols, labels in [
        ("au", [f"{au}_mean" for au in COMMON_AU], COMMON_AU),
        ("emotion", [f"{e}_mean" for e in COMMON_EMO], COMMON_EMO),
    ]:
        # 計算各組統計量
        stats_rows = []
        for col, label in zip(cols, labels):
            row = {"Feature": label, "Region": AU_REGION.get(label, "")}
            for g in FIVE_GROUPS:
                vals = group_subsets[g][col].dropna()
                row[f"{g}_n"] = len(vals)
                row[f"{g}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                row[f"{g}_std"] = vals.std() if len(vals) > 0 else np.nan
            stats_rows.append(row)

        stats_df = pd.DataFrame(stats_rows)

        # --- CSV ---
        csv_path = out_dir / f"{config_key}_{feat_type}_descriptive.csv"
        stats_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # --- LaTeX ---
        tool_display = {"openface": "OpenFace", "pyfeat": "Py-Feat"}.get(tool, tool)
        feat_display = "AU" if feat_type == "au" else "Emotion"
        tex_lines = []
        tex_lines.append(r"\begin{table}[htbp]")
        tex_lines.append(r"\centering")
        tex_lines.append(
            rf"\caption{{Descriptive Statistics of {feat_display} Features — {tool_display} (NoStrat)}}"
        )
        tex_lines.append(r"\begin{tabular}{l" + "c" * len(FIVE_GROUPS) + "}")
        tex_lines.append(r"\toprule")

        # 表頭
        header_parts = ["Feature"]
        for g in FIVE_GROUPS:
            n = stats_rows[0][f"{g}_n"]
            header_parts.append(rf"{g} (n={n})")
        tex_lines.append(" & ".join(header_parts) + r" \\")
        tex_lines.append(r"\midrule")

        # 資料行
        for row in stats_rows:
            parts = [row["Feature"]]
            for g in FIVE_GROUPS:
                m = row[f"{g}_mean"]
                s = row[f"{g}_std"]
                if np.isnan(m):
                    parts.append("—")
                else:
                    parts.append(f"{m:.4f} $\\pm$ {s:.4f}")
            tex_lines.append(" & ".join(parts) + r" \\")

        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append(r"\end{table}")

        tex_path = out_dir / f"{config_key}_{feat_type}_descriptive.tex"
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
        compile_tex_to_png(tex_path)

        # --- Line chart with band-style std ribbon ---
        n_features = len(labels)
        x = np.arange(n_features)

        fig, ax = plt.subplots(figsize=(max(10, n_features * 1.5), 6))
        for g in FIVE_GROUPS:
            means = np.array([stats_rows[j][f"{g}_mean"] for j in range(n_features)])
            stds = np.array([stats_rows[j][f"{g}_std"] for j in range(n_features)])
            n = stats_rows[0][f"{g}_n"]
            color = FIVE_GROUP_COLORS[g]

            upper = means + stds
            lower = means - stds
            # Dashed border + light fill
            ax.fill_between(x, lower, upper, color=color, alpha=0.10)
            ax.plot(x, upper, "--", color=color, linewidth=0.8, alpha=0.6)
            ax.plot(x, lower, "--", color=color, linewidth=0.8, alpha=0.6)
            ax.plot(x, means, "o-", color=color, label=f"{g} (n={n})",
                    linewidth=2, markersize=5, zorder=5)

        ax.set_xlabel(feat_display)
        ax.set_ylabel("Mean Value (subject-level)")
        ax.set_title(f"{feat_display} Descriptive Statistics — {tool_display} (NoStrat)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0 if feat_type == "au" else 25, ha="center" if feat_type == "au" else "right")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"{config_key}_{feat_type}_descriptive.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

        logger.info(f"  (4) {tool} {feat_display}: 5-group descriptive stats saved")


# ═══════════════════════════════════════════════════════════
# (5) 關聯性分析 — per-frame 原始特徵合併 + DropCorrelatedFeatures
# ═══════════════════════════════════════════════════════════

def load_perframe_merged():
    """載入 OpenFace + Py-Feat 所有 per-frame 原始特徵，用 (subject_id, frame) inner join。

    OpenFace: 18 features (8 AU + 8 Emo + 2 gaze)
    Py-Feat:  27 features (20 AU + 7 Emo)
    共通欄位加 _of / _pf suffix → 合計 45 columns
    """
    of_dir = RAW_DIR / "openface"
    pf_dir = RAW_DIR / "pyfeat"
    if not of_dir.exists() or not pf_dir.exists():
        return None

    # 找出兩邊都有的 subject
    of_files = {f.stem: f for f in of_dir.glob("*.csv")}
    pf_files = {f.stem: f for f in pf_dir.glob("*.csv")}
    common_sids = sorted(of_files.keys() & pf_files.keys())
    logger.info(f"    per-frame merge: {len(common_sids)} common subjects "
                f"(OF={len(of_files)}, PF={len(pf_files)})")

    frames = []
    for sid in common_sids:
        of_df = pd.read_csv(of_files[sid])
        pf_df = pd.read_csv(pf_files[sid])

        # Py-Feat 有 _aligned 重複行，去掉
        pf_df = pf_df[~pf_df["frame"].str.contains("_aligned_aligned", na=False)]

        # inner join on frame
        merged = of_df.merge(pf_df, on="frame", suffixes=("_of", "_pf"))
        if len(merged) == 0:
            continue
        merged.insert(0, "subject_id", sid)
        frames.append(merged)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    # 去掉 frame, subject_id 只保留特徵
    feat_cols = [c for c in df.columns if c not in ("subject_id", "frame")]
    X = df[feat_cols].astype(np.float32).fillna(0)
    return X, feat_cols, len(common_sids)


def correlation_analysis(tools, out_dir):
    """合併 OpenFace + Py-Feat per-frame 原始特徵，用不同閾值分析相關性"""
    out_dir.mkdir(parents=True, exist_ok=True)

    result = load_perframe_merged()
    if result is None:
        logger.warning("  (5) 跳過關聯性分析: 無法合併 per-frame 資料")
        return

    X, feat_cols, n_subjects = result
    n_total = len(feat_cols)

    logger.info(f"\n  (5) OpenFace+Py-Feat per-frame 合併關聯性分析 "
                f"(n_frames={len(X)}, n_subjects={n_subjects}, "
                f"n_features={n_total})")

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    tag = "openface_pyfeat_perframe"
    title_tag = "OpenFace + Py-Feat Per-Frame"

    # 各閾值下保留的特徵數
    summary_rows = []
    detail_records = {}
    for thr in thresholds:
        dcf = DropCorrelatedFeatures(
            variables=None, method="pearson", threshold=thr
        )
        dcf.fit(X)
        dropped = dcf.features_to_drop_
        kept = [c for c in feat_cols if c not in dropped]
        n_kept = len(kept)
        n_dropped = len(dropped)

        summary_rows.append({
            "threshold": thr,
            "n_total": n_total,
            "n_kept": n_kept,
            "n_dropped": n_dropped,
            "pct_kept": round(n_kept / n_total * 100, 1),
        })
        detail_records[thr] = {
            "kept": kept, "dropped": list(dropped),
            "correlated_sets": dcf.correlated_feature_sets_,
        }

        logger.info(f"    threshold={thr:.2f}: kept={n_kept}/{n_total} "
                    f"({n_kept/n_total*100:.1f}%), dropped={n_dropped}")

    # --- Summary CSV ---
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"{tag}_correlation_summary.csv",
                      index=False, encoding="utf-8-sig")

    # --- Detail CSV (每個閾值保留/丟棄的特徵) ---
    detail_rows = []
    for thr in thresholds:
        for feat in feat_cols:
            detail_rows.append({
                "threshold": thr,
                "feature": feat,
                "status": "kept" if feat in detail_records[thr]["kept"] else "dropped",
            })
    pd.DataFrame(detail_rows).to_csv(
        out_dir / f"{tag}_correlation_detail.csv",
        index=False, encoding="utf-8-sig",
    )

    # --- 折線圖: 閾值 vs 保留特徵數 ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ts = [r["threshold"] for r in summary_rows]
    ns = [r["n_kept"] for r in summary_rows]
    pcts = [r["pct_kept"] for r in summary_rows]

    ax1.plot(ts, ns, "o-", color="#2980B9", linewidth=2, markersize=7)
    for t, n, p in zip(ts, ns, pcts):
        ax1.annotate(f"{n}\n({p}%)", (t, n), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)

    ax1.axhline(y=n_total, color="gray", linestyle="--", alpha=0.5,
                label=f"Total features = {n_total}")
    ax1.set_xlabel("Correlation Threshold")
    ax1.set_ylabel("Number of Features Kept")
    ax1.set_title(f"DropCorrelatedFeatures — {title_tag}\n"
                  f"(n_frames={len(X)}, n_subjects={n_subjects}, "
                  f"{n_total} features)")
    ax1.set_xticks(ts)
    ax1.set_ylim(0, n_total * 1.15)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_correlation_threshold.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- 相關矩陣 heatmap ---
    corr = X.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(20, 18))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 5}, ax=ax,
                square=True, linewidths=0.2,
                cbar_kws={"shrink": 0.6})
    ax.set_title(f"Feature Correlation Matrix — {title_tag}\n"
                 f"(n_frames={len(X)})", fontsize=12)
    ax.tick_params(axis="both", labelsize=5)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_correlation_matrix.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- LaTeX summary table ---
    tex_lines = []
    tex_lines.append(r"\begin{tabular}{cccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"Threshold & Kept & Dropped & Kept (\%) \\")
    tex_lines.append(r"\midrule")
    for r in summary_rows:
        tex_lines.append(
            f"{r['threshold']:.2f} & {r['n_kept']} & {r['n_dropped']} "
            f"& {r['pct_kept']}\\% \\\\"
        )
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")

    tex_path = out_dir / f"{tag}_correlation_summary.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    compile_tex_to_png(tex_path)

    logger.info("  (5) 關聯性分析完成")


# ═══════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════

def main():
    demo = load_demographics()

    stratifications = [
        ("cdr2", "CDR>=2 vs NAD vs ACS", assign_label_cdr2),
        ("nostrat", "All P vs NAD vs ACS", assign_label_nostrat),
    ]
    age_methods = ["original", "age", "psm"]
    tools = ["openface", "pyfeat", "libreface"]

    for (strat_key, strat_name, label_fn) in stratifications:
        label_map, age_map = build_label_age_map(demo, label_fn)

        # 顯示各組人數
        counts = {}
        for g in GROUPS:
            counts[g] = sum(1 for v in label_map.values() if v == g)
        logger.info(f"\n{'='*60}")
        logger.info(f"分層: {strat_name}")
        logger.info(f"人數: " + ", ".join(f"{g}={counts[g]}" for g in GROUPS))
        logger.info(f"{'='*60}")

        for tool in tools:
            agg_data = load_aggregated_mean(tool, label_map, age_map)
            if agg_data is None:
                logger.warning(f"跳過 {tool}: 無 aggregated 資料")
                continue

            for age_method in age_methods:
                if age_method == "psm":
                    data = psm_match(agg_data.copy(), caliper=2.0)
                    method_label = "PSM"
                elif age_method == "age":
                    data = agg_data.copy()
                    method_label = "+Age"
                else:
                    data = agg_data.copy()
                    method_label = "Original"

                config_key = f"{strat_key}_{tool}_{age_method}"
                group_counts = {g: sum(data["label"] == g) for g in GROUPS}

                logger.info(f"\n{strat_name} | {tool} | {method_label} "
                            f"({', '.join(f'{g}={group_counts[g]}' for g in GROUPS)})")

                # (0) 基礎統計
                group_statistics(data, tool, config_key, OUTPUT_BASE / "group_stats")

                # (2) PCA eigenvalue
                pca_eigenvalues(data, tool, config_key, OUTPUT_BASE / "pca_eigenvalues")

                # (3) PCA loadings
                pca_loadings(data, tool, config_key, OUTPUT_BASE / "pca_loadings")

                # (4) 五組描述統計 (只在 nostrat + original 時跑)
                if strat_key == "nostrat" and age_method == "original":
                    descriptive_bar_plot(data, tool, config_key,
                                        OUTPUT_BASE / "descriptive_stats")

        # (1) 跨工具比較
        for age_method in age_methods:
            config_key = f"{strat_key}_{age_method}"

            if age_method == "psm":
                all_sids = set()
                for t in ["openface", "pyfeat"]:
                    raw_dir = RAW_DIR / t
                    if raw_dir.exists():
                        all_sids.update(f.stem for f in raw_dir.glob("*.csv"))

                records = []
                for sid in all_sids:
                    if sid in label_map and not pd.isna(age_map.get(sid)):
                        records.append({"subject_id": sid, "label": label_map[sid],
                                        "age": age_map[sid]})
                if records:
                    psm_df = psm_match(pd.DataFrame(records), caliper=2.0)
                    psm_label_map = dict(zip(psm_df["subject_id"], psm_df["label"]))
                    psm_age_map = dict(zip(psm_df["subject_id"], psm_df["age"]))
                    cross_tool_comparison(psm_label_map, psm_age_map, config_key,
                                         OUTPUT_BASE / "cross_tool")
            else:
                cross_tool_comparison(label_map, age_map, config_key,
                                     OUTPUT_BASE / "cross_tool")

    # (5) 關聯性分析 (OpenFace+Py-Feat 合併, 所有人次, 不分組)
    correlation_analysis(tools, OUTPUT_BASE / "correlation")

    logger.info("\n全部完成！")


if __name__ == "__main__":
    main()
