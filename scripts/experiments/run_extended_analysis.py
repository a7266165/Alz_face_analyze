"""
Extended Analysis — 五個新分析面向
(1) CDR 嚴重度梯度 (Dose-Response)
(2) MMSE/CASI 連續認知分數相關
(3) 情緒組成分析 (Emotion Profile)
(4) 縱貫性 Within-Subject 變化
(5) 性別分層分析
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

from src.modules.emotion.extractor.au_config import AU_ANALYSIS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
AGG_DIR = PROJECT_ROOT / "workspace" / "au_features" / "aggregated"
OUTPUT_BASE = AU_ANALYSIS_DIR / "au_emo_analysis"

COMMON_AU = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
COMMON_EMO = ["neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger"]
ALL_FEATURES = COMMON_AU + COMMON_EMO
FEATURE_COLS = [f"{f}_mean" for f in ALL_FEATURES]

GROUPS = ["AD", "NAD", "ACS"]
FIVE_GROUPS = ["All", "AD", "NAD", "ACS", "NAD+ACS"]
FIVE_GROUP_COLORS = {
    "All": "#888888", "AD": "#E74C3C", "NAD": "#3498DB",
    "ACS": "#2ECC71", "NAD+ACS": "#9B59B6",
}
CDR_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0]
CDR_COLORS = {0.0: "#2ECC71", 0.5: "#F1C40F", 1.0: "#E67E22", 2.0: "#E74C3C", 3.0: "#8E44AD"}

TOOLS = ["openface", "pyfeat", "libreface"]
TOOL_DISPLAY = {"openface": "OpenFace", "pyfeat": "Py-Feat", "libreface": "LibreFace"}


# ═══════════════════════════════════════════════════════════
# 工具函式
# ═══════════════════════════════════════════════════════════

def compile_tex_to_png(tex_path, dpi=200):
    tex_path = Path(tex_path)
    tmp_dir = tempfile.mkdtemp()
    try:
        tex_content = tex_path.read_text(encoding="utf-8")
        tex_content = tex_content.replace(r"\begin{table}[htbp]", "")
        tex_content = tex_content.replace(r"\end{table}", "")
        tex_content = tex_content.replace(r"\centering", "")
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

        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(tmp_tex)],
            cwd=tmp_dir, capture_output=True, timeout=30,
        )
        tmp_pdf = Path(tmp_dir) / "table.pdf"
        if not tmp_pdf.exists():
            logger.warning(f"  LaTeX 編譯失敗: {tex_path.name}")
            return
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


def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def _cohend(a, b):
    pooled_sd = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0


# ═══════════════════════════════════════════════════════════
# 資料載入
# ═══════════════════════════════════════════════════════════

def load_demographics():
    frames = []
    for g in ["P", "NAD", "ACS"]:
        df = pd.read_csv(DEMOGRAPHICS_DIR / f"{g}.csv")
        df["group"] = g
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_full_data(tool, demo):
    """載入 aggregated features 並合併完整 demographics（含 CDR, MMSE, Sex 等）"""
    csv_path = AGG_DIR / f"{tool}_harmonized.csv"
    if not csv_path.exists():
        return None

    agg = pd.read_csv(csv_path)
    # demo 的 ID 列 = agg 的 subject_id
    merged = agg.merge(demo, left_on="subject_id", right_on="ID", how="inner")

    # 保留需要的欄位
    keep_cols = ["subject_id", "group", "Age", "Sex",
                 "Global_CDR", "CDR_SB", "MMSE", "CASI"]
    keep_cols += [c for c in FEATURE_COLS if c in merged.columns]
    merged = merged[[c for c in keep_cols if c in merged.columns]]
    return merged


# ═══════════════════════════════════════════════════════════
# (1) CDR 嚴重度梯度 (Dose-Response)
# ═══════════════════════════════════════════════════════════

def cdr_gradient_analysis(data, tool, out_dir):
    """P 組內 CDR 0→3 的特徵趨勢分析"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_disp = TOOL_DISPLAY.get(tool, tool)

    # 只取 P 組 + 有 CDR 的
    df = data[(data["group"] == "P") & data["Global_CDR"].notna()].copy()
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(df) == 0:
        logger.warning(f"  (1) {tool}: 無 P 組 CDR 資料")
        return

    logger.info(f"  (1) {tool_disp} CDR gradient: n={len(df)}, "
                f"CDR分佈: {df['Global_CDR'].value_counts().sort_index().to_dict()}")

    # --- A: Spearman correlation (CDR ordinal + CDR_SB continuous) ---
    corr_rows = []
    for feat in feat_cols:
        vals = df[[feat, "Global_CDR", "Age"]].dropna()
        if len(vals) < 10:
            continue

        feat_name = feat.replace("_mean", "")

        # Raw Spearman vs CDR
        rho, p = stats.spearmanr(vals[feat], vals["Global_CDR"])

        # Partial Spearman controlling age
        try:
            pc = pg.partial_corr(data=vals, x=feat, y="Global_CDR",
                                 covar="Age", method="spearman")
            partial_r = pc["r"].values[0]
            partial_p = pc["p_val"].values[0]
        except Exception:
            partial_r, partial_p = np.nan, np.nan

        # CDR_SB (continuous) if available
        rho_sb, p_sb = np.nan, np.nan
        partial_r_sb, partial_p_sb = np.nan, np.nan
        if "CDR_SB" in df.columns:
            vals_sb = df[[feat, "CDR_SB", "Age"]].dropna()
            if len(vals_sb) >= 10:
                rho_sb, p_sb = stats.spearmanr(vals_sb[feat], vals_sb["CDR_SB"])
                try:
                    pc_sb = pg.partial_corr(data=vals_sb, x=feat, y="CDR_SB",
                                            covar="Age", method="spearman")
                    partial_r_sb = pc_sb["r"].values[0]
                    partial_p_sb = pc_sb["p_val"].values[0]
                except Exception:
                    pass

        corr_rows.append({
            "feature": feat_name,
            "n": len(vals),
            "rho_CDR": round(rho, 4),
            "p_CDR": p,
            "sig_CDR": _sig(p),
            "partial_rho_CDR": round(partial_r, 4) if pd.notna(partial_r) else np.nan,
            "partial_p_CDR": partial_p,
            "partial_sig_CDR": _sig(partial_p) if pd.notna(partial_p) else "",
            "rho_CDR_SB": round(rho_sb, 4) if pd.notna(rho_sb) else np.nan,
            "p_CDR_SB": p_sb,
            "partial_rho_CDR_SB": round(partial_r_sb, 4) if pd.notna(partial_r_sb) else np.nan,
            "partial_p_CDR_SB": partial_p_sb,
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / f"{tool}_cdr_spearman.csv",
                   index=False, encoding="utf-8-sig")

    # --- B: Jonckheere-Terpstra trend test ---
    jt_rows = []
    for feat in feat_cols:
        feat_name = feat.replace("_mean", "")
        groups_data = []
        for cdr in CDR_LEVELS:
            subset = df[df["Global_CDR"] == cdr][feat].dropna()
            if len(subset) > 0:
                groups_data.append(subset.values)
        if len(groups_data) < 3:
            continue

        # Manual JT statistic (sum of Mann-Whitney U for ordered pairs)
        jt_stat = 0
        n_comparisons = 0
        for i in range(len(groups_data)):
            for j in range(i + 1, len(groups_data)):
                u, _ = stats.mannwhitneyu(groups_data[i], groups_data[j],
                                          alternative="two-sided")
                jt_stat += u
                n_comparisons += 1

        # Kruskal-Wallis as supplement
        kw_stat, kw_p = stats.kruskal(*groups_data)

        jt_rows.append({
            "feature": feat_name,
            "JT_statistic": jt_stat,
            "KW_statistic": round(kw_stat, 4),
            "KW_p": kw_p,
            "KW_sig": _sig(kw_p),
        })
    jt_df = pd.DataFrame(jt_rows)
    jt_df.to_csv(out_dir / f"{tool}_cdr_trend_test.csv",
                 index=False, encoding="utf-8-sig")

    # --- C: 折線圖 mean ± 95% CI across CDR levels ---
    for feat_type, feats, title_suffix in [
        ("au", [f for f in feat_cols if any(f.startswith(a) for a in ["AU"])], "AU"),
        ("emotion", [f for f in feat_cols if not any(f.startswith(a) for a in ["AU"])], "Emotion"),
    ]:
        if not feats:
            continue
        n_feat = len(feats)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax_idx, (ax, plot_title) in enumerate(zip(axes, ["Raw", "Age-Adjusted"])):
            for fi, feat in enumerate(feats):
                feat_name = feat.replace("_mean", "")
                means, cis_lo, cis_hi = [], [], []

                for cdr in CDR_LEVELS:
                    if ax_idx == 0:
                        vals = df[df["Global_CDR"] == cdr][feat].dropna()
                    else:
                        # Age-adjusted: residuals from age regression
                        subset = df[df["Global_CDR"] == cdr][[feat, "Age"]].dropna()
                        if len(subset) < 3:
                            means.append(np.nan)
                            cis_lo.append(np.nan)
                            cis_hi.append(np.nan)
                            continue
                        # Compute residuals from overall age regression
                        all_valid = df[[feat, "Age"]].dropna()
                        slope, intercept, _, _, _ = stats.linregress(all_valid["Age"], all_valid[feat])
                        residuals = subset[feat] - (slope * subset["Age"] + intercept)
                        vals = residuals

                    if len(vals) < 2:
                        means.append(np.nan)
                        cis_lo.append(np.nan)
                        cis_hi.append(np.nan)
                        continue
                    m = vals.mean()
                    se = vals.std() / np.sqrt(len(vals))
                    ci = 1.96 * se
                    means.append(m)
                    cis_lo.append(m - ci)
                    cis_hi.append(m + ci)

                x = np.arange(len(CDR_LEVELS))
                ax.plot(x, means, "o-", label=feat_name, linewidth=1.5, markersize=4)
                ax.fill_between(x, cis_lo, cis_hi, alpha=0.1)

            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in CDR_LEVELS])
            ax.set_xlabel("CDR Level")
            ax.set_ylabel("Mean Value" if ax_idx == 0 else "Age-Adjusted Residual")
            ax.set_title(f"{plot_title}")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"CDR Severity Gradient — {tool_disp} {title_suffix}\n"
                     f"(P cohort, n={len(df)})", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"{tool}_cdr_gradient_{feat_type}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # --- D: LaTeX summary table ---
    tex_lines = [r"\begin{tabular}{lcccccc}",
                 r"\toprule",
                 r"Feature & $\rho$ (CDR) & $p$ & Partial $\rho$ & Partial $p$ & KW $p$ \\",
                 r"\midrule"]
    for _, row in corr_df.iterrows():
        kw_row = jt_df[jt_df["feature"] == row["feature"]]
        kw_p = kw_row["KW_p"].values[0] if len(kw_row) > 0 else np.nan
        tex_lines.append(
            f"{row['feature']} & {row['rho_CDR']:.3f}{row['sig_CDR']} "
            f"& {row['p_CDR']:.4f} "
            f"& {row['partial_rho_CDR']:.3f}{row['partial_sig_CDR']} "
            f"& {row['partial_p_CDR']:.4f} "
            f"& {kw_p:.4f} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = out_dir / f"{tool}_cdr_gradient_summary.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    compile_tex_to_png(tex_path)

    logger.info(f"  (1) {tool_disp} CDR gradient analysis done")


# ═══════════════════════════════════════════════════════════
# (2) MMSE/CASI 連續認知分數相關
# ═══════════════════════════════════════════════════════════

def cognitive_correlation_analysis(data, tool, out_dir):
    """特徵與 MMSE/CASI 的相關分析（含控制年齡的偏相關）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_disp = TOOL_DISPLAY.get(tool, tool)

    feat_cols = [c for c in FEATURE_COLS if c in data.columns]

    for cog_var in ["MMSE", "CASI"]:
        if cog_var not in data.columns:
            continue

        for scope_name, scope_df in [("all", data), ("P_only", data[data["group"] == "P"])]:
            df = scope_df[scope_df[cog_var].notna()].copy()
            if len(df) < 20:
                continue

            logger.info(f"  (2) {tool_disp} {cog_var} ({scope_name}): n={len(df)}")

            corr_rows = []
            for feat in feat_cols:
                vals = df[[feat, cog_var, "Age"]].dropna()
                if len(vals) < 10:
                    continue
                feat_name = feat.replace("_mean", "")

                # Raw correlation
                r_p, p_p = stats.pearsonr(vals[feat], vals[cog_var])
                r_s, p_s = stats.spearmanr(vals[feat], vals[cog_var])

                # Partial correlation controlling age
                try:
                    pc = pg.partial_corr(data=vals, x=feat, y=cog_var,
                                         covar="Age", method="pearson")
                    partial_r = pc["r"].values[0]
                    partial_p = pc["p_val"].values[0]
                except Exception:
                    partial_r, partial_p = np.nan, np.nan

                # Partial controlling age + sex
                partial_r2, partial_p2 = np.nan, np.nan
                if "Sex" in df.columns:
                    vals_sex = df[[feat, cog_var, "Age", "Sex"]].dropna()
                    vals_sex["Sex_num"] = (vals_sex["Sex"] == "M").astype(int)
                    try:
                        pc2 = pg.partial_corr(data=vals_sex, x=feat, y=cog_var,
                                              covar=["Age", "Sex_num"], method="pearson")
                        partial_r2 = pc2["r"].values[0]
                        partial_p2 = pc2["p_val"].values[0]
                    except Exception:
                        pass

                corr_rows.append({
                    "feature": feat_name,
                    "n": len(vals),
                    "pearson_r": round(r_p, 4),
                    "pearson_p": p_p,
                    "spearman_rho": round(r_s, 4),
                    "spearman_p": p_s,
                    "partial_r_age": round(partial_r, 4) if pd.notna(partial_r) else np.nan,
                    "partial_p_age": partial_p,
                    "partial_sig_age": _sig(partial_p) if pd.notna(partial_p) else "",
                    "partial_r_age_sex": round(partial_r2, 4) if pd.notna(partial_r2) else np.nan,
                    "partial_p_age_sex": partial_p2,
                })

            corr_df = pd.DataFrame(corr_rows)
            tag = f"{tool}_{cog_var}_{scope_name}"
            corr_df.to_csv(out_dir / f"{tag}_correlation.csv",
                           index=False, encoding="utf-8-sig")

            # --- Forest plot of partial r (controlling age) ---
            if len(corr_df) > 0:
                fig, ax = plt.subplots(figsize=(8, max(4, len(corr_df) * 0.4)))
                valid = corr_df[corr_df["partial_r_age"].notna()].copy()
                valid = valid.sort_values("partial_r_age")
                y_pos = np.arange(len(valid))

                colors = ["#E74C3C" if p < 0.05 else "#95A5A6"
                          for p in valid["partial_p_age"]]
                ax.barh(y_pos, valid["partial_r_age"], color=colors, height=0.6, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(valid["feature"])
                ax.axvline(x=0, color="black", linewidth=0.8)
                ax.set_xlabel("Partial r (controlling age)")
                ax.set_title(f"Feature × {cog_var} Partial Correlation — {tool_disp}\n"
                             f"({scope_name}, n={len(df)}, red=p<0.05)")
                ax.grid(True, alpha=0.3, axis="x")
                fig.tight_layout()
                fig.savefig(out_dir / f"{tag}_forest.png",
                            dpi=150, bbox_inches="tight")
                plt.close()

            # --- Scatter plots for top features ---
            if len(corr_df) > 0:
                top = corr_df.reindex(
                    corr_df["partial_r_age"].abs().sort_values(ascending=False).index
                ).head(6)
                n_top = len(top)
                ncols = min(3, n_top)
                nrows = (n_top + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
                if n_top == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                group_colors = {"P": "#E74C3C", "NAD": "#3498DB", "ACS": "#2ECC71"}

                for idx, (_, row) in enumerate(top.iterrows()):
                    ax = axes[idx]
                    feat = f"{row['feature']}_mean"
                    plot_df = df[[feat, cog_var, "group"]].dropna()
                    for grp, color in group_colors.items():
                        g_df = plot_df[plot_df["group"] == grp]
                        if len(g_df) > 0:
                            ax.scatter(g_df[feat], g_df[cog_var], c=color,
                                       alpha=0.3, s=10, label=grp)
                    # Overall regression line
                    x_all = plot_df[feat].values
                    y_all = plot_df[cog_var].values
                    if len(x_all) > 2:
                        slope, intercept, _, _, _ = stats.linregress(x_all, y_all)
                        x_line = np.linspace(x_all.min(), x_all.max(), 100)
                        ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5)
                    ax.set_xlabel(row["feature"])
                    ax.set_ylabel(cog_var)
                    pr = row["partial_r_age"]
                    pp = row["partial_p_age"]
                    ax.set_title(f"partial r={pr:.3f}, p={pp:.4f}", fontsize=9)
                    ax.legend(fontsize=7, markerscale=2)
                for idx in range(n_top, len(axes)):
                    axes[idx].set_visible(False)

                fig.suptitle(f"Top Features × {cog_var} — {tool_disp} ({scope_name})",
                             fontsize=12)
                fig.tight_layout()
                fig.savefig(out_dir / f"{tag}_scatter.png",
                            dpi=150, bbox_inches="tight")
                plt.close()

            # --- LaTeX table ---
            tex_lines = [r"\begin{tabular}{lccccc}",
                         r"\toprule",
                         f"Feature & Pearson $r$ & Spearman $\\rho$ & "
                         f"Partial $r$ (age) & $p$ (partial) \\\\",
                         r"\midrule"]
            for _, row in corr_df.iterrows():
                pr = row["partial_r_age"]
                pp = row["partial_p_age"]
                sig = row["partial_sig_age"]
                tex_lines.append(
                    f"{row['feature']} & {row['pearson_r']:.3f} "
                    f"& {row['spearman_rho']:.3f} "
                    f"& {pr:.3f}{sig} "
                    f"& {pp:.4f} \\\\"
                )
            tex_lines += [r"\bottomrule", r"\end{tabular}"]
            tex_path = out_dir / f"{tag}_summary.tex"
            tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
            compile_tex_to_png(tex_path)

    logger.info(f"  (2) {tool_disp} cognitive correlation done")


# ═══════════════════════════════════════════════════════════
# (3) 情緒組成分析 (Emotion Profile)
# ═══════════════════════════════════════════════════════════

def emotion_profile_analysis(data, tool, out_dir):
    """衍生情緒指標分析"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_disp = TOOL_DISPLAY.get(tool, tool)

    emo_cols = [f"{e}_mean" for e in COMMON_EMO if f"{e}_mean" in data.columns]
    au_cols = [f"{a}_mean" for a in COMMON_AU if f"{a}_mean" in data.columns]
    if not emo_cols:
        return

    df = data.copy()
    eps = 1e-8

    # --- 計算衍生指標 ---
    emo_vals = df[emo_cols].fillna(0).values
    emo_sum = emo_vals.sum(axis=1) + eps

    # Normalize to probabilities
    emo_prob = emo_vals / emo_sum[:, None]

    # Positive / Negative ratio
    pos = df.get("happiness_mean", pd.Series(0, index=df.index)).fillna(0)
    neg = (df.get("sadness_mean", pd.Series(0, index=df.index)).fillna(0) +
           df.get("anger_mean", pd.Series(0, index=df.index)).fillna(0) +
           df.get("fear_mean", pd.Series(0, index=df.index)).fillna(0) +
           df.get("disgust_mean", pd.Series(0, index=df.index)).fillna(0))
    df["pos_neg_ratio"] = pos / (neg + eps)

    # Emotion entropy
    emo_prob_clipped = np.clip(emo_prob, eps, 1.0)
    df["emotion_entropy"] = -np.sum(emo_prob_clipped * np.log2(emo_prob_clipped), axis=1)

    # Neutral dominance
    df["neutral_dominance"] = df.get("neutral_mean", pd.Series(0, index=df.index)).fillna(0) / emo_sum

    # Flat affect index (1 - CV)
    emo_std = emo_vals.std(axis=1)
    emo_mean = emo_vals.mean(axis=1) + eps
    df["flat_affect"] = 1 - (emo_std / emo_mean)

    # Dominant emotion proportion
    df["dominant_emo_prop"] = emo_vals.max(axis=1) / emo_sum

    # Duchenne smile ratio
    au6 = df.get("AU6_mean", pd.Series(eps, index=df.index)).fillna(eps)
    au12 = df.get("AU12_mean", pd.Series(eps, index=df.index)).fillna(eps)
    df["duchenne_ratio"] = np.minimum(au6, au12) / (np.maximum(au6, au12) + eps)

    # Upper/Lower face ratio
    upper = df[[c for c in ["AU1_mean", "AU2_mean", "AU4_mean"] if c in df.columns]].fillna(0).mean(axis=1)
    lower = df[[c for c in ["AU12_mean", "AU25_mean", "AU26_mean"] if c in df.columns]].fillna(0).mean(axis=1)
    df["upper_lower_ratio"] = upper / (lower + eps)

    derived_metrics = [
        ("pos_neg_ratio", "Positive/Negative Ratio"),
        ("emotion_entropy", "Emotion Entropy"),
        ("neutral_dominance", "Neutral Dominance"),
        ("flat_affect", "Flat Affect Index"),
        ("dominant_emo_prop", "Dominant Emotion Proportion"),
        ("duchenne_ratio", "Duchenne Smile Ratio"),
        ("upper_lower_ratio", "Upper/Lower Face Ratio"),
    ]

    # --- 分組標籤 ---
    # 用 group 欄位簡單分：P→AD_pool, NAD, ACS
    group_subsets = {}
    for label in ["P", "NAD", "ACS"]:
        group_subsets[label] = df[df["group"] == label]

    # --- 三組比較 ---
    stat_rows = []
    for metric_col, metric_name in derived_metrics:
        for g1, g2 in [("P", "NAD"), ("P", "ACS"), ("NAD", "ACS")]:
            a = group_subsets[g1][metric_col].dropna()
            b = group_subsets[g2][metric_col].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            d = _cohend(a, b)
            stat_rows.append({
                "metric": metric_name,
                "group1": g1,
                "group2": g2,
                "n1": len(a),
                "n2": len(b),
                "mean1": round(a.mean(), 4),
                "mean2": round(b.mean(), 4),
                "cohens_d": round(d, 4),
                "t": round(t, 4),
                "p": p,
                "sig": _sig(p),
            })
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out_dir / f"{tool}_emotion_profile_stats.csv",
                   index=False, encoding="utf-8-sig")

    # --- Box plots ---
    n_metrics = len(derived_metrics)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    group_order = ["P", "NAD", "ACS"]
    palette = {"P": "#E74C3C", "NAD": "#3498DB", "ACS": "#2ECC71"}

    for idx, (metric_col, metric_name) in enumerate(derived_metrics):
        ax = axes[idx]
        plot_data = df[df["group"].isin(group_order)][[metric_col, "group"]].dropna()
        sns.boxplot(data=plot_data, x="group", y=metric_col, order=group_order,
                    palette=palette, ax=ax, fliersize=2)
        ax.set_title(metric_name, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Emotion Profile Metrics — {tool_disp}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tool}_emotion_profile_boxplots.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- Radar / spider plot of emotion profiles per group ---
    angles = np.linspace(0, 2 * np.pi, len(COMMON_EMO), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for grp, color in palette.items():
        grp_df = df[df["group"] == grp]
        values = [grp_df[f"{e}_mean"].mean() for e in COMMON_EMO]
        values += values[:1]
        ax.plot(angles, values, "o-", label=f"{grp} (n={len(grp_df)})",
                color=color, linewidth=2, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(COMMON_EMO, fontsize=9)
    ax.set_title(f"Emotion Profile — {tool_disp}", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(out_dir / f"{tool}_emotion_radar.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- Partial correlation with MMSE controlling age ---
    if "MMSE" in df.columns:
        pcorr_rows = []
        for metric_col, metric_name in derived_metrics:
            vals = df[[metric_col, "MMSE", "Age"]].dropna()
            if len(vals) < 20:
                continue
            try:
                pc = pg.partial_corr(data=vals, x=metric_col, y="MMSE",
                                     covar="Age", method="pearson")
                pcorr_rows.append({
                    "metric": metric_name,
                    "n": len(vals),
                    "partial_r": round(pc["r"].values[0], 4),
                    "partial_p": pc["p_val"].values[0],
                    "sig": _sig(pc["p_val"].values[0]),
                })
            except Exception:
                pass
        if pcorr_rows:
            pd.DataFrame(pcorr_rows).to_csv(
                out_dir / f"{tool}_emotion_profile_mmse_partial.csv",
                index=False, encoding="utf-8-sig",
            )

    logger.info(f"  (3) {tool_disp} emotion profile analysis done")


# ═══════════════════════════════════════════════════════════
# (4) 縱貫性 Within-Subject 變化
# ═══════════════════════════════════════════════════════════

def longitudinal_analysis(data, tool, out_dir):
    """多次就診的受試者 within-subject 變化分析"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_disp = TOOL_DISPLAY.get(tool, tool)

    feat_cols = [c for c in FEATURE_COLS if c in data.columns]

    # 提取 base_id 和 session
    df = data.copy()
    df["base_id"] = df["subject_id"].str.replace(r"-\d+$", "", regex=True)

    # 只取有多次 session 的 P 組
    df_p = df[df["group"] == "P"].copy()
    session_counts = df_p.groupby("base_id").size()
    multi_ids = session_counts[session_counts >= 2].index
    df_multi = df_p[df_p["base_id"].isin(multi_ids)].copy()

    if len(df_multi) == 0:
        logger.warning(f"  (4) {tool}: 無多次就診資料")
        return

    logger.info(f"  (4) {tool_disp} longitudinal: {len(multi_ids)} subjects, "
                f"{len(df_multi)} rows")

    # --- A: Delta analysis (last - first session) ---
    delta_rows = []
    for bid in multi_ids:
        subj_data = df_multi[df_multi["base_id"] == bid].copy()

        # 用 subject_id 的 session 號排序
        subj_data["session_num"] = subj_data["subject_id"].str.extract(r"-(\d+)$").astype(int)
        subj_data = subj_data.sort_values("session_num")

        first = subj_data.iloc[0]
        last = subj_data.iloc[-1]

        record = {"base_id": bid, "n_sessions": len(subj_data)}

        # CDR change
        if pd.notna(first.get("Global_CDR")) and pd.notna(last.get("Global_CDR")):
            record["delta_CDR"] = last["Global_CDR"] - first["Global_CDR"]
        else:
            record["delta_CDR"] = np.nan

        # MMSE change
        if "MMSE" in df.columns and pd.notna(first.get("MMSE")) and pd.notna(last.get("MMSE")):
            record["delta_MMSE"] = last["MMSE"] - first["MMSE"]
        else:
            record["delta_MMSE"] = np.nan

        # Feature deltas
        for feat in feat_cols:
            if pd.notna(first[feat]) and pd.notna(last[feat]):
                record[f"delta_{feat}"] = last[feat] - first[feat]
            else:
                record[f"delta_{feat}"] = np.nan

        delta_rows.append(record)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / f"{tool}_longitudinal_deltas.csv",
                    index=False, encoding="utf-8-sig")

    # --- B: Correlate delta(feature) with delta(CDR) and delta(MMSE) ---
    delta_corr_rows = []
    for target_var in ["delta_CDR", "delta_MMSE"]:
        for feat in feat_cols:
            delta_feat = f"delta_{feat}"
            vals = delta_df[[delta_feat, target_var]].dropna()
            if len(vals) < 10:
                continue
            r, p = stats.spearmanr(vals[delta_feat], vals[target_var])
            delta_corr_rows.append({
                "feature": feat.replace("_mean", ""),
                "target": target_var,
                "n": len(vals),
                "spearman_rho": round(r, 4),
                "p": p,
                "sig": _sig(p),
            })
    if delta_corr_rows:
        delta_corr_df = pd.DataFrame(delta_corr_rows)
        delta_corr_df.to_csv(out_dir / f"{tool}_longitudinal_delta_corr.csv",
                             index=False, encoding="utf-8-sig")

    # --- C: Progressors vs Stable ---
    has_cdr = delta_df[delta_df["delta_CDR"].notna()]
    progressors = has_cdr[has_cdr["delta_CDR"] > 0]
    stable = has_cdr[has_cdr["delta_CDR"] == 0]

    if len(progressors) >= 5 and len(stable) >= 5:
        comp_rows = []
        for feat in feat_cols:
            delta_feat = f"delta_{feat}"
            a = progressors[delta_feat].dropna()
            b = stable[delta_feat].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            d = _cohend(a, b)
            comp_rows.append({
                "feature": feat.replace("_mean", ""),
                "progressors_n": len(a),
                "stable_n": len(b),
                "progressors_mean_delta": round(a.mean(), 6),
                "stable_mean_delta": round(b.mean(), 6),
                "cohens_d": round(d, 4),
                "t": round(t, 4),
                "p": p,
                "sig": _sig(p),
            })
        if comp_rows:
            pd.DataFrame(comp_rows).to_csv(
                out_dir / f"{tool}_longitudinal_prog_vs_stable.csv",
                index=False, encoding="utf-8-sig",
            )

    # --- D: Visualization: average trajectory ---
    # Split into progressors / stable / regressors
    categories = {}
    if len(progressors) > 0:
        categories["Progressor"] = progressors["base_id"].values
    if len(stable) > 0:
        categories["Stable"] = stable["base_id"].values
    regressors = has_cdr[has_cdr["delta_CDR"] < 0]
    if len(regressors) > 0:
        categories["Regressor"] = regressors["base_id"].values

    cat_colors = {"Progressor": "#E74C3C", "Stable": "#3498DB", "Regressor": "#2ECC71"}

    for feat_type, feats_subset, title_suffix in [
        ("au", [f for f in feat_cols if f.startswith("AU")], "AU"),
        ("emotion", [f for f in feat_cols if not f.startswith("AU")], "Emotion"),
    ]:
        if not feats_subset or not categories:
            continue

        n_feat = len(feats_subset)
        ncols = min(4, n_feat)
        nrows = (n_feat + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n_feat == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes).flatten()

        for fi, feat in enumerate(feats_subset):
            ax = axes[fi]
            for cat_name, cat_ids in categories.items():
                # Get all sessions for these subjects
                cat_data = df_multi[df_multi["base_id"].isin(cat_ids)].copy()
                cat_data["session_num"] = cat_data["subject_id"].str.extract(r"-(\d+)$").astype(int)

                # Average trajectory
                session_means = cat_data.groupby("session_num")[feat].agg(["mean", "sem", "count"])
                session_means = session_means[session_means["count"] >= 3]
                if len(session_means) < 2:
                    continue
                x = session_means.index.values
                y = session_means["mean"].values
                se = session_means["sem"].values
                ax.plot(x, y, "o-", color=cat_colors[cat_name],
                        label=f"{cat_name} (n={len(cat_ids)})", linewidth=1.5, markersize=4)
                ax.fill_between(x, y - 1.96 * se, y + 1.96 * se,
                                color=cat_colors[cat_name], alpha=0.15)

            ax.set_xlabel("Session")
            ax.set_ylabel(feat.replace("_mean", ""))
            ax.set_title(feat.replace("_mean", ""), fontsize=10)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

        for fi in range(n_feat, len(axes)):
            axes[fi].set_visible(False)

        fig.suptitle(f"Longitudinal Trajectory — {tool_disp} {title_suffix}\n"
                     f"(Progressors vs Stable vs Regressors)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{tool}_longitudinal_trajectory_{feat_type}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"  (4) {tool_disp} longitudinal analysis done")


# ═══════════════════════════════════════════════════════════
# (5) 性別分層分析
# ═══════════════════════════════════════════════════════════

def sex_stratified_analysis(data, tool, out_dir):
    """分性別的組間比較"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_disp = TOOL_DISPLAY.get(tool, tool)

    feat_cols = [c for c in FEATURE_COLS if c in data.columns]
    if "Sex" not in data.columns:
        logger.warning(f"  (5) {tool}: 無 Sex 欄位")
        return

    df = data[data["Sex"].isin(["M", "F"])].copy()

    logger.info(f"  (5) {tool_disp} sex stratified: "
                f"M={len(df[df['Sex']=='M'])}, F={len(df[df['Sex']=='F'])}")

    # --- A: 分性別 group stats (P vs NAD) ---
    stat_rows = []
    for sex in ["M", "F"]:
        sex_df = df[df["Sex"] == sex]
        p_df = sex_df[sex_df["group"] == "P"]
        nad_df = sex_df[sex_df["group"] == "NAD"]
        acs_df = sex_df[sex_df["group"] == "ACS"]

        for g1_name, g1_df, g2_name, g2_df in [
            ("P", p_df, "NAD", nad_df),
            ("P", p_df, "ACS", acs_df),
        ]:
            for feat in feat_cols:
                a = g1_df[feat].dropna()
                b = g2_df[feat].dropna()
                if len(a) < 5 or len(b) < 5:
                    continue
                t, p_val = stats.ttest_ind(a, b, equal_var=False)
                d = _cohend(a, b)
                stat_rows.append({
                    "sex": sex,
                    "feature": feat.replace("_mean", ""),
                    "group1": g1_name,
                    "group2": g2_name,
                    "n1": len(a),
                    "n2": len(b),
                    "mean1": round(a.mean(), 4),
                    "mean2": round(b.mean(), 4),
                    "cohens_d": round(d, 4),
                    "t": round(t, 4),
                    "p": p_val,
                    "sig": _sig(p_val),
                })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out_dir / f"{tool}_sex_stratified_stats.csv",
                   index=False, encoding="utf-8-sig")

    # --- B: Two-way ANOVA interaction test ---
    anova_rows = []
    df_pn = df[df["group"].isin(["P", "NAD"])].copy()
    for feat in feat_cols:
        try:
            aov = pg.anova(data=df_pn, dv=feat, between=["group", "Sex"])
            interaction = aov[aov["Source"] == "group * Sex"]
            if len(interaction) > 0:
                f_val = interaction["F"].values[0]
                p_val = interaction["p-unc"].values[0]
                anova_rows.append({
                    "feature": feat.replace("_mean", ""),
                    "interaction_F": round(f_val, 4),
                    "interaction_p": p_val,
                    "interaction_sig": _sig(p_val),
                })
        except Exception:
            pass

    if anova_rows:
        anova_df = pd.DataFrame(anova_rows)
        anova_df.to_csv(out_dir / f"{tool}_sex_interaction_anova.csv",
                        index=False, encoding="utf-8-sig")

    # --- C: Forest plot: Cohen's d by sex ---
    if len(stat_df) > 0:
        p_nad = stat_df[(stat_df["group1"] == "P") & (stat_df["group2"] == "NAD")]
        fig, ax = plt.subplots(figsize=(10, max(4, len(feat_cols) * 0.5)))
        feats_unique = p_nad["feature"].unique()
        y_pos = np.arange(len(feats_unique))
        bar_height = 0.35

        for si, (sex, color, offset) in enumerate([("M", "#3498DB", -bar_height/2),
                                                     ("F", "#E74C3C", bar_height/2)]):
            sex_data = p_nad[p_nad["sex"] == sex].set_index("feature")
            ds = [sex_data.loc[f, "cohens_d"] if f in sex_data.index else 0 for f in feats_unique]
            ax.barh(y_pos + offset, ds, height=bar_height, color=color,
                    alpha=0.7, label=f"{sex}")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats_unique)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Cohen's d (P vs NAD)")
        ax.set_title(f"Effect Size by Sex — {tool_disp}", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(out_dir / f"{tool}_sex_effect_size_forest.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # --- D: LaTeX table ---
    if len(anova_rows) > 0:
        tex_lines = [r"\begin{tabular}{lcccc}",
                     r"\toprule",
                     r"Feature & $d$ (Male) & $d$ (Female) & Interaction $F$ & $p$ \\",
                     r"\midrule"]
        for _, arow in anova_df.iterrows():
            feat = arow["feature"]
            male_d = stat_df[(stat_df["feature"] == feat) & (stat_df["sex"] == "M") &
                             (stat_df["group2"] == "NAD")]["cohens_d"]
            female_d = stat_df[(stat_df["feature"] == feat) & (stat_df["sex"] == "F") &
                               (stat_df["group2"] == "NAD")]["cohens_d"]
            md = male_d.values[0] if len(male_d) > 0 else np.nan
            fd = female_d.values[0] if len(female_d) > 0 else np.nan
            tex_lines.append(
                f"{feat} & {md:.3f} & {fd:.3f} & "
                f"{arow['interaction_F']:.3f} & "
                f"{arow['interaction_p']:.4f}{arow['interaction_sig']} \\\\"
            )
        tex_lines += [r"\bottomrule", r"\end{tabular}"]
        tex_path = out_dir / f"{tool}_sex_interaction.tex"
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
        compile_tex_to_png(tex_path)

    logger.info(f"  (5) {tool_disp} sex stratified analysis done")


# ═══════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════

def main():
    demo = load_demographics()
    logger.info(f"Demographics loaded: {len(demo)} rows")

    for tool in TOOLS:
        data = load_full_data(tool, demo)
        if data is None:
            logger.warning(f"跳過 {tool}: 無 aggregated 資料")
            continue

        tool_disp = TOOL_DISPLAY.get(tool, tool)
        logger.info(f"\n{'='*60}")
        logger.info(f"Extended Analysis — {tool_disp} (n={len(data)})")
        logger.info(f"{'='*60}")

        # (1) CDR 嚴重度梯度
        cdr_gradient_analysis(data, tool, OUTPUT_BASE / "cdr_gradient")

        # (2) MMSE/CASI 認知分數相關
        cognitive_correlation_analysis(data, tool, OUTPUT_BASE / "cognitive_correlation")

        # (3) 情緒組成分析
        emotion_profile_analysis(data, tool, OUTPUT_BASE / "emotion_profiles")

        # (4) 縱貫性分析
        longitudinal_analysis(data, tool, OUTPUT_BASE / "longitudinal")

        # (5) 性別分層分析
        sex_stratified_analysis(data, tool, OUTPUT_BASE / "sex_stratified")

    logger.info("\n全部完成！")


if __name__ == "__main__":
    main()
