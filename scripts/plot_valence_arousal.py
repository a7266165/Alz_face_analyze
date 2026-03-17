"""
Valence-Arousal scatter plot: ACS & NAD (circle) vs AD (x marker).
"""
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.config import DEMOGRAPHICS_DIR, WORKSPACE_DIR

# --- Load data ---
# Demographics
dfs = []
for csv_file in ["ACS.csv", "NAD.csv", "P.csv"]:
    df = pd.read_csv(DEMOGRAPHICS_DIR / csv_file, encoding="utf-8-sig")
    df["group"] = csv_file.replace(".csv", "")
    dfs.append(df[["ID", "Age", "group"]])
demo = pd.concat(dfs, ignore_index=True)

# Emotion scores
emo = pd.read_csv(WORKSPACE_DIR / "emotion_score_EmoNet.csv", encoding="utf-8-sig")

# Predicted ages (to filter to valid subjects)
with open(WORKSPACE_DIR / "predicted_ages.json", "r", encoding="utf-8") as f:
    pred_ages = json.load(f)

# Merge
emo = emo[emo["subject_id"].isin(pred_ages.keys())]
emo = emo.merge(demo, left_on="subject_id", right_on="ID", how="inner")
emo = emo.dropna(subset=["Valence", "Arousal"]).reset_index(drop=True)

# Split groups
acs = emo[emo["group"] == "ACS"]
nad = emo[emo["group"] == "NAD"]
ad = emo[emo["group"] == "P"]

print(f"ACS: {len(acs)}, NAD: {len(nad)}, AD(P): {len(ad)}")

# --- Statistics: 3-group mean, std, tests ---
from scipy import stats

print("\n===== Valence & Arousal: Group Statistics =====")
for feat in ["Valence", "Arousal"]:
    print(f"\n--- {feat} ---")
    for name, grp in [("ACS", acs), ("NAD", nad), ("AD", ad)]:
        print(f"  {name:>3s}: mean={grp[feat].mean():.4f}, std={grp[feat].std():.4f}, n={len(grp)}")

    # Kruskal-Wallis (3-group)
    H, p_kw = stats.kruskal(acs[feat], nad[feat], ad[feat])
    print(f"  Kruskal-Wallis H={H:.4f}, p={p_kw:.2e}")

    # Pairwise Mann-Whitney U
    for (n1, g1), (n2, g2) in [
        (("ACS", acs), ("NAD", nad)),
        (("ACS", acs), ("AD", ad)),
        (("NAD", nad), ("AD", ad)),
    ]:
        U, p_mw = stats.mannwhitneyu(g1[feat], g2[feat], alternative="two-sided")
        # Cohen's d
        nx, ny = len(g1), len(g2)
        pooled = np.sqrt(((nx-1)*g1[feat].std()**2 + (ny-1)*g2[feat].std()**2) / (nx+ny-2))
        d = (g1[feat].mean() - g2[feat].mean()) / pooled if pooled > 0 else 0
        print(f"  {n1} vs {n2}: U={U:.0f}, p={p_mw:.2e}, d={d:+.4f}")

# --- Export statistics to CSV ---
STATS_DIR = WORKSPACE_DIR / "statistics" / "m3m4_deep"
STATS_DIR.mkdir(parents=True, exist_ok=True)

# 1) Group descriptive statistics
desc_rows = []
for feat in ["Valence", "Arousal"]:
    for name, grp in [("ACS", acs), ("NAD", nad), ("AD", ad)]:
        desc_rows.append({
            "feature": feat, "group": name, "n": len(grp),
            "mean": grp[feat].mean(), "std": grp[feat].std(),
            "median": grp[feat].median(),
            "q25": grp[feat].quantile(0.25), "q75": grp[feat].quantile(0.75),
        })
desc_df = pd.DataFrame(desc_rows)
desc_path = STATS_DIR / "valence_arousal_descriptive.csv"
desc_df.to_csv(desc_path, index=False)
print(f"\nSaved: {desc_path}")

# 2) Pairwise comparisons
pair_rows = []
for feat in ["Valence", "Arousal"]:
    # Kruskal-Wallis
    H, p_kw = stats.kruskal(acs[feat], nad[feat], ad[feat])
    pair_rows.append({
        "feature": feat, "test": "Kruskal-Wallis",
        "group1": "ACS+NAD+AD", "group2": "",
        "n1": len(acs)+len(nad)+len(ad), "n2": "",
        "statistic": H, "p_value": p_kw, "cohens_d": "",
    })
    # Pairwise Mann-Whitney
    for (n1, g1), (n2, g2) in [
        (("ACS", acs), ("NAD", nad)),
        (("ACS", acs), ("AD", ad)),
        (("NAD", nad), ("AD", ad)),
    ]:
        U, p_mw = stats.mannwhitneyu(g1[feat], g2[feat], alternative="two-sided")
        nx, ny = len(g1), len(g2)
        pooled = np.sqrt(((nx-1)*g1[feat].std()**2 + (ny-1)*g2[feat].std()**2) / (nx+ny-2))
        d = (g1[feat].mean() - g2[feat].mean()) / pooled if pooled > 0 else 0
        pair_rows.append({
            "feature": feat, "test": "Mann-Whitney U",
            "group1": n1, "group2": n2,
            "n1": nx, "n2": ny,
            "statistic": U, "p_value": p_mw, "cohens_d": d,
        })
pair_df = pd.DataFrame(pair_rows)
pair_path = STATS_DIR / "valence_arousal_tests.csv"
pair_df.to_csv(pair_path, index=False)
print(f"Saved: {pair_path}")

# --- Plot V1: all groups (original) ---
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
try:
    import matplotlib.font_manager as fm
    if any("Arial" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Arial"
except Exception:
    pass

fig, ax = plt.subplots(figsize=(8, 6))

# AD first (background) - X markers
ax.scatter(ad["Valence"], ad["Arousal"],
           marker="x", s=18, alpha=0.3, color="#F44336",
           label=f"AD (n={len(ad)})", linewidths=0.7, zorder=2)

# ACS - circles
ax.scatter(acs["Valence"], acs["Arousal"],
           marker="o", s=20, alpha=0.5, color="#2196F3", edgecolors="none",
           label=f"ACS (n={len(acs)})", zorder=3)

# NAD - circles
ax.scatter(nad["Valence"], nad["Arousal"],
           marker="o", s=20, alpha=0.5, color="#4CAF50", edgecolors="none",
           label=f"NAD (n={len(nad)})", zorder=3)

ax.set_xlabel("Valence")
ax.set_ylabel("Arousal")
ax.set_title("Valence–Arousal Distribution by Diagnostic Group")
ax.legend(loc="upper right", framealpha=0.9)

# Move axes to cross at (0, 0)
ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Place axis labels at the positive ends of each axis
ax.set_xlabel("Valence", loc="right")
ax.set_ylabel("Arousal", loc="top")
ax.xaxis.set_label_coords(1.0, -0.02)
ax.yaxis.set_label_coords(-0.02, 1.0)

ax.grid(True, alpha=0.3)

out_path = project_root / "paper" / "figures" / "fig_valence_arousal_scatter.png"
fig.savefig(out_path)
plt.close(fig)
print(f"\nSaved V1: {out_path}")

# --- Plot V2: ACS (n=223) vs AD random 223 ---
np.random.seed(42)
ad_sample = ad.sample(n=len(acs), random_state=42)
print(f"\nV2: ACS={len(acs)}, AD_sample={len(ad_sample)}")

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(ad_sample["Valence"], ad_sample["Arousal"],
            marker="x", s=25, alpha=0.5, color="#F44336",
            label=f"AD (n={len(ad_sample)})", linewidths=0.8, zorder=2)

ax2.scatter(acs["Valence"], acs["Arousal"],
            marker="o", s=25, alpha=0.6, color="#2196F3", edgecolors="none",
            label=f"ACS (n={len(acs)})", zorder=3)

ax2.set_title("Valence–Arousal: ACS vs AD (matched n=223)")
ax2.legend(loc="upper right", framealpha=0.9)

ax2.spines["left"].set_position("zero")
ax2.spines["bottom"].set_position("zero")
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax2.set_xlabel("Valence", loc="right")
ax2.set_ylabel("Arousal", loc="top")
ax2.xaxis.set_label_coords(1.0, -0.02)
ax2.yaxis.set_label_coords(-0.02, 1.0)
ax2.grid(True, alpha=0.3)

out_path2 = project_root / "paper" / "figures" / "fig_valence_arousal_scatter_v2.png"
fig2.savefig(out_path2)
plt.close(fig2)
print(f"Saved V2: {out_path2}")

# --- Plot V3: NAD random 200 vs AD random 200 ---
nad_sample = nad.sample(n=200, random_state=42)
ad_sample3 = ad.sample(n=200, random_state=42)
print(f"\nV3: NAD_sample={len(nad_sample)}, AD_sample={len(ad_sample3)}")

fig3, ax3 = plt.subplots(figsize=(8, 6))

ax3.scatter(ad_sample3["Valence"], ad_sample3["Arousal"],
            marker="x", s=25, alpha=0.5, color="#F44336",
            label=f"AD (n={len(ad_sample3)})", linewidths=0.8, zorder=2)

ax3.scatter(nad_sample["Valence"], nad_sample["Arousal"],
            marker="o", s=25, alpha=0.6, color="#4CAF50", edgecolors="none",
            label=f"NAD (n={len(nad_sample)})", zorder=3)

ax3.set_title("Valence–Arousal: NAD vs AD (matched n=200)")
ax3.legend(loc="upper right", framealpha=0.9)

ax3.spines["left"].set_position("zero")
ax3.spines["bottom"].set_position("zero")
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)

ax3.set_xlabel("Valence", loc="right")
ax3.set_ylabel("Arousal", loc="top")
ax3.xaxis.set_label_coords(1.0, -0.02)
ax3.yaxis.set_label_coords(-0.02, 1.0)
ax3.grid(True, alpha=0.3)

out_path3 = project_root / "paper" / "figures" / "fig_valence_arousal_scatter_v3.png"
fig3.savefig(out_path3)
plt.close(fig3)
print(f"Saved V3: {out_path3}")

# --- Plot V4: Age-stratified 2x3 subplots (all subjects per bin) ---
age_bins = [(60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 999)]
age_labels = ["60–64", "65–69", "70–74", "75–79", "80–84", "≥85"]

fig4, axes4 = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
axes4 = axes4.flatten()

# Shared axis limits across all subplots
vmin, vmax = emo["Valence"].quantile(0.01), emo["Valence"].quantile(0.99)
amin, amax = emo["Arousal"].quantile(0.01), emo["Arousal"].quantile(0.99)
pad = 0.05
xlim = (min(vmin, -0.1) - pad, max(vmax, 0.1) + pad)
ylim = (min(amin, -0.1) - pad, max(amax, 0.1) + pad)

for i, ((lo, hi), label) in enumerate(zip(age_bins, age_labels)):
    ax4 = axes4[i]
    mask = (emo["Age"] >= lo) & (emo["Age"] <= hi)
    sub = emo[mask]

    for grp_name, marker, color, alpha, s, zorder in [
        ("P",   "x", "#F44336", 0.4, 20, 2),
        ("NAD", "o", "#4CAF50", 0.6, 20, 3),
        ("ACS", "o", "#2196F3", 0.6, 20, 3),
    ]:
        g = sub[sub["group"] == grp_name]
        if len(g) == 0:
            continue
        kwargs = {"marker": marker, "s": s, "alpha": alpha, "color": color, "zorder": zorder}
        if marker == "o":
            kwargs["edgecolors"] = "none"
        else:
            kwargs["linewidths"] = 0.7
        ax4.scatter(g["Valence"], g["Arousal"], label=f"{grp_name} (n={len(g)})", **kwargs)

    # Axes cross at (0, 0)
    ax4.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax4.axvline(0, color="black", linewidth=0.5, zorder=1)
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_title(f"Age {label}", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=7, framealpha=0.8)
    ax4.grid(True, alpha=0.2)

    # Annotate group mean +/- std as text in bottom-left
    txt_lines = []
    for grp_name, grp_label, color in [("ACS", "ACS", "#2196F3"), ("NAD", "NAD", "#4CAF50"), ("P", "AD", "#F44336")]:
        g = sub[sub["group"] == grp_name]
        if len(g) == 0:
            continue
        vm, vs = g["Valence"].mean(), g["Valence"].std()
        am, a_s = g["Arousal"].mean(), g["Arousal"].std()
        txt_lines.append((grp_label, color, f"V={vm:+.3f}\u00b1{vs:.3f}  A={am:+.3f}\u00b1{a_s:.3f}"))
    for j, (glabel, gc, gtxt) in enumerate(txt_lines):
        ax4.text(0.02, 0.02 + j * 0.07, f"{glabel}: {gtxt}",
                 transform=ax4.transAxes, fontsize=6, color=gc,
                 fontweight="bold", va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, edgecolor="none"))

fig4.supxlabel("Valence", fontsize=14)
fig4.supylabel("Arousal", fontsize=14)
fig4.suptitle("Valence–Arousal by Age Stratum", fontsize=15, fontweight="bold")
fig4.tight_layout(rect=[0.02, 0.02, 1, 0.96])

out_path4 = project_root / "paper" / "figures" / "fig_valence_arousal_scatter_v4.png"
fig4.savefig(out_path4)
plt.close(fig4)
print(f"\nSaved V4: {out_path4}")

# --- Age-stratified descriptive statistics CSV ---
strat_rows = []
group_map = {"ACS": "ACS", "NAD": "NAD", "P": "AD"}
for (lo, hi), label in zip(age_bins, age_labels):
    mask = (emo["Age"] >= lo) & (emo["Age"] <= hi)
    sub = emo[mask]
    for grp_code, grp_label in group_map.items():
        g = sub[sub["group"] == grp_code]
        if len(g) == 0:
            continue
        for feat in ["Valence", "Arousal"]:
            strat_rows.append({
                "age_bin": label, "group": grp_label, "feature": feat,
                "n": len(g),
                "mean": g[feat].mean(), "std": g[feat].std(),
                "median": g[feat].median(),
                "q25": g[feat].quantile(0.25), "q75": g[feat].quantile(0.75),
            })
strat_df = pd.DataFrame(strat_rows)
strat_path = STATS_DIR / "valence_arousal_by_age_group.csv"
strat_df.to_csv(strat_path, index=False)
print(f"Saved: {strat_path}")

# Print summary table
print("\n===== Valence & Arousal by Age Stratum × Group =====")
for feat in ["Valence", "Arousal"]:
    print(f"\n--- {feat} ---")
    sf = strat_df[strat_df["feature"] == feat]
    for _, row in sf.iterrows():
        label_safe = row['age_bin'].replace('\u2265', '>=')
        print(f"  {label_safe:>5s} | {row['group']:>3s} | n={row['n']:>4d} | "
              f"mean={row['mean']:+.4f} | std={row['std']:.4f}")
