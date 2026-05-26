"""Project-wide overview pipeline diagram — all 5 analysis modules.

Shows: Raw Images → Preprocessing → 5 parallel streams → Meta-Analysis.

Output:
  workspace/overview/overview_pipeline_mpl.png

Usage:
    python scripts/overview/draw_overview_pipeline_mpl.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(r"c:\Users\4080\Desktop\Alz_face_analyze\workspace\overview")

FONT = 'Microsoft JhengHei'
FS = 12
FS_LG = 14
FS_SM = 11
FS_XS = 10
EC = '#404040'
NODE_H = 1.3
SP = 0.4

C_RAW = dict(nd='#D8D8D8')
C_PRE = dict(bg='#E8EFF5', nd='#B3CCE4')
C_EMB = dict(bg='#FDF3E5', nd='#F5D5A0')
C_AGE = dict(bg='#ECF0E4', nd='#C5D6A8')
C_ASY = dict(bg='#F0E6F0', nd='#D4B0D4')
C_EMO = dict(bg='#FCE8E8', nd='#E8B4B4')
C_BMI = dict(bg='#E8E8FC', nd='#B4B4E8')
C_META = dict(bg='#FBF5E0', nd='#F0D870')

FIG_W = 17.0
COL_W = 2.8
COL_GAP = 0.35
N_COLS = 5
TOTAL_W = N_COLS * COL_W + (N_COLS - 1) * COL_GAP
MARGIN = (FIG_W - TOTAL_W) / 2
CX = FIG_W / 2
COL_X = [MARGIN + COL_W / 2 + i * (COL_W + COL_GAP) for i in range(N_COLS)]


def _box(ax, cx, cy, w, h, fc, lw=1.4, zorder=1):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="square,pad=0", facecolor=fc, edgecolor=EC,
        linewidth=lw, zorder=zorder))


def node(ax, cx, cy, w, h, text, fc, fs=FS, fc_text='#212121'):
    _box(ax, cx, cy, w, h, fc, lw=1.4, zorder=3)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontfamily=FONT, color=fc_text,
            linespacing=1.35, zorder=4)


def cluster(ax, cx, cy, w, h, fc):
    _box(ax, cx, cy, w, h, fc, lw=2.0, zorder=0)


def line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=EC, lw=1.4,
            solid_capstyle='round', zorder=2)


def build():
    # --- Vertical positions ---
    y_raw = 0.5 + SP + NODE_H / 2

    pre_top = y_raw + NODE_H / 2 + SP
    y_pre = pre_top + SP + NODE_H / 2
    pre_bot = y_pre + NODE_H / 2 + SP

    col_top = pre_bot + SP
    y_label = col_top + SP * 0.5
    y_m1 = y_label + 0.55 + NODE_H / 2
    y_m2 = y_m1 + NODE_H + SP
    col_bot = y_m2 + NODE_H / 2 + SP

    y_out = col_bot + SP * 0.6 + NODE_H * 0.35

    meta_top = y_out + NODE_H * 0.35 + SP
    y_meta = meta_top + SP + NODE_H / 2
    meta_bot = y_meta + NODE_H / 2 + SP

    fig_h = meta_bot + 0.5

    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(-0.1, FIG_W + 0.1)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # === Raw Images ===
    node(ax, CX, y_raw, 5.5, NODE_H,
         "Raw Face Images\n(10 photos / subject)", C_RAW['nd'])

    # === Preprocessing ===
    pre_w = 9.0
    cluster(ax, CX, (pre_top + pre_bot) / 2,
            pre_w + 0.6, pre_bot - pre_top, C_PRE['bg'])
    node(ax, CX, y_pre, pre_w, NODE_H,
         "Preprocessing:  Detect  →  Select  →  Align  →  Mirror",
         C_PRE['nd'], fs=FS_LG)
    line(ax, CX, y_raw + NODE_H / 2, CX, y_pre - NODE_H / 2)

    # === 5 Module Columns ===
    colors = [C_EMB, C_AGE, C_ASY, C_EMO, C_BMI]
    labels = [
        "Embedding (M1)", "Age (M3)", "Asymmetry (M2)",
        "Emotion (M4)", "BMI",
    ]
    main_texts = [
        "4 Models\ndlib / TopoFR\nArcFace / VGGFace",
        "MiVOLO v2\nAge Estimation",
        "MediaPipe\n468 Landmarks",
        "10 Extractors\nDAN / FER / HSEmo\nViT / OpenFace / ...",
        "ArcFace\nEmbeddings",
    ]
    detail_texts = [
        "6 Features × 2 BG\nLR / XGB\nFwd / Rev",
        "Bootstrap\nCalibration",
        "L/R Pair Diff\n4 Regions\n66 Pairs",
        "Harmonize\n& Aggregate\n7 Emo + V/A",
        "Ridge / SVR\nRegression",
    ]
    out_texts = [
        "→ lr_score",
        "→ age_error",
        "→ lr_score_asym",
        "→ 10 emo feats",
        "(independent)",
    ]

    for i in range(N_COLS):
        cx = COL_X[i]
        c = colors[i]

        cluster(ax, cx, (col_top + col_bot) / 2,
                COL_W + 0.4, col_bot - col_top, c['bg'])

        ax.text(cx, y_label, labels[i], ha='center', va='top',
                fontsize=FS, fontfamily=FONT, fontweight='bold',
                color='#333333', zorder=4)

        node(ax, cx, y_m1, COL_W, NODE_H, main_texts[i], c['nd'], fs=FS_SM)
        node(ax, cx, y_m2, COL_W, NODE_H, detail_texts[i], c['nd'], fs=FS_SM)
        line(ax, cx, y_m1 + NODE_H / 2, cx, y_m2 - NODE_H / 2)

        line(ax, CX, y_pre + NODE_H / 2, cx, y_m1 - NODE_H / 2)

        fc_out = '#888888' if i == 4 else '#404040'
        ax.text(cx, y_out, out_texts[i], ha='center', va='center',
                fontsize=FS_XS, fontfamily=FONT, color=fc_out,
                style='italic', zorder=4)

    # === Meta-Analysis ===
    meta_w = TOTAL_W * 0.62
    cluster(ax, CX, (meta_top + meta_bot) / 2,
            meta_w + 0.6, meta_bot - meta_top, C_META['bg'])
    node(ax, CX, y_meta, meta_w, NODE_H,
         "Meta-Analysis  (TabPFN)\n"
         "M1 (Emb) + M2 (Asym) + M3 (Age) + M4 (Emo)   ·   "
         "9 Combinations",
         C_META['nd'], fs=FS)

    for i in range(4):
        line(ax, COL_X[i], y_out + NODE_H * 0.35,
             COL_X[i], y_meta - NODE_H / 2)

    out = OUT / "overview_pipeline_mpl.png"
    fig.savefig(out, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
