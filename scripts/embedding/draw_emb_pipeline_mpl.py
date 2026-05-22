"""Embedding classification pipeline diagram — matplotlib version.

Two layout variants for the Training→Eval section:
  v1: one merged cluster, Forward/Reverse as left/right columns
  v2: two separate clusters side-by-side (Forward | Reverse)

Output:
  workspace/embedding/analysis/embedding_classification_pipeline_mpl_v1.png
  workspace/embedding/analysis/embedding_classification_pipeline_mpl_v2.png

Usage:
    python scripts/embedding/draw_emb_pipeline_mpl.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(r"c:\Users\4080\Desktop\Alz_face_analyze\workspace\embedding\analysis")

CX = 6.25
FIG_W = 12.5
NODE_H = 1.15
SP = 0.4

FONT = 'Microsoft JhengHei'
FS = 14
FS_HI = 15
EC = '#404040'

C1 = dict(bg='#E8EFF5', nd='#B3CCE4', hi='#8BB4D9')
C2 = dict(bg='#FDF3E5', nd='#F5D5A0')
C3 = dict(bg='#F0E6F0', nd='#D4B0D4')
C4 = dict(bg='#ECF0E4', nd='#C5D6A8')
C5 = dict(bg='#E0EEF0', nd='#8DC3CB', hi='#5B9BD5')
C6 = dict(bg='#FBF5E0', nd='#F0D870')
CF = dict(bg='#E4F0ED', nd='#A8D4C4')
CR = dict(bg='#F5E8E4', nd='#E8BDB0')


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


def draw_top(ax, SP):
    """Draw Cohort → Models → Features. Returns (fx, y_f) for connecting below."""
    c1_top = 0.5
    y_vs = c1_top + SP + NODE_H / 2
    y_cf = y_vs + NODE_H + SP
    y_fc = y_cf + NODE_H + SP
    c1_bot = y_fc + NODE_H / 2 + SP
    nw1 = 4.2

    c2_top = c1_bot + SP
    y_m = c2_top + SP + NODE_H / 2
    c2_bot = y_m + NODE_H / 2 + SP
    nw2 = 2.2; gap2 = 0.5
    total2 = 3 * nw2 + 2 * gap2

    c3_top = c2_bot + SP
    y_f = c3_top + SP + NODE_H / 2
    c3_bot = y_f + NODE_H / 2 + SP
    nw3 = 2.4; gap3 = 0.65
    total3 = 2 * nw3 + gap3

    cluster(ax, CX, (c1_top + c1_bot) / 2, nw1 + 1.0, c1_bot - c1_top, C1['bg'])
    node(ax, CX, y_vs, nw1, NODE_H, "Visit Selection", C1['nd'])
    node(ax, CX, y_cf, nw1, NODE_H, "CDR / MMSE Filter", C1['nd'])
    node(ax, CX, y_fc, nw1, NODE_H, "Cohort (P + NAD + ACS)",
         C1['hi'], fs=FS_HI, fc_text='#1A3A5C')
    line(ax, CX, y_vs + NODE_H / 2, CX, y_cf - NODE_H / 2)
    line(ax, CX, y_cf + NODE_H / 2, CX, y_fc - NODE_H / 2)

    cluster(ax, CX, (c2_top + c2_bot) / 2, total2 + 0.9, c2_bot - c2_top, C2['bg'])
    mx = [CX - (nw2 + gap2), CX, CX + (nw2 + gap2)]
    for x, lab in zip(mx, ['dlib', 'TopoFR', 'ArcFace']):
        node(ax, x, y_m, nw2, NODE_H, lab, C2['nd'])
    for x in mx:
        line(ax, CX, y_fc + NODE_H / 2, x, y_m - NODE_H / 2)

    cluster(ax, CX, (c3_top + c3_bot) / 2, total3 + 0.9, c3_bot - c3_top, C3['bg'])
    fx = [CX - (nw3 + gap3) / 2, CX + (nw3 + gap3) / 2]
    for x, lab in zip(fx, ['no_background', 'background']):
        node(ax, x, y_f, nw3, NODE_H, lab, C3['nd'])
    for mx_i in mx:
        for fx_i in fx:
            line(ax, mx_i, y_m + NODE_H / 2, fx_i, y_f - NODE_H / 2)

    return fx, y_f, c3_bot


def draw_bottom(ax, y_met, px_positions, y_p_start):
    """Draw Partitions cluster below metrics."""
    c6_top = y_p_start
    y_p = c6_top + SP + NODE_H / 2
    c6_bot = y_p + NODE_H / 2 + SP
    nw6 = 1.65; gap6 = 0.2
    total6 = 5 * nw6 + 4 * gap6

    cluster(ax, CX, (c6_top + c6_bot) / 2, total6 + 0.7, c6_bot - c6_top, C6['bg'])
    p_labels = ['AD vs HC', 'AD vs NAD', 'AD vs ACS', 'MMSE hi/lo', 'CASI hi/lo']
    px = [CX + (i - 2) * (nw6 + gap6) for i in range(5)]
    for x, lab in zip(px, p_labels):
        node(ax, x, y_p, nw6, NODE_H, lab, C6['nd'])
    for px_i in px:
        line(ax, CX, y_met + NODE_H / 2, px_i, y_p - NODE_H / 2)
    return c6_bot


# ══════════════════════════════════════════════════════
# V1: one merged cluster, Forward/Reverse as columns
# ══════════════════════════════════════════════════════
def build_v1():
    nw_col = 3.3; gap_col = 0.5
    total_col = 2 * nw_col + gap_col

    c1_top = 0.5
    y_vs = c1_top + SP + NODE_H / 2
    y_cf = y_vs + NODE_H + SP
    y_fc = y_cf + NODE_H + SP
    c1_bot = y_fc + NODE_H / 2 + SP
    c2_top = c1_bot + SP
    y_m = c2_top + SP + NODE_H / 2
    c2_bot = y_m + NODE_H / 2 + SP
    c3_top = c2_bot + SP
    y_f = c3_top + SP + NODE_H / 2
    c3_bot = y_f + NODE_H / 2 + SP

    cm_top = c3_bot + SP
    y_r1 = cm_top + SP + NODE_H / 2
    y_r2 = y_r1 + NODE_H + SP
    y_r3 = y_r2 + NODE_H + SP
    y_r4 = y_r3 + NODE_H + SP
    y_met = y_r4 + NODE_H + SP
    cm_bot = y_met + NODE_H / 2 + SP

    c6_top = cm_bot + SP
    y_p = c6_top + SP + NODE_H / 2
    c6_bot = y_p + NODE_H / 2 + SP

    fig_h = c6_bot + 0.5
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(-0.1, FIG_W + 0.1)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    fx, y_f, _ = draw_top(ax, SP)

    cluster(ax, CX, (cm_top + cm_bot) / 2, total_col + 0.9, cm_bot - cm_top, C5['bg'])
    xl = CX - (nw_col + gap_col) / 2
    xr = CX + (nw_col + gap_col) / 2

    node(ax, xl, y_r1, nw_col, NODE_H, "Train Full Cohort\n(LR / XGB)", CF['nd'])
    node(ax, xr, y_r1, nw_col, NODE_H, "Train Matched Cohort\n(LR / XGB)", CR['nd'])
    node(ax, xl, y_r2, nw_col, NODE_H, "OOF Scores", CF['nd'])
    node(ax, xr, y_r2, nw_col, NODE_H, "Predict Full Cohort", CR['nd'])
    node(ax, xl, y_r3, nw_col, NODE_H, "Full Cohort Eval", CF['nd'])
    node(ax, xr, y_r3, nw_col, NODE_H, "Matched OOF Eval", CR['nd'])
    node(ax, xl, y_r4, nw_col, NODE_H, "Matched Subset Eval\n(1:1 paired)", CF['nd'])
    node(ax, xr, y_r4, nw_col, NODE_H, "Unmatched Eval", CR['nd'])
    node(ax, CX, y_met, 4.8, NODE_H,
         "Paired Wilcoxon\nAUC / BalAcc / MCC", C5['hi'])

    for fx_i in fx:
        line(ax, fx_i, y_f + NODE_H / 2, xl, y_r1 - NODE_H / 2)
        line(ax, fx_i, y_f + NODE_H / 2, xr, y_r1 - NODE_H / 2)
    for col in [xl, xr]:
        for ya, yb in [(y_r1, y_r2), (y_r2, y_r3), (y_r3, y_r4)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)
    line(ax, xl, y_r4 + NODE_H / 2, CX, y_met - NODE_H / 2)
    line(ax, xr, y_r4 + NODE_H / 2, CX, y_met - NODE_H / 2)

    draw_bottom(ax, y_met, None, c6_top)

    out = OUT / "embedding_classification_pipeline_mpl_v1.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════
# V2: two separate clusters side-by-side
# ══════════════════════════════════════════════════════
def build_v2():
    nw_col = 3.3; gap_col = 0.7
    cw = nw_col + 0.8

    c1_top = 0.5
    y_vs = c1_top + SP + NODE_H / 2
    y_cf = y_vs + NODE_H + SP
    y_fc = y_cf + NODE_H + SP
    c1_bot = y_fc + NODE_H / 2 + SP
    c2_top = c1_bot + SP
    y_m = c2_top + SP + NODE_H / 2
    c2_bot = y_m + NODE_H / 2 + SP
    c3_top = c2_bot + SP
    y_f = c3_top + SP + NODE_H / 2
    c3_bot = y_f + NODE_H / 2 + SP

    c4_top = c3_bot + SP
    y_clf = c4_top + SP + NODE_H / 2
    c4_bot = y_clf + NODE_H / 2 + SP
    nw4 = 4.2; gap4 = 0.5
    total4 = 2 * nw4 + gap4

    cl_top = c4_bot + SP
    y_r1 = cl_top + SP + NODE_H / 2
    y_r2 = y_r1 + NODE_H + SP
    y_r3 = y_r2 + NODE_H + SP
    cl_bot = y_r3 + NODE_H / 2 + SP
    cl_h = cl_bot - cl_top

    y_met = cl_bot + SP + NODE_H / 2

    c6_top = y_met + NODE_H / 2 + SP
    y_p = c6_top + SP + NODE_H / 2
    c6_bot = y_p + NODE_H / 2 + SP

    fig_h = c6_bot + 0.5
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(-0.1, FIG_W + 0.1)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    fx, y_f, _ = draw_top(ax, SP)

    cluster(ax, CX, (c4_top + c4_bot) / 2, total4 + 0.9, c4_bot - c4_top, C4['bg'])
    clx = [CX - (nw4 + gap4) / 2, CX + (nw4 + gap4) / 2]
    for x, lab in zip(clx, ['Logistic Regression\n$C \\in \\{10^{-3}\\,.\\,.\\,10^{2}\\}$',
                             'XGBoost\nn_tree × max_depth × lr\n{200,500,1k}×{3,6,9}×{.05,.1,.2}']):
        node(ax, x, y_clf, nw4, NODE_H, lab, C4['nd'])
    for fx_i in fx:
        for cx_i in clx:
            line(ax, fx_i, y_f + NODE_H / 2, cx_i, y_clf - NODE_H / 2)

    xl = CX - (cw + gap_col) / 2
    xr = CX + (cw + gap_col) / 2

    cluster(ax, xl, (cl_top + cl_bot) / 2, cw, cl_h, CF['bg'])
    node(ax, xl, y_r1, nw_col, NODE_H, "Full Cohort", CF['nd'])
    node(ax, xl, y_r2, nw_col, NODE_H, "OOF Scores", CF['nd'])
    node(ax, xl, y_r3, nw_col, NODE_H, "Full Cohort Eval\nMatched Subset Eval (1:1)", CF['nd'])

    cluster(ax, xr, (cl_top + cl_bot) / 2, cw, cl_h, CR['bg'])
    node(ax, xr, y_r1, nw_col, NODE_H, "Matched Cohort", CR['nd'])
    node(ax, xr, y_r2, nw_col, NODE_H, "Predict Full Cohort", CR['nd'])
    node(ax, xr, y_r3, nw_col, NODE_H, "Matched OOF Eval\nUnmatched Eval", CR['nd'])

    for cx_i in clx:
        line(ax, cx_i, y_clf + NODE_H / 2, xl, y_r1 - NODE_H / 2)
        line(ax, cx_i, y_clf + NODE_H / 2, xr, y_r1 - NODE_H / 2)
    for col in [xl, xr]:
        for ya, yb in [(y_r1, y_r2), (y_r2, y_r3)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)

    node(ax, CX, y_met, 4.8, NODE_H,
         "Paired Wilcoxon\nAUC / BalAcc / MCC", C5['hi'])
    line(ax, xl, y_r3 + NODE_H / 2, CX, y_met - NODE_H / 2)
    line(ax, xr, y_r3 + NODE_H / 2, CX, y_met - NODE_H / 2)

    draw_bottom(ax, y_met, None, c6_top)

    out = OUT / "embedding_classification_pipeline_mpl_v2.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_v1()
    build_v2()
