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
C_FT = dict(bg='#FCE8E8', nd='#E8B4B4')
C_PA = dict(bg='#E8E8FC', nd='#B4B4E8')
C_PD = dict(bg='#F0EDE0', nd='#D0C8A0')
C_TM = dict(bg='#E0EEF0', nd='#8DC3CB')
C_SA = dict(bg='#E8ECF8', nd='#B0B8E0')
C_ES = dict(bg='#F8ECE4', nd='#E0C0A8')
C_MS = dict(bg='#F5F0E0', nd='#D8C890')
C_ML = dict(bg='#EDF0E8', nd='#C0D0A8')


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
    total2 = 4 * nw2 + 3 * gap2

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
    mx = [CX + (i - 1.5) * (nw2 + gap2) for i in range(4)]
    for x, lab in zip(mx, ['dlib', 'TopoFR', 'ArcFace', 'VGGFace']):
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


# ══════════════════════════════════════════════════════
# V3: full 10-variable pipeline
#   Cohort → Background → Embedding → Feature Type
#   → Photo Agg → PCA/Drop → Classifier → Fwd/Rev
# ══════════════════════════════════════════════════════
def _v3v4_top_layout():
    """Shared layout constants for the top section (Cohort → Classifier)."""
    L = {}
    nw1 = 4.2; L['nw1'] = nw1

    c1_top = 0.5
    L['y_vs'] = c1_top + SP + NODE_H / 2
    L['y_cf'] = L['y_vs'] + NODE_H + SP
    L['y_fc'] = L['y_cf'] + NODE_H + SP
    c1_bot = L['y_fc'] + NODE_H / 2 + SP
    L['c1_top'] = c1_top; L['c1_bot'] = c1_bot

    c_bg_top = c1_bot + SP
    L['y_bg'] = c_bg_top + SP + NODE_H / 2
    c_bg_bot = L['y_bg'] + NODE_H / 2 + SP
    nw_bg = 2.4; gap_bg = 0.65
    L['nw_bg'] = nw_bg; L['gap_bg'] = gap_bg
    L['total_bg'] = 2 * nw_bg + gap_bg
    L['c_bg_top'] = c_bg_top; L['c_bg_bot'] = c_bg_bot

    c_emb_top = c_bg_bot + SP
    L['y_emb'] = c_emb_top + SP + NODE_H / 2
    c_emb_bot = L['y_emb'] + NODE_H / 2 + SP
    nw_emb = 2.2; gap_emb = 0.5
    L['nw_emb'] = nw_emb; L['gap_emb'] = gap_emb
    L['total_emb'] = 4 * nw_emb + 3 * gap_emb
    L['c_emb_top'] = c_emb_top; L['c_emb_bot'] = c_emb_bot

    c_ft_top = c_emb_bot + SP
    L['y_ft'] = c_ft_top + SP + NODE_H / 2
    c_ft_bot = L['y_ft'] + NODE_H / 2 + SP
    nw_ft = 1.7; gap_ft = 0.25
    L['nw_ft'] = nw_ft; L['gap_ft'] = gap_ft
    L['total_ft'] = 6 * nw_ft + 5 * gap_ft
    L['c_ft_top'] = c_ft_top; L['c_ft_bot'] = c_ft_bot

    c_pa_top = c_ft_bot + SP
    L['y_pa'] = c_pa_top + SP + NODE_H / 2
    c_pa_bot = L['y_pa'] + NODE_H / 2 + SP
    nw_pa = 2.4; gap_pa = 0.65
    L['nw_pa'] = nw_pa; L['gap_pa'] = gap_pa
    L['total_pa'] = 2 * nw_pa + gap_pa
    L['c_pa_top'] = c_pa_top; L['c_pa_bot'] = c_pa_bot

    c_pd_top = c_pa_bot + SP
    L['y_pd'] = c_pd_top + SP + NODE_H / 2
    c_pd_bot = L['y_pd'] + NODE_H / 2 + SP
    nw_pd = 2.0; gap_pd = 0.5
    L['nw_pd'] = nw_pd; L['gap_pd'] = gap_pd
    L['total_pd'] = 3 * nw_pd + 2 * gap_pd
    L['c_pd_top'] = c_pd_top; L['c_pd_bot'] = c_pd_bot

    c4_top = c_pd_bot + SP
    L['y_clf'] = c4_top + SP + NODE_H / 2
    c4_bot = L['y_clf'] + NODE_H / 2 + SP
    nw4 = 4.2; gap4 = 0.5
    L['nw4'] = nw4; L['gap4'] = gap4
    L['total4'] = 2 * nw4 + gap4
    L['c4_top'] = c4_top; L['c4_bot'] = c4_bot

    return L


def _draw_v3v4_top(ax, L):
    """Draw the shared top section; return (clx, y_clf) for connecting below."""
    cluster(ax, CX, (L['c1_top'] + L['c1_bot']) / 2,
            L['nw1'] + 1.0, L['c1_bot'] - L['c1_top'], C1['bg'])
    node(ax, CX, L['y_vs'], L['nw1'], NODE_H, "Visit Selection", C1['nd'])
    node(ax, CX, L['y_cf'], L['nw1'], NODE_H, "CDR / MMSE Filter", C1['nd'])
    node(ax, CX, L['y_fc'], L['nw1'], NODE_H, "Cohort (P + NAD + ACS)",
         C1['hi'], fs=FS_HI, fc_text='#1A3A5C')
    line(ax, CX, L['y_vs'] + NODE_H / 2, CX, L['y_cf'] - NODE_H / 2)
    line(ax, CX, L['y_cf'] + NODE_H / 2, CX, L['y_fc'] - NODE_H / 2)

    cluster(ax, CX, (L['c_bg_top'] + L['c_bg_bot']) / 2,
            L['total_bg'] + 0.9, L['c_bg_bot'] - L['c_bg_top'], C3['bg'])
    bgx = [CX - (L['nw_bg'] + L['gap_bg']) / 2,
           CX + (L['nw_bg'] + L['gap_bg']) / 2]
    for x, lab in zip(bgx, ['no_background', 'background']):
        node(ax, x, L['y_bg'], L['nw_bg'], NODE_H, lab, C3['nd'])
    for bx in bgx:
        line(ax, CX, L['y_fc'] + NODE_H / 2, bx, L['y_bg'] - NODE_H / 2)

    cluster(ax, CX, (L['c_emb_top'] + L['c_emb_bot']) / 2,
            L['total_emb'] + 0.9, L['c_emb_bot'] - L['c_emb_top'], C2['bg'])
    emx = [CX + (i - 1.5) * (L['nw_emb'] + L['gap_emb']) for i in range(4)]
    for x, lab in zip(emx, ['dlib', 'TopoFR', 'ArcFace', 'VGGFace']):
        node(ax, x, L['y_emb'], L['nw_emb'], NODE_H, lab, C2['nd'])
    for bx in bgx:
        for ex in emx:
            line(ax, bx, L['y_bg'] + NODE_H / 2, ex, L['y_emb'] - NODE_H / 2)

    cluster(ax, CX, (L['c_ft_top'] + L['c_ft_bot']) / 2,
            L['total_ft'] + 0.9, L['c_ft_bot'] - L['c_ft_top'], C_FT['bg'])
    ftx = [CX + (i - 2.5) * (L['nw_ft'] + L['gap_ft']) for i in range(6)]
    for x, lab in zip(ftx, ['original', 'diff', '|diff|', 'average',
                             'rel_diff', '|rel_diff|']):
        node(ax, x, L['y_ft'], L['nw_ft'], NODE_H, lab, C_FT['nd'])
    for ex in emx:
        for fx in ftx:
            line(ax, ex, L['y_emb'] + NODE_H / 2, fx, L['y_ft'] - NODE_H / 2)

    cluster(ax, CX, (L['c_pa_top'] + L['c_pa_bot']) / 2,
            L['total_pa'] + 0.9, L['c_pa_bot'] - L['c_pa_top'], C_PA['bg'])
    pax = [CX - (L['nw_pa'] + L['gap_pa']) / 2,
           CX + (L['nw_pa'] + L['gap_pa']) / 2]
    for x, lab in zip(pax, ['mean', 'all']):
        node(ax, x, L['y_pa'], L['nw_pa'], NODE_H, lab, C_PA['nd'])
    for fx in ftx:
        for px in pax:
            line(ax, fx, L['y_ft'] + NODE_H / 2, px, L['y_pa'] - NODE_H / 2)

    cluster(ax, CX, (L['c_pd_top'] + L['c_pd_bot']) / 2,
            L['total_pd'] + 0.9, L['c_pd_bot'] - L['c_pd_top'], C_PD['bg'])
    pdx = [CX - (L['nw_pd'] + L['gap_pd']), CX,
           CX + (L['nw_pd'] + L['gap_pd'])]
    for x, lab in zip(pdx, ['no_drop', 'PCA', 'DropCorr']):
        node(ax, x, L['y_pd'], L['nw_pd'], NODE_H, lab, C_PD['nd'])
    for px in pax:
        for dx in pdx:
            line(ax, px, L['y_pa'] + NODE_H / 2, dx, L['y_pd'] - NODE_H / 2)

    cluster(ax, CX, (L['c4_top'] + L['c4_bot']) / 2,
            L['total4'] + 0.9, L['c4_bot'] - L['c4_top'], C4['bg'])
    clx = [CX - (L['nw4'] + L['gap4']) / 2,
           CX + (L['nw4'] + L['gap4']) / 2]
    for x, lab in zip(clx, [
        'Logistic Regression\n$C \\in \\{10^{-3}\\,.\\,.\\,10^{2}\\}$',
        'XGBoost\nn_tree × max_depth × lr\n'
        '{200,500,1k}×{3,6,9}×{.05,.1,.2}']):
        node(ax, x, L['y_clf'], L['nw4'], NODE_H, lab, C4['nd'])
    for dx in pdx:
        for ci in clx:
            line(ax, dx, L['y_pd'] + NODE_H / 2, ci, L['y_clf'] - NODE_H / 2)

    return clx


def build_v3():
    """V3: 10-variable pipeline with Forward/Reverse detail columns."""
    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
    L = _v3v4_top_layout()
    c4_bot = L['c4_bot']

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

    clx = _draw_v3v4_top(ax, L)
    y_clf = L['y_clf']

    # === Training: Forward / Reverse ===
    xl = CX - (cw + gap_col) / 2
    xr = CX + (cw + gap_col) / 2

    cluster(ax, xl, (cl_top + cl_bot) / 2, cw, cl_h, CF['bg'])
    node(ax, xl, y_r1, nw_col, NODE_H, "Full Cohort", CF['nd'])
    node(ax, xl, y_r2, nw_col, NODE_H, "OOF Scores", CF['nd'])
    node(ax, xl, y_r3, nw_col, NODE_H,
         "Full Cohort Eval\nMatched Subset Eval (1:1)", CF['nd'])

    cluster(ax, xr, (cl_top + cl_bot) / 2, cw, cl_h, CR['bg'])
    node(ax, xr, y_r1, nw_col, NODE_H, "Matched Cohort", CR['nd'])
    node(ax, xr, y_r2, nw_col, NODE_H, "Predict Full Cohort", CR['nd'])
    node(ax, xr, y_r3, nw_col, NODE_H,
         "Matched OOF Eval\nUnmatched Eval", CR['nd'])

    for cx_i in clx:
        line(ax, cx_i, y_clf + NODE_H / 2, xl, y_r1 - NODE_H / 2)
        line(ax, cx_i, y_clf + NODE_H / 2, xr, y_r1 - NODE_H / 2)
    for col in [xl, xr]:
        for ya, yb in [(y_r1, y_r2), (y_r2, y_r3)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)

    # === Metrics ===
    node(ax, CX, y_met, 4.8, NODE_H,
         "Paired Wilcoxon\nAUC / BalAcc / MCC", C5['hi'])
    line(ax, xl, y_r3 + NODE_H / 2, CX, y_met - NODE_H / 2)
    line(ax, xr, y_r3 + NODE_H / 2, CX, y_met - NODE_H / 2)

    # === Partitions ===
    draw_bottom(ax, y_met, None, c6_top)

    out = OUT / "embedding_classification_pipeline_mpl_v3.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════
# V4: full pipeline with Fwd/Rev detail + score aggregation
#   → evaluation flow
# ══════════════════════════════════════════════════════
def build_v4():
    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
    L = _v3v4_top_layout()
    c4_bot = L['c4_bot']

    # --- Forward / Reverse detail columns ---
    cl_top = c4_bot + SP
    y_r1 = cl_top + SP + NODE_H / 2
    y_r2 = y_r1 + NODE_H + SP
    y_r3 = y_r2 + NODE_H + SP
    cl_bot = y_r3 + NODE_H / 2 + SP
    cl_h = cl_bot - cl_top

    # --- Score Aggregation (C_SA) ---
    c_sa_top = cl_bot + SP
    y_sa = c_sa_top + SP + NODE_H / 2
    c_sa_bot = y_sa + NODE_H / 2 + SP
    nw_sa = 2.4; gap_sa = 0.65
    total_sa = 2 * nw_sa + gap_sa

    # --- Eval Scope (C_ES) ---
    c_es_top = c_sa_bot + SP
    y_es = c_es_top + SP + NODE_H / 2
    c_es_bot = y_es + NODE_H / 2 + SP
    nw_es = 2.4; gap_es = 0.65
    total_es = 2 * nw_es + gap_es

    # --- Matching Strategy (C_MS) ---
    c_ms_top = c_es_bot + SP
    y_ms = c_ms_top + SP + NODE_H / 2
    c_ms_bot = y_ms + NODE_H / 2 + SP
    nw_ms = 2.4; gap_ms = 0.65
    total_ms = 3 * nw_ms + 2 * gap_ms

    # --- Partitions (C6) ---
    c6_top = c_ms_bot + SP
    y_p = c6_top + SP + NODE_H / 2
    c6_bot = y_p + NODE_H / 2 + SP

    fig_h = c6_bot + 0.5
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(-0.1, FIG_W + 0.1)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    clx = _draw_v3v4_top(ax, L)
    y_clf = L['y_clf']

    # === Training: Forward / Reverse detail columns ===
    xl = CX - (cw + gap_col) / 2
    xr = CX + (cw + gap_col) / 2

    cluster(ax, xl, (cl_top + cl_bot) / 2, cw, cl_h, CF['bg'])
    node(ax, xl, y_r1, nw_col, NODE_H, "Full Cohort", CF['nd'])
    node(ax, xl, y_r2, nw_col, NODE_H, "OOF Scores", CF['nd'])
    node(ax, xl, y_r3, nw_col, NODE_H,
         "Full Cohort Eval\nMatched Subset Eval (1:1)", CF['nd'])

    cluster(ax, xr, (cl_top + cl_bot) / 2, cw, cl_h, CR['bg'])
    node(ax, xr, y_r1, nw_col, NODE_H, "Matched Cohort", CR['nd'])
    node(ax, xr, y_r2, nw_col, NODE_H, "Predict Full Cohort", CR['nd'])
    node(ax, xr, y_r3, nw_col, NODE_H,
         "Matched OOF Eval\nUnmatched Eval", CR['nd'])

    for cx_i in clx:
        line(ax, cx_i, y_clf + NODE_H / 2, xl, y_r1 - NODE_H / 2)
        line(ax, cx_i, y_clf + NODE_H / 2, xr, y_r1 - NODE_H / 2)
    for col in [xl, xr]:
        for ya, yb in [(y_r1, y_r2), (y_r2, y_r3)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)

    # === Score Aggregation ===
    cluster(ax, CX, (c_sa_top + c_sa_bot) / 2,
            total_sa + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    sax = [CX - (nw_sa + gap_sa) / 2, CX + (nw_sa + gap_sa) / 2]
    for x, lab in zip(sax, ['subject', 'visit']):
        node(ax, x, y_sa, nw_sa, NODE_H, lab, C_SA['nd'])
    line(ax, xl, y_r3 + NODE_H / 2, sax[0], y_sa - NODE_H / 2)
    line(ax, xl, y_r3 + NODE_H / 2, sax[1], y_sa - NODE_H / 2)
    line(ax, xr, y_r3 + NODE_H / 2, sax[0], y_sa - NODE_H / 2)
    line(ax, xr, y_r3 + NODE_H / 2, sax[1], y_sa - NODE_H / 2)

    # === Eval Scope ===
    cluster(ax, CX, (c_es_top + c_es_bot) / 2,
            total_es + 0.9, c_es_bot - c_es_top, C_ES['bg'])
    esx = [CX - (nw_es + gap_es) / 2, CX + (nw_es + gap_es) / 2]
    for x, lab in zip(esx, ['1:1 matched', 'caliper_group']):
        node(ax, x, y_es, nw_es, NODE_H, lab, C_ES['nd'])
    for sx in sax:
        for ex in esx:
            line(ax, sx, y_sa + NODE_H / 2, ex, y_es - NODE_H / 2)

    # === Matching Strategy ===
    cluster(ax, CX, (c_ms_top + c_ms_bot) / 2,
            total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX - (nw_ms + gap_ms), CX, CX + (nw_ms + gap_ms)]
    for x, lab in zip(msx, ['random', 'ACS_first', 'NAD_first']):
        node(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'])
    for ex in esx:
        for mx in msx:
            line(ax, ex, y_es + NODE_H / 2, mx, y_ms - NODE_H / 2)

    # === Partitions ===
    nw6 = 1.65; gap6 = 0.2
    total6 = 5 * nw6 + 4 * gap6
    cluster(ax, CX, (c6_top + c6_bot) / 2,
            total6 + 0.7, c6_bot - c6_top, C6['bg'])
    p_labels = ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                'MMSE hi/lo', 'CASI hi/lo']
    ppx = [CX + (i - 2) * (nw6 + gap6) for i in range(5)]
    for x, lab in zip(ppx, p_labels):
        node(ax, x, y_p, nw6, NODE_H, lab, C6['nd'])
    for mx in msx:
        for px_i in ppx:
            line(ax, mx, y_ms + NODE_H / 2, px_i, y_p - NODE_H / 2)

    out = OUT / "embedding_classification_pipeline_mpl_v4.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════
# V5: Fwd/Rev detail + match_level + eval flow
# ══════════════════════════════════════════════════════
def build_v5():
    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
    L = _v3v4_top_layout()
    c4_bot = L['c4_bot']

    # --- Forward / Reverse detail columns ---
    cl_top = c4_bot + SP
    y_r1 = cl_top + SP + NODE_H / 2
    y_r2 = y_r1 + NODE_H + SP
    y_r3 = y_r2 + NODE_H + SP
    cl_bot = y_r3 + NODE_H / 2 + SP
    cl_h = cl_bot - cl_top

    # --- Match Level (C_ML) ---
    c_ml_top = cl_bot + SP
    y_ml = c_ml_top + SP + NODE_H / 2
    c_ml_bot = y_ml + NODE_H / 2 + SP
    nw_ml = 2.4; gap_ml = 0.65
    total_ml = 2 * nw_ml + gap_ml

    # --- Eval Unit (C_SA) ---
    c_sa_top = c_ml_bot + SP
    y_sa = c_sa_top + SP + NODE_H / 2
    c_sa_bot = y_sa + NODE_H / 2 + SP
    nw_sa = 2.4; gap_sa = 0.65
    total_sa = 2 * nw_sa + gap_sa

    # --- Eval Method (C_ES) ---
    c_es_top = c_sa_bot + SP
    y_es = c_es_top + SP + NODE_H / 2
    c_es_bot = y_es + NODE_H / 2 + SP
    nw_es = 2.4; gap_es = 0.65
    total_es = 2 * nw_es + gap_es

    # --- Match Strategy (C_MS) ---
    c_ms_top = c_es_bot + SP
    y_ms = c_ms_top + SP + NODE_H / 2
    c_ms_bot = y_ms + NODE_H / 2 + SP
    nw_ms = 2.4; gap_ms = 0.65
    total_ms = 3 * nw_ms + 2 * gap_ms

    # --- Partitions (C6) ---
    c6_top = c_ms_bot + SP
    y_p = c6_top + SP + NODE_H / 2
    c6_bot = y_p + NODE_H / 2 + SP

    fig_h = c6_bot + 0.5
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(-0.1, FIG_W + 0.1)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    clx = _draw_v3v4_top(ax, L)
    y_clf = L['y_clf']

    # === Training: Forward / Reverse detail columns ===
    xl = CX - (cw + gap_col) / 2
    xr = CX + (cw + gap_col) / 2

    cluster(ax, xl, (cl_top + cl_bot) / 2, cw, cl_h, CF['bg'])
    node(ax, xl, y_r1, nw_col, NODE_H, "Full Cohort", CF['nd'])
    node(ax, xl, y_r2, nw_col, NODE_H, "OOF Scores", CF['nd'])
    node(ax, xl, y_r3, nw_col, NODE_H,
         "Full Cohort Eval\nMatched Subset Eval (1:1)", CF['nd'])

    cluster(ax, xr, (cl_top + cl_bot) / 2, cw, cl_h, CR['bg'])
    node(ax, xr, y_r1, nw_col, NODE_H, "Matched Cohort", CR['nd'])
    node(ax, xr, y_r2, nw_col, NODE_H, "Predict Full Cohort", CR['nd'])
    node(ax, xr, y_r3, nw_col, NODE_H,
         "Matched OOF Eval\nUnmatched Eval", CR['nd'])

    for cx_i in clx:
        line(ax, cx_i, y_clf + NODE_H / 2, xl, y_r1 - NODE_H / 2)
        line(ax, cx_i, y_clf + NODE_H / 2, xr, y_r1 - NODE_H / 2)
    for col in [xl, xr]:
        for ya, yb in [(y_r1, y_r2), (y_r2, y_r3)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)

    # === Eval Method (1by1matched / caliper_group) ===
    cluster(ax, CX, (c_ml_top + c_ml_bot) / 2,
            total_ml + 0.9, c_ml_bot - c_ml_top, C_ES['bg'])
    esx = [CX - (nw_ml + gap_ml) / 2, CX + (nw_ml + gap_ml) / 2]
    for x, lab in zip(esx, ['1by1matched', 'caliper_group']):
        node(ax, x, y_ml, nw_ml, NODE_H, lab, C_ES['nd'])
    for src in [xl, xr]:
        for ex in esx:
            line(ax, src, y_r3 + NODE_H / 2, ex, y_ml - NODE_H / 2)

    # === Match Level (subject_match / visit_match) ===
    cluster(ax, CX, (c_sa_top + c_sa_bot) / 2,
            total_sa + 0.9, c_sa_bot - c_sa_top, C_ML['bg'])
    mlx = [CX - (nw_sa + gap_sa) / 2, CX + (nw_sa + gap_sa) / 2]
    for x, lab in zip(mlx, ['subject_match', 'visit_match']):
        node(ax, x, y_sa, nw_sa, NODE_H, lab, C_ML['nd'])
    for ex in esx:
        for mx in mlx:
            line(ax, ex, y_ml + NODE_H / 2, mx, y_sa - NODE_H / 2)

    # === Eval Unit (eval_by_subject / eval_by_visit) ===
    cluster(ax, CX, (c_es_top + c_es_bot) / 2,
            total_es + 0.9, c_es_bot - c_es_top, C_SA['bg'])
    sax = [CX - (nw_es + gap_es) / 2, CX + (nw_es + gap_es) / 2]
    for x, lab in zip(sax, ['eval_by_subject', 'eval_by_visit']):
        node(ax, x, y_es, nw_es, NODE_H, lab, C_SA['nd'])
    for mx in mlx:
        for sx in sax:
            line(ax, mx, y_sa + NODE_H / 2, sx, y_es - NODE_H / 2)

    # === Match Strategy ===
    cluster(ax, CX, (c_ms_top + c_ms_bot) / 2,
            total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX - (nw_ms + gap_ms), CX, CX + (nw_ms + gap_ms)]
    ms_labels = ['no_priority', 'priority_acs', 'priority_nad']
    for x, lab in zip(msx, ms_labels):
        node(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'])
    for sx in sax:
        for mx in msx:
            line(ax, sx, y_es + NODE_H / 2, mx, y_ms - NODE_H / 2)

    # === Partitions ===
    nw6 = 1.65; gap6 = 0.2
    total6 = 5 * nw6 + 4 * gap6
    cluster(ax, CX, (c6_top + c6_bot) / 2,
            total6 + 0.7, c6_bot - c6_top, C6['bg'])
    p_labels = ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                'MMSE hi/lo', 'CASI hi/lo']
    ppx = [CX + (i - 2) * (nw6 + gap6) for i in range(5)]
    for x, lab in zip(ppx, p_labels):
        node(ax, x, y_p, nw6, NODE_H, lab, C6['nd'])
    for mx in msx:
        for px_i in ppx:
            line(ax, mx, y_ms + NODE_H / 2, px_i, y_p - NODE_H / 2)

    out = OUT / "embedding_classification_pipeline_mpl_v5.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_v1()
    build_v2()
    build_v3()
    build_v4()
    build_v5()
