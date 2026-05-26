"""Combined Age + Embedding classification pipeline diagram.

Shared cohort section forks into:
  Left:  Age prediction (MiVOLO → calibration → window classifier)
  Right: Embedding classification (full v5 pipeline)

Output:
  workspace/overview/age_emb_pipeline_mpl.png

Usage:
    python scripts/overview/draw_age_emb_pipeline_mpl.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(r"c:\Users\4080\Desktop\Alz_face_analyze\workspace\overview")

FONT = 'Microsoft JhengHei'
FS = 14
FS_HI = 15
FS_SM = 12
FS_XS = 10
EC = '#404040'
NODE_H = 1.15
SP = 0.4

C1 = dict(bg='#E8EFF5', nd='#B3CCE4', hi='#8BB4D9')
C2 = dict(bg='#FDF3E5', nd='#F5D5A0')
C3 = dict(bg='#F0E6F0', nd='#D4B0D4')
C4 = dict(bg='#ECF0E4', nd='#C5D6A8')
C6 = dict(bg='#FBF5E0', nd='#F0D870')
CF = dict(bg='#E4F0ED', nd='#A8D4C4')
CR = dict(bg='#F5E8E4', nd='#E8BDB0')
C_FT = dict(bg='#FCE8E8', nd='#E8B4B4')
C_PA = dict(bg='#E8E8FC', nd='#B4B4E8')
C_PD = dict(bg='#F0EDE0', nd='#D0C8A0')
C_SA = dict(bg='#E8ECF8', nd='#B0B8E0')
C_ES = dict(bg='#F8ECE4', nd='#E0C0A8')
C_ML = dict(bg='#EDF0E8', nd='#C0D0A8')
C_MS = dict(bg='#F5F0E0', nd='#D8C890')
C_AGE = dict(bg='#FFF3E0', nd='#FFD180')
C_EACS = dict(bg='#FFFDE8', nd='#F0D870')
C_AOUT = dict(bg='#ECEEF8', nd='#C0C8E4')
G = dict(bg='#E0E0E0', nd='#C0C0C0')

FIG_W = 28.0
CX = 16.5
X_AGE = 5.0
CX_TOP = 16.5


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


def _cohort_layout():
    """Shared layout constants for the 3-row Cohort section."""
    nw_vs = 2.6; gap_vs = 0.3
    nw_cdr = 3.6; gap_cdr = 0.3
    nw_co = 4.2
    total_vs = 3 * nw_vs + 2 * gap_vs
    total_cdr = 3 * nw_cdr + 2 * gap_cdr

    c1_top = 0.5
    y_vs = c1_top + SP + NODE_H / 2
    y_cf = y_vs + NODE_H + SP
    y_fc = y_cf + NODE_H + SP
    c1_bot = y_fc + NODE_H / 2 + SP
    c1_w = max(total_vs, total_cdr, nw_co) + 0.9

    return dict(
        c1_top=c1_top, y_vs=y_vs, y_cf=y_cf, y_fc=y_fc, c1_bot=c1_bot,
        c1_w=c1_w,
        nw_vs=nw_vs, gap_vs=gap_vs, total_vs=total_vs,
        nw_cdr=nw_cdr, gap_cdr=gap_cdr, total_cdr=total_cdr,
        nw_co=nw_co,
    )


VS_LABELS = ['P: all\nHC: all', 'P: first\nHC: all', 'P: first\nHC: first']
CDR_LABELS = [
    'P: CDR none\nHC: CDR none / MMSE none',
    'P: CDR >0.5\nHC: CDR none / MMSE none',
    'P: CDR >0.5\nHC: CDR =0 / MMSE >26',
]


def build():
    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
    age_w = 4.5
    age_h = NODE_H

    # ══════════════════════════════════════
    # Shared Cohort
    # ══════════════════════════════════════
    CL = _cohort_layout()
    c1_top = CL['c1_top']; y_vs = CL['y_vs']; y_cf = CL['y_cf']
    y_fc = CL['y_fc']; c1_bot = CL['c1_bot']

    # ══════════════════════════════════════
    # Age branch (left)
    # ══════════════════════════════════════
    a_top = c1_bot + SP
    y_a1 = a_top + SP + age_h / 2
    y_a2 = y_a1 + age_h + SP
    y_a3 = y_a2 + age_h + SP
    a_bot = y_a3 + age_h / 2 + SP

    # calibration cluster
    ac_top = a_bot + SP
    y_ac1 = ac_top + SP + age_h / 2
    y_ac2 = y_ac1 + age_h + SP
    y_ac3 = y_ac2 + age_h + SP
    ac_bot = y_ac3 + age_h / 2 + SP

    # age outputs
    c_ao_top = ac_bot + SP
    y_a_out = c_ao_top + SP + NODE_H / 2
    c_ao_bot = y_a_out + NODE_H / 2 + SP

    # age eval chain
    y_a_join = y_a_out + NODE_H / 2 + 3 * SP

    # ══════════════════════════════════════
    # Embedding branch (right)
    # ══════════════════════════════════════
    c_bg_top = c1_bot + SP
    y_bg = c_bg_top + SP + NODE_H / 2
    c_bg_bot = y_bg + NODE_H / 2 + SP
    nw_bg = 2.4; gap_bg = 0.65
    total_bg = 2 * nw_bg + gap_bg

    c_emb_top = c_bg_bot + SP
    y_emb = c_emb_top + SP + NODE_H / 2
    c_emb_bot = y_emb + NODE_H / 2 + SP
    nw_emb = 2.2; gap_emb = 0.5
    total_emb = 4 * nw_emb + 3 * gap_emb

    c_ft_top = c_emb_bot + SP
    y_ft = c_ft_top + SP + NODE_H / 2
    c_ft_bot = y_ft + NODE_H / 2 + SP
    nw_ft = 1.7; gap_ft = 0.25
    total_ft = 6 * nw_ft + 5 * gap_ft

    c_pa_top = c_ft_bot + SP
    y_pa = c_pa_top + SP + NODE_H / 2
    c_pa_bot = y_pa + NODE_H / 2 + SP
    nw_pa = 2.4; gap_pa = 0.65
    total_pa = 2 * nw_pa + gap_pa

    c_pd_top = c_pa_bot + SP
    y_pd = c_pd_top + SP + NODE_H / 2
    c_pd_bot = y_pd + NODE_H / 2 + SP
    nw_pd = 2.0; gap_pd = 0.5
    total_pd = 3 * nw_pd + 2 * gap_pd

    c4_top = c_pd_bot + SP
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

    nw_ev = 2.4; gap_ev = 0.65

    c_em_top = cl_bot + SP
    y_em = c_em_top + SP + NODE_H / 2
    c_em_bot = y_em + NODE_H / 2 + SP

    c_ml_top = c_em_bot + SP
    y_ml = c_ml_top + SP + NODE_H / 2
    c_ml_bot = y_ml + NODE_H / 2 + SP

    c_sa_top = c_ml_bot + SP
    y_sa = c_sa_top + SP + NODE_H / 2
    c_sa_bot = y_sa + NODE_H / 2 + SP

    c_ms_top = c_sa_bot + SP
    y_ms = c_ms_top + SP + NODE_H / 2
    c_ms_bot = y_ms + NODE_H / 2 + SP
    nw_ms = 2.4; gap_ms = 0.65
    total_ms = 3 * nw_ms + 2 * gap_ms

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

    # ════════════════════════════════════
    # DRAW: Shared Cohort
    # ════════════════════════════════════
    cluster(ax, CX_TOP, (c1_top + c1_bot) / 2,
            CL['c1_w'], c1_bot - c1_top, C1['bg'])

    vsx = [CX_TOP + (i - 1) * (CL['nw_vs'] + CL['gap_vs']) for i in range(3)]
    for x, lab in zip(vsx, VS_LABELS):
        node(ax, x, y_vs, CL['nw_vs'], NODE_H, lab, C1['nd'])

    cdrx = [CX_TOP + (i - 1) * (CL['nw_cdr'] + CL['gap_cdr']) for i in range(3)]
    for x, lab in zip(cdrx, CDR_LABELS):
        node(ax, x, y_cf, CL['nw_cdr'], NODE_H, lab, C1['nd'])

    node(ax, CX_TOP, y_fc, CL['nw_co'], NODE_H, "Cohort (P + NAD + ACS)",
         C1['nd'])

    for vx in vsx:
        for cx in cdrx:
            line(ax, vx, y_vs + NODE_H / 2, cx, y_cf - NODE_H / 2)
    for cx in cdrx:
        line(ax, cx, y_cf + NODE_H / 2, CX_TOP, y_fc - NODE_H / 2)

    # ════════════════════════════════════
    # DRAW: External Datasets (same level as Cohort)
    # ════════════════════════════════════
    eacs_w = age_w + 0.6
    cluster(ax, X_AGE, y_fc, eacs_w + 0.6, NODE_H + 2 * SP, C_EACS['bg'])
    node(ax, X_AGE, y_fc, eacs_w, NODE_H,
         "External Datasets\nUTKFace / AgeDB / APPA-REAL / ...",
         C_EACS['nd'])

    # ════════════════════════════════════
    # DRAW: Age branch (left)
    # ════════════════════════════════════

    # -- prediction --
    cluster(ax, X_AGE, (a_top + a_bot) / 2,
            age_w + 0.6, a_bot - a_top, C_AGE['bg'])
    node(ax, X_AGE, y_a1, age_w, age_h,
         "MiVOLO v2\nAge Prediction", C_AGE['nd'])
    node(ax, X_AGE, y_a2, age_w, age_h,
         "Mean Aggregation\n(10 photos - 1)", C_AGE['nd'])
    node(ax, X_AGE, y_a3, age_w, age_h,
         "age_error\n= real_age - predicted_age", C_AGE['nd'])
    line(ax, CX_TOP, y_fc + NODE_H / 2, X_AGE, y_a1 - age_h / 2)
    line(ax, X_AGE, y_fc + NODE_H / 2, X_AGE, y_a1 - age_h / 2)
    line(ax, X_AGE, y_a1 + age_h / 2, X_AGE, y_a2 - age_h / 2)
    line(ax, X_AGE, y_a2 + age_h / 2, X_AGE, y_a3 - age_h / 2)

    # -- calibration --
    cluster(ax, X_AGE, (ac_top + ac_bot) / 2,
            age_w + 0.6, ac_bot - ac_top, C_AGE['bg'])
    node(ax, X_AGE, y_ac1, age_w, age_h,
         "Logistic 10-fold\n(90/10 + 10/90, 30 seeds)", C_AGE['nd'])
    node(ax, X_AGE, y_ac2, age_w, age_h,
         "Bootstrap\n(NAD age>=60, x1000)", C_AGE['nd'])
    node(ax, X_AGE, y_ac3, age_w, age_h,
         "Mean Correction\n(33 age bins, single fit)", C_AGE['nd'])
    line(ax, X_AGE, y_a3 + age_h / 2, X_AGE, y_ac1 - age_h / 2)
    line(ax, X_AGE, y_ac1 + age_h / 2, X_AGE, y_ac2 - age_h / 2)
    line(ax, X_AGE, y_ac2 + age_h / 2, X_AGE, y_ac3 - age_h / 2)

    # -- age outputs --
    out_w = 2.4; out_gap = 0.25
    out_labels = ['predicted_age', 'corrected_age', 'age_error']
    n_out = len(out_labels)
    total_out = n_out * out_w + (n_out - 1) * out_gap
    ox = [X_AGE - total_out / 2 + out_w / 2 + i * (out_w + out_gap)
          for i in range(n_out)]
    cluster(ax, X_AGE, y_a_out, total_out + 0.7,
            c_ao_bot - c_ao_top, C_AOUT['bg'])
    for x, lab in zip(ox, out_labels):
        node(ax, x, y_a_out, out_w, NODE_H, lab, C_AOUT['nd'])
    for x in ox:
        line(ax, X_AGE, y_ac3 + age_h / 2, x, y_a_out - NODE_H / 2)

    # Lines from each output node converging to a point (triangle)
    for x in ox:
        line(ax, x, y_a_out + NODE_H / 2, X_AGE, y_a_join)

    # ════════════════════════════════════
    # DRAW: Age eval chain (left, aligned to embedding y)
    # ════════════════════════════════════
    total_em_a = 2 * nw_ev + gap_ev
    total_ms_a = 3 * nw_ms + 2 * gap_ms
    nw6a = 1.65; gap6a = 0.2
    total_p_a = 5 * nw6a + 4 * gap6a

    cluster(ax, X_AGE, (c_em_top + c_em_bot) / 2,
            total_em_a + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    ae_esx = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    for x, lab in zip(ae_esx, ['1by1matched', 'caliper_group']):
        node(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'])
    for ex in ae_esx:
        line(ax, X_AGE, y_a_join, ex, y_em - NODE_H / 2)

    cluster(ax, X_AGE, (c_ml_top + c_ml_bot) / 2,
            total_em_a + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    ae_mlx = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    for x, lab in zip(ae_mlx, ['subject_match', 'visit_match']):
        node(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'])
    for ex in ae_esx:
        for mx in ae_mlx:
            line(ax, ex, y_em + NODE_H / 2, mx, y_ml - NODE_H / 2)

    cluster(ax, X_AGE, (c_sa_top + c_sa_bot) / 2,
            total_em_a + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    ae_sax = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    for x, lab in zip(ae_sax, ['eval_by_subject', 'eval_by_visit']):
        node(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'])
    for mx in ae_mlx:
        for sx in ae_sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    cluster(ax, X_AGE, (c_ms_top + c_ms_bot) / 2,
            total_ms_a + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    ae_msx = [X_AGE - (nw_ms + gap_ms), X_AGE, X_AGE + (nw_ms + gap_ms)]
    for x, lab in zip(ae_msx, ['match_randomly', 'match_acs_first',
                                 'match_nad_first']):
        node(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'])
    for sx in ae_sax:
        for mx in ae_msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    cluster(ax, X_AGE, (c6_top + c6_bot) / 2,
            total_p_a + 0.7, c6_bot - c6_top, C6['bg'])
    ae_ppx = [X_AGE + (i - 2) * (nw6a + gap6a) for i in range(5)]
    for x, lab in zip(ae_ppx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                                 'MMSE hi/lo', 'CASI hi/lo']):
        node(ax, x, y_p, nw6a, NODE_H, lab, C6['nd'])
    for mx in ae_msx:
        for px in ae_ppx:
            line(ax, mx, y_ms + NODE_H / 2, px, y_p - NODE_H / 2)

    # ════════════════════════════════════
    # DRAW: Embedding branch (right)
    # ════════════════════════════════════

    # Background
    cluster(ax, CX, (c_bg_top + c_bg_bot) / 2,
            total_bg + 0.9, c_bg_bot - c_bg_top, C3['bg'])
    bgx = [CX - (nw_bg + gap_bg) / 2, CX + (nw_bg + gap_bg) / 2]
    for x, lab in zip(bgx, ['no_background', 'background']):
        node(ax, x, y_bg, nw_bg, NODE_H, lab, C3['nd'])
    for bx in bgx:
        line(ax, CX_TOP, y_fc + NODE_H / 2, bx, y_bg - NODE_H / 2)

    # Embedding Models
    cluster(ax, CX, (c_emb_top + c_emb_bot) / 2,
            total_emb + 0.9, c_emb_bot - c_emb_top, C2['bg'])
    emx = [CX + (i - 1.5) * (nw_emb + gap_emb) for i in range(4)]
    for x, lab in zip(emx, ['dlib', 'TopoFR', 'ArcFace', 'VGGFace']):
        node(ax, x, y_emb, nw_emb, NODE_H, lab, C2['nd'])
    for bx in bgx:
        for ex in emx:
            line(ax, bx, y_bg + NODE_H / 2, ex, y_emb - NODE_H / 2)

    # Feature Type
    cluster(ax, CX, (c_ft_top + c_ft_bot) / 2,
            total_ft + 0.9, c_ft_bot - c_ft_top, C_FT['bg'])
    ftx = [CX + (i - 2.5) * (nw_ft + gap_ft) for i in range(6)]
    for x, lab in zip(ftx, ['original', 'diff', '|diff|', 'average',
                             'rel_diff', '|rel_diff|']):
        node(ax, x, y_ft, nw_ft, NODE_H, lab, C_FT['nd'])
    for ex in emx:
        for fx in ftx:
            line(ax, ex, y_emb + NODE_H / 2, fx, y_ft - NODE_H / 2)

    # Photo Aggregation
    cluster(ax, CX, (c_pa_top + c_pa_bot) / 2,
            total_pa + 0.9, c_pa_bot - c_pa_top, C_PA['bg'])
    pax = [CX - (nw_pa + gap_pa) / 2, CX + (nw_pa + gap_pa) / 2]
    for x, lab in zip(pax, ['mean', 'all']):
        node(ax, x, y_pa, nw_pa, NODE_H, lab, C_PA['nd'])
    for fx in ftx:
        for px in pax:
            line(ax, fx, y_ft + NODE_H / 2, px, y_pa - NODE_H / 2)

    # PCA / Drop
    cluster(ax, CX, (c_pd_top + c_pd_bot) / 2,
            total_pd + 0.9, c_pd_bot - c_pd_top, C_PD['bg'])
    pdx = [CX - (nw_pd + gap_pd), CX, CX + (nw_pd + gap_pd)]
    for x, lab in zip(pdx, ['no_drop', 'PCA', 'DropCorr']):
        node(ax, x, y_pd, nw_pd, NODE_H, lab, C_PD['nd'])
    for px in pax:
        for dx in pdx:
            line(ax, px, y_pa + NODE_H / 2, dx, y_pd - NODE_H / 2)

    # Classifier
    cluster(ax, CX, (c4_top + c4_bot) / 2,
            total4 + 0.9, c4_bot - c4_top, C4['bg'])
    clx = [CX - (nw4 + gap4) / 2, CX + (nw4 + gap4) / 2]
    for x, lab in zip(clx, [
        'Logistic Regression\n$C \\in \\{10^{-3}\\,.\\,.\\,10^{2}\\}$',
        'XGBoost\nn_tree × max_depth × lr\n'
        '{200,500,1k}×{3,6,9}×{.05,.1,.2}']):
        node(ax, x, y_clf, nw4, NODE_H, lab, C4['nd'])
    for dx in pdx:
        for ci in clx:
            line(ax, dx, y_pd + NODE_H / 2, ci, y_clf - NODE_H / 2)

    # Forward / Reverse columns
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

    for ci in clx:
        line(ax, ci, y_clf + NODE_H / 2, xl, y_r1 - NODE_H / 2)
        line(ax, ci, y_clf + NODE_H / 2, xr, y_r1 - NODE_H / 2)
    for col in [xl, xr]:
        for ya, yb in [(y_r1, y_r2), (y_r2, y_r3)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)

    # Eval Method
    total_em = 2 * nw_ev + gap_ev
    cluster(ax, CX, (c_em_top + c_em_bot) / 2,
            total_em + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    esx = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    for x, lab in zip(esx, ['1by1matched', 'caliper_group']):
        node(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'])
    for src in [xl, xr]:
        for ex in esx:
            line(ax, src, y_r3 + NODE_H / 2, ex, y_em - NODE_H / 2)

    # Match Level
    total_ml = 2 * nw_ev + gap_ev
    cluster(ax, CX, (c_ml_top + c_ml_bot) / 2,
            total_ml + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    mlx = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    for x, lab in zip(mlx, ['subject_match', 'visit_match']):
        node(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'])
    for ex in esx:
        for mx in mlx:
            line(ax, ex, y_em + NODE_H / 2, mx, y_ml - NODE_H / 2)

    # Eval Unit
    total_sa = 2 * nw_ev + gap_ev
    cluster(ax, CX, (c_sa_top + c_sa_bot) / 2,
            total_sa + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    sax = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    for x, lab in zip(sax, ['eval_by_subject', 'eval_by_visit']):
        node(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'])
    for mx in mlx:
        for sx in sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    # Match Strategy
    cluster(ax, CX, (c_ms_top + c_ms_bot) / 2,
            total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX - (nw_ms + gap_ms), CX, CX + (nw_ms + gap_ms)]
    for x, lab in zip(msx, ['match_randomly', 'match_acs_first',
                             'match_nad_first']):
        node(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'])
    for sx in sax:
        for mx in msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    # Partitions
    nw6 = 1.65; gap6 = 0.2
    total6 = 5 * nw6 + 4 * gap6
    cluster(ax, CX, (c6_top + c6_bot) / 2,
            total6 + 0.7, c6_bot - c6_top, C6['bg'])
    ppx = [CX + (i - 2) * (nw6 + gap6) for i in range(5)]
    for x, lab in zip(ppx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                             'MMSE hi/lo', 'CASI hi/lo']):
        node(ax, x, y_p, nw6, NODE_H, lab, C6['nd'])
    for mx in msx:
        for px_i in ppx:
            line(ax, mx, y_ms + NODE_H / 2, px_i, y_p - NODE_H / 2)

    # ── Save ──
    out = OUT / "age_emb_pipeline_mpl.png"
    fig.savefig(out, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


def build_show():
    """Highlighted-path version: only selected nodes keep color."""
    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
    age_w = 4.5
    age_h = NODE_H

    # ── Layout (identical to build()) ──
    CL = _cohort_layout()
    c1_top = CL['c1_top']; y_vs = CL['y_vs']; y_cf = CL['y_cf']
    y_fc = CL['y_fc']; c1_bot = CL['c1_bot']

    a_top = c1_bot + SP
    y_a1 = a_top + SP + age_h / 2
    y_a2 = y_a1 + age_h + SP
    y_a3 = y_a2 + age_h + SP
    a_bot = y_a3 + age_h / 2 + SP

    ac_top = a_bot + SP
    y_ac1 = ac_top + SP + age_h / 2
    y_ac2 = y_ac1 + age_h + SP
    y_ac3 = y_ac2 + age_h + SP
    ac_bot = y_ac3 + age_h / 2 + SP

    c_ao_top = ac_bot + SP
    y_a_out = c_ao_top + SP + NODE_H / 2
    c_ao_bot = y_a_out + NODE_H / 2 + SP

    y_a_join = y_a_out + NODE_H / 2 + 3 * SP

    c_bg_top = c1_bot + SP
    y_bg = c_bg_top + SP + NODE_H / 2
    c_bg_bot = y_bg + NODE_H / 2 + SP
    nw_bg = 2.4; gap_bg = 0.65
    total_bg = 2 * nw_bg + gap_bg

    c_emb_top = c_bg_bot + SP
    y_emb = c_emb_top + SP + NODE_H / 2
    c_emb_bot = y_emb + NODE_H / 2 + SP
    nw_emb = 2.2; gap_emb = 0.5
    total_emb = 4 * nw_emb + 3 * gap_emb

    c_ft_top = c_emb_bot + SP
    y_ft = c_ft_top + SP + NODE_H / 2
    c_ft_bot = y_ft + NODE_H / 2 + SP
    nw_ft = 1.7; gap_ft = 0.25
    total_ft = 6 * nw_ft + 5 * gap_ft

    c_pa_top = c_ft_bot + SP
    y_pa = c_pa_top + SP + NODE_H / 2
    c_pa_bot = y_pa + NODE_H / 2 + SP
    nw_pa = 2.4; gap_pa = 0.65
    total_pa = 2 * nw_pa + gap_pa

    c_pd_top = c_pa_bot + SP
    y_pd = c_pd_top + SP + NODE_H / 2
    c_pd_bot = y_pd + NODE_H / 2 + SP
    nw_pd = 2.0; gap_pd = 0.5
    total_pd = 3 * nw_pd + 2 * gap_pd

    c4_top = c_pd_bot + SP
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

    nw_ev = 2.4; gap_ev = 0.65

    c_em_top = cl_bot + SP
    y_em = c_em_top + SP + NODE_H / 2
    c_em_bot = y_em + NODE_H / 2 + SP

    c_ml_top = c_em_bot + SP
    y_ml = c_ml_top + SP + NODE_H / 2
    c_ml_bot = y_ml + NODE_H / 2 + SP

    c_sa_top = c_ml_bot + SP
    y_sa = c_sa_top + SP + NODE_H / 2
    c_sa_bot = y_sa + NODE_H / 2 + SP

    c_ms_top = c_sa_bot + SP
    y_ms = c_ms_top + SP + NODE_H / 2
    c_ms_bot = y_ms + NODE_H / 2 + SP
    nw_ms = 2.4; gap_ms = 0.65
    total_ms = 3 * nw_ms + 2 * gap_ms

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

    # ── helper ──
    def _n(ax, cx, cy, w, h, text, fc_on, on=True):
        node(ax, cx, cy, w, h, text, fc_on if on else G['nd'])

    def _cl(ax, cx, cy, w, h, fc_on, on=True):
        cluster(ax, cx, cy, w, h, fc_on if on else G['bg'])

    # ════ Cohort (selective highlight) ════
    _cl(ax, CX_TOP, (c1_top + c1_bot) / 2,
        CL['c1_w'], c1_bot - c1_top, C1['bg'])

    vsx = [CX_TOP + (i - 1) * (CL['nw_vs'] + CL['gap_vs']) for i in range(3)]
    vs_on = [False, True, False]
    for x, lab, on in zip(vsx, VS_LABELS, vs_on):
        _n(ax, x, y_vs, CL['nw_vs'], NODE_H, lab, C1['nd'], on)

    cdrx = [CX_TOP + (i - 1) * (CL['nw_cdr'] + CL['gap_cdr']) for i in range(3)]
    cdr_on = [True, False, False]
    for x, lab, on in zip(cdrx, CDR_LABELS, cdr_on):
        _n(ax, x, y_cf, CL['nw_cdr'], NODE_H, lab, C1['nd'], on)

    _n(ax, CX_TOP, y_fc, CL['nw_co'], NODE_H, "Cohort (P + NAD + ACS)", C1['nd'])

    for vx in vsx:
        for cx in cdrx:
            line(ax, vx, y_vs + NODE_H / 2, cx, y_cf - NODE_H / 2)
    for cx in cdrx:
        line(ax, cx, y_cf + NODE_H / 2, CX_TOP, y_fc - NODE_H / 2)

    # ════ EACS (gray) ════
    eacs_w = age_w + 0.6
    _cl(ax, X_AGE, y_fc, eacs_w + 0.6, NODE_H + 2 * SP, C_EACS['bg'], False)
    _n(ax, X_AGE, y_fc, eacs_w, NODE_H,
       "External Datasets\nUTKFace / AgeDB / APPA-REAL / ...",
       C_EACS['nd'], False)

    # ════ Age prediction (colored) ════
    _cl(ax, X_AGE, (a_top + a_bot) / 2,
        age_w + 0.6, a_bot - a_top, C_AGE['bg'])
    _n(ax, X_AGE, y_a1, age_w, age_h, "MiVOLO v2\nAge Prediction", C_AGE['nd'])
    _n(ax, X_AGE, y_a2, age_w, age_h, "Mean Aggregation\n(10 photos - 1)", C_AGE['nd'])
    _n(ax, X_AGE, y_a3, age_w, age_h,
       "age_error\n= real_age - predicted_age", C_AGE['nd'])
    line(ax, CX_TOP, y_fc + NODE_H / 2, X_AGE, y_a1 - age_h / 2)
    line(ax, X_AGE, y_fc + NODE_H / 2, X_AGE, y_a1 - age_h / 2)
    line(ax, X_AGE, y_a1 + age_h / 2, X_AGE, y_a2 - age_h / 2)
    line(ax, X_AGE, y_a2 + age_h / 2, X_AGE, y_a3 - age_h / 2)

    # ════ Calibration (gray) ════
    _cl(ax, X_AGE, (ac_top + ac_bot) / 2,
        age_w + 0.6, ac_bot - ac_top, C_AGE['bg'], False)
    _n(ax, X_AGE, y_ac1, age_w, age_h,
       "Logistic 10-fold\n(90/10 + 10/90, 30 seeds)", C_AGE['nd'], False)
    _n(ax, X_AGE, y_ac2, age_w, age_h,
       "Bootstrap\n(NAD age>=60, x1000)", C_AGE['nd'], False)
    _n(ax, X_AGE, y_ac3, age_w, age_h,
       "Mean Correction\n(33 age bins, single fit)", C_AGE['nd'], False)
    line(ax, X_AGE, y_a3 + age_h / 2, X_AGE, y_ac1 - age_h / 2)
    line(ax, X_AGE, y_ac1 + age_h / 2, X_AGE, y_ac2 - age_h / 2)
    line(ax, X_AGE, y_ac2 + age_h / 2, X_AGE, y_ac3 - age_h / 2)

    # ════ Age outputs (predicted_age + age_error colored, corrected gray) ════
    out_w = 2.4; out_gap = 0.25
    out_labels = ['predicted_age', 'corrected_age', 'age_error']
    out_on = [True, False, True]
    n_out = len(out_labels)
    total_out = n_out * out_w + (n_out - 1) * out_gap
    ox = [X_AGE - total_out / 2 + out_w / 2 + i * (out_w + out_gap)
          for i in range(n_out)]
    cluster(ax, X_AGE, y_a_out, total_out + 0.7,
            c_ao_bot - c_ao_top, C_AOUT['bg'])
    for x, lab, on in zip(ox, out_labels, out_on):
        _n(ax, x, y_a_out, out_w, NODE_H, lab, C_AOUT['nd'], on)
    for x in ox:
        line(ax, X_AGE, y_ac3 + age_h / 2, x, y_a_out - NODE_H / 2)

    # Lines from each output node converging to a point (triangle)
    for x in ox:
        line(ax, x, y_a_out + NODE_H / 2, X_AGE, y_a_join)

    # ════ Age eval chain (left, aligned to embedding y, selective highlight) ════
    total_em_a = 2 * nw_ev + gap_ev
    total_ms_a = 3 * nw_ms + 2 * gap_ms
    nw6a = 1.65; gap6a = 0.2
    total_p_a = 5 * nw6a + 4 * gap6a

    _cl(ax, X_AGE, (c_em_top + c_em_bot) / 2,
        total_em_a + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    ae_esx = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    ae_es_on = [True, False]
    for x, lab, on in zip(ae_esx, ['1by1matched', 'caliper_group'], ae_es_on):
        _n(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'], on)
    for ex in ae_esx:
        line(ax, X_AGE, y_a_join, ex, y_em - NODE_H / 2)

    _cl(ax, X_AGE, (c_ml_top + c_ml_bot) / 2,
        total_em_a + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    ae_mlx = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    ae_ml_on = [True, False]
    for x, lab, on in zip(ae_mlx, ['subject_match', 'visit_match'], ae_ml_on):
        _n(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'], on)
    for ex in ae_esx:
        for mx in ae_mlx:
            line(ax, ex, y_em + NODE_H / 2, mx, y_ml - NODE_H / 2)

    _cl(ax, X_AGE, (c_sa_top + c_sa_bot) / 2,
        total_em_a + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    ae_sax = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    ae_sa_on = [False, True]
    for x, lab, on in zip(ae_sax, ['eval_by_subject', 'eval_by_visit'], ae_sa_on):
        _n(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'], on)
    for mx in ae_mlx:
        for sx in ae_sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    _cl(ax, X_AGE, (c_ms_top + c_ms_bot) / 2,
        total_ms_a + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    ae_msx = [X_AGE - (nw_ms + gap_ms), X_AGE, X_AGE + (nw_ms + gap_ms)]
    ae_ms_on = [False, True, False]
    for x, lab, on in zip(ae_msx, ['match_randomly', 'match_acs_first',
                                     'match_nad_first'], ae_ms_on):
        _n(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'], on)
    for sx in ae_sax:
        for mx in ae_msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    _cl(ax, X_AGE, (c6_top + c6_bot) / 2,
        total_p_a + 0.7, c6_bot - c6_top, C6['bg'])
    ae_ppx = [X_AGE + (i - 2) * (nw6a + gap6a) for i in range(5)]
    ae_p_on = [True, True, True, True, True]
    for x, lab, on in zip(ae_ppx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                                     'MMSE hi/lo', 'CASI hi/lo'], ae_p_on):
        _n(ax, x, y_p, nw6a, NODE_H, lab, C6['nd'], on)
    for mx in ae_msx:
        for px in ae_ppx:
            line(ax, mx, y_ms + NODE_H / 2, px, y_p - NODE_H / 2)

    # ════ Background (background=on, no_background=off) ════
    _cl(ax, CX, (c_bg_top + c_bg_bot) / 2,
        total_bg + 0.9, c_bg_bot - c_bg_top, C3['bg'])
    bgx = [CX - (nw_bg + gap_bg) / 2, CX + (nw_bg + gap_bg) / 2]
    bg_on = [False, True]
    for x, lab, on in zip(bgx, ['no_background', 'background'], bg_on):
        _n(ax, x, y_bg, nw_bg, NODE_H, lab, C3['nd'], on)
    for bx in bgx:
        line(ax, CX_TOP, y_fc + NODE_H / 2, bx, y_bg - NODE_H / 2)

    # ════ Embedding Models (ArcFace=on) ════
    _cl(ax, CX, (c_emb_top + c_emb_bot) / 2,
        total_emb + 0.9, c_emb_bot - c_emb_top, C2['bg'])
    emx = [CX + (i - 1.5) * (nw_emb + gap_emb) for i in range(4)]
    em_on = [False, False, True, False]
    for x, lab, on in zip(emx, ['dlib', 'TopoFR', 'ArcFace', 'VGGFace'], em_on):
        _n(ax, x, y_emb, nw_emb, NODE_H, lab, C2['nd'], on)
    for bx in bgx:
        for ex in emx:
            line(ax, bx, y_bg + NODE_H / 2, ex, y_emb - NODE_H / 2)

    # ════ Feature Type (original=on) ════
    _cl(ax, CX, (c_ft_top + c_ft_bot) / 2,
        total_ft + 0.9, c_ft_bot - c_ft_top, C_FT['bg'])
    ftx = [CX + (i - 2.5) * (nw_ft + gap_ft) for i in range(6)]
    ft_labs = ['original', 'diff', '|diff|', 'average', 'rel_diff', '|rel_diff|']
    ft_on = [True, True, True, False, True, True]
    for x, lab, on in zip(ftx, ft_labs, ft_on):
        _n(ax, x, y_ft, nw_ft, NODE_H, lab, C_FT['nd'], on)
    for ex in emx:
        for fx in ftx:
            line(ax, ex, y_emb + NODE_H / 2, fx, y_ft - NODE_H / 2)

    # ════ Photo Aggregation (mean=on) ════
    _cl(ax, CX, (c_pa_top + c_pa_bot) / 2,
        total_pa + 0.9, c_pa_bot - c_pa_top, C_PA['bg'])
    pax = [CX - (nw_pa + gap_pa) / 2, CX + (nw_pa + gap_pa) / 2]
    pa_on = [True, False]
    for x, lab, on in zip(pax, ['mean', 'all'], pa_on):
        _n(ax, x, y_pa, nw_pa, NODE_H, lab, C_PA['nd'], on)
    for fx in ftx:
        for px in pax:
            line(ax, fx, y_ft + NODE_H / 2, px, y_pa - NODE_H / 2)

    # ════ PCA/Drop (no_drop=on) ════
    _cl(ax, CX, (c_pd_top + c_pd_bot) / 2,
        total_pd + 0.9, c_pd_bot - c_pd_top, C_PD['bg'])
    pdx = [CX - (nw_pd + gap_pd), CX, CX + (nw_pd + gap_pd)]
    pd_on = [True, False, False]
    for x, lab, on in zip(pdx, ['no_drop', 'PCA', 'DropCorr'], pd_on):
        _n(ax, x, y_pd, nw_pd, NODE_H, lab, C_PD['nd'], on)
    for px in pax:
        for dx in pdx:
            line(ax, px, y_pa + NODE_H / 2, dx, y_pd - NODE_H / 2)

    # ════ Classifier (both on) ════
    _cl(ax, CX, (c4_top + c4_bot) / 2,
        total4 + 0.9, c4_bot - c4_top, C4['bg'])
    clx = [CX - (nw4 + gap4) / 2, CX + (nw4 + gap4) / 2]
    for x, lab in zip(clx, [
        'Logistic Regression\n$C \\in \\{10^{-3}\\,.\\,.\\,10^{2}\\}$',
        'XGBoost\nn_tree × max_depth × lr\n'
        '{200,500,1k}×{3,6,9}×{.05,.1,.2}']):
        _n(ax, x, y_clf, nw4, NODE_H, lab, C4['nd'])
    for dx in pdx:
        for ci in clx:
            line(ax, dx, y_pd + NODE_H / 2, ci, y_clf - NODE_H / 2)

    # ════ Fwd (on) / Rev (off) ════
    xl = CX - (cw + gap_col) / 2
    xr = CX + (cw + gap_col) / 2

    _cl(ax, xl, (cl_top + cl_bot) / 2, cw, cl_h, CF['bg'])
    _n(ax, xl, y_r1, nw_col, NODE_H, "Full Cohort", CF['nd'])
    _n(ax, xl, y_r2, nw_col, NODE_H, "OOF Scores", CF['nd'])
    _n(ax, xl, y_r3, nw_col, NODE_H,
       "Full Cohort Eval\nMatched Subset Eval (1:1)", CF['nd'])

    _cl(ax, xr, (cl_top + cl_bot) / 2, cw, cl_h, CR['bg'], False)
    _n(ax, xr, y_r1, nw_col, NODE_H, "Matched Cohort", CR['nd'], False)
    _n(ax, xr, y_r2, nw_col, NODE_H, "Predict Full Cohort", CR['nd'], False)
    _n(ax, xr, y_r3, nw_col, NODE_H,
       "Matched OOF Eval\nUnmatched Eval", CR['nd'], False)

    for ci in clx:
        line(ax, ci, y_clf + NODE_H / 2, xl, y_r1 - NODE_H / 2)
        line(ax, ci, y_clf + NODE_H / 2, xr, y_r1 - NODE_H / 2)
    for col in [xl, xr]:
        for ya, yb in [(y_r1, y_r2), (y_r2, y_r3)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)

    # ════ Eval Method (1by1matched=on) ════
    total_em = 2 * nw_ev + gap_ev
    _cl(ax, CX, (c_em_top + c_em_bot) / 2,
        total_em + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    esx = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    es_on = [True, False]
    for x, lab, on in zip(esx, ['1by1matched', 'caliper_group'], es_on):
        _n(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'], on)
    for src in [xl, xr]:
        for ex in esx:
            line(ax, src, y_r3 + NODE_H / 2, ex, y_em - NODE_H / 2)

    # ════ Match Level (subject_match=on) ════
    total_ml = 2 * nw_ev + gap_ev
    _cl(ax, CX, (c_ml_top + c_ml_bot) / 2,
        total_ml + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    mlx = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    ml_on = [True, False]
    for x, lab, on in zip(mlx, ['subject_match', 'visit_match'], ml_on):
        _n(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'], on)
    for ex in esx:
        for mx in mlx:
            line(ax, ex, y_em + NODE_H / 2, mx, y_ml - NODE_H / 2)

    # ════ Eval Unit (eval_by_visit=on) ════
    total_sa = 2 * nw_ev + gap_ev
    _cl(ax, CX, (c_sa_top + c_sa_bot) / 2,
        total_sa + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    sax = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    sa_on = [False, True]
    for x, lab, on in zip(sax, ['eval_by_subject', 'eval_by_visit'], sa_on):
        _n(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'], on)
    for mx in mlx:
        for sx in sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    # ════ Match Strategy (match_acs_first=on) ════
    total_ms = 3 * nw_ms + 2 * gap_ms
    _cl(ax, CX, (c_ms_top + c_ms_bot) / 2,
        total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX - (nw_ms + gap_ms), CX, CX + (nw_ms + gap_ms)]
    ms_on = [False, True, False]
    for x, lab, on in zip(msx,
                          ['match_randomly', 'match_acs_first', 'match_nad_first'],
                          ms_on):
        _n(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'], on)
    for sx in sax:
        for mx in msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    # ════ Partitions (HC/NAD/ACS=on, MMSE/CASI=off) ════
    nw6 = 1.65; gap6 = 0.2
    total6 = 5 * nw6 + 4 * gap6
    _cl(ax, CX, (c6_top + c6_bot) / 2,
        total6 + 0.7, c6_bot - c6_top, C6['bg'])
    ppx = [CX + (i - 2) * (nw6 + gap6) for i in range(5)]
    p_labs = ['AD vs HC', 'AD vs NAD', 'AD vs ACS', 'MMSE hi/lo', 'CASI hi/lo']
    p_on = [True, True, True, False, False]
    for x, lab, on in zip(ppx, p_labs, p_on):
        _n(ax, x, y_p, nw6, NODE_H, lab, C6['nd'], on)
    for mx in msx:
        for px_i in ppx:
            line(ax, mx, y_ms + NODE_H / 2, px_i, y_p - NODE_H / 2)

    out = OUT / "age_emb_pipeline_mpl_show.png"
    fig.savefig(out, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
    build_show()
