"""Combined Age + Embedding classification pipeline diagram.

Shared cohort section forks into:
  Left:  Age prediction (MiVOLO -> calibration -> window classifier)
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
C_PRE = dict(bg='#E0EEF0', nd='#8DC3CB')
C_EACS = dict(bg='#FFFDE8', nd='#F0D870')
C_AOUT = dict(bg='#ECEEF8', nd='#C0C8E4')
C_ASY = dict(bg='#F3E8F8', nd='#C8A8E0')
G = dict(bg='#E0E0E0', nd='#C0C0C0')

AGE_MODELS = ['MiVOLO', 'InsightFace', 'DeepFace', 'FairFace', 'OpenCV\nDNN']
NW_AM = 2.2; GAP_AM = 0.5
TOTAL_AM = 5 * NW_AM + 4 * GAP_AM

FIG_W = 30.5
X_LIM_LEFT = -4.5
X_LIM_RIGHT = 28.1
CX = 16.5
X_AGE = 3.0
CX_TOP = 16.5

# Preprocessing: 3 stacked nodes + 2 side-by-side bg nodes
PRE_LABELS = ['Detect', 'Select', 'Align']
NW_PRE = 2.2


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


def _common_layout():
    """Compute all y-positions and layout values shared by build() and build_show()."""
    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
    age_w = 4.5
    age_h = NODE_H

    CL = _cohort_layout()
    c1_top = CL['c1_top']; y_vs = CL['y_vs']; y_cf = CL['y_cf']
    y_fc = CL['y_fc']; c1_bot = CL['c1_bot']

    # Preprocessing (shared, 3 stacked + 2 side-by-side bg nodes)
    nw_bg = 2.4; gap_bg = 0.65
    c_pre_top = c1_bot + SP
    y_pre1 = c_pre_top + SP + NODE_H / 2
    y_pre2 = y_pre1 + NODE_H + SP
    y_pre3 = y_pre2 + NODE_H + SP
    y_pre_list = [y_pre1, y_pre2, y_pre3]  # Detect, Select, Align
    y_pre_bg = y_pre3 + NODE_H + SP  # new row for bg options (side by side)
    c_pre_bot = y_pre_bg + NODE_H / 2 + SP
    pre_cluster_w = max(NW_PRE, 2 * nw_bg + gap_bg) + 0.9

    # face/mirrored_face (side by side, after preprocessing)
    c_mir_top = c_pre_bot + SP
    y_mir = c_mir_top + SP + NODE_H / 2
    c_mir_bot = y_mir + NODE_H / 2 + SP
    nw_mir = 2.4

    # Embedding Models (after face/mirror)
    c_emb_top = c_mir_bot + SP
    y_emb = c_emb_top + SP + NODE_H / 2
    c_emb_bot = y_emb + NODE_H / 2 + SP
    nw_emb = 2.2; gap_emb = 0.5
    total_emb = 4 * nw_emb + 3 * gap_emb

    # Feature Type: original + asymmetry side by side (after embedding)
    c_ft_top = c_emb_bot + SP
    y_ft = c_ft_top + SP + NODE_H / 2
    c_ft_bot = y_ft + NODE_H / 2 + SP
    nw_ft = 1.7; gap_ft = 0.25
    total_asym = 4 * nw_ft + 3 * gap_ft  # 7.55
    ft_orig_cl_w = nw_ft + 0.6
    ft_asym_cl_w = total_asym + 0.6

    # x-positions: center original+asymmetry around CX
    gap_between = 1.0
    total_ft_w = nw_ft + gap_between + total_asym
    x_orig = CX - total_ft_w / 2 + nw_ft / 2
    x_asym_center = CX + total_ft_w / 2 - total_asym / 2
    x_ft_asym = [x_asym_center - total_asym/2 + nw_ft/2 + i*(nw_ft+gap_ft) for i in range(4)]

    # face/mirrored_face: standard centered layout
    gap_mir = gap_between
    x_face = CX - (nw_mir + gap_mir) / 2
    x_mirr = CX + (nw_mir + gap_mir) / 2

    # Photo Aggregation
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

    # Age branch positions (aligned to embedding y)
    # Age Models at y_emb
    # Predict Age at y_ft (single feature type row)
    # Mean Aggregation at y_pa
    # age_error at y_pd
    # Calibration: evenly spaced starting from c4_top
    cal_top = c4_top
    y_cal1 = cal_top + SP + NODE_H / 2
    y_cal2 = y_cal1 + NODE_H + SP
    y_cal3 = y_cal2 + NODE_H + SP
    cal_bot = y_cal3 + NODE_H / 2 + SP

    # Age outputs at y_r3
    y_a_out = y_r3
    c_ao_top = y_a_out - NODE_H / 2 - SP
    c_ao_bot = y_a_out + NODE_H / 2 + SP

    # Age model x-positions
    amx = [X_AGE + (i - 2) * (NW_AM + GAP_AM) for i in range(5)]

    fig_h = c6_bot + 0.5

    return dict(
        CL=CL, nw_col=nw_col, gap_col=gap_col, cw=cw,
        age_w=age_w, age_h=age_h,
        c1_top=c1_top, y_vs=y_vs, y_cf=y_cf, y_fc=y_fc, c1_bot=c1_bot,
        c_pre_top=c_pre_top, c_pre_bot=c_pre_bot, y_pre_list=y_pre_list,
        y_pre_bg=y_pre_bg, nw_bg=nw_bg, gap_bg=gap_bg,
        pre_cluster_w=pre_cluster_w,
        y_mir=y_mir, c_mir_top=c_mir_top, c_mir_bot=c_mir_bot,
        nw_mir=nw_mir,
        x_face=x_face, x_mirr=x_mirr,
        c_emb_top=c_emb_top, y_emb=y_emb, c_emb_bot=c_emb_bot,
        nw_emb=nw_emb, gap_emb=gap_emb, total_emb=total_emb,
        y_ft=y_ft, c_ft_top=c_ft_top, c_ft_bot=c_ft_bot,
        nw_ft=nw_ft, gap_ft=gap_ft, total_asym=total_asym,
        ft_orig_cl_w=ft_orig_cl_w, ft_asym_cl_w=ft_asym_cl_w,
        x_orig=x_orig, x_asym_center=x_asym_center, x_ft_asym=x_ft_asym,
        c_pa_top=c_pa_top, y_pa=y_pa, c_pa_bot=c_pa_bot,
        nw_pa=nw_pa, gap_pa=gap_pa, total_pa=total_pa,
        c_pd_top=c_pd_top, y_pd=y_pd, c_pd_bot=c_pd_bot,
        nw_pd=nw_pd, gap_pd=gap_pd, total_pd=total_pd,
        c4_top=c4_top, y_clf=y_clf, c4_bot=c4_bot,
        nw4=nw4, gap4=gap4, total4=total4,
        cl_top=cl_top, y_r1=y_r1, y_r2=y_r2, y_r3=y_r3,
        cl_bot=cl_bot, cl_h=cl_h,
        nw_ev=nw_ev, gap_ev=gap_ev,
        c_em_top=c_em_top, y_em=y_em, c_em_bot=c_em_bot,
        c_ml_top=c_ml_top, y_ml=y_ml, c_ml_bot=c_ml_bot,
        c_sa_top=c_sa_top, y_sa=y_sa, c_sa_bot=c_sa_bot,
        c_ms_top=c_ms_top, y_ms=y_ms, c_ms_bot=c_ms_bot,
        nw_ms=nw_ms, gap_ms=gap_ms, total_ms=total_ms,
        c6_top=c6_top, y_p=y_p, c6_bot=c6_bot,
        cal_top=cal_top, y_cal1=y_cal1, y_cal2=y_cal2, y_cal3=y_cal3,
        cal_bot=cal_bot,
        y_a_out=y_a_out, c_ao_top=c_ao_top, c_ao_bot=c_ao_bot,
        amx=amx, fig_h=fig_h,
    )


def build():
    L = _common_layout()
    CL = L['CL']
    nw_col = L['nw_col']; gap_col = L['gap_col']; cw = L['cw']
    age_w = L['age_w']; age_h = L['age_h']
    c1_top = L['c1_top']; y_vs = L['y_vs']; y_cf = L['y_cf']
    y_fc = L['y_fc']; c1_bot = L['c1_bot']
    c_pre_top = L['c_pre_top']; c_pre_bot = L['c_pre_bot']
    y_pre_list = L['y_pre_list']
    y_pre_bg = L['y_pre_bg']; nw_bg = L['nw_bg']; gap_bg = L['gap_bg']
    pre_cluster_w = L['pre_cluster_w']
    y_mir = L['y_mir']; c_mir_top = L['c_mir_top']; c_mir_bot = L['c_mir_bot']
    nw_mir = L['nw_mir']
    x_face = L['x_face']; x_mirr = L['x_mirr']
    c_emb_top = L['c_emb_top']; y_emb = L['y_emb']; c_emb_bot = L['c_emb_bot']
    nw_emb = L['nw_emb']; gap_emb = L['gap_emb']; total_emb = L['total_emb']
    y_ft = L['y_ft']; c_ft_top = L['c_ft_top']; c_ft_bot = L['c_ft_bot']
    nw_ft = L['nw_ft']; gap_ft = L['gap_ft']; total_asym = L['total_asym']
    ft_orig_cl_w = L['ft_orig_cl_w']; ft_asym_cl_w = L['ft_asym_cl_w']
    x_orig = L['x_orig']; x_asym_center = L['x_asym_center']
    x_ft_asym = L['x_ft_asym']
    c_pa_top = L['c_pa_top']; y_pa = L['y_pa']; c_pa_bot = L['c_pa_bot']
    nw_pa = L['nw_pa']; gap_pa = L['gap_pa']; total_pa = L['total_pa']
    c_pd_top = L['c_pd_top']; y_pd = L['y_pd']; c_pd_bot = L['c_pd_bot']
    nw_pd = L['nw_pd']; gap_pd = L['gap_pd']; total_pd = L['total_pd']
    c4_top = L['c4_top']; y_clf = L['y_clf']; c4_bot = L['c4_bot']
    nw4 = L['nw4']; gap4 = L['gap4']; total4 = L['total4']
    cl_top = L['cl_top']; y_r1 = L['y_r1']; y_r2 = L['y_r2']
    y_r3 = L['y_r3']; cl_bot = L['cl_bot']; cl_h = L['cl_h']
    nw_ev = L['nw_ev']; gap_ev = L['gap_ev']
    c_em_top = L['c_em_top']; y_em = L['y_em']; c_em_bot = L['c_em_bot']
    c_ml_top = L['c_ml_top']; y_ml = L['y_ml']; c_ml_bot = L['c_ml_bot']
    c_sa_top = L['c_sa_top']; y_sa = L['y_sa']; c_sa_bot = L['c_sa_bot']
    c_ms_top = L['c_ms_top']; y_ms = L['y_ms']; c_ms_bot = L['c_ms_bot']
    nw_ms = L['nw_ms']; gap_ms = L['gap_ms']; total_ms = L['total_ms']
    c6_top = L['c6_top']; y_p = L['y_p']; c6_bot = L['c6_bot']
    cal_top = L['cal_top']; y_cal1 = L['y_cal1']; y_cal2 = L['y_cal2']
    y_cal3 = L['y_cal3']; cal_bot = L['cal_bot']
    y_a_out = L['y_a_out']; c_ao_top = L['c_ao_top']; c_ao_bot = L['c_ao_bot']
    amx = L['amx']; fig_h = L['fig_h']

    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(X_LIM_LEFT, X_LIM_RIGHT)
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
    # DRAW: Preprocessing (3 stacked + 2 side-by-side bg nodes)
    # ════════════════════════════════════
    cluster(ax, CX_TOP, (c_pre_top + c_pre_bot) / 2,
            pre_cluster_w, c_pre_bot - c_pre_top, C_PRE['bg'])
    for y, lab in zip(y_pre_list, PRE_LABELS):
        node(ax, CX_TOP, y, NW_PRE, NODE_H, lab, C_PRE['nd'])
    line(ax, CX_TOP, y_fc + NODE_H / 2, CX_TOP, y_pre_list[0] - NODE_H / 2)
    for i in range(len(y_pre_list) - 1):
        line(ax, CX_TOP, y_pre_list[i] + NODE_H / 2,
             CX_TOP, y_pre_list[i + 1] - NODE_H / 2)
    # Side-by-side bg nodes
    bgx = [CX_TOP - (nw_bg + gap_bg) / 2, CX_TOP + (nw_bg + gap_bg) / 2]
    bg_labels = ['no_background', 'background']
    for x, lab in zip(bgx, bg_labels):
        node(ax, x, y_pre_bg, nw_bg, NODE_H, lab, C_PRE['nd'])
    # Align -> both bg nodes
    for bx in bgx:
        line(ax, CX_TOP, y_pre_list[-1] + NODE_H / 2, bx, y_pre_bg - NODE_H / 2)

    # ════════════════════════════════════
    # DRAW: External Datasets (same level as Cohort)
    # ════════════════════════════════════
    eacs_w = age_w + 0.6
    eacs_h = NODE_H * 1.6
    cluster(ax, X_AGE, y_fc, eacs_w + 0.6, eacs_h + 2 * SP, C_EACS['bg'])
    node(ax, X_AGE, y_fc, eacs_w, eacs_h,
         "External Datasets\nUTKFace / AgeDB / APPA-REAL / IMDB\nMegaAge / FairFace / SZU-EmoDage\nAFAD / DiverseAsian",
         C_EACS['nd'])

    # ════════════════════════════════════
    # DRAW: Age branch (left, aligned to embedding y)
    # ════════════════════════════════════

    # -- Age Models at y_emb --
    pred_cw = TOTAL_AM + 0.9
    am_cluster_top = y_emb - age_h / 2 - SP
    am_cluster_bot = y_emb + age_h / 2 + SP
    cluster(ax, X_AGE, y_emb,
            pred_cw, am_cluster_bot - am_cluster_top, C_AGE['bg'])

    for x, lab in zip(amx, AGE_MODELS):
        node(ax, x, y_emb, NW_AM, age_h, lab, C_AGE['nd'])
    # Both bg nodes -> each Age Model
    for bx in bgx:
        for x in amx:
            line(ax, bx, y_pre_bg + NODE_H / 2, x, y_emb - NODE_H / 2)
    # EACS -> each Age Model
    for x in amx:
        line(ax, X_AGE, y_fc + NODE_H / 2, x, y_emb - age_h / 2)

    # -- Predict Age at y_ft --
    pa_cluster_top = y_ft - NODE_H / 2 - SP
    pa_cluster_bot = y_ft + NODE_H / 2 + SP
    cluster(ax, X_AGE, y_ft,
            age_w + 0.6, pa_cluster_bot - pa_cluster_top, C_AGE['bg'])
    node(ax, X_AGE, y_ft, age_w, age_h, "Predict Age", C_AGE['nd'])
    for x in amx:
        line(ax, x, y_emb + age_h / 2, X_AGE, y_ft - age_h / 2)

    # -- Mean Aggregation at y_pa --
    ma_cluster_top = y_pa - NODE_H / 2 - SP
    ma_cluster_bot = y_pa + NODE_H / 2 + SP
    cluster(ax, X_AGE, y_pa,
            age_w + 0.6, ma_cluster_bot - ma_cluster_top, C_AGE['bg'])
    node(ax, X_AGE, y_pa, age_w, age_h, "mean", C_AGE['nd'])
    line(ax, X_AGE, y_ft + age_h / 2, X_AGE, y_pa - age_h / 2)

    # -- age_error at y_pd --
    ae_node_w = 5.5
    ae_cluster_top = y_pd - NODE_H / 2 - SP
    ae_cluster_bot = y_pd + NODE_H / 2 + SP
    cluster(ax, X_AGE, y_pd,
            ae_node_w + 0.9, ae_cluster_bot - ae_cluster_top, C_AGE['bg'])
    node(ax, X_AGE, y_pd, ae_node_w, age_h,
         "age_error = real_age - predicted_age", C_AGE['nd'])
    line(ax, X_AGE, y_pa + age_h / 2, X_AGE, y_pd - age_h / 2)

    # -- Calibration cluster (evenly spaced from c4_top) --
    cal_cluster_h = cal_bot - cal_top
    cluster(ax, X_AGE, (cal_top + cal_bot) / 2,
            age_w + 0.6, cal_cluster_h, C_AGE['bg'])
    node(ax, X_AGE, y_cal1, age_w, age_h,
         "Logistic 10-fold\n(90/10 + 10/90, 30 seeds)", C_AGE['nd'])
    node(ax, X_AGE, y_cal2, age_w, age_h,
         "Bootstrap\n(NAD age>=60, x1000)", C_AGE['nd'])
    node(ax, X_AGE, y_cal3, age_w, age_h,
         "Mean Correction\n(33 age bins, single fit)", C_AGE['nd'])
    line(ax, X_AGE, y_pd + age_h / 2, X_AGE, y_cal1 - age_h / 2)
    line(ax, X_AGE, y_cal1 + age_h / 2, X_AGE, y_cal2 - age_h / 2)
    line(ax, X_AGE, y_cal2 + age_h / 2, X_AGE, y_cal3 - age_h / 2)

    # -- Age outputs at y_r3 --
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
        line(ax, X_AGE, y_cal3 + age_h / 2, x, y_a_out - NODE_H / 2)

    # Each output node directly connects to each eval method node

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
    # each output -> each eval method (3x2=6 lines)
    for o in ox:
        for ex in ae_esx:
            line(ax, o, y_a_out + NODE_H / 2, ex, y_em - NODE_H / 2)

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
    for x, lab in zip(ae_msx, ['no_priority', 'priority_acs',
                                 'priority_nad']):
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

    # face (independent cluster at x_face, y_mir)
    cluster(ax, x_face, (c_mir_top + c_mir_bot) / 2,
            nw_mir + 0.6, c_mir_bot - c_mir_top, C3['bg'])
    node(ax, x_face, y_mir, nw_mir, NODE_H, 'face', C3['nd'])

    # mirrored_face (independent cluster at x_mirr, y_mir)
    cluster(ax, x_mirr, (c_mir_top + c_mir_bot) / 2,
            nw_mir + 0.6, c_mir_bot - c_mir_top, C3['bg'])
    node(ax, x_mirr, y_mir, nw_mir, NODE_H, 'mirrored_face', C3['nd'])

    # bg nodes -> face and mirrored_face
    for bx in bgx:
        line(ax, bx, y_pre_bg + NODE_H / 2, x_face, y_mir - NODE_H / 2)
        line(ax, bx, y_pre_bg + NODE_H / 2, x_mirr, y_mir - NODE_H / 2)

    # Embedding Models
    cluster(ax, CX, (c_emb_top + c_emb_bot) / 2,
            total_emb + 0.9, c_emb_bot - c_emb_top, C2['bg'])
    emx = [CX + (i - 1.5) * (nw_emb + gap_emb) for i in range(4)]
    for x, lab in zip(emx, ['dlib', 'TopoFR', 'ArcFace', 'VGGFace']):
        node(ax, x, y_emb, nw_emb, NODE_H, lab, C2['nd'])
    # face -> Embedding Models, mirrored_face -> Embedding Models
    for src in [x_face, x_mirr]:
        for ex in emx:
            line(ax, src, y_mir + NODE_H / 2, ex, y_emb - NODE_H / 2)

    # original cluster (at x_orig, y_ft)
    cluster(ax, x_orig, (c_ft_top + c_ft_bot) / 2,
            ft_orig_cl_w, c_ft_bot - c_ft_top, C_FT['bg'])
    node(ax, x_orig, y_ft, nw_ft, NODE_H, 'original', C_FT['nd'])

    # asymmetry cluster (centered at x_asym_center, y_ft)
    cluster(ax, x_asym_center, (c_ft_top + c_ft_bot) / 2,
            ft_asym_cl_w, c_ft_bot - c_ft_top, C_FT['bg'])
    for x, lab in zip(x_ft_asym, ['diff', '|diff|', 'rel_diff', '|rel_diff|']):
        node(ax, x, y_ft, nw_ft, NODE_H, lab, C_FT['nd'])

    # Embedding models -> all feature nodes
    for ex in emx:
        line(ax, ex, y_emb + NODE_H / 2, x_orig, y_ft - NODE_H / 2)
        for fx in x_ft_asym:
            line(ax, ex, y_emb + NODE_H / 2, fx, y_ft - NODE_H / 2)

    # Photo Aggregation
    cluster(ax, CX, (c_pa_top + c_pa_bot) / 2,
            total_pa + 0.9, c_pa_bot - c_pa_top, C_PA['bg'])
    pax = [CX - (nw_pa + gap_pa) / 2, CX + (nw_pa + gap_pa) / 2]
    for x, lab in zip(pax, ['mean', 'all']):
        node(ax, x, y_pa, nw_pa, NODE_H, lab, C_PA['nd'])
    # original -> photo agg, asymmetry nodes -> photo agg
    for px in pax:
        line(ax, x_orig, y_ft + NODE_H / 2, px, y_pa - NODE_H / 2)
        for fx in x_ft_asym:
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
    for x, lab in zip(msx, ['no_priority', 'priority_acs',
                             'priority_nad']):
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
    L = _common_layout()
    CL = L['CL']
    nw_col = L['nw_col']; gap_col = L['gap_col']; cw = L['cw']
    age_w = L['age_w']; age_h = L['age_h']
    c1_top = L['c1_top']; y_vs = L['y_vs']; y_cf = L['y_cf']
    y_fc = L['y_fc']; c1_bot = L['c1_bot']
    c_pre_top = L['c_pre_top']; c_pre_bot = L['c_pre_bot']
    y_pre_list = L['y_pre_list']
    y_pre_bg = L['y_pre_bg']; nw_bg = L['nw_bg']; gap_bg = L['gap_bg']
    pre_cluster_w = L['pre_cluster_w']
    y_mir = L['y_mir']; c_mir_top = L['c_mir_top']; c_mir_bot = L['c_mir_bot']
    nw_mir = L['nw_mir']
    x_face = L['x_face']; x_mirr = L['x_mirr']
    c_emb_top = L['c_emb_top']; y_emb = L['y_emb']; c_emb_bot = L['c_emb_bot']
    nw_emb = L['nw_emb']; gap_emb = L['gap_emb']; total_emb = L['total_emb']
    y_ft = L['y_ft']; c_ft_top = L['c_ft_top']; c_ft_bot = L['c_ft_bot']
    nw_ft = L['nw_ft']; gap_ft = L['gap_ft']; total_asym = L['total_asym']
    ft_orig_cl_w = L['ft_orig_cl_w']; ft_asym_cl_w = L['ft_asym_cl_w']
    x_orig = L['x_orig']; x_asym_center = L['x_asym_center']
    x_ft_asym = L['x_ft_asym']
    c_pa_top = L['c_pa_top']; y_pa = L['y_pa']; c_pa_bot = L['c_pa_bot']
    nw_pa = L['nw_pa']; gap_pa = L['gap_pa']; total_pa = L['total_pa']
    c_pd_top = L['c_pd_top']; y_pd = L['y_pd']; c_pd_bot = L['c_pd_bot']
    nw_pd = L['nw_pd']; gap_pd = L['gap_pd']; total_pd = L['total_pd']
    c4_top = L['c4_top']; y_clf = L['y_clf']; c4_bot = L['c4_bot']
    nw4 = L['nw4']; gap4 = L['gap4']; total4 = L['total4']
    cl_top = L['cl_top']; y_r1 = L['y_r1']; y_r2 = L['y_r2']
    y_r3 = L['y_r3']; cl_bot = L['cl_bot']; cl_h = L['cl_h']
    nw_ev = L['nw_ev']; gap_ev = L['gap_ev']
    c_em_top = L['c_em_top']; y_em = L['y_em']; c_em_bot = L['c_em_bot']
    c_ml_top = L['c_ml_top']; y_ml = L['y_ml']; c_ml_bot = L['c_ml_bot']
    c_sa_top = L['c_sa_top']; y_sa = L['y_sa']; c_sa_bot = L['c_sa_bot']
    c_ms_top = L['c_ms_top']; y_ms = L['y_ms']; c_ms_bot = L['c_ms_bot']
    nw_ms = L['nw_ms']; gap_ms = L['gap_ms']; total_ms = L['total_ms']
    c6_top = L['c6_top']; y_p = L['y_p']; c6_bot = L['c6_bot']
    cal_top = L['cal_top']; y_cal1 = L['y_cal1']; y_cal2 = L['y_cal2']
    y_cal3 = L['y_cal3']; cal_bot = L['cal_bot']
    y_a_out = L['y_a_out']; c_ao_top = L['c_ao_top']; c_ao_bot = L['c_ao_bot']
    amx = L['amx']; fig_h = L['fig_h']

    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(X_LIM_LEFT, X_LIM_RIGHT)
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

    # ════ Preprocessing (3 stacked + 2 side-by-side bg, highlighted) ════
    _cl(ax, CX_TOP, (c_pre_top + c_pre_bot) / 2,
        pre_cluster_w, c_pre_bot - c_pre_top, C_PRE['bg'])
    for y, lab in zip(y_pre_list, PRE_LABELS):
        _n(ax, CX_TOP, y, NW_PRE, NODE_H, lab, C_PRE['nd'])
    line(ax, CX_TOP, y_fc + NODE_H / 2, CX_TOP, y_pre_list[0] - NODE_H / 2)
    for i in range(len(y_pre_list) - 1):
        line(ax, CX_TOP, y_pre_list[i] + NODE_H / 2,
             CX_TOP, y_pre_list[i + 1] - NODE_H / 2)
    # Side-by-side bg nodes: no_background=gray, background=highlighted
    bgx = [CX_TOP - (nw_bg + gap_bg) / 2, CX_TOP + (nw_bg + gap_bg) / 2]
    bg_labels = ['no_background', 'background']
    bg_on = [False, True]
    for x, lab, on in zip(bgx, bg_labels, bg_on):
        _n(ax, x, y_pre_bg, nw_bg, NODE_H, lab, C_PRE['nd'], on)
    for bx in bgx:
        line(ax, CX_TOP, y_pre_list[-1] + NODE_H / 2, bx, y_pre_bg - NODE_H / 2)

    # ════ EACS (gray) ════
    eacs_w = age_w + 0.6
    _cl(ax, X_AGE, y_fc, eacs_w + 0.6, NODE_H + 2 * SP, C_EACS['bg'], False)
    _n(ax, X_AGE, y_fc, eacs_w, NODE_H,
       "External Datasets\nUTKFace / AgeDB / APPA-REAL / ...",
       C_EACS['nd'], False)

    # ════ Age Models (MiVOLO highlighted, others gray) ════
    pred_cw = TOTAL_AM + 0.9
    am_cluster_top = y_emb - age_h / 2 - SP
    am_cluster_bot = y_emb + age_h / 2 + SP
    _cl(ax, X_AGE, y_emb,
        pred_cw, am_cluster_bot - am_cluster_top, C_AGE['bg'])

    am_on = [True, False, False, False, False]
    for x, lab, on in zip(amx, AGE_MODELS, am_on):
        _n(ax, x, y_emb, NW_AM, age_h, lab, C_AGE['nd'], on)
    # Both bg nodes -> each Age Model
    for bx in bgx:
        for x in amx:
            line(ax, bx, y_pre_bg + NODE_H / 2, x, y_emb - NODE_H / 2)
    # EACS -> each Age Model
    for x in amx:
        line(ax, X_AGE, y_fc + NODE_H / 2, x, y_emb - age_h / 2)

    # ════ Predict Age (highlighted) at y_ft ════
    pa_cluster_top = y_ft - NODE_H / 2 - SP
    pa_cluster_bot = y_ft + NODE_H / 2 + SP
    _cl(ax, X_AGE, y_ft,
        age_w + 0.6, pa_cluster_bot - pa_cluster_top, C_AGE['bg'])
    _n(ax, X_AGE, y_ft, age_w, age_h, "Predict Age", C_AGE['nd'])
    for x in amx:
        line(ax, x, y_emb + age_h / 2, X_AGE, y_ft - age_h / 2)

    # ════ Mean Aggregation (highlighted) ════
    ma_cluster_top = y_pa - NODE_H / 2 - SP
    ma_cluster_bot = y_pa + NODE_H / 2 + SP
    _cl(ax, X_AGE, y_pa,
        age_w + 0.6, ma_cluster_bot - ma_cluster_top, C_AGE['bg'])
    _n(ax, X_AGE, y_pa, age_w, age_h, "mean", C_AGE['nd'])
    line(ax, X_AGE, y_ft + age_h / 2, X_AGE, y_pa - age_h / 2)

    # ════ age_error (highlighted, wider) ════
    ae_node_w = 5.5
    ae_cluster_top = y_pd - NODE_H / 2 - SP
    ae_cluster_bot = y_pd + NODE_H / 2 + SP
    _cl(ax, X_AGE, y_pd,
        ae_node_w + 0.9, ae_cluster_bot - ae_cluster_top, C_AGE['bg'])
    _n(ax, X_AGE, y_pd, ae_node_w, age_h,
       "age_error = real_age - predicted_age", C_AGE['nd'])
    line(ax, X_AGE, y_pa + age_h / 2, X_AGE, y_pd - age_h / 2)

    # ════ Calibration (gray, evenly spaced from c4_top) ════
    cal_cluster_h = cal_bot - cal_top
    _cl(ax, X_AGE, (cal_top + cal_bot) / 2,
        age_w + 0.6, cal_cluster_h, C_AGE['bg'], False)
    _n(ax, X_AGE, y_cal1, age_w, age_h,
       "Logistic 10-fold\n(90/10 + 10/90, 30 seeds)", C_AGE['nd'], False)
    _n(ax, X_AGE, y_cal2, age_w, age_h,
       "Bootstrap\n(NAD age>=60, x1000)", C_AGE['nd'], False)
    _n(ax, X_AGE, y_cal3, age_w, age_h,
       "Mean Correction\n(33 age bins, single fit)", C_AGE['nd'], False)
    line(ax, X_AGE, y_pd + age_h / 2, X_AGE, y_cal1 - age_h / 2)
    line(ax, X_AGE, y_cal1 + age_h / 2, X_AGE, y_cal2 - age_h / 2)
    line(ax, X_AGE, y_cal2 + age_h / 2, X_AGE, y_cal3 - age_h / 2)

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
        line(ax, X_AGE, y_cal3 + age_h / 2, x, y_a_out - NODE_H / 2)

    # each output -> each eval method (3x2=6 lines)

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
    # each output -> each eval method (3x2=6 lines)
    for o in ox:
        for ex in ae_esx:
            line(ax, o, y_a_out + NODE_H / 2, ex, y_em - NODE_H / 2)

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
    for x, lab, on in zip(ae_msx, ['no_priority', 'priority_acs',
                                     'priority_nad'], ae_ms_on):
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

    # ════ face (highlighted) ════
    _cl(ax, x_face, (c_mir_top + c_mir_bot) / 2,
        nw_mir + 0.6, c_mir_bot - c_mir_top, C3['bg'])
    _n(ax, x_face, y_mir, nw_mir, NODE_H, 'face', C3['nd'])

    # ════ mirrored_face (highlighted) ════
    _cl(ax, x_mirr, (c_mir_top + c_mir_bot) / 2,
        nw_mir + 0.6, c_mir_bot - c_mir_top, C3['bg'])
    _n(ax, x_mirr, y_mir, nw_mir, NODE_H, 'mirrored_face', C3['nd'])

    # Preprocessing bg nodes -> face and mirrored_face
    for bx in bgx:
        line(ax, bx, y_pre_bg + NODE_H / 2, x_face, y_mir - NODE_H / 2)
        line(ax, bx, y_pre_bg + NODE_H / 2, x_mirr, y_mir - NODE_H / 2)

    # ════ Embedding Models (ArcFace=on) ════
    _cl(ax, CX, (c_emb_top + c_emb_bot) / 2,
        total_emb + 0.9, c_emb_bot - c_emb_top, C2['bg'])
    emx = [CX + (i - 1.5) * (nw_emb + gap_emb) for i in range(4)]
    em_on = [False, False, True, False]
    for x, lab, on in zip(emx, ['dlib', 'TopoFR', 'ArcFace', 'VGGFace'], em_on):
        _n(ax, x, y_emb, nw_emb, NODE_H, lab, C2['nd'], on)
    for src in [x_face, x_mirr]:
        for ex in emx:
            line(ax, src, y_mir + NODE_H / 2, ex, y_emb - NODE_H / 2)

    # ════ original (on) ════
    _cl(ax, x_orig, (c_ft_top + c_ft_bot) / 2,
        ft_orig_cl_w, c_ft_bot - c_ft_top, C_FT['bg'])
    _n(ax, x_orig, y_ft, nw_ft, NODE_H, 'original', C_FT['nd'], True)

    # ════ asymmetry (all on) ════
    _cl(ax, x_asym_center, (c_ft_top + c_ft_bot) / 2,
        ft_asym_cl_w, c_ft_bot - c_ft_top, C_FT['bg'])
    asym_labels = ['diff', '|diff|', 'rel_diff', '|rel_diff|']
    asym_on = [True, True, True, True]
    for x, lab, on in zip(x_ft_asym, asym_labels, asym_on):
        _n(ax, x, y_ft, nw_ft, NODE_H, lab, C_FT['nd'], on)

    # Embedding models -> all feature nodes
    for ex in emx:
        line(ax, ex, y_emb + NODE_H / 2, x_orig, y_ft - NODE_H / 2)
        for fx in x_ft_asym:
            line(ax, ex, y_emb + NODE_H / 2, fx, y_ft - NODE_H / 2)

    # ════ Photo Aggregation (mean=on) ════
    _cl(ax, CX, (c_pa_top + c_pa_bot) / 2,
        total_pa + 0.9, c_pa_bot - c_pa_top, C_PA['bg'])
    pax = [CX - (nw_pa + gap_pa) / 2, CX + (nw_pa + gap_pa) / 2]
    pa_on = [True, False]
    for x, lab, on in zip(pax, ['mean', 'all'], pa_on):
        _n(ax, x, y_pa, nw_pa, NODE_H, lab, C_PA['nd'], on)
    for px in pax:
        line(ax, x_orig, y_ft + NODE_H / 2, px, y_pa - NODE_H / 2)
        for fx in x_ft_asym:
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

    # ════ Match Strategy (priority_acs=on) ════
    total_ms = 3 * nw_ms + 2 * gap_ms
    _cl(ax, CX, (c_ms_top + c_ms_bot) / 2,
        total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX - (nw_ms + gap_ms), CX, CX + (nw_ms + gap_ms)]
    ms_on = [False, True, False]
    for x, lab, on in zip(msx,
                          ['no_priority', 'priority_acs', 'priority_nad'],
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


def build_v2():
    """v2: Age (left) + Embedding/original (center) + Asymmetry (right)."""
    CX_V2 = 16.5
    X_ASYM = 27.0
    age_w = 4.5; age_h = NODE_H

    CL = _cohort_layout()
    c1_top = CL['c1_top']; y_vs = CL['y_vs']; y_cf = CL['y_cf']
    y_fc = CL['y_fc']; c1_bot = CL['c1_bot']

    nw_bg = 2.4; gap_bg = 0.65
    c_pre_top = c1_bot + SP
    y_pre1 = c_pre_top + SP + NODE_H / 2
    y_pre2 = y_pre1 + NODE_H + SP
    y_pre3 = y_pre2 + NODE_H + SP
    y_pre_list = [y_pre1, y_pre2, y_pre3]
    y_pre_bg = y_pre3 + NODE_H + SP
    c_pre_bot = y_pre_bg + NODE_H / 2 + SP
    pre_cluster_w = max(NW_PRE, 2 * nw_bg + gap_bg) + 0.9

    nw_mir = 2.4; gap_mir = 1.0
    c_mir_top = c_pre_bot + SP
    y_mir = c_mir_top + SP + NODE_H / 2
    c_mir_bot = y_mir + NODE_H / 2 + SP
    x_face = CX_V2 - (nw_mir + gap_mir) / 2
    x_mirr = CX_V2 + (nw_mir + gap_mir) / 2

    nw_emb = 2.2; gap_emb = 0.5
    total_emb = 4 * nw_emb + 3 * gap_emb
    c_emb_top = c_mir_bot + SP
    y_emb = c_emb_top + SP + NODE_H / 2
    c_emb_bot = y_emb + NODE_H / 2 + SP
    emx = [CX_V2 + (i - 1.5) * (nw_emb + gap_emb) for i in range(4)]

    # ═══ FORK ═══
    nw_ft = 1.7; gap_ft = 0.25
    c_ft_top = c_emb_bot + SP
    y_ft = c_ft_top + SP + NODE_H / 2
    c_ft_bot = y_ft + NODE_H / 2 + SP

    nw_pa = 2.4; gap_pa = 0.65
    total_pa = 2 * nw_pa + gap_pa
    c_pa_top = c_ft_bot + SP
    y_pa = c_pa_top + SP + NODE_H / 2
    c_pa_bot = y_pa + NODE_H / 2 + SP

    nw_pd = 2.0; gap_pd = 0.5
    total_pd = 3 * nw_pd + 2 * gap_pd
    c_pd_top = c_pa_bot + SP
    y_pd = c_pd_top + SP + NODE_H / 2
    c_pd_bot = y_pd + NODE_H / 2 + SP

    nw4 = 4.2; gap4 = 0.5
    total4 = 2 * nw4 + gap4
    c4_top = c_pd_bot + SP
    y_clf = c4_top + SP + NODE_H / 2
    c4_bot = y_clf + NODE_H / 2 + SP

    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
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

    nw_ms = 2.4; gap_ms = 0.65
    total_ms = 3 * nw_ms + 2 * gap_ms
    c_ms_top = c_sa_bot + SP
    y_ms = c_ms_top + SP + NODE_H / 2
    c_ms_bot = y_ms + NODE_H / 2 + SP

    c6_top = c_ms_bot + SP
    y_p = c6_top + SP + NODE_H / 2
    c6_bot = y_p + NODE_H / 2 + SP

    # Age branch y-positions (aligned to embedding rows)
    amx = [X_AGE + (i - 2) * (NW_AM + GAP_AM) for i in range(5)]
    cal_top = c4_top
    y_cal1 = cal_top + SP + NODE_H / 2
    y_cal2 = y_cal1 + NODE_H + SP
    y_cal3 = y_cal2 + NODE_H + SP
    cal_bot = y_cal3 + NODE_H / 2 + SP
    y_a_out = y_r3
    c_ao_top = y_a_out - NODE_H / 2 - SP
    c_ao_bot = y_a_out + NODE_H / 2 + SP

    # Asymmetry branch positions
    total_asym = 4 * nw_ft + 3 * gap_ft
    x_ft_asym = [X_ASYM - total_asym / 2 + nw_ft / 2
                 + i * (nw_ft + gap_ft) for i in range(4)]
    asym_nw = 3.5

    fig_h = c6_bot + 0.5
    fig, ax = plt.subplots(figsize=(36.0, fig_h))
    ax.set_xlim(-4.5, 34)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ════════════════════════════════════
    # Shared Cohort
    # ════════════════════════════════════
    cluster(ax, CX_TOP, (c1_top + c1_bot) / 2,
            CL['c1_w'], c1_bot - c1_top, C1['bg'])
    vsx = [CX_TOP + (i - 1) * (CL['nw_vs'] + CL['gap_vs']) for i in range(3)]
    for x, lab in zip(vsx, VS_LABELS):
        node(ax, x, y_vs, CL['nw_vs'], NODE_H, lab, C1['nd'])
    cdrx = [CX_TOP + (i - 1) * (CL['nw_cdr'] + CL['gap_cdr'])
            for i in range(3)]
    for x, lab in zip(cdrx, CDR_LABELS):
        node(ax, x, y_cf, CL['nw_cdr'], NODE_H, lab, C1['nd'])
    node(ax, CX_TOP, y_fc, CL['nw_co'], NODE_H,
         "Cohort (P + NAD + ACS)", C1['nd'])
    for vx in vsx:
        for cx in cdrx:
            line(ax, vx, y_vs + NODE_H / 2, cx, y_cf - NODE_H / 2)
    for cx in cdrx:
        line(ax, cx, y_cf + NODE_H / 2, CX_TOP, y_fc - NODE_H / 2)

    # ════════════════════════════════════
    # Preprocessing
    # ════════════════════════════════════
    cluster(ax, CX_TOP, (c_pre_top + c_pre_bot) / 2,
            pre_cluster_w, c_pre_bot - c_pre_top, C_PRE['bg'])
    for y, lab in zip(y_pre_list, PRE_LABELS):
        node(ax, CX_TOP, y, NW_PRE, NODE_H, lab, C_PRE['nd'])
    line(ax, CX_TOP, y_fc + NODE_H / 2, CX_TOP, y_pre_list[0] - NODE_H / 2)
    for i in range(2):
        line(ax, CX_TOP, y_pre_list[i] + NODE_H / 2,
             CX_TOP, y_pre_list[i + 1] - NODE_H / 2)
    bgx = [CX_TOP - (nw_bg + gap_bg) / 2, CX_TOP + (nw_bg + gap_bg) / 2]
    for x, lab in zip(bgx, ['no_background', 'background']):
        node(ax, x, y_pre_bg, nw_bg, NODE_H, lab, C_PRE['nd'])
    for bx in bgx:
        line(ax, CX_TOP, y_pre_list[-1] + NODE_H / 2,
             bx, y_pre_bg - NODE_H / 2)

    # ════════════════════════════════════
    # External Datasets
    # ════════════════════════════════════
    eacs_w = age_w + 0.6
    eacs_h = NODE_H * 1.6
    cluster(ax, X_AGE, y_fc, eacs_w + 0.6, eacs_h + 2 * SP, C_EACS['bg'])
    node(ax, X_AGE, y_fc, eacs_w, eacs_h,
         "External Datasets\nUTKFace / AgeDB / APPA-REAL / IMDB\n"
         "MegaAge / FairFace / SZU-EmoDage\nAFAD / DiverseAsian",
         C_EACS['nd'])

    # ════════════════════════════════════
    # Age branch (same as v1)
    # ════════════════════════════════════
    pred_cw = TOTAL_AM + 0.9
    am_top = y_emb - age_h / 2 - SP
    am_bot = y_emb + age_h / 2 + SP
    cluster(ax, X_AGE, y_emb, pred_cw, am_bot - am_top, C_AGE['bg'])
    for x, lab in zip(amx, AGE_MODELS):
        node(ax, x, y_emb, NW_AM, age_h, lab, C_AGE['nd'])
    for bx in bgx:
        for x in amx:
            line(ax, bx, y_pre_bg + NODE_H / 2, x, y_emb - NODE_H / 2)
    for x in amx:
        line(ax, X_AGE, y_fc + NODE_H / 2, x, y_emb - age_h / 2)

    cluster(ax, X_AGE, y_ft, age_w + 0.6, NODE_H + 2 * SP, C_AGE['bg'])
    node(ax, X_AGE, y_ft, age_w, age_h, "Predict Age", C_AGE['nd'])
    for x in amx:
        line(ax, x, y_emb + age_h / 2, X_AGE, y_ft - age_h / 2)

    cluster(ax, X_AGE, y_pa, age_w + 0.6, NODE_H + 2 * SP, C_AGE['bg'])
    node(ax, X_AGE, y_pa, age_w, age_h, "mean", C_AGE['nd'])
    line(ax, X_AGE, y_ft + age_h / 2, X_AGE, y_pa - age_h / 2)

    ae_node_w = 5.5
    cluster(ax, X_AGE, y_pd, ae_node_w + 0.9, NODE_H + 2 * SP, C_AGE['bg'])
    node(ax, X_AGE, y_pd, ae_node_w, age_h,
         "age_error = real_age - predicted_age", C_AGE['nd'])
    line(ax, X_AGE, y_pa + age_h / 2, X_AGE, y_pd - age_h / 2)

    cal_h = cal_bot - cal_top
    cluster(ax, X_AGE, (cal_top + cal_bot) / 2,
            age_w + 0.6, cal_h, C_AGE['bg'])
    node(ax, X_AGE, y_cal1, age_w, age_h,
         "Logistic 10-fold\n(90/10 + 10/90, 30 seeds)", C_AGE['nd'])
    node(ax, X_AGE, y_cal2, age_w, age_h,
         "Bootstrap\n(NAD age>=60, x1000)", C_AGE['nd'])
    node(ax, X_AGE, y_cal3, age_w, age_h,
         "Mean Correction\n(33 age bins, single fit)", C_AGE['nd'])
    line(ax, X_AGE, y_pd + age_h / 2, X_AGE, y_cal1 - age_h / 2)
    line(ax, X_AGE, y_cal1 + age_h / 2, X_AGE, y_cal2 - age_h / 2)
    line(ax, X_AGE, y_cal2 + age_h / 2, X_AGE, y_cal3 - age_h / 2)

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
        line(ax, X_AGE, y_cal3 + age_h / 2, x, y_a_out - NODE_H / 2)

    # Age eval chain
    total_em_a = 2 * nw_ev + gap_ev
    total_ms_a = 3 * nw_ms + 2 * gap_ms
    nw6a = 1.65; gap6a = 0.2
    total_p_a = 5 * nw6a + 4 * gap6a

    cluster(ax, X_AGE, (c_em_top + c_em_bot) / 2,
            total_em_a + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    ae_esx = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    for x, lab in zip(ae_esx, ['1by1matched', 'caliper_group']):
        node(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'])
    for o in ox:
        for ex in ae_esx:
            line(ax, o, y_a_out + NODE_H / 2, ex, y_em - NODE_H / 2)

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
    for x, lab in zip(ae_msx,
                      ['no_priority', 'priority_acs', 'priority_nad']):
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
    # face / mirrored_face (shared)
    # ════════════════════════════════════
    cluster(ax, x_face, (c_mir_top + c_mir_bot) / 2,
            nw_mir + 0.6, c_mir_bot - c_mir_top, C3['bg'])
    node(ax, x_face, y_mir, nw_mir, NODE_H, 'face', C3['nd'])
    cluster(ax, x_mirr, (c_mir_top + c_mir_bot) / 2,
            nw_mir + 0.6, c_mir_bot - c_mir_top, C3['bg'])
    node(ax, x_mirr, y_mir, nw_mir, NODE_H, 'mirrored_face', C3['nd'])
    for bx in bgx:
        line(ax, bx, y_pre_bg + NODE_H / 2, x_face, y_mir - NODE_H / 2)
        line(ax, bx, y_pre_bg + NODE_H / 2, x_mirr, y_mir - NODE_H / 2)

    # ════════════════════════════════════
    # Embedding Models (shared)
    # ════════════════════════════════════
    cluster(ax, CX_V2, (c_emb_top + c_emb_bot) / 2,
            total_emb + 0.9, c_emb_bot - c_emb_top, C2['bg'])
    for x, lab in zip(emx, ['dlib', 'TopoFR', 'ArcFace', 'VGGFace']):
        node(ax, x, y_emb, nw_emb, NODE_H, lab, C2['nd'])
    for src in [x_face, x_mirr]:
        for ex in emx:
            line(ax, src, y_mir + NODE_H / 2, ex, y_emb - NODE_H / 2)

    # ════════════════════════════════════
    # Embedding branch: original (center)
    # ════════════════════════════════════
    ft_orig_cl_w = nw_ft + 0.6
    cluster(ax, CX_V2, (c_ft_top + c_ft_bot) / 2,
            ft_orig_cl_w, c_ft_bot - c_ft_top, C_FT['bg'])
    node(ax, CX_V2, y_ft, nw_ft, NODE_H, 'original', C_FT['nd'])
    for ex in emx:
        line(ax, ex, y_emb + NODE_H / 2, CX_V2, y_ft - NODE_H / 2)

    cluster(ax, CX_V2, (c_pa_top + c_pa_bot) / 2,
            total_pa + 0.9, c_pa_bot - c_pa_top, C_PA['bg'])
    pax = [CX_V2 - (nw_pa + gap_pa) / 2, CX_V2 + (nw_pa + gap_pa) / 2]
    for x, lab in zip(pax, ['mean', 'all']):
        node(ax, x, y_pa, nw_pa, NODE_H, lab, C_PA['nd'])
    for px in pax:
        line(ax, CX_V2, y_ft + NODE_H / 2, px, y_pa - NODE_H / 2)

    cluster(ax, CX_V2, (c_pd_top + c_pd_bot) / 2,
            total_pd + 0.9, c_pd_bot - c_pd_top, C_PD['bg'])
    pdx = [CX_V2 - (nw_pd + gap_pd), CX_V2, CX_V2 + (nw_pd + gap_pd)]
    for x, lab in zip(pdx, ['no_drop', 'PCA', 'DropCorr']):
        node(ax, x, y_pd, nw_pd, NODE_H, lab, C_PD['nd'])
    for px in pax:
        for dx in pdx:
            line(ax, px, y_pa + NODE_H / 2, dx, y_pd - NODE_H / 2)

    cluster(ax, CX_V2, (c4_top + c4_bot) / 2,
            total4 + 0.9, c4_bot - c4_top, C4['bg'])
    clx = [CX_V2 - (nw4 + gap4) / 2, CX_V2 + (nw4 + gap4) / 2]
    for x, lab in zip(clx, [
        'Logistic Regression\n$C \\in \\{10^{-3}\\,.\\,.\\,10^{2}\\}$',
        'XGBoost\nn_tree x max_depth x lr\n'
        '{200,500,1k}x{3,6,9}x{.05,.1,.2}']):
        node(ax, x, y_clf, nw4, NODE_H, lab, C4['nd'])
    for dx in pdx:
        for ci in clx:
            line(ax, dx, y_pd + NODE_H / 2, ci, y_clf - NODE_H / 2)

    xl = CX_V2 - (cw + gap_col) / 2
    xr = CX_V2 + (cw + gap_col) / 2
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

    # Embedding eval chain
    total_em = 2 * nw_ev + gap_ev
    cluster(ax, CX_V2, (c_em_top + c_em_bot) / 2,
            total_em + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    esx = [CX_V2 - (nw_ev + gap_ev) / 2, CX_V2 + (nw_ev + gap_ev) / 2]
    for x, lab in zip(esx, ['1by1matched', 'caliper_group']):
        node(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'])
    for src in [xl, xr]:
        for ex in esx:
            line(ax, src, y_r3 + NODE_H / 2, ex, y_em - NODE_H / 2)

    total_ml = 2 * nw_ev + gap_ev
    cluster(ax, CX_V2, (c_ml_top + c_ml_bot) / 2,
            total_ml + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    mlx = [CX_V2 - (nw_ev + gap_ev) / 2, CX_V2 + (nw_ev + gap_ev) / 2]
    for x, lab in zip(mlx, ['subject_match', 'visit_match']):
        node(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'])
    for ex in esx:
        for mx in mlx:
            line(ax, ex, y_em + NODE_H / 2, mx, y_ml - NODE_H / 2)

    total_sa = 2 * nw_ev + gap_ev
    cluster(ax, CX_V2, (c_sa_top + c_sa_bot) / 2,
            total_sa + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    sax = [CX_V2 - (nw_ev + gap_ev) / 2, CX_V2 + (nw_ev + gap_ev) / 2]
    for x, lab in zip(sax, ['eval_by_subject', 'eval_by_visit']):
        node(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'])
    for mx in mlx:
        for sx in sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    cluster(ax, CX_V2, (c_ms_top + c_ms_bot) / 2,
            total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX_V2 - (nw_ms + gap_ms), CX_V2, CX_V2 + (nw_ms + gap_ms)]
    for x, lab in zip(msx, ['no_priority', 'priority_acs', 'priority_nad']):
        node(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'])
    for sx in sax:
        for mx in msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    nw6 = 1.65; gap6 = 0.2
    total6 = 5 * nw6 + 4 * gap6
    cluster(ax, CX_V2, (c6_top + c6_bot) / 2,
            total6 + 0.7, c6_bot - c6_top, C6['bg'])
    ppx = [CX_V2 + (i - 2) * (nw6 + gap6) for i in range(5)]
    for x, lab in zip(ppx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                             'MMSE hi/lo', 'CASI hi/lo']):
        node(ax, x, y_p, nw6, NODE_H, lab, C6['nd'])
    for mx in msx:
        for px_i in ppx:
            line(ax, mx, y_ms + NODE_H / 2, px_i, y_p - NODE_H / 2)

    # ════════════════════════════════════
    # Asymmetry branch (right)
    # ════════════════════════════════════
    asym_cl_w = total_asym + 0.6
    cluster(ax, X_ASYM, (c_ft_top + c_ft_bot) / 2,
            asym_cl_w, c_ft_bot - c_ft_top, C_FT['bg'])
    for x, lab in zip(x_ft_asym,
                      ['diff', '|diff|', 'rel_diff', '|rel_diff|']):
        node(ax, x, y_ft, nw_ft, NODE_H, lab, C_FT['nd'])
    for ex in emx:
        for fx in x_ft_asym:
            line(ax, ex, y_emb + NODE_H / 2, fx, y_ft - NODE_H / 2)

    cluster(ax, X_ASYM, (c_pa_top + c_pa_bot) / 2,
            asym_nw + 0.6, c_pa_bot - c_pa_top, C_ASY['bg'])
    node(ax, X_ASYM, y_pa, asym_nw, NODE_H, 'mean', C_ASY['nd'])
    for fx in x_ft_asym:
        line(ax, fx, y_ft + NODE_H / 2, X_ASYM, y_pa - NODE_H / 2)

    nw_sc = 3.0; gap_sc = 0.4
    total_sc = 3 * nw_sc + 2 * gap_sc
    cluster(ax, X_ASYM, (c_pd_top + c_pd_bot) / 2,
            total_sc + 0.6, c_pd_bot - c_pd_top, C_ASY['bg'])
    scx = [X_ASYM - (nw_sc + gap_sc), X_ASYM, X_ASYM + (nw_sc + gap_sc)]
    sc_labels = [
        'L2 Norm\n$\\sqrt{\\Sigma_i f_i^2}$',
        'Centroid Distance\n$\\Delta\\cos(x, \\mu)$',
        'LDA Projection\nFisher 1D',
    ]
    for x, lab in zip(scx, sc_labels):
        node(ax, x, y_pd, nw_sc, NODE_H, lab, C_ASY['nd'], fs=FS_SM)
    for sx in scx:
        line(ax, X_ASYM, y_pa + NODE_H / 2, sx, y_pd - NODE_H / 2)

    # Asymmetry Fwd / Rev
    xl_a = X_ASYM - (cw + gap_col) / 2
    xr_a = X_ASYM + (cw + gap_col) / 2
    cluster(ax, xl_a, (cl_top + cl_bot) / 2, cw, cl_h, CF['bg'])
    node(ax, xl_a, y_r1, nw_col, NODE_H, "Full Cohort", CF['nd'])
    node(ax, xl_a, y_r2, nw_col, NODE_H, "OOF Scores", CF['nd'])
    node(ax, xl_a, y_r3, nw_col, NODE_H,
         "Full Cohort Eval\nMatched Subset Eval (1:1)", CF['nd'])
    cluster(ax, xr_a, (cl_top + cl_bot) / 2, cw, cl_h, CR['bg'])
    node(ax, xr_a, y_r1, nw_col, NODE_H, "Matched Cohort", CR['nd'])
    node(ax, xr_a, y_r2, nw_col, NODE_H, "Predict Full Cohort", CR['nd'])
    node(ax, xr_a, y_r3, nw_col, NODE_H,
         "Matched OOF Eval\nUnmatched Eval", CR['nd'])
    for sx in scx:
        line(ax, sx, y_pd + NODE_H / 2, xl_a, y_r1 - NODE_H / 2)
        line(ax, sx, y_pd + NODE_H / 2, xr_a, y_r1 - NODE_H / 2)
    for col in [xl_a, xr_a]:
        for ya, yb in [(y_r1, y_r2), (y_r2, y_r3)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)

    # Asymmetry eval chain
    cluster(ax, X_ASYM, (c_em_top + c_em_bot) / 2,
            total_em + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    a_esx = [X_ASYM - (nw_ev + gap_ev) / 2,
             X_ASYM + (nw_ev + gap_ev) / 2]
    for x, lab in zip(a_esx, ['1by1matched', 'caliper_group']):
        node(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'])
    for src in [xl_a, xr_a]:
        for ex in a_esx:
            line(ax, src, y_r3 + NODE_H / 2, ex, y_em - NODE_H / 2)

    cluster(ax, X_ASYM, (c_ml_top + c_ml_bot) / 2,
            total_ml + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    a_mlx = [X_ASYM - (nw_ev + gap_ev) / 2,
             X_ASYM + (nw_ev + gap_ev) / 2]
    for x, lab in zip(a_mlx, ['subject_match', 'visit_match']):
        node(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'])
    for ex in a_esx:
        for mx in a_mlx:
            line(ax, ex, y_em + NODE_H / 2, mx, y_ml - NODE_H / 2)

    cluster(ax, X_ASYM, (c_sa_top + c_sa_bot) / 2,
            total_sa + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    a_sax = [X_ASYM - (nw_ev + gap_ev) / 2,
             X_ASYM + (nw_ev + gap_ev) / 2]
    for x, lab in zip(a_sax, ['eval_by_subject', 'eval_by_visit']):
        node(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'])
    for mx in a_mlx:
        for sx in a_sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    cluster(ax, X_ASYM, (c_ms_top + c_ms_bot) / 2,
            total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    a_msx = [X_ASYM - (nw_ms + gap_ms), X_ASYM,
             X_ASYM + (nw_ms + gap_ms)]
    for x, lab in zip(a_msx,
                      ['no_priority', 'priority_acs', 'priority_nad']):
        node(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'])
    for sx in a_sax:
        for mx in a_msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    cluster(ax, X_ASYM, (c6_top + c6_bot) / 2,
            total6 + 0.7, c6_bot - c6_top, C6['bg'])
    a_ppx = [X_ASYM + (i - 2) * (nw6 + gap6) for i in range(5)]
    for x, lab in zip(a_ppx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                                'MMSE hi/lo', 'CASI hi/lo']):
        node(ax, x, y_p, nw6, NODE_H, lab, C6['nd'])
    for mx in a_msx:
        for px_i in a_ppx:
            line(ax, mx, y_ms + NODE_H / 2, px_i, y_p - NODE_H / 2)

    out = OUT / "age_emb_pipeline_mpl_v2.png"
    fig.savefig(out, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


def build_v2_show():
    """v2 highlighted-path version."""
    CX_V2 = 16.5
    X_ASYM = 27.0
    age_w = 4.5; age_h = NODE_H

    CL = _cohort_layout()
    c1_top = CL['c1_top']; y_vs = CL['y_vs']; y_cf = CL['y_cf']
    y_fc = CL['y_fc']; c1_bot = CL['c1_bot']

    nw_bg = 2.4; gap_bg = 0.65
    c_pre_top = c1_bot + SP
    y_pre1 = c_pre_top + SP + NODE_H / 2
    y_pre2 = y_pre1 + NODE_H + SP
    y_pre3 = y_pre2 + NODE_H + SP
    y_pre_list = [y_pre1, y_pre2, y_pre3]
    y_pre_bg = y_pre3 + NODE_H + SP
    c_pre_bot = y_pre_bg + NODE_H / 2 + SP
    pre_cluster_w = max(NW_PRE, 2 * nw_bg + gap_bg) + 0.9

    nw_mir = 2.4; gap_mir = 1.0
    c_mir_top = c_pre_bot + SP
    y_mir = c_mir_top + SP + NODE_H / 2
    c_mir_bot = y_mir + NODE_H / 2 + SP
    x_face = CX_V2 - (nw_mir + gap_mir) / 2
    x_mirr = CX_V2 + (nw_mir + gap_mir) / 2

    nw_emb = 2.2; gap_emb = 0.5
    total_emb = 4 * nw_emb + 3 * gap_emb
    c_emb_top = c_mir_bot + SP
    y_emb = c_emb_top + SP + NODE_H / 2
    c_emb_bot = y_emb + NODE_H / 2 + SP
    emx = [CX_V2 + (i - 1.5) * (nw_emb + gap_emb) for i in range(4)]

    nw_ft = 1.7; gap_ft = 0.25
    c_ft_top = c_emb_bot + SP
    y_ft = c_ft_top + SP + NODE_H / 2
    c_ft_bot = y_ft + NODE_H / 2 + SP

    nw_pa = 2.4; gap_pa = 0.65
    total_pa = 2 * nw_pa + gap_pa
    c_pa_top = c_ft_bot + SP
    y_pa = c_pa_top + SP + NODE_H / 2
    c_pa_bot = y_pa + NODE_H / 2 + SP

    nw_pd = 2.0; gap_pd = 0.5
    total_pd = 3 * nw_pd + 2 * gap_pd
    c_pd_top = c_pa_bot + SP
    y_pd = c_pd_top + SP + NODE_H / 2
    c_pd_bot = y_pd + NODE_H / 2 + SP

    nw4 = 4.2; gap4 = 0.5
    total4 = 2 * nw4 + gap4
    c4_top = c_pd_bot + SP
    y_clf = c4_top + SP + NODE_H / 2
    c4_bot = y_clf + NODE_H / 2 + SP

    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
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
    nw_ms = 2.4; gap_ms = 0.65
    total_ms = 3 * nw_ms + 2 * gap_ms
    c_ms_top = c_sa_bot + SP
    y_ms = c_ms_top + SP + NODE_H / 2
    c_ms_bot = y_ms + NODE_H / 2 + SP
    c6_top = c_ms_bot + SP
    y_p = c6_top + SP + NODE_H / 2
    c6_bot = y_p + NODE_H / 2 + SP

    amx = [X_AGE + (i - 2) * (NW_AM + GAP_AM) for i in range(5)]
    cal_top = c4_top
    y_cal1 = cal_top + SP + NODE_H / 2
    y_cal2 = y_cal1 + NODE_H + SP
    y_cal3 = y_cal2 + NODE_H + SP
    cal_bot = y_cal3 + NODE_H / 2 + SP
    y_a_out = y_r3
    c_ao_top = y_a_out - NODE_H / 2 - SP
    c_ao_bot = y_a_out + NODE_H / 2 + SP

    total_asym = 4 * nw_ft + 3 * gap_ft
    x_ft_asym = [X_ASYM - total_asym / 2 + nw_ft / 2
                 + i * (nw_ft + gap_ft) for i in range(4)]
    asym_nw = 3.5

    total_em = 2 * nw_ev + gap_ev
    total_ml = 2 * nw_ev + gap_ev
    total_sa = 2 * nw_ev + gap_ev

    fig_h = c6_bot + 0.5
    fig, ax = plt.subplots(figsize=(36.0, fig_h))
    ax.set_xlim(-4.5, 34)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def _n(ax, cx, cy, w, h, text, fc_on, on=True):
        node(ax, cx, cy, w, h, text, fc_on if on else G['nd'])

    def _cl(ax, cx, cy, w, h, fc_on, on=True):
        cluster(ax, cx, cy, w, h, fc_on if on else G['bg'])

    # ════ Cohort ════
    _cl(ax, CX_TOP, (c1_top + c1_bot) / 2,
        CL['c1_w'], c1_bot - c1_top, C1['bg'])
    vsx = [CX_TOP + (i - 1) * (CL['nw_vs'] + CL['gap_vs'])
           for i in range(3)]
    for x, lab, on in zip(vsx, VS_LABELS, [False, True, False]):
        _n(ax, x, y_vs, CL['nw_vs'], NODE_H, lab, C1['nd'], on)
    cdrx = [CX_TOP + (i - 1) * (CL['nw_cdr'] + CL['gap_cdr'])
            for i in range(3)]
    for x, lab, on in zip(cdrx, CDR_LABELS, [True, False, False]):
        _n(ax, x, y_cf, CL['nw_cdr'], NODE_H, lab, C1['nd'], on)
    _n(ax, CX_TOP, y_fc, CL['nw_co'], NODE_H,
       "Cohort (P + NAD + ACS)", C1['nd'])
    for vx in vsx:
        for cx in cdrx:
            line(ax, vx, y_vs + NODE_H / 2, cx, y_cf - NODE_H / 2)
    for cx in cdrx:
        line(ax, cx, y_cf + NODE_H / 2, CX_TOP, y_fc - NODE_H / 2)

    # ════ Preprocessing ════
    _cl(ax, CX_TOP, (c_pre_top + c_pre_bot) / 2,
        pre_cluster_w, c_pre_bot - c_pre_top, C_PRE['bg'])
    for y, lab in zip(y_pre_list, PRE_LABELS):
        _n(ax, CX_TOP, y, NW_PRE, NODE_H, lab, C_PRE['nd'])
    line(ax, CX_TOP, y_fc + NODE_H / 2, CX_TOP, y_pre_list[0] - NODE_H / 2)
    for i in range(2):
        line(ax, CX_TOP, y_pre_list[i] + NODE_H / 2,
             CX_TOP, y_pre_list[i + 1] - NODE_H / 2)
    bgx = [CX_TOP - (nw_bg + gap_bg) / 2, CX_TOP + (nw_bg + gap_bg) / 2]
    for x, lab, on in zip(bgx, ['no_background', 'background'],
                          [False, True]):
        _n(ax, x, y_pre_bg, nw_bg, NODE_H, lab, C_PRE['nd'], on)
    for bx in bgx:
        line(ax, CX_TOP, y_pre_list[-1] + NODE_H / 2,
             bx, y_pre_bg - NODE_H / 2)

    # ════ EACS (gray) ════
    eacs_w = age_w + 0.6
    _cl(ax, X_AGE, y_fc, eacs_w + 0.6, NODE_H + 2 * SP,
        C_EACS['bg'], False)
    _n(ax, X_AGE, y_fc, eacs_w, NODE_H,
       "External Datasets\nUTKFace / AgeDB / APPA-REAL / ...",
       C_EACS['nd'], False)

    # ════ Age branch ════
    pred_cw = TOTAL_AM + 0.9
    am_top = y_emb - age_h / 2 - SP
    am_bot = y_emb + age_h / 2 + SP
    _cl(ax, X_AGE, y_emb, pred_cw, am_bot - am_top, C_AGE['bg'])
    for x, lab, on in zip(amx, AGE_MODELS,
                          [True, False, False, False, False]):
        _n(ax, x, y_emb, NW_AM, age_h, lab, C_AGE['nd'], on)
    for bx in bgx:
        for x in amx:
            line(ax, bx, y_pre_bg + NODE_H / 2, x, y_emb - NODE_H / 2)
    for x in amx:
        line(ax, X_AGE, y_fc + NODE_H / 2, x, y_emb - age_h / 2)

    _cl(ax, X_AGE, y_ft, age_w + 0.6, NODE_H + 2 * SP, C_AGE['bg'])
    _n(ax, X_AGE, y_ft, age_w, age_h, "Predict Age", C_AGE['nd'])
    for x in amx:
        line(ax, x, y_emb + age_h / 2, X_AGE, y_ft - age_h / 2)

    _cl(ax, X_AGE, y_pa, age_w + 0.6, NODE_H + 2 * SP, C_AGE['bg'])
    _n(ax, X_AGE, y_pa, age_w, age_h, "mean", C_AGE['nd'])
    line(ax, X_AGE, y_ft + age_h / 2, X_AGE, y_pa - age_h / 2)

    ae_node_w = 5.5
    _cl(ax, X_AGE, y_pd, ae_node_w + 0.9, NODE_H + 2 * SP, C_AGE['bg'])
    _n(ax, X_AGE, y_pd, ae_node_w, age_h,
       "age_error = real_age - predicted_age", C_AGE['nd'])
    line(ax, X_AGE, y_pa + age_h / 2, X_AGE, y_pd - age_h / 2)

    cal_h = cal_bot - cal_top
    _cl(ax, X_AGE, (cal_top + cal_bot) / 2,
        age_w + 0.6, cal_h, C_AGE['bg'], False)
    _n(ax, X_AGE, y_cal1, age_w, age_h,
       "Logistic 10-fold\n(90/10 + 10/90, 30 seeds)", C_AGE['nd'], False)
    _n(ax, X_AGE, y_cal2, age_w, age_h,
       "Bootstrap\n(NAD age>=60, x1000)", C_AGE['nd'], False)
    _n(ax, X_AGE, y_cal3, age_w, age_h,
       "Mean Correction\n(33 age bins, single fit)", C_AGE['nd'], False)
    line(ax, X_AGE, y_pd + age_h / 2, X_AGE, y_cal1 - age_h / 2)
    line(ax, X_AGE, y_cal1 + age_h / 2, X_AGE, y_cal2 - age_h / 2)
    line(ax, X_AGE, y_cal2 + age_h / 2, X_AGE, y_cal3 - age_h / 2)

    out_w = 2.4; out_gap = 0.25
    out_labels = ['predicted_age', 'corrected_age', 'age_error']
    n_out = len(out_labels)
    total_out = n_out * out_w + (n_out - 1) * out_gap
    ox = [X_AGE - total_out / 2 + out_w / 2 + i * (out_w + out_gap)
          for i in range(n_out)]
    cluster(ax, X_AGE, y_a_out, total_out + 0.7,
            c_ao_bot - c_ao_top, C_AOUT['bg'])
    for x, lab, on in zip(ox, out_labels, [True, False, True]):
        _n(ax, x, y_a_out, out_w, NODE_H, lab, C_AOUT['nd'], on)
    for x in ox:
        line(ax, X_AGE, y_cal3 + age_h / 2, x, y_a_out - NODE_H / 2)

    # Age eval chain
    total_em_a = 2 * nw_ev + gap_ev
    total_ms_a = 3 * nw_ms + 2 * gap_ms
    nw6a = 1.65; gap6a = 0.2
    total_p_a = 5 * nw6a + 4 * gap6a

    _cl(ax, X_AGE, (c_em_top + c_em_bot) / 2,
        total_em_a + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    ae_esx = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    for x, lab, on in zip(ae_esx, ['1by1matched', 'caliper_group'],
                          [True, False]):
        _n(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'], on)
    for o in ox:
        for ex in ae_esx:
            line(ax, o, y_a_out + NODE_H / 2, ex, y_em - NODE_H / 2)

    _cl(ax, X_AGE, (c_ml_top + c_ml_bot) / 2,
        total_em_a + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    ae_mlx = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    for x, lab, on in zip(ae_mlx, ['subject_match', 'visit_match'],
                          [True, False]):
        _n(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'], on)
    for ex in ae_esx:
        for mx in ae_mlx:
            line(ax, ex, y_em + NODE_H / 2, mx, y_ml - NODE_H / 2)

    _cl(ax, X_AGE, (c_sa_top + c_sa_bot) / 2,
        total_em_a + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    ae_sax = [X_AGE - (nw_ev + gap_ev) / 2, X_AGE + (nw_ev + gap_ev) / 2]
    for x, lab, on in zip(ae_sax, ['eval_by_subject', 'eval_by_visit'],
                          [False, True]):
        _n(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'], on)
    for mx in ae_mlx:
        for sx in ae_sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    _cl(ax, X_AGE, (c_ms_top + c_ms_bot) / 2,
        total_ms_a + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    ae_msx = [X_AGE - (nw_ms + gap_ms), X_AGE, X_AGE + (nw_ms + gap_ms)]
    for x, lab, on in zip(ae_msx,
                          ['no_priority', 'priority_acs', 'priority_nad'],
                          [False, True, False]):
        _n(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'], on)
    for sx in ae_sax:
        for mx in ae_msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    _cl(ax, X_AGE, (c6_top + c6_bot) / 2,
        total_p_a + 0.7, c6_bot - c6_top, C6['bg'])
    ae_ppx = [X_AGE + (i - 2) * (nw6a + gap6a) for i in range(5)]
    for x, lab in zip(ae_ppx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                                 'MMSE hi/lo', 'CASI hi/lo']):
        _n(ax, x, y_p, nw6a, NODE_H, lab, C6['nd'])
    for mx in ae_msx:
        for px in ae_ppx:
            line(ax, mx, y_ms + NODE_H / 2, px, y_p - NODE_H / 2)

    # ════ face / mirrored_face ════
    _cl(ax, x_face, (c_mir_top + c_mir_bot) / 2,
        nw_mir + 0.6, c_mir_bot - c_mir_top, C3['bg'])
    _n(ax, x_face, y_mir, nw_mir, NODE_H, 'face', C3['nd'])
    _cl(ax, x_mirr, (c_mir_top + c_mir_bot) / 2,
        nw_mir + 0.6, c_mir_bot - c_mir_top, C3['bg'])
    _n(ax, x_mirr, y_mir, nw_mir, NODE_H, 'mirrored_face', C3['nd'])
    for bx in bgx:
        line(ax, bx, y_pre_bg + NODE_H / 2, x_face, y_mir - NODE_H / 2)
        line(ax, bx, y_pre_bg + NODE_H / 2, x_mirr, y_mir - NODE_H / 2)

    # ════ Embedding Models (ArcFace on) ════
    _cl(ax, CX_V2, (c_emb_top + c_emb_bot) / 2,
        total_emb + 0.9, c_emb_bot - c_emb_top, C2['bg'])
    for x, lab, on in zip(emx, ['dlib', 'TopoFR', 'ArcFace', 'VGGFace'],
                          [False, False, True, False]):
        _n(ax, x, y_emb, nw_emb, NODE_H, lab, C2['nd'], on)
    for src in [x_face, x_mirr]:
        for ex in emx:
            line(ax, src, y_mir + NODE_H / 2, ex, y_emb - NODE_H / 2)

    # ════ Embedding branch: original (on) ════
    ft_orig_cl_w = nw_ft + 0.6
    _cl(ax, CX_V2, (c_ft_top + c_ft_bot) / 2,
        ft_orig_cl_w, c_ft_bot - c_ft_top, C_FT['bg'])
    _n(ax, CX_V2, y_ft, nw_ft, NODE_H, 'original', C_FT['nd'])
    for ex in emx:
        line(ax, ex, y_emb + NODE_H / 2, CX_V2, y_ft - NODE_H / 2)

    _cl(ax, CX_V2, (c_pa_top + c_pa_bot) / 2,
        total_pa + 0.9, c_pa_bot - c_pa_top, C_PA['bg'])
    pax = [CX_V2 - (nw_pa + gap_pa) / 2, CX_V2 + (nw_pa + gap_pa) / 2]
    for x, lab, on in zip(pax, ['mean', 'all'], [True, False]):
        _n(ax, x, y_pa, nw_pa, NODE_H, lab, C_PA['nd'], on)
    for px in pax:
        line(ax, CX_V2, y_ft + NODE_H / 2, px, y_pa - NODE_H / 2)

    _cl(ax, CX_V2, (c_pd_top + c_pd_bot) / 2,
        total_pd + 0.9, c_pd_bot - c_pd_top, C_PD['bg'])
    pdx = [CX_V2 - (nw_pd + gap_pd), CX_V2, CX_V2 + (nw_pd + gap_pd)]
    for x, lab, on in zip(pdx, ['no_drop', 'PCA', 'DropCorr'],
                          [True, False, False]):
        _n(ax, x, y_pd, nw_pd, NODE_H, lab, C_PD['nd'], on)
    for px in pax:
        for dx in pdx:
            line(ax, px, y_pa + NODE_H / 2, dx, y_pd - NODE_H / 2)

    _cl(ax, CX_V2, (c4_top + c4_bot) / 2,
        total4 + 0.9, c4_bot - c4_top, C4['bg'])
    clx = [CX_V2 - (nw4 + gap4) / 2, CX_V2 + (nw4 + gap4) / 2]
    for x, lab in zip(clx, [
        'Logistic Regression\n$C \\in \\{10^{-3}\\,.\\,.\\,10^{2}\\}$',
        'XGBoost\nn_tree x max_depth x lr\n'
        '{200,500,1k}x{3,6,9}x{.05,.1,.2}']):
        _n(ax, x, y_clf, nw4, NODE_H, lab, C4['nd'])
    for dx in pdx:
        for ci in clx:
            line(ax, dx, y_pd + NODE_H / 2, ci, y_clf - NODE_H / 2)

    xl = CX_V2 - (cw + gap_col) / 2
    xr = CX_V2 + (cw + gap_col) / 2
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

    # Embedding eval chain
    _cl(ax, CX_V2, (c_em_top + c_em_bot) / 2,
        total_em + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    esx = [CX_V2 - (nw_ev + gap_ev) / 2, CX_V2 + (nw_ev + gap_ev) / 2]
    for x, lab, on in zip(esx, ['1by1matched', 'caliper_group'],
                          [True, False]):
        _n(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'], on)
    for src in [xl, xr]:
        for ex in esx:
            line(ax, src, y_r3 + NODE_H / 2, ex, y_em - NODE_H / 2)

    _cl(ax, CX_V2, (c_ml_top + c_ml_bot) / 2,
        total_ml + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    mlx = [CX_V2 - (nw_ev + gap_ev) / 2, CX_V2 + (nw_ev + gap_ev) / 2]
    for x, lab, on in zip(mlx, ['subject_match', 'visit_match'],
                          [True, False]):
        _n(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'], on)
    for ex in esx:
        for mx in mlx:
            line(ax, ex, y_em + NODE_H / 2, mx, y_ml - NODE_H / 2)

    _cl(ax, CX_V2, (c_sa_top + c_sa_bot) / 2,
        total_sa + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    sax = [CX_V2 - (nw_ev + gap_ev) / 2, CX_V2 + (nw_ev + gap_ev) / 2]
    for x, lab, on in zip(sax, ['eval_by_subject', 'eval_by_visit'],
                          [False, True]):
        _n(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'], on)
    for mx in mlx:
        for sx in sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    _cl(ax, CX_V2, (c_ms_top + c_ms_bot) / 2,
        total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX_V2 - (nw_ms + gap_ms), CX_V2, CX_V2 + (nw_ms + gap_ms)]
    for x, lab, on in zip(msx,
                          ['no_priority', 'priority_acs', 'priority_nad'],
                          [False, True, False]):
        _n(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'], on)
    for sx in sax:
        for mx in msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    nw6 = 1.65; gap6 = 0.2
    total6 = 5 * nw6 + 4 * gap6
    _cl(ax, CX_V2, (c6_top + c6_bot) / 2,
        total6 + 0.7, c6_bot - c6_top, C6['bg'])
    ppx = [CX_V2 + (i - 2) * (nw6 + gap6) for i in range(5)]
    for x, lab, on in zip(ppx,
                          ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                           'MMSE hi/lo', 'CASI hi/lo'],
                          [True, True, True, False, False]):
        _n(ax, x, y_p, nw6, NODE_H, lab, C6['nd'], on)
    for mx in msx:
        for px_i in ppx:
            line(ax, mx, y_ms + NODE_H / 2, px_i, y_p - NODE_H / 2)

    # ════ Asymmetry branch (all on) ════
    asym_cl_w = total_asym + 0.6
    _cl(ax, X_ASYM, (c_ft_top + c_ft_bot) / 2,
        asym_cl_w, c_ft_bot - c_ft_top, C_FT['bg'])
    for x, lab in zip(x_ft_asym,
                      ['diff', '|diff|', 'rel_diff', '|rel_diff|']):
        _n(ax, x, y_ft, nw_ft, NODE_H, lab, C_FT['nd'])
    for ex in emx:
        for fx in x_ft_asym:
            line(ax, ex, y_emb + NODE_H / 2, fx, y_ft - NODE_H / 2)

    _cl(ax, X_ASYM, (c_pa_top + c_pa_bot) / 2,
        asym_nw + 0.6, c_pa_bot - c_pa_top, C_ASY['bg'])
    _n(ax, X_ASYM, y_pa, asym_nw, NODE_H, 'mean', C_ASY['nd'])
    for fx in x_ft_asym:
        line(ax, fx, y_ft + NODE_H / 2, X_ASYM, y_pa - NODE_H / 2)

    nw_sc = 3.0; gap_sc = 0.4
    total_sc = 3 * nw_sc + 2 * gap_sc
    _cl(ax, X_ASYM, (c_pd_top + c_pd_bot) / 2,
        total_sc + 0.6, c_pd_bot - c_pd_top, C_ASY['bg'])
    scx = [X_ASYM - (nw_sc + gap_sc), X_ASYM, X_ASYM + (nw_sc + gap_sc)]
    sc_labels = [
        'L2 Norm\n$\\sqrt{\\Sigma_i f_i^2}$',
        'Centroid Distance\n$\\Delta\\cos(x, \\mu)$',
        'LDA Projection\nFisher 1D',
    ]
    for x, lab in zip(scx, sc_labels):
        _n(ax, x, y_pd, nw_sc, NODE_H, lab, C_ASY['nd'])
    for sx in scx:
        line(ax, X_ASYM, y_pa + NODE_H / 2, sx, y_pd - NODE_H / 2)

    # Asymmetry Fwd (on) / Rev (off)
    xl_a = X_ASYM - (cw + gap_col) / 2
    xr_a = X_ASYM + (cw + gap_col) / 2
    _cl(ax, xl_a, (cl_top + cl_bot) / 2, cw, cl_h, CF['bg'])
    _n(ax, xl_a, y_r1, nw_col, NODE_H, "Full Cohort", CF['nd'])
    _n(ax, xl_a, y_r2, nw_col, NODE_H, "OOF Scores", CF['nd'])
    _n(ax, xl_a, y_r3, nw_col, NODE_H,
       "Full Cohort Eval\nMatched Subset Eval (1:1)", CF['nd'])
    _cl(ax, xr_a, (cl_top + cl_bot) / 2, cw, cl_h, CR['bg'], False)
    _n(ax, xr_a, y_r1, nw_col, NODE_H, "Matched Cohort", CR['nd'], False)
    _n(ax, xr_a, y_r2, nw_col, NODE_H,
       "Predict Full Cohort", CR['nd'], False)
    _n(ax, xr_a, y_r3, nw_col, NODE_H,
       "Matched OOF Eval\nUnmatched Eval", CR['nd'], False)
    for sx in scx:
        line(ax, sx, y_pd + NODE_H / 2, xl_a, y_r1 - NODE_H / 2)
        line(ax, sx, y_pd + NODE_H / 2, xr_a, y_r1 - NODE_H / 2)
    for col in [xl_a, xr_a]:
        for ya, yb in [(y_r1, y_r2), (y_r2, y_r3)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)

    # Asymmetry eval chain
    _cl(ax, X_ASYM, (c_em_top + c_em_bot) / 2,
        total_em + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    a_esx = [X_ASYM - (nw_ev + gap_ev) / 2,
             X_ASYM + (nw_ev + gap_ev) / 2]
    for x, lab, on in zip(a_esx, ['1by1matched', 'caliper_group'],
                          [True, False]):
        _n(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'], on)
    for src in [xl_a, xr_a]:
        for ex in a_esx:
            line(ax, src, y_r3 + NODE_H / 2, ex, y_em - NODE_H / 2)

    _cl(ax, X_ASYM, (c_ml_top + c_ml_bot) / 2,
        total_ml + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    a_mlx = [X_ASYM - (nw_ev + gap_ev) / 2,
             X_ASYM + (nw_ev + gap_ev) / 2]
    for x, lab, on in zip(a_mlx, ['subject_match', 'visit_match'],
                          [True, False]):
        _n(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'], on)
    for ex in a_esx:
        for mx in a_mlx:
            line(ax, ex, y_em + NODE_H / 2, mx, y_ml - NODE_H / 2)

    _cl(ax, X_ASYM, (c_sa_top + c_sa_bot) / 2,
        total_sa + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    a_sax = [X_ASYM - (nw_ev + gap_ev) / 2,
             X_ASYM + (nw_ev + gap_ev) / 2]
    for x, lab, on in zip(a_sax, ['eval_by_subject', 'eval_by_visit'],
                          [False, True]):
        _n(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'], on)
    for mx in a_mlx:
        for sx in a_sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    _cl(ax, X_ASYM, (c_ms_top + c_ms_bot) / 2,
        total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    a_msx = [X_ASYM - (nw_ms + gap_ms), X_ASYM,
             X_ASYM + (nw_ms + gap_ms)]
    for x, lab, on in zip(a_msx,
                          ['no_priority', 'priority_acs', 'priority_nad'],
                          [False, True, False]):
        _n(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'], on)
    for sx in a_sax:
        for mx in a_msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    _cl(ax, X_ASYM, (c6_top + c6_bot) / 2,
        total6 + 0.7, c6_bot - c6_top, C6['bg'])
    a_ppx = [X_ASYM + (i - 2) * (nw6 + gap6) for i in range(5)]
    for x, lab, on in zip(a_ppx,
                          ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                           'MMSE hi/lo', 'CASI hi/lo'],
                          [True, True, True, False, False]):
        _n(ax, x, y_p, nw6, NODE_H, lab, C6['nd'], on)
    for mx in a_msx:
        for px_i in a_ppx:
            line(ax, mx, y_ms + NODE_H / 2, px_i, y_p - NODE_H / 2)

    out = OUT / "age_emb_pipeline_mpl_v2_show.png"
    fig.savefig(out, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
    build_show()
    build_v2()
    build_v2_show()
