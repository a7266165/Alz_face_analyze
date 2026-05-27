"""Shared constants and helpers for age/embedding pipeline diagrams."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

__all__ = [
    'plt', 'FancyBboxPatch', 'Path',
    'OUT', 'FONT', 'FS', 'FS_HI', 'FS_SM', 'FS_XS', 'EC', 'NODE_H', 'SP',
    'C1', 'C2', 'C3', 'C4', 'C6', 'CF', 'CR',
    'C_FT', 'C_PA', 'C_PD', 'C_SA', 'C_ES', 'C_ML', 'C_MS',
    'C_AGE', 'C_PRE', 'C_EACS', 'C_AOUT', 'C_ASY', 'C_EMO', 'G',
    'AGE_MODELS', 'NW_AM', 'GAP_AM', 'TOTAL_AM',
    'FIG_W', 'X_LIM_LEFT', 'X_LIM_RIGHT', 'CX', 'X_AGE', 'CX_TOP',
    'PRE_LABELS', 'NW_PRE',
    '_box', 'node', 'cluster', 'line',
    '_cohort_layout', '_draw_eval_chain', '_common_layout',
    'VS_LABELS', 'CDR_LABELS',
]

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
C_EMO = dict(bg='#FCE8E8', nd='#E8B4B4')
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


def _draw_eval_chain(ax, cx, src_xs, src_ys, ey):
    """Draw per-branch eval chain (5 rows) + partition cluster at center cx."""
    nw_ev = 2.4; gap_ev = 0.65
    nw_ms = 2.4; gap_ms = 0.65
    nw6 = 1.65; gap6 = 0.2
    total_em = 2 * nw_ev + gap_ev
    total_ms = 3 * nw_ms + 2 * gap_ms
    total6 = 5 * nw6 + 4 * gap6

    cluster(ax, cx, (ey['c_em_top'] + ey['c_em_bot']) / 2,
            total_em + 0.9, ey['c_em_bot'] - ey['c_em_top'], C_ES['bg'])
    esx = [cx - (nw_ev + gap_ev) / 2, cx + (nw_ev + gap_ev) / 2]
    for x, lab in zip(esx, ['1by1matched', 'caliper_group']):
        node(ax, x, ey['y_em'], nw_ev, NODE_H, lab, C_ES['nd'])
    for sx, sy in zip(src_xs, src_ys):
        for ex in esx:
            line(ax, sx, sy, ex, ey['y_em'] - NODE_H / 2)

    cluster(ax, cx, (ey['c_ml_top'] + ey['c_ml_bot']) / 2,
            total_em + 0.9, ey['c_ml_bot'] - ey['c_ml_top'], C_ML['bg'])
    mlx = [cx - (nw_ev + gap_ev) / 2, cx + (nw_ev + gap_ev) / 2]
    for x, lab in zip(mlx, ['subject_match', 'visit_match']):
        node(ax, x, ey['y_ml'], nw_ev, NODE_H, lab, C_ML['nd'])
    for ex in esx:
        for mx in mlx:
            line(ax, ex, ey['y_em'] + NODE_H / 2, mx, ey['y_ml'] - NODE_H / 2)

    cluster(ax, cx, (ey['c_sa_top'] + ey['c_sa_bot']) / 2,
            total_em + 0.9, ey['c_sa_bot'] - ey['c_sa_top'], C_SA['bg'])
    sax = [cx - (nw_ev + gap_ev) / 2, cx + (nw_ev + gap_ev) / 2]
    for x, lab in zip(sax, ['eval_by_subject', 'eval_by_visit']):
        node(ax, x, ey['y_sa'], nw_ev, NODE_H, lab, C_SA['nd'])
    for mx in mlx:
        for sx in sax:
            line(ax, mx, ey['y_ml'] + NODE_H / 2, sx, ey['y_sa'] - NODE_H / 2)

    cluster(ax, cx, (ey['c_ms_top'] + ey['c_ms_bot']) / 2,
            total_ms + 0.9, ey['c_ms_bot'] - ey['c_ms_top'], C_MS['bg'])
    msx = [cx - (nw_ms + gap_ms), cx, cx + (nw_ms + gap_ms)]
    for x, lab in zip(msx, ['no_priority', 'priority_acs', 'priority_nad']):
        node(ax, x, ey['y_ms'], nw_ms, NODE_H, lab, C_MS['nd'])
    for sx in sax:
        for mx in msx:
            line(ax, sx, ey['y_sa'] + NODE_H / 2, mx, ey['y_ms'] - NODE_H / 2)

    cluster(ax, cx, (ey['c6_top'] + ey['c6_bot']) / 2,
            total6 + 0.7, ey['c6_bot'] - ey['c6_top'], C6['bg'])
    ppx = [cx + (i - 2) * (nw6 + gap6) for i in range(5)]
    for x, lab in zip(ppx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS',
                             'MMSE hi/lo', 'CASI hi/lo']):
        node(ax, x, ey['y_p'], nw6, NODE_H, lab, C6['nd'])
    for mx in msx:
        for px in ppx:
            line(ax, mx, ey['y_ms'] + NODE_H / 2, px, ey['y_p'] - NODE_H / 2)


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


