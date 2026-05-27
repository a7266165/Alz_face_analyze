"""v2: Three-branch pipeline diagram (Age + Embedding + Asymmetry).

Output:
  workspace/overview/age_emb_pipeline_mpl_v2.png
  workspace/overview/age_emb_pipeline_mpl_v2_show.png

Usage:
    python scripts/overview/draw_age_emb_pipeline_v2.py
"""
import matplotlib.pyplot as plt
from draw_age_emb_pipeline_common import *


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
    build_v2()
    build_v2_show()
