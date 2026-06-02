"""scripts/overview/age_emb_pipeline_mpl_refactor_v2.py
Refactor v2 (matplotlib): three parallel top-down branches (Age / embedding /
emotion) sharing one Preprocessing head.

Built incrementally — current scope stops at step 3 (per-branch features):
  step 1  Preprocessing cluster (shared head: Detect..face/mirror)
  step 2  per-branch models   (Age x5 / embedding x4 / emotion x9)
  step 3  per-branch features (Predict Age x10 / 5 emb feats / 10 emo feats)

(steps 4+ — demographic/cohort, outputs, Fwd/Rev, eval chain — to be added.)

Output:
  workspace/overview/age_emb_pipeline_mpl_refactor_v2.png

Usage:
    python scripts/overview/age_emb_pipeline_mpl_refactor_v2.py
"""
import matplotlib.pyplot as plt
from draw_age_emb_pipeline_common import *


_DC_PAD = 0.7
# height of the whole shared block (Demographic + Cohort + matched sub-tree:
# 1by1matched/caliper -> subject/visit match -> priority -> matched cohort),
# incl. outer pads. Mirrors the y-spacing in _demo_cohort.
_DC_H = 2 * _DC_PAD + 12 * NODE_H + 17 * SP + 0.6


def _rowx(cx, n, w, gap):
    """Return (x-positions, total-width) for n nodes centred on cx."""
    tot = n * w + (n - 1) * gap
    xs = [cx - tot / 2 + w / 2 + i * (w + gap) for i in range(n)]
    return xs, tot


def _demo_cohort(ax, cx, s4_top):
    """Draw the standalone Demographic+Cohort block (big outer cluster wrapping
    a Demographic sub-cluster on top and a Cohort sub-cluster below).
    `s4_top` is the top edge of the outer wrapper. Returns its bottom edge."""
    pad = _DC_PAD
    dg_top = s4_top + pad
    dy_above = dg_top + SP + NODE_H / 2
    dy_block = dy_above + NODE_H + SP
    dy_below = dy_block + NODE_H + SP
    dg_bot = dy_below + NODE_H / 2 + SP
    co_top = dg_bot + SP + 0.3
    cy_pv = co_top + SP + NODE_H / 2
    cy_hv = cy_pv + NODE_H + SP
    cy_ps = cy_hv + NODE_H + SP
    cy_hs = cy_ps + NODE_H + SP
    cy_fc = cy_hs + NODE_H + SP
    co_bot = cy_fc + NODE_H / 2 + SP
    mt_top = co_bot + SP + 0.3
    my = mt_top + SP + NODE_H / 2          # 1 by 1 matched / caliper_group
    my2 = my + NODE_H + SP                  # subject_match / visit_match
    my3 = my2 + NODE_H + SP                 # no_priority / priority_acs / priority_nad
    my4 = my3 + NODE_H + SP                 # matched cohort (output)
    mt_bot = my4 + NODE_H / 2 + SP
    s4_bot = mt_bot + pad

    DEMO_ABOVE = ['Group', 'Number', 'Photo_Session', 'Sex']
    DEMO_BELOW = ['Age', 'BMI', 'MMSE', 'CASI', 'Global_CDR']
    nw_dc = 2.4; gap_dc = 0.3
    dax, da_tot = _rowx(cx, len(DEMO_ABOVE), nw_dc, gap_dc)
    dbx, db_tot = _rowx(cx, len(DEMO_BELOW), nw_dc, gap_dc)
    demo_w = max(da_tot, db_tot) + 0.9
    psx, ps_tot = _rowx(cx, 4, 2.6, 0.3)
    hsx, hs_tot = _rowx(cx, 2, 3.6, 0.3)
    cohort_w = max(ps_tot, hs_tot, 4.2) + 0.9
    big_w = max(demo_w, cohort_w) + 1.6

    # outer wrapper (cool lavender frame — harmonises the purple Demographic
    # and blue Cohort sub-clusters; shared by all branches)
    cluster(ax, cx, (s4_top + s4_bot) / 2, big_w, s4_bot - s4_top, '#DCDDEE')

    # Demographic sub-cluster (top)
    cluster(ax, cx, (dg_top + dg_bot) / 2, demo_w, dg_bot - dg_top, C3['bg'])
    node(ax, cx, dy_block, 3.0, NODE_H, 'Demographic', C3['nd'])
    for x, lab in zip(dax, DEMO_ABOVE):
        node(ax, x, dy_above, nw_dc, NODE_H, lab, C3['nd'])
        line(ax, cx, dy_block - NODE_H / 2, x, dy_above + NODE_H / 2)
    for x, lab in zip(dbx, DEMO_BELOW):
        node(ax, x, dy_below, nw_dc, NODE_H, lab, C3['nd'])
        line(ax, cx, dy_block + NODE_H / 2, x, dy_below - NODE_H / 2)

    # Cohort sub-cluster (bottom)
    cluster(ax, cx, (co_top + co_bot) / 2, cohort_w, co_bot - co_top, C1['bg'])
    pvx, _ = _rowx(cx, 2, 2.0, 0.4)
    for x, lab in zip(pvx, ['P: first', 'P: all']):
        node(ax, x, cy_pv, 2.0, NODE_H, lab, C1['nd'])
    for x, lab in zip(pvx, ['HC: first', 'HC: all']):
        node(ax, x, cy_hv, 2.0, NODE_H, lab, C1['nd'])
    for px in pvx:
        for hx in pvx:
            line(ax, px, cy_pv + NODE_H / 2, hx, cy_hv - NODE_H / 2)
    for x, lab in zip(psx, ['P: CDR all', 'P: CDR >=0.5',
                            'P: CDR >=1', 'P: CDR >=2']):
        node(ax, x, cy_ps, 2.6, NODE_H, lab, C1['nd'])
    for hx in pvx:
        for sx in psx:
            line(ax, hx, cy_hv + NODE_H / 2, sx, cy_ps - NODE_H / 2)
    for x, lab in zip(hsx, ['HC: CDR all\nor MMSE all',
                            'HC: CDR =0\nor MMSE >=26']):
        node(ax, x, cy_hs, 3.6, NODE_H, lab, C1['nd'])
    for sx in psx:
        for hx2 in hsx:
            line(ax, sx, cy_ps + NODE_H / 2, hx2, cy_hs - NODE_H / 2)
    node(ax, cx, cy_fc, 4.2, NODE_H, 'Cohort (P + NAD + ACS)', C1['nd'])
    for hx2 in hsx:
        line(ax, hx2, cy_hs + NODE_H / 2, cx, cy_fc - NODE_H / 2)

    # matched sub-cluster: 1by1matched/caliper, then 1by1matched's config
    # (subject/visit match -> priority) and a final matched-cohort output
    mtx, mt_tot = _rowx(cx, 2, 2.6, 0.6)
    prx, pr_tot = _rowx(cx, 3, 2.4, 0.65)
    mt_w = max(mt_tot, pr_tot) + 0.9
    mt_box_top = my - NODE_H / 2 - SP
    mt_box_bot = my4 + NODE_H / 2 + SP
    cluster(ax, cx, (mt_box_top + mt_box_bot) / 2, mt_w,
            mt_box_bot - mt_box_top, C_ES['bg'])
    for x, lab in zip(mtx, ['1 by 1 matched', 'caliper_group']):
        node(ax, x, my, 2.6, NODE_H, lab, C_ES['nd'])
    for x, lab in zip(mtx, ['subject_match', 'visit_match']):
        node(ax, x, my2, 2.6, NODE_H, lab, C_ES['nd'])
    for x, lab in zip(prx, ['no_priority', 'priority_acs', 'priority_nad']):
        node(ax, x, my3, 2.4, NODE_H, lab, C_ES['nd'])
    node(ax, cx, my4, 3.6, NODE_H, 'matched cohort', C_ES['nd'])
    # 1by1matched -> subject/visit match -> priority -> matched cohort
    for sx in mtx:
        line(ax, mtx[0], my + NODE_H / 2, sx, my2 - NODE_H / 2)
    for sx in mtx:
        for px in prx:
            line(ax, sx, my2 + NODE_H / 2, px, my3 - NODE_H / 2)
    for px in prx:
        line(ax, px, my3 + NODE_H / 2, cx, my4 - NODE_H / 2)
    return s4_bot


def build(mode='center'):
    """mode='center' : Demographic+Cohort sits below the features (central spine).
       mode='side'   : Demographic+Cohort parked beside the whole chain (left)."""
    # ════ column centres (left -> right) ════
    if mode == 'side':
        X_AGE_C, X_EMB_C, X_EMO_C = 24.0, 45.0, 70.0
    else:
        X_AGE_C, X_EMB_C, X_EMO_C = 7.0, 30.0, 55.0
    PRE_CX = X_EMB_C        # preprocess aligned to the embedding centre-line
    # embedding splits into two aligned sub-columns: each feature cluster sits
    # directly above its own downstream path.
    X_ORIG = X_EMB_C - 6.0     # 'original' -> dim-reduce -> classifier
    X_ASYM = X_EMB_C + 6.0     # 'asymmetry' -> L2/CD/LDA scoring

    # ════ vertical bands (top -> bottom) ════
    top = 0.5
    y_pre1 = top + SP + NODE_H / 2          # Detect
    y_pre2 = y_pre1 + NODE_H + SP           # Select
    y_pre3 = y_pre2 + NODE_H + SP           # no_background / background
    y_pre4 = y_pre3 + NODE_H + SP           # Align
    y_pre5 = y_pre4 + NODE_H + SP           # face / mirrored_face
    pre_top = top
    pre_bot = y_pre5 + NODE_H / 2 + SP

    mod_top = pre_bot + SP
    y_mod = mod_top + SP + NODE_H / 2
    mod_bot = y_mod + NODE_H / 2 + SP

    feat_top = mod_bot + SP
    y_feat = feat_top + SP + NODE_H / 2
    feat_bot = y_feat + NODE_H / 2 + SP

    # ════ Age step-5 output chain (mean -> age_error -> violin/lines/...) ════
    d1_top = feat_bot + SP
    y_d1 = d1_top + SP + NODE_H / 2          # Predict Age mean x 1
    d2_top = y_d1 + NODE_H / 2 + SP + SP
    y_d2 = d2_top + SP + NODE_H / 2          # age_error
    d3_top = y_d2 + NODE_H / 2 + SP + SP
    y_d3 = d3_top + SP + NODE_H / 2          # violin / lines / scatter / stat
    d3_bot = y_d3 + NODE_H / 2 + SP

    # ════ Demographic+Cohort placement (depends on mode) ════
    if mode == 'side':
        dc_cx = 8.5                       # left spine, beside the chain
        s4_top = top                      # top-aligned with Preprocessing
        x_left, x_right = -1.5, 88.0
    else:
        dc_cx = X_EMB_C                   # central, below the features
        s4_top = d3_bot + SP
        x_left, x_right = -1.5, 66.0
    dc_bot = s4_top + _DC_H

    # ════ embedding Fwd/Rev -> eval-result -> eval_by — below the step-5 ════
    e_fr_top = d3_bot + SP + 0.6
    e_fr_y1 = e_fr_top + SP + NODE_H / 2       # Fwd Kfold / Rev 1by1matched
    e_fr_y2 = e_fr_y1 + NODE_H + SP            # Rev Kfold
    e_er_y = e_fr_y2 + NODE_H + SP + 0.5       # all+1by1 / 1by1+other (dimmed)
    e_eb_y = e_er_y + NODE_H + SP + 0.5        # eval_by_subject / eval_by_visit
    e_cmp_y = e_eb_y + NODE_H + SP + 0.5       # AD-vs-HC / NAD / ACS contrasts
    e_fr_bot = e_cmp_y + NODE_H / 2 + SP

    fig_h = max(dc_bot, e_fr_bot) + 0.5

    fig, ax = plt.subplots(figsize=((x_right - x_left), fig_h))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # dimmed connector colour for greyed-out (embedding / emotion) branches
    DIM_LN = '#CCCCCF'

    def _dln(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color=DIM_LN, lw=1.4,
                solid_capstyle='round', zorder=2)

    def _uline(x1, y1, x2, y2):           # connector routed UNDER cluster fills
        ax.plot([x1, x2], [y1, y2], color=EC, lw=1.4,
                solid_capstyle='round', zorder=-1)

    # ════════════════════════════════════════════════════════
    # step 1 — Preprocessing (shared head)
    # ════════════════════════════════════════════════════════
    x_face = PRE_CX - 2.2
    x_mirr = PRE_CX + 2.2
    nw_mir = 2.4
    pre_w = 2 * (PRE_CX - (x_face - nw_mir / 2)) + 1.0
    cluster(ax, PRE_CX, (pre_top + pre_bot) / 2, pre_w,
            pre_bot - pre_top, C_PRE['bg'])
    node(ax, PRE_CX, y_pre1, NW_PRE, NODE_H, 'Detect', C_PRE['nd'])
    node(ax, PRE_CX, y_pre2, NW_PRE, NODE_H, 'Select', C_PRE['nd'])
    line(ax, PRE_CX, y_pre1 + NODE_H / 2, PRE_CX, y_pre2 - NODE_H / 2)
    bgx, _ = _rowx(PRE_CX, 2, 2.4, 0.65)
    for x, lab in zip(bgx, ['no_background', 'background']):
        node(ax, x, y_pre3, 2.4, NODE_H, lab, C_PRE['nd'])
        line(ax, PRE_CX, y_pre2 + NODE_H / 2, x, y_pre3 - NODE_H / 2)
    node(ax, PRE_CX, y_pre4, NW_PRE, NODE_H, 'Align', C_PRE['nd'])
    for x in bgx:
        line(ax, x, y_pre3 + NODE_H / 2, PRE_CX, y_pre4 - NODE_H / 2)
    node(ax, x_face, y_pre5, nw_mir, NODE_H, 'face x 10', C_PRE['nd'])
    node(ax, x_mirr, y_pre5, nw_mir, NODE_H, 'mirrored_face\nx 20', C_PRE['nd'])
    line(ax, PRE_CX, y_pre4 + NODE_H / 2, x_face, y_pre5 - NODE_H / 2)
    line(ax, PRE_CX, y_pre4 + NODE_H / 2, x_mirr, y_pre5 - NODE_H / 2)

    # ════════════════════════════════════════════════════════
    # step 2 — per-branch models
    # ════════════════════════════════════════════════════════
    # Age models
    amx, am_tot = _rowx(X_AGE_C, len(AGE_MODELS), NW_AM, GAP_AM)
    for x, lab in zip(amx, AGE_MODELS):
        node(ax, x, y_mod, NW_AM, NODE_H, lab, C_AGE['nd'])
        line(ax, x_face, y_pre5 + NODE_H / 2, x, y_mod - NODE_H / 2)

    # embedding models  (LIT)
    EMB_MODELS = ['dlib', 'TopoFR', 'ArcFace', 'VGGFace']
    nw_em = 2.2; gap_em = 0.5
    emx, em_tot = _rowx(X_EMB_C, len(EMB_MODELS), nw_em, gap_em)
    for x, lab in zip(emx, EMB_MODELS):
        node(ax, x, y_mod, nw_em, NODE_H, lab, C2['nd'])
        for src in [x_face, x_mirr]:
            line(ax, src, y_pre5 + NODE_H / 2, x, y_mod - NODE_H / 2)

    # emotion models  (LIT)
    EMO_MODELS = ['EmoNet', 'Open\nFace', 'FER', 'HS\nEmotion', 'Libre\nFace',
                  'DAN', 'POSTER\n++', 'Py-Feat', 'ViT']
    nw_om = 2.2; gap_om = 0.3
    omx, om_tot = _rowx(X_EMO_C, len(EMO_MODELS), nw_om, gap_om)
    for x, lab in zip(omx, EMO_MODELS):
        node(ax, x, y_mod, nw_om, NODE_H, lab, C_EMO['nd'])
        line(ax, x_face, y_pre5 + NODE_H / 2, x, y_mod - NODE_H / 2)

    # ════════════════════════════════════════════════════════
    # step 3 — per-branch features
    # ════════════════════════════════════════════════════════
    # Age: Predict Age x 10
    age_w = 4.5
    node(ax, X_AGE_C, y_feat, age_w, NODE_H, 'Predict Age x 10', C_AGE['nd'])
    for x in amx:
        line(ax, x, y_mod + NODE_H / 2, X_AGE_C, y_feat - NODE_H / 2)

    # ── Age step 5: mean -> age_error -> violin/lines/scatter/stat (LIT) ──
    node(ax, X_AGE_C, y_d1, age_w, NODE_H, 'Predict Age mean x 1', C_AOUT['nd'])
    line(ax, X_AGE_C, y_feat + NODE_H / 2, X_AGE_C, y_d1 - NODE_H / 2)

    ae_w = 5.5
    node(ax, X_AGE_C, y_d2, ae_w, NODE_H,
         'age_error = real_age - predicted_age', C_AOUT['nd'])
    line(ax, X_AGE_C, y_d1 + NODE_H / 2, X_AGE_C, y_d2 - NODE_H / 2)

    vls_x, vls_tot = _rowx(X_AGE_C, 4, 2.6, 0.5)
    for x, lab in zip(vls_x, ['violin', 'lines', 'scatter', 'stat']):
        node(ax, x, y_d3, 2.6, NODE_H, lab, C_AOUT['nd'])
        line(ax, X_AGE_C, y_d2 + NODE_H / 2, x, y_d3 - NODE_H / 2)

    # embedding features split into TWO clusters, each centred on its sub-column
    # (mirrors age_emb_pipeline_mpl): 'original' above X_ORIG, asymmetry above X_ASYM
    nw_ef = 1.9; gap_ef = 0.3
    ASYM_FEATS = ['diff', '|diff|', 'rel_diff', '|rel_diff|']
    x_orig = X_ORIG
    asymx, asym_tot = _rowx(X_ASYM, len(ASYM_FEATS), nw_ef, gap_ef)

    node(ax, x_orig, y_feat, nw_ef, NODE_H, 'original', C2['nd'])
    for x, lab in zip(asymx, ASYM_FEATS):
        node(ax, x, y_feat, nw_ef, NODE_H, lab, C2['nd'])
    for fx in [x_orig] + asymx:
        for ex in emx:
            line(ax, ex, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # ── embedding step 5 (shares Age's y_d1/y_d2/y_d3 bands) ──
    #   shared:    original + asymmetry -> mean/all (avg feature vectors?)
    #   original:  mean/all -> no_drop/PCA/DropCorr (dim-reduce) -> LR/XGB
    #   asymmetry: mean/all -> L2/CD/LDA (three scoring methods)
    # Each path is centred on its sub-column (X_ORIG / X_ASYM), aligned with the
    # feature cluster above it. Cluster details kept (cf. age_emb_pipeline_mpl).

    # mean/all is LIT (C_PA); the original path (dim-reduce + classifier) is LIT
    # and shares ONE colour family (C4); only the asymmetry scoring stays dimmed.
    # shared mean/all aggregation (both feature clusters feed it)
    eax, ea_tot = _rowx(X_EMB_C, 2, 2.4, 0.65)
    cluster(ax, X_EMB_C, y_d1, ea_tot + 0.9, NODE_H + 2 * SP, C_PA['bg'])
    for x, lab in zip(eax, ['mean', 'all']):
        node(ax, x, y_d1, 2.4, NODE_H, lab, C_PA['nd'])
    for fx in [x_orig] + asymx:
        for px in eax:
            line(ax, fx, y_feat + NODE_H / 2, px, y_d1 - NODE_H / 2)

    # original path: dimensionality reduction (aligned under 'original')
    pdx, pd_tot = _rowx(X_ORIG, 3, 2.0, 0.5)
    for x, lab in zip(pdx, ['no_drop', 'PCA', 'DropCorr']):
        node(ax, x, y_d2, 2.0, NODE_H, lab, C4['nd'])
        for px in eax:
            line(ax, px, y_d1 + NODE_H / 2, x, y_d2 - NODE_H / 2)

    # asymmetry path: three scoring methods (aligned under the asymmetry feats), LIT
    scx, sc_tot = _rowx(X_ASYM, 3, 2.9, 0.4)
    cluster(ax, X_ASYM, y_d2, sc_tot + 0.8, NODE_H + 2 * SP, C_ASY['bg'])
    sc_labels = ['Centroid Dist\n$\\Delta\\cos(x,\\mu)$',
                 'LDA Proj\nFisher 1D',
                 'L2 Norm\n$\\sqrt{\\Sigma_i f_i^2}$']
    sc_colors = [C_ASY['nd'], C_ASY['nd'], C6['nd']]   # L2 Norm singled out, rightmost
    for x, lab, nc in zip(scx, sc_labels, sc_colors):
        node(ax, x, y_d2, 2.9, NODE_H, lab, nc)
        for px in eax:
            line(ax, px, y_d1 + NODE_H / 2, x, y_d2 - NODE_H / 2)

    # original path: classifier (aligned under the dim-reduce row)
    clfx, clf_tot = _rowx(X_ORIG, 2, 4.2, 0.5)
    for x, lab in zip(clfx, [
            'Logistic Regression\n$C \\in \\{10^{-3}\\,.\\,.\\,10^{2}\\}$',
            'XGBoost\nn_tree x max_depth x lr\n'
            '{200,500,1k}x{3,6,9}x{.05,.1,.2}']):
        node(ax, x, y_d3, 4.2, NODE_H, lab, C4['nd'])
        for dx in pdx:
            line(ax, dx, y_d2 + NODE_H / 2, x, y_d3 - NODE_H / 2)

    # emotion: 10 features in 3 sub-groups (V/A | contempt | 7 shared)
    EMO_VA = ['valence', 'arousal']
    EMO_CT = ['contempt']
    EMO_SH = ['anger', 'disgust', 'fear', 'happiness',
              'sadness', 'surprise', 'neutral']
    nw_of = 1.7; gap_of = 0.2; sub_gap = 0.8
    _, va_tot = _rowx(0, len(EMO_VA), nw_of, gap_of)
    _, ct_tot = _rowx(0, len(EMO_CT), nw_of, gap_of)
    _, sh_tot = _rowx(0, len(EMO_SH), nw_of, gap_of)
    total_of = va_tot + sub_gap + ct_tot + sub_gap + sh_tot
    of_left = X_EMO_C - total_of / 2
    va_c = of_left + va_tot / 2
    ct_c = of_left + va_tot + sub_gap + ct_tot / 2
    sh_c = of_left + va_tot + sub_gap + ct_tot + sub_gap + sh_tot / 2
    vax, _ = _rowx(va_c, len(EMO_VA), nw_of, gap_of)
    ctx, _ = _rowx(ct_c, len(EMO_CT), nw_of, gap_of)
    shx, _ = _rowx(sh_c, len(EMO_SH), nw_of, gap_of)
    for c, tot, xs, labs in [(va_c, va_tot, vax, EMO_VA),
                             (ct_c, ct_tot, ctx, EMO_CT),
                             (sh_c, sh_tot, shx, EMO_SH)]:
        for x, lab in zip(xs, labs):
            node(ax, x, y_feat, nw_of, NODE_H, lab, C_EMO['nd'])
    # EmoNet -> V/A ; EmoNet+OpenFace -> contempt ; all 9 -> 7 shared  (LIT)
    for fx in vax:
        line(ax, omx[0], y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)
    for fx in ctx:
        for src in [omx[0], omx[1]]:
            line(ax, src, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)
    for fx in shx:
        for mx in omx:
            line(ax, mx, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # ════════════════════════════════════════════════════════
    # group wrappers — one big cluster per same-colour group (drawn behind:
    # cluster() is zorder 0, so it sits under the already-placed nodes/lines)
    # ════════════════════════════════════════════════════════
    PADX = 0.5; PADY = SP

    def _grp(x0, x1, y0, y1, fc):
        cluster(ax, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, fc)

    y_mr_top = y_mod - NODE_H / 2 - PADY     # models row top
    y_ft_bot = y_feat + NODE_H / 2 + PADY    # features row bottom

    # age: models + Predict Age x10
    half = max(am_tot, age_w) / 2 + PADX
    _grp(X_AGE_C - half, X_AGE_C + half, y_mr_top, y_ft_bot, C_AGE['bg'])

    # age_error: Predict Age mean -> age_error -> violin/lines/scatter/stat
    half = max(age_w, ae_w, vls_tot) / 2 + PADX
    _grp(X_AGE_C - half, X_AGE_C + half,
         y_d1 - NODE_H / 2 - PADY, y_d3 + NODE_H / 2 + PADY, C_AOUT['bg'])

    # embedding: models + original/asymmetry features
    eL = min(X_EMB_C - em_tot / 2, X_ORIG - nw_ef / 2) - PADX
    eR = max(X_EMB_C + em_tot / 2, X_ASYM + asym_tot / 2) + PADX
    _grp(eL, eR, y_mr_top, y_ft_bot, C2['bg'])

    # emotion: models + features
    half = max(om_tot, total_of) / 2 + PADX
    _grp(X_EMO_C - half, X_EMO_C + half, y_mr_top, y_ft_bot, C_EMO['bg'])

    # embedding original path: dim-reduce + classifier wrapped together (lit, C4)
    half = max(pd_tot, clf_tot) / 2 + PADX
    _grp(X_ORIG - half, X_ORIG + half,
         y_d2 - NODE_H / 2 - PADY, y_d3 + NODE_H / 2 + PADY, C4['bg'])
    # ...and one inner sub-cluster per row (drawn after the wrapper so they show)
    cluster(ax, X_ORIG, y_d2, pd_tot + 0.6, NODE_H + 0.6, C4['bg'])
    cluster(ax, X_ORIG, y_d3, clf_tot + 0.6, NODE_H + 0.6, C4['bg'])

    # ════════════════════════════════════════════════════════
    # step 4 — Demographic + Cohort  (shared by all branches; standalone for now)
    # ════════════════════════════════════════════════════════
    _demo_cohort(ax, dc_cx, s4_top)

    # ════════════════════════════════════════════════════════
    # embedding Fwd / Rev (lit) -> two independent eval-result clusters
    #   (all+1by1 / 1by1+other) -> central eval_by (dimmed)
    # ════════════════════════════════════════════════════════
    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
    fxl = X_EMB_C - (cw + gap_col) / 2
    fxr = X_EMB_C + (cw + gap_col) / 2

    # Fwd: K fold only (1 by 1 matched removed)
    fwd_top = e_fr_y1 - NODE_H / 2 - SP
    fwd_bot = e_fr_y1 + NODE_H / 2 + SP
    cluster(ax, fxl, e_fr_y1, cw, fwd_bot - fwd_top, CF['bg'])
    node(ax, fxl, e_fr_y1, nw_col, NODE_H, 'K fold(K=10)', CF['nd'])

    # Rev: 1 by 1 matched -> K fold
    rev_top = e_fr_y1 - NODE_H / 2 - SP
    rev_bot = e_fr_y2 + NODE_H / 2 + SP
    cluster(ax, fxr, (rev_top + rev_bot) / 2, cw, rev_bot - rev_top, CR['bg'])
    node(ax, fxr, e_fr_y1, nw_col, NODE_H, '1 by 1 matched', CR['nd'])
    node(ax, fxr, e_fr_y2, nw_col, NODE_H, 'K fold(K=10)', CR['nd'])
    line(ax, fxr, e_fr_y1 + NODE_H / 2, fxr, e_fr_y2 - NODE_H / 2)

    # two independent eval-result clusters (LIT, matching their feeders)
    cluster(ax, fxl, e_er_y, nw_col + 0.8, NODE_H + 2 * SP, CF['bg'])
    node(ax, fxl, e_er_y, nw_col, NODE_H, 'all + 1 by 1', CF['nd'])
    cluster(ax, fxr, e_er_y, nw_col + 0.8, NODE_H + 2 * SP, CR['bg'])
    node(ax, fxr, e_er_y, nw_col, NODE_H, '1 by 1 + other', CR['nd'])
    line(ax, fxl, e_fr_y1 + NODE_H / 2, fxl, e_er_y - NODE_H / 2)   # Fwd  -> all+1by1
    line(ax, fxr, e_fr_y2 + NODE_H / 2, fxr, e_er_y - NODE_H / 2)   # Rev  -> 1by1+other

    # central eval_by (LIT) + the 3 contrasts below share ONE C_ES cluster
    nw_ev = 2.4; gap_ev = 0.65
    esx, es_tot = _rowx(X_EMB_C, 2, nw_ev, gap_ev)
    cmpx, cmp_tot = _rowx(X_EMB_C, 3, 3.0, 0.5)
    eb_cy = (e_eb_y + e_cmp_y) / 2
    eb_h = (e_cmp_y - e_eb_y) + NODE_H + 2 * SP
    cluster(ax, X_EMB_C, eb_cy, max(es_tot, cmp_tot) + 0.9, eb_h, C_ES['bg'])
    for x, lab in zip(esx, ['eval_by_subject', 'eval_by_visit']):
        node(ax, x, e_eb_y, nw_ev, NODE_H, lab, C_ES['nd'])
    for srcx in [fxl, fxr]:
        for ex in esx:
            line(ax, srcx, e_er_y + NODE_H / 2, ex, e_eb_y - NODE_H / 2)

    # ── wire embedding step-5 outputs into the eval protocols ──
    #   every classifier/scoring box EXCEPT L2 Norm -> Fwd & Rev tops
    #   (K fold / 1 by 1 matched); L2 Norm bypasses the folds and connects
    #   straight to the final eval_by_subject / eval_by_visit.
    fr_tops = [(fxl, e_fr_y1), (fxr, e_fr_y1)]          # Fwd K fold / Rev 1by1matched
    for sx in clfx:                                     # LR, XGBoost   (y_d3)
        for tx, ty in fr_tops:
            line(ax, sx, y_d3 + NODE_H / 2, tx, ty - NODE_H / 2)
    for sx in scx[:2]:                                  # Centroid Dist, LDA Proj (y_d2)
        for tx, ty in fr_tops:
            line(ax, sx, y_d2 + NODE_H / 2, tx, ty - NODE_H / 2)
    for ex in esx:                                      # L2 Norm (scx[2]) -> eval_by
        _uline(scx[2], y_d2 + NODE_H / 2, ex, e_eb_y - NODE_H / 2)

    # ── eval_by -> three diagnostic contrasts (inside the shared C_ES cluster) ──
    for x, lab in zip(cmpx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS']):
        node(ax, x, e_cmp_y, 3.0, NODE_H, lab, C_ES['nd'])
    for ex in esx:
        for cx2 in cmpx:
            line(ax, ex, e_eb_y + NODE_H / 2, cx2, e_cmp_y - NODE_H / 2)

    # ── Save ── (side is the chosen layout -> the primary file)
    suffix = '' if mode == 'side' else '_center'
    out = OUT / f"age_emb_pipeline_mpl_refactor_v2{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build('side')
