"""scripts/overview/age_emb_pipeline_v2.py
Refactor V2 (matplotlib): same 3-branch pipeline as age_emb_pipeline_mpl_refactor,
re-drawn on the stricter drawer toolkit (draw_pipeline_v2_common: indexed palette,
content-driven node sizing, linespacing 1.2, uniform GAP/PADDING).

Pipeline (top -> bottom):
  step 1  Preprocessing cluster (shared head: Detect..face/mirror)
  step 2  per-branch models   (Age x5 / embedding x4 / emotion x9)
  step 3  per-branch features (Predict Age x10 / 5 emb feats / 10 emo feats)
  step 4  demographic + cohort (left shared spine)
  step 5  per-branch outputs + eval chains
          (Age/emo: -> full/1by1matched -> violin;
           embedding: Fwd/Rev train/CV -> eval-result -> eval_by -> contrasts
           -> aggregator -> full/1by1matched -> violin)

Output:
  workspace_refactor/overview/age_emb_pipeline_v2.png

Usage:
    python scripts/overview/age_emb_pipeline_v2.py
"""
import matplotlib.pyplot as plt
from draw_pipeline_v2_common import *


_DC_PAD = 0.7
# height of the whole shared block (Demographic + Cohort + matched sub-tree:
# 1by1matched/caliper -> subject/visit match -> priority -> matched cohort),
# incl. outer pads. Mirrors the y-spacing in _demo_cohort.
_DC_H = 2 * _DC_PAD + 12 * NODE_H + 17 * SP + 0.6


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
    PS = ['P: CDR all', 'P: CDR >=0.5', 'P: CDR >=1', 'P: CDR >=2']
    HS = ['HC: CDR all\nor MMSE all', 'HC: CDR =0\nor MMSE >=26']
    _rw = lambda labs: sum(node_w(l) for l in labs) + (len(labs) - 1) * GAP
    demo_w = max(_rw(DEMO_ABOVE), _rw(DEMO_BELOW), node_w('Demographic')) + PADDING
    cohort_w = max(_rw(PS), _rw(HS), _rw(['P: first', 'P: all']),
                   node_w('Cohort (P + NAD + ACS)')) + PADDING
    big_w = max(demo_w, cohort_w) + 1.6

    # outer wrapper (cool lavender frame — harmonises the purple Demographic
    # and blue Cohort sub-clusters; shared by all branches)
    cluster(ax, cx, (s4_top + s4_bot) / 2, big_w, s4_bot - s4_top, '#DCDDEE')

    # Demographic sub-cluster (top)
    cluster(ax, cx, (dg_top + dg_bot) / 2, demo_w, dg_bot - dg_top, C3['bg'])
    node(ax, cx, dy_block, node_w('Demographic'), NODE_H, 'Demographic', C3['nd'])
    dax, _, _ = prow(ax, cx, dy_above, DEMO_ABOVE, C3['nd'])
    for x in dax:
        line(ax, cx, dy_block - NODE_H / 2, x, dy_above + NODE_H / 2)
    dbx, _, _ = prow(ax, cx, dy_below, DEMO_BELOW, C3['nd'])
    for x in dbx:
        line(ax, cx, dy_block + NODE_H / 2, x, dy_below - NODE_H / 2)

    # Cohort sub-cluster (bottom)
    cluster(ax, cx, (co_top + co_bot) / 2, cohort_w, co_bot - co_top, C1['bg'])
    pvx, _, _ = prow(ax, cx, cy_pv, ['P: first', 'P: all'], C1['nd'])
    hvx, _, _ = prow(ax, cx, cy_hv, ['HC: first', 'HC: all'], C1['nd'])
    for px in pvx:
        for hx in hvx:
            line(ax, px, cy_pv + NODE_H / 2, hx, cy_hv - NODE_H / 2)
    psx, _, _ = prow(ax, cx, cy_ps, PS, C1['nd'])
    for hx in hvx:
        for sx in psx:
            line(ax, hx, cy_hv + NODE_H / 2, sx, cy_ps - NODE_H / 2)
    hsx, _, _ = prow(ax, cx, cy_hs, HS, C1['nd'])
    for sx in psx:
        for hx2 in hsx:
            line(ax, sx, cy_ps + NODE_H / 2, hx2, cy_hs - NODE_H / 2)
    node(ax, cx, cy_fc, node_w('Cohort (P + NAD + ACS)'), NODE_H,
         'Cohort (P + NAD + ACS)', C1['nd'])
    for hx2 in hsx:
        line(ax, hx2, cy_hs + NODE_H / 2, cx, cy_fc - NODE_H / 2)

    # matched sub-cluster: 1by1matched/caliper, then 1by1matched's config
    # (subject/visit match -> priority) and a final matched-cohort output
    MT = ['1 by 1 matched', 'caliper_group']
    SV = ['subject_match', 'visit_match']
    PR = ['no_priority', 'priority_acs', 'priority_nad']
    mt_w = max(_rw(MT), _rw(SV), _rw(PR), node_w('matched cohort')) + PADDING
    mt_box_top = my - NODE_H / 2 - SP
    mt_box_bot = my4 + NODE_H / 2 + SP
    cluster(ax, cx, (mt_box_top + mt_box_bot) / 2, mt_w,
            mt_box_bot - mt_box_top, C_ES['bg'])
    mtx, _, _ = prow(ax, cx, my, MT, C_ES['nd'])
    svx, _, _ = prow(ax, cx, my2, SV, C_ES['nd'])
    prx, _, _ = prow(ax, cx, my3, PR, C_ES['nd'])
    node(ax, cx, my4, node_w('matched cohort'), NODE_H, 'matched cohort', C_ES['nd'])
    # 1by1matched -> subject/visit match -> priority -> matched cohort
    for sx in svx:
        line(ax, mtx[0], my + NODE_H / 2, sx, my2 - NODE_H / 2)
    for sx in svx:
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
    y_d3 = d3_top + SP + NODE_H / 2          # embedding classifier / emo violin
    d3_bot = y_d3 + NODE_H / 2 + SP

    # Age branch grows two extra rows below age_error:
    #   full / 1 by 1 matched, then the 4 stat plots
    y_a_fm = y_d2 + NODE_H + 2 * SP          # full / 1 by 1 matched  (Age)
    y_a_stat = y_a_fm + NODE_H + 2 * SP      # violin / lines / scatter / stat
    a_bot = y_a_stat + NODE_H / 2 + SP

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
    e_agg_y = e_cmp_y + NODE_H + SP + 0.5      # aggregator
    e_fm_y = e_agg_y + NODE_H + SP + 0.5       # full / 1 by 1 matched
    e_vio_y = e_fm_y + NODE_H + SP + 0.5       # violin
    e_fr_bot = e_vio_y + NODE_H / 2 + SP

    fig_h = max(dc_bot, e_fr_bot, a_bot) + 0.5

    fig, ax = plt.subplots(figsize=((x_right - x_left), fig_h))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ════════════════════════════════════════════════════════
    # step 1 — Preprocessing (shared head)
    # ════════════════════════════════════════════════════════
    _rw = lambda labs: sum(node_w(l) for l in labs) + (len(labs) - 1) * GAP
    MIR = ['face x 10', 'mirrored_face\nx 20']
    BG = ['no_background', 'background']
    pre_w = max(_rw(MIR), _rw(BG), node_w('Detect')) + PADDING
    cluster(ax, PRE_CX, (pre_top + pre_bot) / 2, pre_w,
            pre_bot - pre_top, C_PRE['bg'])
    node(ax, PRE_CX, y_pre1, node_w('Detect'), NODE_H, 'Detect', C_PRE['nd'])
    node(ax, PRE_CX, y_pre2, node_w('Select'), NODE_H, 'Select', C_PRE['nd'])
    line(ax, PRE_CX, y_pre1 + NODE_H / 2, PRE_CX, y_pre2 - NODE_H / 2)
    bgx, _, _ = prow(ax, PRE_CX, y_pre3, BG, C_PRE['nd'])
    for x in bgx:
        line(ax, PRE_CX, y_pre2 + NODE_H / 2, x, y_pre3 - NODE_H / 2)
    node(ax, PRE_CX, y_pre4, node_w('Align'), NODE_H, 'Align', C_PRE['nd'])
    for x in bgx:
        line(ax, x, y_pre3 + NODE_H / 2, PRE_CX, y_pre4 - NODE_H / 2)
    (x_face, x_mirr), _, _ = prow(ax, PRE_CX, y_pre5, MIR, C_PRE['nd'])
    line(ax, PRE_CX, y_pre4 + NODE_H / 2, x_face, y_pre5 - NODE_H / 2)
    line(ax, PRE_CX, y_pre4 + NODE_H / 2, x_mirr, y_pre5 - NODE_H / 2)

    # ════════════════════════════════════════════════════════
    # step 2 — per-branch models
    # ════════════════════════════════════════════════════════
    # Age models
    amx, _, am_tot = prow(ax, X_AGE_C, y_mod, AGE_MODELS, C_AGE['nd'])
    for x in amx:
        line(ax, x_face, y_pre5 + NODE_H / 2, x, y_mod - NODE_H / 2)

    # embedding models  (LIT)
    EMB_MODELS = ['dlib', 'TopoFR', 'ArcFace', 'VGGFace']
    emx, _, em_tot = prow(ax, X_EMB_C, y_mod, EMB_MODELS, C2['nd'])
    for x in emx:
        for src in [x_face, x_mirr]:
            line(ax, src, y_pre5 + NODE_H / 2, x, y_mod - NODE_H / 2)

    # emotion models  (LIT)
    EMO_MODELS = ['EmoNet', 'Open\nFace', 'FER', 'HS\nEmotion', 'Libre\nFace',
                  'DAN', 'POSTER\n++', 'Py-Feat', 'ViT']
    omx, _, om_tot = prow(ax, X_EMO_C, y_mod, EMO_MODELS, C_EMO['nd'])
    for x in omx:
        line(ax, x_face, y_pre5 + NODE_H / 2, x, y_mod - NODE_H / 2)

    # ════════════════════════════════════════════════════════
    # step 3 — per-branch features
    # ════════════════════════════════════════════════════════
    # Age: Predict Age x 10
    age_w = node_w('Predict Age mean x 1')
    node(ax, X_AGE_C, y_feat, node_w('Predict Age x 10'), NODE_H,
         'Predict Age x 10', C_AGE['nd'])
    for x in amx:
        line(ax, x, y_mod + NODE_H / 2, X_AGE_C, y_feat - NODE_H / 2)

    # ── Age step 5: mean -> age_error -> full/1by1matched -> violin/.../stat ──
    node(ax, X_AGE_C, y_d1, age_w, NODE_H, 'Predict Age mean x 1', C_AOUT['nd'])
    line(ax, X_AGE_C, y_feat + NODE_H / 2, X_AGE_C, y_d1 - NODE_H / 2)

    ae_w = node_w('age_error = real_age - predicted_age')
    node(ax, X_AGE_C, y_d2, ae_w, NODE_H,
         'age_error = real_age - predicted_age', C_AOUT['nd'])
    line(ax, X_AGE_C, y_d1 + NODE_H / 2, X_AGE_C, y_d2 - NODE_H / 2)

    # full / 1 by 1 matched (1by1matched syncs to the left matched cluster colour)
    afmx, _, afm_tot = prow(ax, X_AGE_C, y_a_fm, ['full', '1 by 1 matched'],
                            [C_SA['nd'], C_ES['nd']])
    for x in afmx:
        line(ax, X_AGE_C, y_d2 + NODE_H / 2, x, y_a_fm - NODE_H / 2)

    vls_x, _, vls_tot = prow(ax, X_AGE_C, y_a_stat,
                             ['violin', 'lines', 'scatter', 'stat'], C_AOUT['nd'])
    for x in vls_x:
        for sx in afmx:
            line(ax, sx, y_a_fm + NODE_H / 2, x, y_a_stat - NODE_H / 2)

    # embedding features: 'original' above X_ORIG, asymmetry above X_ASYM
    ASYM_FEATS = ['diff', '|diff|', 'rel_diff', '|rel_diff|']
    x_orig = X_ORIG
    node(ax, x_orig, y_feat, node_w('original'), NODE_H, 'original', C2['nd'])
    asymx, _, asym_tot = prow(ax, X_ASYM, y_feat, ASYM_FEATS, C2['nd'])
    for fx in [x_orig] + asymx:
        for ex in emx:
            line(ax, ex, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # ── embedding step 5 (shares Age's y_d1/y_d2/y_d3 bands) ──
    #   shared:    original + asymmetry -> mean/all
    #   original:  mean/all -> no_drop/PCA/DropCorr (dim-reduce) -> LR/XGB
    #   asymmetry: mean/all -> Centroid Dist / LDA Proj / L2 Norm (scoring)
    # shared mean/all aggregation (both feature clusters feed it)
    eax, _, ea_tot = prow(ax, X_EMB_C, y_d1, ['mean', 'all'], C_PA['nd'])
    cluster(ax, X_EMB_C, y_d1, ea_tot + PADDING, NODE_H + 2 * SP, C_PA['bg'])
    for fx in [x_orig] + asymx:
        for px in eax:
            line(ax, fx, y_feat + NODE_H / 2, px, y_d1 - NODE_H / 2)

    # original path: dimensionality reduction (aligned under 'original')
    pdx, _, pd_tot = prow(ax, X_ORIG, y_d2, ['no_drop', 'PCA', 'DropCorr'], C4['nd'])
    for x in pdx:
        for px in eax:
            line(ax, px, y_d1 + NODE_H / 2, x, y_d2 - NODE_H / 2)

    # asymmetry path: three scoring methods (LaTeX labels -> explicit width)
    sc_labels = ['Centroid Dist\n$\\Delta\\cos(x,\\mu)$',
                 'LDA Proj\nFisher 1D',
                 'L2 Norm\n$\\sqrt{\\Sigma_i f_i^2}$']
    sc_colors = [C_ASY['nd'], C_ASY['nd'], C6['nd']]   # L2 Norm singled out, rightmost
    sc_w = 2.9
    scx, sc_tot = rowx(X_ASYM, 3, sc_w, GAP)
    cluster(ax, X_ASYM, y_d2, sc_tot + PADDING, NODE_H + 2 * SP, C_ASY['bg'])
    for x, lab, nc in zip(scx, sc_labels, sc_colors):
        node(ax, x, y_d2, sc_w, NODE_H, lab, nc)
        for px in eax:
            line(ax, px, y_d1 + NODE_H / 2, x, y_d2 - NODE_H / 2)

    # original path: classifier (LaTeX labels -> explicit width; XGBoost is 3-line)
    clf_labels = ['Logistic Regression\n$C \\in \\{10^{-3}\\,.\\,.\\,10^{2}\\}$',
                  'XGBoost\nn_tree x max_depth x lr\n'
                  '{200,500,1k}x{3,6,9}x{.05,.1,.2}']
    clf_w = 4.2
    clfx, clf_tot = rowx(X_ORIG, 2, clf_w, GAP)
    for x, lab in zip(clfx, clf_labels):
        node(ax, x, y_d3, clf_w, hgt(lab), lab, C4['nd'])
        for dx in pdx:
            line(ax, dx, y_d2 + NODE_H / 2, x, y_d3 - NODE_H / 2)

    # emotion: 10 features in 3 sub-groups (V/A | contempt | 7 shared)
    EMO_VA = ['valence', 'arousal']
    EMO_CT = ['contempt']
    EMO_SH = ['anger', 'disgust', 'fear', 'happiness',
              'sadness', 'surprise', 'neutral']
    sub_gap = 0.8
    va_tot = _rw(EMO_VA); ct_tot = _rw(EMO_CT); sh_tot = _rw(EMO_SH)
    total_of = va_tot + sub_gap + ct_tot + sub_gap + sh_tot
    of_left = X_EMO_C - total_of / 2
    va_c = of_left + va_tot / 2
    ct_c = of_left + va_tot + sub_gap + ct_tot / 2
    sh_c = of_left + va_tot + sub_gap + ct_tot + sub_gap + sh_tot / 2
    vax, _, _ = prow(ax, va_c, y_feat, EMO_VA, C_EMO['nd'])
    ctx, _, _ = prow(ax, ct_c, y_feat, EMO_CT, C_EMO['nd'])
    shx, _, _ = prow(ax, sh_c, y_feat, EMO_SH, C_EMO['nd'])
    # EmoNet -> V/A ; EmoNet+OpenFace -> contempt ; all 9 -> 7 shared  (LIT)
    for fx in vax:
        line(ax, omx[0], y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)
    for fx in ctx:
        for src in [omx[0], omx[1]]:
            line(ax, src, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)
    for fx in shx:
        for mx in omx:
            line(ax, mx, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # ── emo step 5: features -> mean/all -> full/1by1matched -> violin ──
    emo_all_feats = vax + ctx + shx
    emax, _, ema_tot = prow(ax, X_EMO_C, y_d1, ['mean', 'all'], C_PA['nd'])
    cluster(ax, X_EMO_C, y_d1, ema_tot + PADDING, NODE_H + 2 * SP, C_PA['bg'])
    for fx in emo_all_feats:
        for px in emax:
            line(ax, fx, y_feat + NODE_H / 2, px, y_d1 - NODE_H / 2)

    eofmx, _, eofm_tot = prow(ax, X_EMO_C, y_d2, ['full', '1 by 1 matched'],
                              [C_SA['nd'], C_ES['nd']])
    cluster(ax, X_EMO_C, y_d2, eofm_tot + PADDING, NODE_H + 2 * SP, C_SA['bg'])
    for x in eofmx:
        for px in emax:
            line(ax, px, y_d1 + NODE_H / 2, x, y_d2 - NODE_H / 2)

    node(ax, X_EMO_C, y_d3, node_w('violin'), NODE_H, 'violin', C_AOUT['nd'])
    for sx in eofmx:
        line(ax, sx, y_d2 + NODE_H / 2, X_EMO_C, y_d3 - NODE_H / 2)

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

    # age_error: Predict Age mean -> age_error -> full/1by1 -> violin/.../stat
    half = max(age_w, ae_w, afm_tot, vls_tot) / 2 + PADX
    _grp(X_AGE_C - half, X_AGE_C + half,
         y_d1 - NODE_H / 2 - PADY, y_a_stat + NODE_H / 2 + PADY, C_AOUT['bg'])

    # embedding: models + original/asymmetry features
    eL = min(X_EMB_C - em_tot / 2, X_ORIG - node_w('original') / 2) - PADX
    eR = max(X_EMB_C + em_tot / 2, X_ASYM + asym_tot / 2) + PADX
    _grp(eL, eR, y_mr_top, y_ft_bot, C2['bg'])

    # emotion: models + features
    half = max(om_tot, total_of) / 2 + PADX
    _grp(X_EMO_C - half, X_EMO_C + half, y_mr_top, y_ft_bot, C_EMO['bg'])

    # embedding original path: dim-reduce + classifier wrapped together (lit, C4)
    clf_h = hgt(clf_labels[1])          # XGBoost is 3-line -> taller
    half = max(pd_tot, clf_tot) / 2 + PADX
    _grp(X_ORIG - half, X_ORIG + half,
         y_d2 - NODE_H / 2 - PADY, y_d3 + clf_h / 2 + PADY, C4['bg'])

    # ════════════════════════════════════════════════════════
    # step 4 — Demographic + Cohort  (shared by all branches; standalone for now)
    # ════════════════════════════════════════════════════════
    _demo_cohort(ax, dc_cx, s4_top)

    # ════════════════════════════════════════════════════════
    # embedding eval: train/CV cluster (Fwd K fold + Rev 1by1matched+K fold)
    #   -> eval-result cluster (all+1by1 / 1by1+other) -> central eval_by
    # ════════════════════════════════════════════════════════
    EVAL_LABELS = ['K fold(K=10)', '1 by 1 matched', 'all + 1 by 1', '1 by 1 + other']
    nw_col = max(node_w(l) for l in EVAL_LABELS)
    fxl = X_EMB_C - (nw_col + GAP) / 2
    fxr = X_EMB_C + (nw_col + GAP) / 2
    grp_w = 2 * nw_col + GAP + PADDING          # spans both columns

    # train/CV cluster (one colour, cf. classification/train.py resampling):
    #   Fwd = K fold; Rev = 1 by 1 matched -> K fold
    cluster(ax, X_EMB_C, (e_fr_y1 + e_fr_y2) / 2, grp_w,
            (e_fr_y2 - e_fr_y1) + NODE_H + 2 * SP, CF['bg'])
    node(ax, fxl, e_fr_y1, nw_col, NODE_H, 'K fold(K=10)', CF['nd'])      # Fwd
    node(ax, fxr, e_fr_y1, nw_col, NODE_H, '1 by 1 matched', C_ES['nd'])  # Rev (matched)
    node(ax, fxr, e_fr_y2, nw_col, NODE_H, 'K fold(K=10)', CF['nd'])      # Rev
    line(ax, fxr, e_fr_y1 + NODE_H / 2, fxr, e_fr_y2 - NODE_H / 2)

    # eval-result cluster (another colour): all+1by1 + 1by1+other
    cluster(ax, X_EMB_C, e_er_y, grp_w, NODE_H + 2 * SP, C1['bg'])
    node(ax, fxl, e_er_y, nw_col, NODE_H, 'all + 1 by 1', C1['nd'])
    node(ax, fxr, e_er_y, nw_col, NODE_H, '1 by 1 + other', C1['nd'])
    line(ax, fxl, e_fr_y1 + NODE_H / 2, fxl, e_er_y - NODE_H / 2)   # Fwd  -> all+1by1
    line(ax, fxr, e_fr_y2 + NODE_H / 2, fxr, e_er_y - NODE_H / 2)   # Rev  -> 1by1+other

    # central eval_by (LIT) + the 3 contrasts below share ONE C_ES cluster
    EB = ['eval_by_subject', 'eval_by_visit']
    CMP = ['AD vs HC', 'AD vs NAD', 'AD vs ACS']
    es_w = max(node_w(l) for l in EB)
    cmp_w = max(node_w(l) for l in CMP)
    esx, es_tot = rowx(X_EMB_C, 2, es_w, GAP)
    cmpx, cmp_tot = rowx(X_EMB_C, 3, cmp_w, GAP)
    eb_cy = (e_eb_y + e_cmp_y) / 2
    eb_h = (e_cmp_y - e_eb_y) + NODE_H + 2 * SP
    cluster(ax, X_EMB_C, eb_cy, max(es_tot, cmp_tot) + PADDING, eb_h, C_ES['bg'])
    for x, lab in zip(esx, EB):
        node(ax, x, e_eb_y, es_w, NODE_H, lab, C_ES['nd'])
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
    for ex in esx:                          # L2 Norm (scx[2]) -> eval_by, under fills
        line(ax, scx[2], y_d2 + NODE_H / 2, ex, e_eb_y - NODE_H / 2, zorder=-1)

    # ── eval_by -> three diagnostic contrasts (inside the shared C_ES cluster) ──
    for x, lab in zip(cmpx, CMP):
        node(ax, x, e_cmp_y, cmp_w, NODE_H, lab, C_ES['nd'])
    for ex in esx:
        for cx2 in cmpx:
            line(ax, ex, e_eb_y + NODE_H / 2, cx2, e_cmp_y - NODE_H / 2)

    # ── contrasts -> aggregator -> full/1by1matched -> violin ──
    agg_w = node_w('aggregator')
    cluster(ax, X_EMB_C, e_agg_y, agg_w + PADDING, NODE_H + 2 * SP, C3['bg'])
    node(ax, X_EMB_C, e_agg_y, agg_w, NODE_H, 'aggregator', C3['nd'])
    for cx2 in cmpx:
        line(ax, cx2, e_cmp_y + NODE_H / 2, X_EMB_C, e_agg_y - NODE_H / 2)

    # full / 1 by 1 matched selector (1by1matched syncs to left matched colour)
    efmx, _, efm_tot = prow(ax, X_EMB_C, e_fm_y, ['full', '1 by 1 matched'],
                            [C_SA['nd'], C_ES['nd']])
    cluster(ax, X_EMB_C, e_fm_y, efm_tot + PADDING, NODE_H + 2 * SP, C_SA['bg'])
    for x in efmx:
        line(ax, X_EMB_C, e_agg_y + NODE_H / 2, x, e_fm_y - NODE_H / 2)

    node(ax, X_EMB_C, e_vio_y, node_w('violin'), NODE_H, 'violin', C_AOUT['nd'])
    for sx in efmx:
        line(ax, sx, e_fm_y + NODE_H / 2, X_EMB_C, e_vio_y - NODE_H / 2)

    # ── Save ── (side is the chosen layout -> the primary file)
    suffix = '' if mode == 'side' else '_center'
    out = OUT / f"age_emb_pipeline_v2{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build('side')
