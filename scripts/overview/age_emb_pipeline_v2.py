"""scripts/overview/age_emb_pipeline_v2.py
V2 (matplotlib): the trimmed 3-branch pipeline. Same head/branches as the
refactor baseline (age_emb_pipeline_refactor), but the embedding branch is
simplified and its asymmetry features carry a full downstream (integrated from
asymmetry_pipeline.py):
  asymmetry features -> L1/L2 scorers (right) + Logistic Regression (left)
    right: scorers -> average 10 score -> full / 1 by 1 matched / ANCOVA
    left:  Logistic Regression -> full / 1 by 1 matched
The figure is exported as high-res PNG + vector (SVG/PDF).

Embedding simplifications vs the refactor baseline (Age / emotion unchanged):
  (1) mean/all        -> all only
  (2) reducer         -> no_drop only      (PCA / DropCorr dropped)
  (3) classifier      -> Logistic Regression C=1e-3 only   (XGBoost dropped)
  (4) original branch keeps only the Logistic Regression classifier path; the
      asymmetry branch keeps the L1 / L2 norm scorers (Centroid Dist / LDA dropped)
  (5) embedding model -> ArcFace only       (dlib / TopoFR / VGGFace dropped)
  (6) the original/asymmetry feature fan stays left-right symmetric: ArcFace sits
      on the centre-line and fans to 'original' (left) and the 2 asymmetry
      features (right). 'original' runs down the left spine (all -> no_drop -> LR
      -> eval chain); the asymmetry features now fan to L1/L2 scorers (right) and
      a Logistic Regression classifier (left), each ending in its own analyses
      row (integrated from asymmetry_pipeline.py).
  (7) eval chain trimmed to one forward column: the reverse path and
      eval_by_visit are removed, leaving
      (K fold -> mean 10 pic score -> eval_by_subject -> contrasts
       -> full/1by1matched).

Reuses the V2 drawer toolkit (draw_pipeline_v2_common) unchanged.

Output (workspace/overview/):
  age_emb_pipeline_v2.png   (dpi=300)
  age_emb_pipeline_v2.svg   (vector)
  age_emb_pipeline_v2.pdf   (vector)

Usage:
    python scripts/overview/age_emb_pipeline_v2.py
"""
import matplotlib.pyplot as plt
from draw_pipeline_v2_common import *

# ── v2-only recolour: shadow a few shared aliases in THIS module so that
#    neighbouring blocks no longer share/approximate a hue. The shared toolkit
#    and the other scripts (refactor / asymmetry_pipeline) are unaffected.
#    Anchor kept by request: the matched block and every "1 by 1 matched" stay
#    C_ES (peach), matching the standalone matched sub-cluster on the left. ──
C_AGE = P[20]     # Age branch   : amber -> magenta (extension set; ≠ Demographic purple)
C_AOUT = P[17]    # outputs/desc : grey-blue -> sage (was ~ steel "full" it wraps)
CF = P[5]         # K fold       : green  -> yellow  (was ~ lime classifier above it)
C_EMA = P[21]     # mean/all (emotion) : lavender -> emerald (≠ coral above, steel below)
C_EVAL = P[22]    # eval_by_subject + contrasts : peach -> raspberry (≠ violet scorers, ≠ 1by1matched)
C_POOL = P[23]    # mean 10 pic score  : blue -> indigo (frees blue; ≠ yellow K-fold / raspberry eval)
C4 = P[24]        # classifier (no_drop + LR, both branches) : lime -> azure
#                   (was a 3rd green ≈ sage outputs / emerald emo-mean)


_DC_PAD = 0.7
# height of the whole shared block (Demographic + Cohort + matched sub-tree),
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
    my = mt_top + SP + NODE_H / 2          # 1 by 1 matched
    my2 = my + NODE_H + SP                  # visit_match
    my3 = my2 + NODE_H + SP                 # priority_acs
    my4 = my3 + NODE_H + SP                 # matched cohort (output)
    mt_bot = my4 + NODE_H / 2 + SP
    s4_bot = mt_bot + pad

    DEMO_ABOVE = ['Group', 'Number', 'Photo_Session', 'Sex']
    DEMO_BELOW = ['Age', 'BMI', 'MMSE', 'CASI', 'Global_CDR']
    PS = ['P: CDR all']
    HS = ['HC: CDR all\nor MMSE all']
    _rw = lambda labs: sum(node_w(l) for l in labs) + (len(labs) - 1) * GAP
    demo_w = max(_rw(DEMO_ABOVE), _rw(DEMO_BELOW), node_w('Demographic')) + PADDING
    cohort_w = max(_rw(PS), _rw(HS), node_w('P: first'), node_w('HC: all'),
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
    pvx, _, _ = prow(ax, cx, cy_pv, ['P: first'], C1['nd'])
    hvx, _, _ = prow(ax, cx, cy_hv, ['HC: all'], C1['nd'])
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

    # matched sub-cluster: the chosen config only
    # (1 by 1 matched -> visit_match -> priority_acs) -> matched-cohort output
    MT = ['1 by 1 matched']
    SV = ['visit_match']
    PR = ['priority_acs']
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


def _save(fig, suffix):
    """Write the figure as 300-dpi PNG plus SVG/PDF vectors (same tight bbox)."""
    stem = f"age_emb_pipeline_v2{suffix}"
    common = dict(bbox_inches='tight', pad_inches=0.15, facecolor='white')
    png = OUT / f"{stem}.png"
    fig.savefig(png, dpi=300, **common)
    for ext in ('svg', 'pdf'):
        fig.savefig(OUT / f"{stem}.{ext}", **common)
    print(f"Saved: {png}  (+ .svg / .pdf)")


def build(mode='center'):
    """mode='center' : Demographic+Cohort sits below the features (central spine).
       mode='side'   : Demographic+Cohort parked beside the whole chain (left)."""
    # ════ column centres (left -> right) ════
    if mode == 'side':
        X_AGE_C, X_EMB_C, X_EMO_C = 22.4, 37.6, 61.3
    else:
        X_AGE_C, X_EMB_C, X_EMO_C = 7.0, 30.0, 55.0
    PRE_CX = X_EMB_C        # preprocess aligned to the embedding centre-line
    # keep the original/asymmetry feature fan left-right symmetric — ArcFace sits
    # on the centre-line (X_EMB_C) and fans to 'original' (left, X_ORIG) and the
    # 2 asymmetry features (right, X_ASYM). 'original' runs the classifier/eval
    # chain down the LEFT spine X_ORIG; the asymmetry features fan to their own
    # scorer + Logistic Regression downstream (right, X_ASYM).
    X_ORIG = X_EMB_C - 6.0
    X_ASYM = X_EMB_C + 6.0

    # ════ vertical bands (top -> bottom) ════
    # BAND = uniform pitch for the clustered bands (models down through the eval
    # chain): tight enough that adjacent cluster wrappers leave a small even gap.
    # The preprocessing rows are plain nodes inside one cluster, so they keep the
    # base NODE_H + SP pitch.
    BAND = NODE_H + 2 * SP + 0.3   # cluster height (NODE_H + 2*SP) + 0.3 gap
    top = 0.5
    y_pre1 = top + SP + NODE_H / 2          # Detect
    y_pre2 = y_pre1 + NODE_H + SP           # Select
    y_pre3 = y_pre2 + NODE_H + SP           # no_background / background
    y_pre4 = y_pre3 + NODE_H + SP           # Align
    y_pre5 = y_pre4 + NODE_H + SP           # face / mirrored_face
    pre_top = top
    pre_bot = y_pre5 + NODE_H / 2 + SP

    y_mod = y_pre5 + BAND                   # per-branch models
    y_feat = y_mod + BAND                   # per-branch features

    # ════ shared step-5 bands (Age: mean -> age_error -> ...; emb: all -> no_drop
    #      -> LR; emo: all -> full/1by1 -> violin) ════
    y_d1 = y_feat + BAND                    # Predict Age mean / all / emo all
    y_d2 = y_d1 + BAND                      # age_error / no_drop / emo full-1by1
    y_d3 = y_d2 + BAND                      # LR / emo violin
    d3_bot = y_d3 + NODE_H / 2 + SP

    # Age branch grows two extra rows below age_error:
    #   full / 1 by 1 matched, then the 4 stat plots
    y_a_fm = y_d2 + BAND                    # full / 1 by 1 matched  (Age)
    y_a_stat = y_a_fm + BAND               # violin / lines / scatter / stat
    a_bot = y_a_stat + NODE_H / 2 + SP

    # ════ Demographic+Cohort placement (depends on mode) ════
    if mode == 'side':
        dc_cx = 8.5                       # left spine, beside the chain
        s4_top = top                      # top-aligned with Preprocessing
        x_left, x_right = 0.5, 74.0       # crop to content (branches tightened)
    else:
        dc_cx = X_EMB_C                   # central, below the features
        s4_top = d3_bot + SP
        x_left, x_right = -1.5, 66.0
    dc_bot = s4_top + _DC_H

    # ════ embedding eval chain — single column below the step-5 (uniform BAND) ══
    e_k_y = y_d3 + BAND                        # K fold(K=10)
    e_sc_y = e_k_y + BAND                      # mean 10 pic score
    e_eb_y = e_sc_y + BAND                     # eval_by_subject
    e_cmp_y = e_eb_y + BAND                    # AD-vs-HC / NAD / ACS contrasts
    e_fm_y = e_cmp_y + BAND                    # full / 1 by 1 matched (chain ends here)
    e_bot = e_fm_y + NODE_H / 2 + SP

    fig_h = max(dc_bot, e_bot, a_bot) + 0.5

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
    MIR = ['face x 10', 'mirrored_face\nx 10']
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

    # embedding model — ArcFace only (V3), on the centre-line
    arc_w = node_w('ArcFace')
    node(ax, X_EMB_C, y_mod, arc_w, NODE_H, 'ArcFace', C2['nd'])
    for src in [x_face, x_mirr]:
        line(ax, src, y_pre5 + NODE_H / 2, X_EMB_C, y_mod - NODE_H / 2)

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

    afmx, _, afm_tot = prow(ax, X_AGE_C, y_a_fm, ['full', '1 by 1 matched'],
                            [C_SA['nd'], C_ES['nd']])
    for x in afmx:
        line(ax, X_AGE_C, y_d2 + NODE_H / 2, x, y_a_fm - NODE_H / 2)

    vls_x, _, vls_tot = prow(ax, X_AGE_C, y_a_stat,
                             ['violin', 'lines', 'scatter', 'stat'], C_AOUT['nd'])
    for x in vls_x:
        for sx in afmx:
            line(ax, sx, y_a_fm + NODE_H / 2, x, y_a_stat - NODE_H / 2)

    # embedding features: ArcFace -> face / mirrored_face embedding vectors. The
    # 'original' box was dropped (it is exactly the face embedding vector); the
    # face embedding vector now feeds BOTH the classifier/eval spine ('all', left)
    # and the asymmetry sub-branch (right). The asymmetry side mirrors
    # asymmetry_pipeline: vectors -> asymmetry features -> L1/L2 scorers +
    # a Logistic Regression classifier. The asymmetry sub-branch is one band
    # deeper than the others, so its rows shift down from y_feat.

    # embedding vectors (y_feat): ArcFace -> face / mirrored_face embedding
    # vectors, centred under ArcFace so the fan is symmetric (they then feed the
    # classifier spine on the left and the asymmetry sub-branch on the right)
    VECS = ['face embedding\nvector x 10', 'mirrored_face embedding\nvector x 10']
    vec_w = 3.6
    vec_tot = 2 * vec_w + GAP
    vecx, _ = rowx(X_EMB_C, 2, vec_w, GAP)
    for vx, lab in zip(vecx, VECS):
        node(ax, vx, y_feat, vec_w, NODE_H, lab, C2['nd'])
        line(ax, X_EMB_C, y_mod + NODE_H / 2, vx, y_feat - NODE_H / 2)

    # asym features (y_af): vectors -> 2 formula boxes (fixed width — node_w can't
    # size LaTeX), each carrying the per-photo "vector x 10"
    y_af = y_d1
    ASYM_FEATS = ['$L_i - R_i$\nvector x 10',
                  '$(L_i - R_i)/\\sqrt{L_i^2 + R_i^2}$\nvector x 10']
    asym_w = 4.5
    asym_tot = 2 * asym_w + GAP
    asymx, _ = rowx(X_ASYM, 2, asym_w, GAP)
    cluster(ax, X_ASYM, y_af, asym_tot + PADDING, NODE_H + 2 * SP, C2['bg'])
    for fx, lab in zip(asymx, ASYM_FEATS):
        node(ax, fx, y_af, asym_w, NODE_H, lab, C2['nd'])
        for vx in vecx:
            line(ax, vx, y_feat + NODE_H / 2, fx, y_af - NODE_H / 2)

    # ── asymmetry downstream (from asymmetry_pipeline): the 2 features fan to ──
    #   right: L1/L2 scorers -> average 10 score -> full/1by1matched/ANCOVA
    #   left:  Logistic Regression -> full/1by1matched
    y_asc = y_d2          # scorers / Logistic Regression
    y_aav = y_d3          # average 10 score / left full-1by1
    y_aan = y_d3 + BAND   # analyses (full / 1 by 1 matched / ANCOVA)
    A_SC = ['L1 Norm\n$\\Sigma_i |f_i|$', 'L2 Norm\n$\\sqrt{\\Sigma_i f_i^2}$']
    a_sc_w = 3.0
    a_sc_grp = 2 * a_sc_w + GAP
    A_TRN = 'Logistic Regression'
    A_RIGHT = ['full', '1 by 1 matched', 'ANCOVA']
    A_LEFT = ['full', '1 by 1 matched']
    a_avg = 'average 10 score'
    a_anc_tot = _rw(A_RIGHT)
    a_left_tot = _rw(A_LEFT)
    a_right_vis = max(a_sc_grp, node_w(a_avg), a_anc_tot)
    a_left_vis = max(node_w(A_TRN), a_left_tot)
    a_col_gap = 2.0
    a_branch = a_left_vis + a_col_gap + a_right_vis
    a_x0 = X_ASYM - a_branch / 2          # centre both sub-columns under X_ASYM
    X_ASYM_L = a_x0 + a_left_vis / 2
    X_ASYM_R = a_x0 + a_left_vis + a_col_gap + a_right_vis / 2

    # split row (y_asc): scorers (right, violet) + Logistic Regression (left, lime)
    a_scx, _ = rowx(X_ASYM_R, 2, a_sc_w, GAP)
    cluster(ax, X_ASYM_R, y_asc, a_sc_grp + PADDING, NODE_H + 2 * SP, C_ASY['bg'])
    for x, lab in zip(a_scx, A_SC):
        node(ax, x, y_asc, a_sc_w, NODE_H, lab, C_ASY['nd'])
    a_trn_w = node_w(A_TRN)
    cluster(ax, X_ASYM_L, y_asc, a_trn_w + PADDING, NODE_H + 2 * SP, C4['bg'])
    node(ax, X_ASYM_L, y_asc, a_trn_w, NODE_H, A_TRN, C4['nd'])
    for fx in asymx:                      # 2 features feed both sub-columns
        line(ax, fx, y_af + NODE_H / 2, X_ASYM_L, y_asc - NODE_H / 2)
        for sx in a_scx:
            line(ax, fx, y_af + NODE_H / 2, sx, y_asc - NODE_H / 2)

    # right sub-column: scorers -> average 10 score -> analyses
    # (amber, not blue: sits beside the steel left full/1by1 at this band)
    cluster(ax, X_ASYM_R, y_aav, node_w(a_avg) + PADDING, NODE_H + 2 * SP, P[10]['bg'])
    node(ax, X_ASYM_R, y_aav, node_w(a_avg), NODE_H, a_avg, P[10]['nd'])
    for sx in a_scx:
        line(ax, sx, y_asc + NODE_H / 2, X_ASYM_R, y_aav - NODE_H / 2)
    a_ancx, _, _ = prow(ax, X_ASYM_R, y_aan, A_RIGHT,
                        [C_SA['nd'], C_ES['nd'], C_AOUT['nd']])
    cluster(ax, X_ASYM_R, y_aan, a_anc_tot + PADDING, NODE_H + 2 * SP, C_AOUT['bg'])
    for tx in a_ancx:
        line(ax, X_ASYM_R, y_aav + NODE_H / 2, tx, y_aan - NODE_H / 2)

    # left sub-column: Logistic Regression -> full/1by1matched
    a_lfx, _, _ = prow(ax, X_ASYM_L, y_aav, A_LEFT, [C_SA['nd'], C_ES['nd']])
    cluster(ax, X_ASYM_L, y_aav, a_left_tot + PADDING, NODE_H + 2 * SP, C_SA['bg'])
    for tx in a_lfx:
        line(ax, X_ASYM_L, y_asc + NODE_H / 2, tx, y_aav - NODE_H / 2)

    # ── embedding step 5 (left spine X_ORIG): face embedding vector -> all ->
    #    no_drop -> LR (the face embedding vector — formerly 'original' — feeds
    #    this classifier spine) ──
    all_w = node_w('all')
    cluster(ax, X_ORIG, y_d1, all_w + PADDING, NODE_H + 2 * SP, C_PA['bg'])
    node(ax, X_ORIG, y_d1, all_w, NODE_H, 'all', C_PA['nd'])
    line(ax, vecx[0], y_feat + NODE_H / 2, X_ORIG, y_d1 - NODE_H / 2)

    node(ax, X_ORIG, y_d2, node_w('no_drop'), NODE_H, 'no_drop', C4['nd'])
    line(ax, X_ORIG, y_d1 + NODE_H / 2, X_ORIG, y_d2 - NODE_H / 2)

    clf_label = 'Logistic Regression\n$C = 10^{-3}$'
    clf_w = node_w(clf_label)
    clf_h = hgt(clf_label)
    node(ax, X_ORIG, y_d3, clf_w, clf_h, clf_label, C4['nd'])
    line(ax, X_ORIG, y_d2 + NODE_H / 2, X_ORIG, y_d3 - NODE_H / 2)

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
    emax, _, ema_tot = prow(ax, X_EMO_C, y_d1, ['mean', 'all'], C_EMA['nd'])
    cluster(ax, X_EMO_C, y_d1, ema_tot + PADDING, NODE_H + 2 * SP, C_EMA['bg'])
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
    # group wrappers — one big cluster per same-colour group (drawn behind)
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

    # embedding: ArcFace (centre) + embedding vectors (centred below it); the
    # asymmetry features sit one band lower in their own C2 sub-cluster
    eL = X_EMB_C - vec_tot / 2 - PADX
    eR = X_EMB_C + vec_tot / 2 + PADX
    _grp(eL, eR, y_mr_top, y_ft_bot, C2['bg'])

    # emotion: models + features
    half = max(om_tot, total_of) / 2 + PADX
    _grp(X_EMO_C - half, X_EMO_C + half, y_mr_top, y_ft_bot, C_EMO['bg'])

    # embedding classifier path: no_drop + LR wrapped together (C4)
    half = max(node_w('no_drop'), clf_w) / 2 + PADX
    _grp(X_ORIG - half, X_ORIG + half,
         y_d2 - NODE_H / 2 - PADY, y_d3 + clf_h / 2 + PADY, C4['bg'])

    # ════════════════════════════════════════════════════════
    # step 4 — Demographic + Cohort  (shared by all branches; standalone for now)
    # ════════════════════════════════════════════════════════
    _demo_cohort(ax, dc_cx, s4_top)

    # ════════════════════════════════════════════════════════
    # embedding eval — single left-spine column under the classifier (X_ORIG):
    #   K fold -> all+1by1 -> eval_by_subject -> contrasts -> aggregator
    #   -> full/1by1matched -> violin   (reverse path + eval_by_visit removed)
    # ════════════════════════════════════════════════════════
    # K fold (fed by LR; reverse path removed)
    nw_k = node_w('K fold(K=10)')
    cluster(ax, X_ORIG, e_k_y, nw_k + PADDING, NODE_H + 2 * SP, CF['bg'])
    node(ax, X_ORIG, e_k_y, nw_k, NODE_H, 'K fold(K=10)', CF['nd'])
    line(ax, X_ORIG, y_d3 + clf_h / 2, X_ORIG, e_k_y - NODE_H / 2)   # LR -> K fold

    # per-subject score aggregation: mean of the 10 pics' scores
    sc_label = 'mean 10 pic score'
    nw_sc = node_w(sc_label)
    cluster(ax, X_ORIG, e_sc_y, nw_sc + PADDING, NODE_H + 2 * SP, C_POOL['bg'])
    node(ax, X_ORIG, e_sc_y, nw_sc, NODE_H, sc_label, C_POOL['nd'])
    line(ax, X_ORIG, e_k_y + NODE_H / 2, X_ORIG, e_sc_y - NODE_H / 2)   # K fold -> mean score

    # eval_by_subject (eval_by_visit removed) + the 3 contrasts share ONE C_ES cluster
    EB = 'eval_by_subject'
    CMP = ['AD vs HC', 'AD vs NAD', 'AD vs ACS']
    es_w = node_w(EB)
    cmp_w = max(node_w(l) for l in CMP)
    cmpx, cmp_tot = rowx(X_ORIG, 3, cmp_w, GAP)
    eb_cy = (e_eb_y + e_cmp_y) / 2
    eb_h = (e_cmp_y - e_eb_y) + NODE_H + 2 * SP
    cluster(ax, X_ORIG, eb_cy, max(es_w, cmp_tot) + PADDING, eb_h, C_EVAL['bg'])
    node(ax, X_ORIG, e_eb_y, es_w, NODE_H, EB, C_EVAL['nd'])
    line(ax, X_ORIG, e_sc_y + NODE_H / 2, X_ORIG, e_eb_y - NODE_H / 2)   # mean score -> eval_by_subject
    for x, lab in zip(cmpx, CMP):
        node(ax, x, e_cmp_y, cmp_w, NODE_H, lab, C_EVAL['nd'])
    for cx2 in cmpx:
        line(ax, X_ORIG, e_eb_y + NODE_H / 2, cx2, e_cmp_y - NODE_H / 2)

    # ── contrasts -> full/1by1matched -> violin (aggregator removed) ──
    efmx, _, efm_tot = prow(ax, X_ORIG, e_fm_y, ['full', '1 by 1 matched'],
                            [C_SA['nd'], C_ES['nd']])
    cluster(ax, X_ORIG, e_fm_y, efm_tot + PADDING, NODE_H + 2 * SP, C_SA['bg'])
    for cx2 in cmpx:
        for x in efmx:
            line(ax, cx2, e_cmp_y + NODE_H / 2, x, e_fm_y - NODE_H / 2)

    # ── Save ── (side is the chosen layout -> the primary file)
    suffix = '' if mode == 'side' else '_center'
    _save(fig, suffix)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build('side')
