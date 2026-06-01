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
# height of the whole shared block (Demographic + Cohort + matched head, incl.
# outer pads); used to size/position it. Mirrors the y-spacing in _demo_cohort.
_DC_H = 2 * _DC_PAD + 9 * NODE_H + 14 * SP + 0.6


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
    my = mt_top + SP + NODE_H / 2          # 1by1matched / caliper_group
    mt_bot = my + NODE_H / 2 + SP
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

    # matched-head sub-cluster (shared matching primitives)
    mtx, mt_tot = _rowx(cx, 2, 2.6, 0.6)
    cluster(ax, cx, my, mt_tot + 0.9, NODE_H + 2 * SP, C_ES['bg'])
    for x, lab in zip(mtx, ['1by1matched', 'caliper_group']):
        node(ax, x, my, 2.6, NODE_H, lab, C_ES['nd'])
    return s4_bot


def build(mode='center'):
    """mode='center' : Demographic+Cohort sits below the features (central spine).
       mode='side'   : Demographic+Cohort parked beside the whole chain (left)."""
    # ════ column centres (left -> right) ════
    if mode == 'side':
        X_AGE_C, X_EMB_C, X_EMO_C = 24.0, 40.0, 60.0
    else:
        X_AGE_C, X_EMB_C, X_EMO_C = 7.0, 24.0, 45.0
    PRE_CX = X_EMB_C        # preprocess aligned to the embedding centre-line

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
        x_left, x_right = -1.5, 73.0
    else:
        dc_cx = X_EMB_C                   # central, below the features
        s4_top = d3_bot + SP
        x_left, x_right = -1.5, 58.5
    dc_bot = s4_top + _DC_H

    # ════ parked matched-logic region (eval chain + Fwd/Rev), below everything ════
    park_top = max(d3_bot, dc_bot) + SP + 0.6
    ev_y1 = park_top + SP + NODE_H / 2       # subject_match / visit_match
    ev_y2 = ev_y1 + NODE_H + SP              # eval_by_subject / eval_by_visit
    ev_y3 = ev_y2 + NODE_H + SP              # priority
    ev_bot = ev_y3 + NODE_H / 2 + SP
    fr_y1 = park_top + SP + NODE_H / 2
    fr_y2 = fr_y1 + NODE_H + SP
    fr_y3 = fr_y2 + NODE_H + SP
    fr_bot = fr_y3 + NODE_H / 2 + SP
    park_bot = max(ev_bot, fr_bot)

    fig_h = max(dc_bot, d3_bot, park_bot) + 0.5

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
    cluster(ax, X_AGE_C, y_mod, am_tot + 0.9, NODE_H + 2 * SP, C_AGE['bg'])
    for x, lab in zip(amx, AGE_MODELS):
        node(ax, x, y_mod, NW_AM, NODE_H, lab, C_AGE['nd'])
        line(ax, x_face, y_pre5 + NODE_H / 2, x, y_mod - NODE_H / 2)

    # embedding models  (LIT)
    EMB_MODELS = ['dlib', 'TopoFR', 'ArcFace', 'VGGFace']
    nw_em = 2.2; gap_em = 0.5
    emx, em_tot = _rowx(X_EMB_C, len(EMB_MODELS), nw_em, gap_em)
    cluster(ax, X_EMB_C, y_mod, em_tot + 0.9, NODE_H + 2 * SP, C2['bg'])
    for x, lab in zip(emx, EMB_MODELS):
        node(ax, x, y_mod, nw_em, NODE_H, lab, C2['nd'])
        for src in [x_face, x_mirr]:
            line(ax, src, y_pre5 + NODE_H / 2, x, y_mod - NODE_H / 2)

    # emotion models  (LIT)
    EMO_MODELS = ['EmoNet', 'Open\nFace', 'FER', 'HS\nEmotion', 'Libre\nFace',
                  'DAN', 'POSTER\n++', 'Py-Feat', 'ViT']
    nw_om = 2.2; gap_om = 0.3
    omx, om_tot = _rowx(X_EMO_C, len(EMO_MODELS), nw_om, gap_om)
    cluster(ax, X_EMO_C, y_mod, om_tot + 0.9, NODE_H + 2 * SP, C_EMO['bg'])
    for x, lab in zip(omx, EMO_MODELS):
        node(ax, x, y_mod, nw_om, NODE_H, lab, C_EMO['nd'])
        line(ax, x_face, y_pre5 + NODE_H / 2, x, y_mod - NODE_H / 2)

    # ════════════════════════════════════════════════════════
    # step 3 — per-branch features
    # ════════════════════════════════════════════════════════
    # Age: Predict Age x 10
    age_w = 4.5
    cluster(ax, X_AGE_C, y_feat, age_w + 0.9, NODE_H + 2 * SP, C_AGE['bg'])
    node(ax, X_AGE_C, y_feat, age_w, NODE_H, 'Predict Age x 10', C_AGE['nd'])
    for x in amx:
        line(ax, x, y_mod + NODE_H / 2, X_AGE_C, y_feat - NODE_H / 2)

    # ── Age step 5: mean -> age_error -> violin/lines/scatter/stat (LIT) ──
    cluster(ax, X_AGE_C, y_d1, age_w + 0.9, NODE_H + 2 * SP, C_AOUT['bg'])
    node(ax, X_AGE_C, y_d1, age_w, NODE_H, 'Predict Age mean x 1', C_AOUT['nd'])
    line(ax, X_AGE_C, y_feat + NODE_H / 2, X_AGE_C, y_d1 - NODE_H / 2)

    ae_w = 5.5
    cluster(ax, X_AGE_C, y_d2, ae_w + 0.9, NODE_H + 2 * SP, C_AOUT['bg'])
    node(ax, X_AGE_C, y_d2, ae_w, NODE_H,
         'age_error = real_age - predicted_age', C_AOUT['nd'], fs=FS_SM)
    line(ax, X_AGE_C, y_d1 + NODE_H / 2, X_AGE_C, y_d2 - NODE_H / 2)

    vls_x, vls_tot = _rowx(X_AGE_C, 4, 2.6, 0.5)
    cluster(ax, X_AGE_C, y_d3, vls_tot + 0.9, NODE_H + 2 * SP, C_AOUT['bg'])
    for x, lab in zip(vls_x, ['violin', 'lines', 'scatter', 'stat']):
        node(ax, x, y_d3, 2.6, NODE_H, lab, C_AOUT['nd'])
        line(ax, X_AGE_C, y_d2 + NODE_H / 2, x, y_d3 - NODE_H / 2)

    # embedding features split into TWO clusters (mirrors age_emb_pipeline_mpl):
    #   - 'original'                       (single-node cluster)
    #   - 'diff / |diff| / rel_diff / |rel_diff|'  (asymmetry cluster)
    nw_ef = 1.9; gap_ef = 0.3
    ASYM_FEATS = ['diff', '|diff|', 'rel_diff', '|rel_diff|']
    orig_cl_w = nw_ef + 0.6
    _, asym_tot = _rowx(0, len(ASYM_FEATS), nw_ef, gap_ef)
    asym_cl_w = asym_tot + 0.6
    gap_cl = 1.0
    pair_w = orig_cl_w + gap_cl + asym_cl_w
    pair_left = X_EMB_C - pair_w / 2
    x_orig = pair_left + orig_cl_w / 2
    asym_c = pair_left + orig_cl_w + gap_cl + asym_cl_w / 2
    asymx, _ = _rowx(asym_c, len(ASYM_FEATS), nw_ef, gap_ef)

    cluster(ax, x_orig, y_feat, orig_cl_w, NODE_H + 2 * SP, C2['bg'])
    node(ax, x_orig, y_feat, nw_ef, NODE_H, 'original', C2['nd'])
    cluster(ax, asym_c, y_feat, asym_cl_w, NODE_H + 2 * SP, C2['bg'])
    for x, lab in zip(asymx, ASYM_FEATS):
        node(ax, x, y_feat, nw_ef, NODE_H, lab, C2['nd'])
    for fx in [x_orig] + asymx:
        for ex in emx:
            line(ax, ex, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

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
        cluster(ax, c, y_feat, tot + 0.6, NODE_H + 2 * SP, C_EMO['bg'])
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
    # step 4 — Demographic + Cohort  (shared by all branches; standalone for now)
    # ════════════════════════════════════════════════════════
    _demo_cohort(ax, dc_cx, s4_top)

    # ════════════════════════════════════════════════════════
    # matched logic — parked in the bottom-right corner, all DIMMED
    #   (1by1matched / caliper_group now live in the shared cluster on the left)
    # ════════════════════════════════════════════════════════
    EV_CX = x_right - 17.0
    FR_CX = x_right - 6.0

    # eval chain: subject_match..priority wrapped in ONE big dimmed cluster
    nw_ev = 2.4; gap_ev = 0.65
    esx, es_tot = _rowx(EV_CX, 2, nw_ev, gap_ev)
    msx, ms_tot = _rowx(EV_CX, 3, 2.4, 0.65)
    cluster(ax, EV_CX, (park_top + ev_bot) / 2,
            max(es_tot, ms_tot) + 1.4, ev_bot - park_top, G['bg'])
    for x, lab in zip(esx, ['subject_match', 'visit_match']):
        node(ax, x, ev_y1, nw_ev, NODE_H, lab, G['nd'])
    for x, lab in zip(esx, ['eval_by_subject', 'eval_by_visit']):
        node(ax, x, ev_y2, nw_ev, NODE_H, lab, G['nd'])
    for x, lab in zip(msx, ['no_priority', 'priority_acs', 'priority_nad']):
        node(ax, x, ev_y3, 2.4, NODE_H, lab, G['nd'])
    for ya, yb, xbs in [(ev_y1, ev_y2, esx), (ev_y2, ev_y3, msx)]:
        for xa in esx:
            for xb in xbs:
                _dln(xa, ya + NODE_H / 2, xb, yb - NODE_H / 2)

    # Fwd / Rev cohort clusters — dimmed
    nw_col = 3.3; gap_col = 0.7; cw = nw_col + 0.8
    xl = FR_CX - (cw + gap_col) / 2
    xr = FR_CX + (cw + gap_col) / 2
    cl_h = fr_bot - park_top
    cluster(ax, xl, (park_top + fr_bot) / 2, cw, cl_h, G['bg'])
    node(ax, xl, fr_y1, nw_col, NODE_H, 'Full Cohort', G['nd'])
    node(ax, xl, fr_y2, nw_col, NODE_H, 'OOF Scores', G['nd'])
    node(ax, xl, fr_y3, nw_col, NODE_H,
         'Full Cohort Eval\nMatched Subset Eval (1:1)', G['nd'], fs=FS_SM)
    cluster(ax, xr, (park_top + fr_bot) / 2, cw, cl_h, G['bg'])
    node(ax, xr, fr_y1, nw_col, NODE_H, 'Matched Cohort', G['nd'])
    node(ax, xr, fr_y2, nw_col, NODE_H, 'Predict Full Cohort', G['nd'], fs=FS_SM)
    node(ax, xr, fr_y3, nw_col, NODE_H,
         'Matched OOF Eval\nUnmatched Eval', G['nd'], fs=FS_SM)
    for col in [xl, xr]:
        for ya, yb in [(fr_y1, fr_y2), (fr_y2, fr_y3)]:
            _dln(col, ya + NODE_H / 2, col, yb - NODE_H / 2)

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
