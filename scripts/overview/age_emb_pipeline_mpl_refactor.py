"""scripts/overview/age_emb_pipeline_mpl_refactor.py
Deconstructed age-pipeline overview (matplotlib): Preprocess -> Demographic ->
Cohort centre column, lit Age branch through age_error, isolated eval-head and
Fwd/Rev clusters, per-branch partitions removed.

Output:
  workspace/overview/age_emb_pipeline_mpl_refactor.png

Usage:
    python scripts/overview/age_emb_pipeline_mpl_refactor.py
"""
import matplotlib.pyplot as plt
from draw_age_emb_pipeline_common import *


def build():
    """v6 (work-in-progress): v5_refactor with the embedding, emotion AND
    asymmetry branch modules stripped, and the per-branch partitions removed.
    Two clusters are pulled out as isolated floating blocks:
      - the embedding-side Fwd/Rev cluster (nudged to the lower-right)
      - the eval-head 1by1matched/caliper_group cluster (no in/out)

    Center column, top→bottom: Preprocessing → Demographic (own cluster, 8 key
    columns) → Cohort. Age branch (Part A) lit through age_error.

    Output:
      workspace/overview/age_emb_pipeline_mpl_refactor.png
    """
    CX_S = 20.0; CX_E = 14.5; X_A = 26.0; X_EMO = 42.0
    FIG_W_V3 = 56.0
    age_w = 4.5; age_h = NODE_H

    nw_dem = 3.0
    nw_pv = 2.0; gap_pv = 0.4
    nw_ps = 2.6; gap_ps = 0.3
    nw_hs = 3.6; gap_hs = 0.3
    nw_co = 4.2
    total_ps = 4 * nw_ps + 3 * gap_ps
    total_hs = 2 * nw_hs + gap_hs

    c1_top = 0.5
    y_dem = c1_top + SP + NODE_H / 2
    y_pv  = y_dem + NODE_H + SP
    y_hv  = y_pv  + NODE_H + SP
    y_ps  = y_hv  + NODE_H + SP
    y_hs  = y_ps  + NODE_H + SP
    y_fc  = y_hs  + NODE_H + SP
    c1_bot = y_fc + NODE_H / 2 + SP
    c1_w = max(total_ps, total_hs, nw_co) + 0.9

    nw_bg = 2.4; gap_bg = 0.65
    nw_mir = 2.4; gap_mir = 1.0
    x_face = CX_E
    x_mirr = X_A
    c_pre_top = c1_bot + SP
    y_pre1 = c_pre_top + SP + NODE_H / 2
    y_pre2 = y_pre1 + NODE_H + SP
    y_pre_bg = y_pre2 + NODE_H + SP
    y_pre_align = y_pre_bg + NODE_H + SP
    y_mir = y_pre_align + NODE_H + SP
    c_pre_bot = y_mir + NODE_H / 2 + SP
    pre_cluster_w = 2 * max(CX_S - (x_face - nw_mir / 2),
                            (x_mirr + nw_mir / 2) - CX_S) + 0.9

    nw_emb = 2.2; gap_emb = 0.5
    total_emb = 4 * nw_emb + 3 * gap_emb
    c_emb_top = c_pre_bot + SP
    y_emb = c_emb_top + SP + NODE_H / 2
    c_emb_bot = y_emb + NODE_H / 2 + SP
    emx = [CX_S + (i - 1.5) * (nw_emb + gap_emb) for i in range(4)]

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
    x_ft_asym = [X_A - total_asym / 2 + nw_ft / 2
                 + i * (nw_ft + gap_ft) for i in range(4)]
    asym_nw = 3.5

    pre_right = CX_S + pre_cluster_w / 2
    COH_X = pre_right + 2.0 + c1_w / 2
    coh_dy = c_pre_bot - c1_bot
    cy_dem = y_dem + coh_dy
    cy_pv  = y_pv  + coh_dy
    cy_hv  = y_hv  + coh_dy
    cy_ps  = y_ps  + coh_dy
    cy_hs  = y_hs  + coh_dy
    cy_fc  = y_fc  + coh_dy
    coh_cy = (c1_top + c1_bot) / 2 + coh_dy

    fig_h = c6_bot + 0.5
    fig, ax = plt.subplots(figsize=(FIG_W_V3, fig_h))
    ax.set_xlim(-5.0, 54)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    lit = [True]

    def _n(ax, cx, cy, w, h, text, fc, on=None):
        use = lit[0] if on is None else on
        node(ax, cx, cy, w, h, text, fc if use else G['nd'])

    def _cl(ax, cx, cy, w, h, fc, on=None):
        use = lit[0] if on is None else on
        cluster(ax, cx, cy, w, h, fc if use else G['bg'])

    # ════ Demographic — own cluster between preprocess and cohort ════
    # Group~Sex fan UP above the Demographic block; Age~Global_CDR fan DOWN below.
    # Own colour scheme (C3, distinct from the C1 Cohort below).
    # Rows differ in length (4 vs 5) → centre each row independently.
    DEMO_ABOVE = ['Group', 'Number', 'Photo_Session', 'Sex']
    DEMO_BELOW = ['Age', 'BMI', 'MMSE', 'CASI', 'Global_CDR']
    nw_dc = 2.4; gap_dc = 0.3

    def _demo_rowx(n):
        tot = n * nw_dc + (n - 1) * gap_dc
        return [CX_S - tot / 2 + nw_dc / 2 + i * (nw_dc + gap_dc)
                for i in range(n)]
    above_x = _demo_rowx(len(DEMO_ABOVE))
    below_x = _demo_rowx(len(DEMO_BELOW))
    n_wide = max(len(DEMO_ABOVE), len(DEMO_BELOW))
    total_dc = n_wide * nw_dc + (n_wide - 1) * gap_dc
    demo_top = c_pre_bot + SP
    dy_above = demo_top + SP + NODE_H / 2
    dy_block = dy_above + NODE_H + SP
    dy_below = dy_block + NODE_H + SP
    demo_bot = dy_below + NODE_H / 2 + SP
    _cl(ax, CX_S, (demo_top + demo_bot) / 2, total_dc + 0.9,
        demo_bot - demo_top, C3['bg'])
    _n(ax, CX_S, dy_block, nw_dem, NODE_H, 'Demographic', C3['nd'])
    for x, lab in zip(above_x, DEMO_ABOVE):
        _n(ax, x, dy_above, nw_dc, NODE_H, lab, C3['nd'])
        line(ax, CX_S, dy_block - NODE_H / 2, x, dy_above + NODE_H / 2)
    for x, lab in zip(below_x, DEMO_BELOW):
        _n(ax, x, dy_below, nw_dc, NODE_H, lab, C3['nd'])
        line(ax, CX_S, dy_block + NODE_H / 2, x, dy_below - NODE_H / 2)

    # ════ Cohort — below Demographic (LIT, detached: no wiring) ════
    coh_top = demo_bot + SP
    cy_pv = coh_top + SP + NODE_H / 2
    cy_hv = cy_pv + NODE_H + SP
    cy_ps = cy_hv + NODE_H + SP
    cy_hs = cy_ps + NODE_H + SP
    cy_fc = cy_hs + NODE_H + SP
    coh_bot = cy_fc + NODE_H / 2 + SP
    _cl(ax, CX_S, (coh_top + coh_bot) / 2, c1_w, coh_bot - coh_top, C1['bg'])
    pvx = [CX_S - (nw_pv + gap_pv) / 2, CX_S + (nw_pv + gap_pv) / 2]
    for x, lab in zip(pvx, ['P: first', 'P: all']):
        _n(ax, x, cy_pv, nw_pv, NODE_H, lab, C1['nd'])
    hvx = [CX_S - (nw_pv + gap_pv) / 2, CX_S + (nw_pv + gap_pv) / 2]
    for x, lab in zip(hvx, ['HC: first', 'HC: all']):
        _n(ax, x, cy_hv, nw_pv, NODE_H, lab, C1['nd'])
    for px in pvx:
        for hx in hvx:
            line(ax, px, cy_pv + NODE_H / 2, hx, cy_hv - NODE_H / 2)
    psx = [CX_S + (i - 1.5) * (nw_ps + gap_ps) for i in range(4)]
    P_SCORE_LABELS = ['P: CDR all', 'P: CDR >=0.5', 'P: CDR >=1', 'P: CDR >=2']
    for x, lab in zip(psx, P_SCORE_LABELS):
        _n(ax, x, cy_ps, nw_ps, NODE_H, lab, C1['nd'])
    for hx in hvx:
        for sx in psx:
            line(ax, hx, cy_hv + NODE_H / 2, sx, cy_ps - NODE_H / 2)
    hsx = [CX_S - (nw_hs + gap_hs) / 2, CX_S + (nw_hs + gap_hs) / 2]
    HC_SCORE_LABELS = ['HC: CDR all\nor MMSE all',
                       'HC: CDR =0\nor MMSE >=26']
    for x, lab in zip(hsx, HC_SCORE_LABELS):
        _n(ax, x, cy_hs, nw_hs, NODE_H, lab, C1['nd'])
    for sx in psx:
        for hx2 in hsx:
            line(ax, sx, cy_ps + NODE_H / 2, hx2, cy_hs - NODE_H / 2)
    _n(ax, CX_S, cy_fc, nw_co, NODE_H, "Cohort (P + NAD + ACS)", C1['nd'])
    for hx2 in hsx:
        line(ax, hx2, cy_hs + NODE_H / 2, CX_S, cy_fc - NODE_H / 2)

    # ════ Demographic → Cohort cluster FRAME ════
    # Group/Number/Photo_Session/MMSE/Global_CDR feed cohort construction; the
    # cohort has no single target block, so connect to its top frame
    # (intentionally block→frame, unlike the rest — kept simple for now).
    coh_left = CX_S - c1_w / 2
    coh_right = CX_S + c1_w / 2
    for cx, cyy in [(above_x[0], dy_above), (above_x[1], dy_above),
                    (above_x[2], dy_above), (below_x[2], dy_below),
                    (below_x[4], dy_below)]:
        tx = min(max(cx, coh_left + 0.3), coh_right - 0.3)
        line(ax, cx, cyy, tx, coh_top)

    # ════ Preprocessing (LIT) ════
    _cl(ax, CX_S, (c_pre_top + c_pre_bot) / 2,
        pre_cluster_w, c_pre_bot - c_pre_top, C_PRE['bg'])
    _n(ax, CX_S, y_pre1, NW_PRE, NODE_H, 'Detect', C_PRE['nd'])
    _n(ax, CX_S, y_pre2, NW_PRE, NODE_H, 'Select', C_PRE['nd'])
    line(ax, CX_S, y_pre1 + NODE_H / 2, CX_S, y_pre2 - NODE_H / 2)
    bgx = [CX_S - (nw_bg + gap_bg) / 2, CX_S + (nw_bg + gap_bg) / 2]
    for x, lab in zip(bgx, ['no_background', 'background']):
        _n(ax, x, y_pre_bg, nw_bg, NODE_H, lab, C_PRE['nd'])
    for bx in bgx:
        line(ax, CX_S, y_pre2 + NODE_H / 2, bx, y_pre_bg - NODE_H / 2)
    _n(ax, CX_S, y_pre_align, NW_PRE, NODE_H, 'Align', C_PRE['nd'])
    for bx in bgx:
        line(ax, bx, y_pre_bg + NODE_H / 2, CX_S, y_pre_align - NODE_H / 2)
    _n(ax, x_face, y_mir, nw_mir, NODE_H, 'face x 10', C_PRE['nd'])
    _n(ax, x_mirr, y_mir, nw_mir, NODE_H, 'mirrored_face\nx 20', C_PRE['nd'])
    line(ax, CX_S, y_pre_align + NODE_H / 2, x_face, y_mir - NODE_H / 2)
    line(ax, CX_S, y_pre_align + NODE_H / 2, x_mirr, y_mir - NODE_H / 2)

    lit[0] = False

    # External Datasets — re-anchored alongside preprocess (left)
    eacs_w = age_w + 0.6
    eacs_h = NODE_H * 1.6
    y_ext = (c_pre_top + c_pre_bot) / 2
    _cl(ax, X_AGE, y_ext, eacs_w + 0.6, eacs_h + 2 * SP, C_EACS['bg'])
    _n(ax, X_AGE, y_ext, eacs_w, eacs_h,
       "External Datasets\nUTKFace / AgeDB / APPA-REAL / IMDB\n"
       "MegaAge / FairFace / SZU-EmoDage\nAFAD / DiverseAsian",
       C_EACS['nd'])

    # Embedding Models — REMOVED (sole consumer Asymmetry is gone;
    # mirrored_face now feeds nothing).

    # ════ Age branch: ONE big cluster around models → Predict Age x10 → mean x1 ════
    # (age_error is kept as its OWN separate cluster, below.)
    pred_cw = TOTAL_AM + 0.9
    am_top = y_emb - age_h / 2 - SP
    _cl(ax, X_AGE, (am_top + c_pa_bot) / 2, pred_cw, c_pa_bot - am_top,
        C_AGE['bg'], on=True)
    for x, lab in zip(amx, AGE_MODELS):
        _n(ax, x, y_emb, NW_AM, age_h, lab, C_AGE['nd'], on=True)
    for x in amx:
        line(ax, x_face, y_mir + NODE_H / 2, x, y_emb - NODE_H / 2)
    for x in amx:
        line(ax, X_AGE, y_ext + eacs_h / 2, x, y_emb - age_h / 2)

    _n(ax, X_AGE, y_ft, age_w, age_h, "Predict Age x 10", C_AGE['nd'], on=True)
    for x in amx:
        line(ax, x, y_emb + age_h / 2, X_AGE, y_ft - age_h / 2)

    _n(ax, X_AGE, y_pa, age_w, age_h, "Predict Age mean x 1", C_AGE['nd'], on=True)
    line(ax, X_AGE, y_ft + age_h / 2, X_AGE, y_pa - age_h / 2)

    ae_node_w = 5.5
    _n(ax, X_AGE, y_pd, ae_node_w, age_h,
       "age_error = real_age - predicted_age", C_AOUT['nd'], on=True)
    line(ax, X_AGE, y_pa + age_h / 2, X_AGE, y_pd - age_h / 2)
    # real_age feed: demographic 'Age' column → age_error (right edge)
    line(ax, below_x[0], dy_below + NODE_H / 2, X_AGE + ae_node_w / 2, y_pd)

    # ════ age_error + violin / lines / scatter — ONE shared cluster (C_AOUT, lit) ════
    # age_error is the source (orange node); real age also feeds the three plots.
    nw_vls = 2.6; gap_vls = 0.5
    total_vls = 3 * nw_vls + 2 * gap_vls
    vls_top = c_pd_bot + SP
    y_vls = vls_top + SP + NODE_H / 2
    vls_bot = y_vls + NODE_H / 2 + SP
    vls_x = [X_AGE + (i - 1) * (nw_vls + gap_vls) for i in range(3)]
    ae_merge_top = y_pd - NODE_H / 2 - SP        # extend up to enclose age_error
    _cl(ax, X_AGE, (ae_merge_top + vls_bot) / 2, total_vls + 0.9,
        vls_bot - ae_merge_top, C_AOUT['bg'], on=True)
    for x, lab in zip(vls_x, ['violin', 'lines', 'scatter']):
        _n(ax, x, y_vls, nw_vls, NODE_H, lab, C_AOUT['nd'], on=True)
        line(ax, X_AGE, y_pd, x, y_vls)               # age_error → consumer
        line(ax, below_x[0], dy_below, x, y_vls)      # real age → consumer

    # age → histogram ; age + Sex + BMI → stat. Placed in the gap between the
    # age-prediction column and the demographic/cohort column.
    x_mid = ((X_AGE + pred_cw / 2) + (CX_S - (total_dc + 0.9) / 2)) / 2
    nw_o = 2.4
    y_stat = dy_block
    y_hist = dy_below
    _n(ax, x_mid, y_stat, nw_o, NODE_H, 'stat', C4['nd'], on=True)
    _n(ax, x_mid, y_hist, nw_o, NODE_H, 'histogram', C4['nd'], on=True)
    line(ax, below_x[0], dy_below, x_mid, y_hist)     # Age → histogram
    line(ax, below_x[0], dy_below, x_mid, y_stat)     # Age → stat
    line(ax, below_x[1], dy_below, x_mid, y_stat)     # BMI → stat
    line(ax, above_x[3], dy_above, x_mid, y_stat)     # Sex → stat

    # (age calibration & age outputs removed)

    # ════ Fwd/Rev — nudged further to the lower-right (isolated) ════
    fr_cx = (CX_S + (2 * nw_ev + gap_ev + 0.9) / 2
             + 1.5 + (cw + gap_col) / 2 + cw / 2) + 4.0
    fr_cy = y_sa
    fr_rp = NODE_H + SP
    fr_y1 = fr_cy - fr_rp
    fr_y2 = fr_cy
    fr_y3 = fr_cy + fr_rp
    xl = fr_cx - (cw + gap_col) / 2
    xr = fr_cx + (cw + gap_col) / 2
    _cl(ax, xl, fr_cy, cw, cl_h, CF['bg'])
    _n(ax, xl, fr_y1, nw_col, NODE_H, "Full Cohort", CF['nd'])
    _n(ax, xl, fr_y2, nw_col, NODE_H, "OOF Scores", CF['nd'])
    _n(ax, xl, fr_y3, nw_col, NODE_H,
       "Full Cohort Eval\nMatched Subset Eval (1:1)", CF['nd'])
    _cl(ax, xr, fr_cy, cw, cl_h, CR['bg'])
    _n(ax, xr, fr_y1, nw_col, NODE_H, "Matched Cohort", CR['nd'])
    _n(ax, xr, fr_y2, nw_col, NODE_H, "Predict Full Cohort", CR['nd'])
    _n(ax, xr, fr_y3, nw_col, NODE_H,
       "Matched OOF Eval\nUnmatched Eval", CR['nd'])
    for col in [xl, xr]:
        for ya, yb in [(fr_y1, fr_y2), (fr_y2, fr_y3)]:
            line(ax, col, ya + NODE_H / 2, col, yb - NODE_H / 2)

    # ════ Asymmetry branch — REMOVED ════

    # ════ Emotion branch — REMOVED ════

    # ════ Eval head 1by1matched/caliper — ISOLATED (no in/out) ════
    CX_EV = CX_S
    total_em_ev = 2 * nw_ev + gap_ev
    _cl(ax, CX_EV, (c_em_top + c_em_bot) / 2,
        total_em_ev + 0.9, c_em_bot - c_em_top, C_ES['bg'])
    esx = [CX_EV - (nw_ev + gap_ev) / 2, CX_EV + (nw_ev + gap_ev) / 2]
    for x, lab in zip(esx, ['1by1matched', 'caliper_group']):
        _n(ax, x, y_em, nw_ev, NODE_H, lab, C_ES['nd'])

    # ════ Rest of eval chain — kept, now headless (no eval-head feed) ════
    _cl(ax, CX_EV, (c_ml_top + c_ml_bot) / 2,
        total_em_ev + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    mlx = [CX_EV - (nw_ev + gap_ev) / 2, CX_EV + (nw_ev + gap_ev) / 2]
    for x, lab in zip(mlx, ['subject_match', 'visit_match']):
        _n(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'])

    _cl(ax, CX_EV, (c_sa_top + c_sa_bot) / 2,
        total_em_ev + 0.9, c_sa_bot - c_sa_top, C_SA['bg'])
    sax = [CX_EV - (nw_ev + gap_ev) / 2, CX_EV + (nw_ev + gap_ev) / 2]
    for x, lab in zip(sax, ['eval_by_subject', 'eval_by_visit']):
        _n(ax, x, y_sa, nw_ev, NODE_H, lab, C_SA['nd'])
    for mx in mlx:
        for sx in sax:
            line(ax, mx, y_ml + NODE_H / 2, sx, y_sa - NODE_H / 2)

    _cl(ax, CX_EV, (c_ms_top + c_ms_bot) / 2,
        total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX_EV - (nw_ms + gap_ms), CX_EV, CX_EV + (nw_ms + gap_ms)]
    for x, lab in zip(msx, ['no_priority', 'priority_acs', 'priority_nad']):
        _n(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'])
    for sx in sax:
        for mx in msx:
            line(ax, sx, y_sa + NODE_H / 2, mx, y_ms - NODE_H / 2)

    # (all branch → eval-head feeds cut: eval head is isolated)

    # ── Per-branch partitions ── REMOVED (AD vs HC ... CASI hi/lo; all 4 clusters)

    # ── Save ──
    out = OUT / "age_emb_pipeline_mpl_refactor.png"
    fig.savefig(out, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
