"""scripts/overview/asymmetry_pipeline.py
Standalone diagram of the asymmetry scoring branch only (no original/classifier
path, no emotion/age branches, no eval chain):

  preprocessing (Detect..face/mirror)
    -> ArcFace embedding model
    -> face / mirrored_face embedding vectors (x 10 each)
    -> 2 asymmetry features: diff (Li-Ri) / rel_diff ((Li-Ri)/sqrt(Li^2+Ri^2))
    -> 2 norm scorers: L1 / L2
    -> downstream splits into two columns:
         right: scorers -> average 10 score -> full / 1 by 1 matched / ANCOVA
         left:  Logistic Regression -> full / 1 by 1 matched

Reuses the V2 drawer toolkit (draw_pipeline_v2_common) unchanged.

Output (workspace/overview/):
  asymmetry_pipeline.png (dpi=300) / .svg / .pdf

Usage:
    python scripts/overview/asymmetry_pipeline.py
"""
import matplotlib.pyplot as plt
from draw_pipeline_v2_common import *


def _save(fig, stem):
    """Write the figure as 300-dpi PNG plus SVG/PDF vectors (same tight bbox)."""
    common = dict(bbox_inches='tight', pad_inches=0.15, facecolor='white')
    png = OUT / f"{stem}.png"
    fig.savefig(png, dpi=300, **common)
    for ext in ('svg', 'pdf'):
        fig.savefig(OUT / f"{stem}.{ext}", **common)
    print(f"Saved: {png}  (+ .svg / .pdf)")


def build():
    BAND = NODE_H + 2 * SP + 0.3          # cluster band pitch (0.3 gap), as in V3
    _rw = lambda labs: sum(node_w(l) for l in labs) + (len(labs) - 1) * GAP

    MIR = ['face x 10', 'mirrored_face\nx 10']
    BG = ['no_background', 'background']
    EMB_MODELS = ['ArcFace']
    VECS = ['face embedding\nvector x 10', 'mirrored_face embedding\nvector x 10']
    ASYM = ['$L_i - R_i$\nvector x 10',
            '$(L_i - R_i)/\\sqrt{L_i^2 + R_i^2}$\nvector x 10']
    # order follows src/embedding/classification/scorer.py ASYMMETRY_METHODS
    # (l1_norm, l2_norm, ...): only the two norm scorers kept.
    SCORERS = ['L1 Norm\n$\\Sigma_i |f_i|$',
               'L2 Norm\n$\\sqrt{\\Sigma_i f_i^2}$']
    sc_w = 3.0                            # uniform scorer width (LaTeX labels)
    grp_w = 2 * sc_w + GAP                # the L1/L2 norm group
    vec_w = 3.4                           # uniform embedding-vector width
    vec_tot = 2 * vec_w + GAP
    asym_w = 4.5                          # uniform asymmetry-feature width (LaTeX)
    asym_tot = 2 * asym_w + GAP

    # downstream splits in two: scorer column (right) | classifier column (left)
    TRAINER = 'Logistic Regression'
    LEFT_ANALYSES = ['full', '1 by 1 matched']
    ANALYSES = ['full', '1 by 1 matched', 'ANCOVA']
    avg_label = 'average 10 score'
    anc_tot = _rw(ANALYSES)
    left_anc_tot = _rw(LEFT_ANALYSES)
    avg_w = node_w(avg_label)
    right_vis = max(grp_w, avg_w, anc_tot) + PADDING       # scorer/avg/analyses col
    left_vis = max(node_w(TRAINER), left_anc_tot) + PADDING
    col_gap = 3.0                         # gap between the two branch columns
    branch_total = left_vis + col_gap + right_vis

    pre_w = max(_rw(MIR), _rw(BG), node_w('Detect')) + PADDING
    em_tot = _rw(EMB_MODELS)

    # ── canvas: centre everything on CX; xlim hugs the widest row (branch split) ──
    half = max(branch_total, asym_tot + PADDING, vec_tot + PADDING,
               em_tot + PADDING, pre_w) / 2
    CX = half + 1.0
    x_left = CX - half - 0.8
    x_right = CX + half + 0.8

    branch_left = CX - branch_total / 2   # centre the two columns under CX
    X_L = branch_left + left_vis / 2
    X_R = branch_left + left_vis + col_gap + right_vis / 2

    # ── vertical bands (preprocessing at base pitch, then BAND) ──
    top = 0.5
    y_pre1 = top + SP + NODE_H / 2          # Detect
    y_pre2 = y_pre1 + NODE_H + SP           # Select
    y_pre3 = y_pre2 + NODE_H + SP           # no_background / background
    y_pre4 = y_pre3 + NODE_H + SP           # Align
    y_pre5 = y_pre4 + NODE_H + SP           # face / mirrored_face
    pre_top = top
    pre_bot = y_pre5 + NODE_H / 2 + SP

    y_mod = y_pre5 + BAND                   # embedding models
    y_vec = y_mod + BAND                    # face / mirrored_face embedding vectors
    y_feat = y_vec + BAND                   # asymmetry features (diff, rel_diff)
    y_sc = y_feat + BAND                    # scorers
    y_avg = y_sc + BAND                     # average 10 score (single box)
    y_anc = y_avg + BAND                    # full / 1 by 1 matched / ANCOVA
    bot = y_anc + NODE_H / 2 + SP

    fig_h = bot + 0.5
    fig, ax = plt.subplots(figsize=(x_right - x_left, fig_h))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── preprocessing (shared head) ──
    cluster(ax, CX, (pre_top + pre_bot) / 2, pre_w, pre_bot - pre_top, C_PRE['bg'])
    node(ax, CX, y_pre1, node_w('Detect'), NODE_H, 'Detect', C_PRE['nd'])
    node(ax, CX, y_pre2, node_w('Select'), NODE_H, 'Select', C_PRE['nd'])
    line(ax, CX, y_pre1 + NODE_H / 2, CX, y_pre2 - NODE_H / 2)
    bgx, _, _ = prow(ax, CX, y_pre3, BG, C_PRE['nd'])
    for x in bgx:
        line(ax, CX, y_pre2 + NODE_H / 2, x, y_pre3 - NODE_H / 2)
    node(ax, CX, y_pre4, node_w('Align'), NODE_H, 'Align', C_PRE['nd'])
    for x in bgx:
        line(ax, x, y_pre3 + NODE_H / 2, CX, y_pre4 - NODE_H / 2)
    mirx, _, _ = prow(ax, CX, y_pre5, MIR, C_PRE['nd'])
    for mx in mirx:
        line(ax, CX, y_pre4 + NODE_H / 2, mx, y_pre5 - NODE_H / 2)

    # ── embedding models -> vectors -> asymmetry features (one C2 group) ──
    emx, _, _ = prow(ax, CX, y_mod, EMB_MODELS, C2['nd'])
    for ex in emx:
        for mx in mirx:                    # face / mirrored_face -> each model
            line(ax, mx, y_pre5 + NODE_H / 2, ex, y_mod - NODE_H / 2)

    vecx, _ = rowx(CX, 2, vec_w, GAP)       # face / mirrored_face embedding vectors
    for vx, lab in zip(vecx, VECS):
        node(ax, vx, y_vec, vec_w, NODE_H, lab, C2['nd'])
    for ex in emx:
        for vx in vecx:
            line(ax, ex, y_mod + NODE_H / 2, vx, y_vec - NODE_H / 2)

    asymx, _ = rowx(CX, 2, asym_w, GAP)     # diff (Li-Ri) / rel_diff
    for fx, lab in zip(asymx, ASYM):
        node(ax, fx, y_feat, asym_w, NODE_H, lab, C2['nd'])
    for vx in vecx:
        for fx in asymx:
            line(ax, vx, y_vec + NODE_H / 2, fx, y_feat - NODE_H / 2)

    g_top = y_mod - NODE_H / 2 - SP
    g_bot = y_feat + NODE_H / 2 + SP
    cluster(ax, CX, (g_top + g_bot) / 2,
            max(em_tot, vec_tot, asym_tot) + PADDING, g_bot - g_top, C2['bg'])

    # ── branch split: scorers (right) vs Logistic Regression (left), fed by asymmetry ──
    SC_NORM = C_ASY
    scx, _ = rowx(X_R, 2, sc_w, GAP)
    cluster(ax, X_R, y_sc, grp_w + PADDING, NODE_H + 2 * SP, SC_NORM['bg'])
    for x, lab in zip(scx, SCORERS):
        node(ax, x, y_sc, sc_w, NODE_H, lab, SC_NORM['nd'])
    trn_w = node_w(TRAINER)
    cluster(ax, X_L, y_sc, trn_w + PADDING, NODE_H + 2 * SP, CF['bg'])
    node(ax, X_L, y_sc, trn_w, NODE_H, TRAINER, CF['nd'])
    for fx in asymx:                       # asymmetry features feed both columns
        line(ax, fx, y_feat + NODE_H / 2, X_L, y_sc - NODE_H / 2)
        for sx in scx:
            line(ax, fx, y_feat + NODE_H / 2, sx, y_sc - NODE_H / 2)

    # ── right column: scorers -> average 10 score -> full/1by1matched/ANCOVA ──
    cluster(ax, X_R, y_avg, avg_w + PADDING, NODE_H + 2 * SP, C1['bg'])
    node(ax, X_R, y_avg, avg_w, NODE_H, avg_label, C1['nd'])
    for sx in scx:
        line(ax, sx, y_sc + NODE_H / 2, X_R, y_avg - NODE_H / 2)
    anc_nd = [C_SA['nd'], C_ES['nd'], C_AOUT['nd']]
    ancx, _, _ = prow(ax, X_R, y_anc, ANALYSES, anc_nd)
    cluster(ax, X_R, y_anc, anc_tot + PADDING, NODE_H + 2 * SP, C_AOUT['bg'])
    for tx in ancx:
        line(ax, X_R, y_avg + NODE_H / 2, tx, y_anc - NODE_H / 2)

    # ── left column: Logistic Regression -> full / 1 by 1 matched (2 boxes, as right) ──
    le_nd = [C_SA['nd'], C_ES['nd']]
    lex, _, _ = prow(ax, X_L, y_avg, LEFT_ANALYSES, le_nd)
    cluster(ax, X_L, y_avg, left_anc_tot + PADDING, NODE_H + 2 * SP, C_AOUT['bg'])
    for tx in lex:
        line(ax, X_L, y_sc + NODE_H / 2, tx, y_avg - NODE_H / 2)

    _save(fig, 'asymmetry_pipeline')
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
