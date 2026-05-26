"""Meta-Learner pipeline diagram (matplotlib, same style as age_emb_pipeline_mpl).

Flat feature view with all upstream models expanded.
Font sizes match age_emb_pipeline_mpl_show.png.

Output:
  workspace/meta/meta_pipeline_mpl.png

Usage:
    python scripts/meta/draw_meta_pipeline_mpl.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(r"c:\Users\4080\Desktop\Alz_face_analyze\workspace\meta")

FONT = 'Microsoft JhengHei'
FS = 14
FS_HI = 15
FS_SM = 12
FS_XS = 10
EC = '#404040'
NODE_H = 1.15
SP = 0.4

C_SRC = dict(bg='#E8EFF5', nd='#B3CCE4')
C_EMB = dict(bg='#FDF3E5', nd='#F5D5A0')
C_AGE = dict(bg='#FFF3E0', nd='#FFD180')
C_EMO = dict(bg='#F0E6F0', nd='#D4B0D4')
C_DEM = dict(bg='#E8EFF5', nd='#B3CCE4')
C_TRN = dict(bg='#E4F0ED', nd='#A8D4C4')
C_EVL = dict(bg='#F5E8E4', nd='#E8BDB0')
C_FEAT = dict(bg='#F5F5F5')

FIG_W = 42.0
CX = FIG_W / 2


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


def _lay(cx, n, nw, gap):
    total = n * nw + (n - 1) * gap
    return [cx - total / 2 + nw / 2 + i * (nw + gap) for i in range(n)]


def _cluster_w(n, nw, gap, pad=0.6):
    return n * nw + (n - 1) * gap + pad


def build():
    # Section centres
    # Model row positions (equal model-level gaps ~0.8)
    xm_emb = 5.2
    xm_dem = 11.4
    xm_age = 18.5
    xm_emo = 31.6

    # Feature row positions (equal feature-group gaps ~1.0)
    xf_emb = 6.0
    xf_dem = 13.85
    xf_age = 17.45
    xf_emo = 30.5

    # Model node sizes (match reference FS_XS=10)
    mnw = 1.7
    mgap = 0.15

    y = 0.5

    # Row 1: Model clusters + Demographic
    c_mod_top = y
    y_mod = y + SP + NODE_H / 2
    c_mod_bot = y_mod + NODE_H / 2 + SP
    y = c_mod_bot

    # Row 2: All features
    c_feat_top = y + SP
    y_feat = c_feat_top + SP + NODE_H / 2
    c_feat_bot = y_feat + NODE_H / 2 + SP
    y = c_feat_bot

    # Row 3: K-Fold CV
    c_cv_top = y + SP
    y_cv = c_cv_top + SP + NODE_H / 2
    c_cv_bot = y_cv + NODE_H / 2 + SP
    y = c_cv_bot

    # Row 4: Classifier
    c_clf_top = y + SP
    y_clf = c_clf_top + SP + NODE_H / 2
    c_clf_bot = y_clf + NODE_H / 2 + SP
    y = c_clf_bot

    # Row 5: Eval Strategy
    c_es_top = y + SP
    y_es = c_es_top + SP + NODE_H / 2
    c_es_bot = y_es + NODE_H / 2 + SP
    y = c_es_bot

    # Row 6: Match Level
    c_ml_top = y + SP
    y_ml = c_ml_top + SP + NODE_H / 2
    c_ml_bot = y_ml + NODE_H / 2 + SP
    y = c_ml_bot

    # Row 7: Eval Unit
    c_eu_top = y + SP
    y_eu = c_eu_top + SP + NODE_H / 2
    c_eu_bot = y_eu + NODE_H / 2 + SP
    y = c_eu_bot

    # Row 8: Match Strategy
    c_ms_top = y + SP
    y_ms = c_ms_top + SP + NODE_H / 2
    c_ms_bot = y_ms + NODE_H / 2 + SP
    y = c_ms_bot

    # Row 9: Partitions
    c_pt_top = y + SP
    y_pt = c_pt_top + SP + NODE_H / 2
    c_pt_bot = y_pt + NODE_H / 2 + SP
    y = c_pt_bot

    fig_h = y + 0.5
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(-0.5, FIG_W + 0.5)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ═══════════════════════════════════
    # Row 1: Model clusters + Demographic
    # ═══════════════════════════════════

    # Embedding models (4)
    emb_labels = ['dlib', 'ArcFace', 'TopoFR', 'VGGFace']
    emb_xs = _lay(xm_emb, len(emb_labels), mnw, mgap)
    cluster(ax, xm_emb, (c_mod_top + c_mod_bot) / 2,
            _cluster_w(len(emb_labels), mnw, mgap),
            c_mod_bot - c_mod_top, C_EMB['bg'])
    for x, lab in zip(emb_xs, emb_labels):
        node(ax, x, y_mod, mnw, NODE_H, lab, C_EMB['nd'], fs=FS_XS)

    # Demographic (single node)
    dem_w = 2.5
    cluster(ax, xm_dem, (c_mod_top + c_mod_bot) / 2,
            dem_w + 0.4, c_mod_bot - c_mod_top, C_DEM['bg'])
    node(ax, xm_dem, y_mod, dem_w, NODE_H, 'Demographic', C_DEM['nd'], fs=FS_XS)

    # Age prediction models (5)
    age_labels = ['MiVOLO', 'Insight\nFace', 'Deep\nFace', 'Fair\nFace', 'OpenCV\nDNN']
    age_xs = _lay(xm_age, len(age_labels), mnw, mgap)
    cluster(ax, xm_age, (c_mod_top + c_mod_bot) / 2,
            _cluster_w(len(age_labels), mnw, mgap),
            c_mod_bot - c_mod_top, C_AGE['bg'])
    for x, lab in zip(age_xs, age_labels):
        node(ax, x, y_mod, mnw, NODE_H, lab, C_AGE['nd'], fs=FS_XS)

    # Emotion models (9 total, one cluster)
    emo_labels = ['EmoNet', 'Open\nFace', 'FER', 'HS\nEmotion', 'Libre\nFace',
                  'DAN', 'POSTER\n++', 'Py-Feat', 'ViT']
    mnw_emo = 1.5
    mgap_emo = 0.2
    emo_xs = _lay(xm_emo, len(emo_labels), mnw_emo, mgap_emo)
    cluster(ax, xm_emo, (c_mod_top + c_mod_bot) / 2,
            _cluster_w(len(emo_labels), mnw_emo, mgap_emo),
            c_mod_bot - c_mod_top, C_EMO['bg'])
    for x, lab in zip(emo_xs, emo_labels):
        node(ax, x, y_mod, mnw_emo, NODE_H, lab, C_EMO['nd'], fs=FS_XS)

    idx_emonet = 0    # emo_xs[0] = EmoNet
    idx_openface = 1  # emo_xs[1] = OpenFace

    # ═══════════════════════════════════
    # Row 2: Features — 4 separate clusters, vertically aligned
    # ═══════════════════════════════════
    feat_nw = 2.0
    feat_gap = 0.15
    feat_h = c_feat_bot - c_feat_top

    # -- Cluster A1: original + A2: asymmetry (centred on x_emb) --
    asym_labels = ['diff', '|diff|', 'rel_diff', '|rel_diff|']
    asym_nw = 1.7
    asym_gap = 0.15
    w_orig = feat_nw + 0.6
    w_asym = _cluster_w(len(asym_labels), asym_nw, asym_gap)
    sub_gap = 0.6
    total_emb_feat = w_orig + sub_gap + w_asym
    emb_feat_left = xf_emb - total_emb_feat / 2
    x_orig = emb_feat_left + w_orig / 2
    x_asym = emb_feat_left + w_orig + sub_gap + w_asym / 2

    cluster(ax, x_orig, (c_feat_top + c_feat_bot) / 2,
            w_orig, feat_h, C_EMB['bg'])
    node(ax, x_orig, y_feat, feat_nw, NODE_H, 'original', C_EMB['nd'], fs=FS_XS)
    orig_feat_xs = [x_orig]

    asym_feat_xs = _lay(x_asym, len(asym_labels), asym_nw, asym_gap)
    cluster(ax, x_asym, (c_feat_top + c_feat_bot) / 2,
            w_asym, feat_h, C_EMB['bg'])
    for x, lab in zip(asym_feat_xs, asym_labels):
        node(ax, x, y_feat, asym_nw, NODE_H, lab, C_EMB['nd'], fs=FS_XS)

    emb_feat_xs = orig_feat_xs + asym_feat_xs

    # -- Cluster B: Demographic feature (under x_dem) --
    dem_feat_xs = [xf_dem]
    cluster(ax, xf_dem, (c_feat_top + c_feat_bot) / 2,
            feat_nw + 0.6, feat_h, C_DEM['bg'])
    node(ax, xf_dem, y_feat, feat_nw, NODE_H, 'real_age', C_DEM['nd'], fs=FS_XS)

    # -- Cluster C: Age feature (under x_age) --
    age_feat_xs = [xf_age]
    cluster(ax, xf_age, (c_feat_top + c_feat_bot) / 2,
            feat_nw + 0.6, feat_h, C_AGE['bg'])
    node(ax, xf_age, y_feat, feat_nw, NODE_H, 'age_error', C_AGE['nd'], fs=FS_XS)

    # -- Emotion feature sub-clusters (3 groups, spaced under emo model cluster) --
    emo_fnw = 1.7
    emo_fgap = 0.15
    emo_sub_gap = 0.8  # gap between the 3 sub-clusters

    # Calculate total widths of the 3 sub-clusters
    va_labels = ['valence', 'arousal']
    shared_emo_labels = ['anger', 'disgust', 'fear', 'happiness',
                         'sadness', 'surprise', 'neutral']
    w_va = _cluster_w(len(va_labels), emo_fnw, emo_fgap)
    w_cont = emo_fnw + 0.6
    w_shared = _cluster_w(len(shared_emo_labels), emo_fnw, emo_fgap)
    total_emo_feat = w_va + emo_sub_gap + w_cont + emo_sub_gap + w_shared

    # Centre the 3 sub-clusters on x_emo
    emo_feat_left = xf_emo - total_emo_feat / 2

    x_va = emo_feat_left + w_va / 2
    x_contempt = emo_feat_left + w_va + emo_sub_gap + w_cont / 2
    x_shared = emo_feat_left + w_va + emo_sub_gap + w_cont + emo_sub_gap + w_shared / 2

    # D1: V/A (EmoNet only)
    va_feat_xs = _lay(x_va, len(va_labels), emo_fnw, emo_fgap)
    cluster(ax, x_va, (c_feat_top + c_feat_bot) / 2, w_va, feat_h, C_EMO['bg'])
    for x, lab in zip(va_feat_xs, va_labels):
        node(ax, x, y_feat, emo_fnw, NODE_H, lab, C_EMO['nd'], fs=FS_XS)

    # D2: contempt (OpenFace + EmoNet)
    contempt_feat_xs = [x_contempt]
    cluster(ax, x_contempt, (c_feat_top + c_feat_bot) / 2, w_cont, feat_h, C_EMO['bg'])
    node(ax, x_contempt, y_feat, emo_fnw, NODE_H, 'contempt', C_EMO['nd'], fs=FS_XS)

    # D3: 7 shared emotions (all 9 models)
    shared_emo_feat_xs = _lay(x_shared, len(shared_emo_labels), emo_fnw, emo_fgap)
    cluster(ax, x_shared, (c_feat_top + c_feat_bot) / 2, w_shared, feat_h, C_EMO['bg'])
    for x, lab in zip(shared_emo_feat_xs, shared_emo_labels):
        node(ax, x, y_feat, emo_fnw, NODE_H, lab, C_EMO['nd'], fs=FS_XS)

    # Collect all feature x positions for downstream connections
    feat_xs = (emb_feat_xs + dem_feat_xs + age_feat_xs
               + va_feat_xs + contempt_feat_xs + shared_emo_feat_xs)

    # ── Connections: models → features ──

    # Embedding models → lr_score_original, lr_score_asymmetry
    for ex in emb_xs:
        for fx in emb_feat_xs:
            line(ax, ex, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # Demographic → real_age
    line(ax, xm_dem, y_mod + NODE_H / 2, xf_dem, y_feat - NODE_H / 2)

    # Age prediction models → age_error
    for ax_ in age_xs:
        line(ax, ax_, y_mod + NODE_H / 2, xf_age, y_feat - NODE_H / 2)

    # EmoNet → V/A + contempt + 7 shared emotions (all 10)
    for fx in va_feat_xs:
        line(ax, emo_xs[idx_emonet], y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)
    line(ax, emo_xs[idx_emonet], y_mod + NODE_H / 2, x_contempt, y_feat - NODE_H / 2)
    for fx in shared_emo_feat_xs:
        line(ax, emo_xs[idx_emonet], y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # OpenFace → contempt + 7 shared emotions (8)
    line(ax, emo_xs[idx_openface], y_mod + NODE_H / 2, x_contempt, y_feat - NODE_H / 2)
    for fx in shared_emo_feat_xs:
        line(ax, emo_xs[idx_openface], y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # Other 7 emotion models → 7 shared emotions only
    for i, ex in enumerate(emo_xs):
        if i in (idx_emonet, idx_openface):
            continue
        for fx in shared_emo_feat_xs:
            line(ax, ex, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # ═══════════════════════════════════
    # Row 3: K-Fold CV
    # ═══════════════════════════════════
    cv_w = 12.0
    cluster(ax, CX, (c_cv_top + c_cv_bot) / 2,
            cv_w + 0.8, c_cv_bot - c_cv_top, C_TRN['bg'])
    node(ax, CX, y_cv, cv_w, NODE_H,
         "Fold-aligned K-Fold CV\n(test = fold K, train = all others)",
         C_TRN['nd'], fs=FS_SM)

    for fx in feat_xs:
        line(ax, fx, y_feat + NODE_H / 2, CX, y_cv - NODE_H / 2)

    # ═══════════════════════════════════
    # Row 4: Classifiers (3 choices)
    # ═══════════════════════════════════
    C_CLF = dict(bg='#ECF0E4', nd='#C5D6A8')
    clf_labels = ['TabPFN', 'Logistic\nRegression', 'XGBoost']
    clf_nw = 2.8
    clf_gap = 0.4
    clf_xs = _lay(CX, len(clf_labels), clf_nw, clf_gap)
    cluster(ax, CX, (c_clf_top + c_clf_bot) / 2,
            _cluster_w(len(clf_labels), clf_nw, clf_gap),
            c_clf_bot - c_clf_top, C_CLF['bg'])
    for x, lab in zip(clf_xs, clf_labels):
        node(ax, x, y_clf, clf_nw, NODE_H, lab, C_CLF['nd'], fs=FS_XS)

    for cx in clf_xs:
        line(ax, CX, y_cv + NODE_H / 2, cx, y_clf - NODE_H / 2)

    # ═══════════════════════════════════
    # Row 5–9: Eval chain (same as age_emb_pipeline)
    # ═══════════════════════════════════
    C_ES = dict(bg='#F8ECE4', nd='#E0C0A8')
    C_ML = dict(bg='#EDF0E8', nd='#C0D0A8')
    C_SA = dict(bg='#E8ECF8', nd='#B0B8E0')
    C_MS = dict(bg='#F5F0E0', nd='#D8C890')
    C_PT = dict(bg='#FBF5E0', nd='#F0D870')
    nw_ev = 2.4
    gap_ev = 0.65

    # Eval Strategy
    total_es = 2 * nw_ev + gap_ev
    cluster(ax, CX, (c_es_top + c_es_bot) / 2,
            total_es + 0.9, c_es_bot - c_es_top, C_ES['bg'])
    esx = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    for x, lab in zip(esx, ['1by1matched', 'caliper_group']):
        node(ax, x, y_es, nw_ev, NODE_H, lab, C_ES['nd'], fs=FS_XS)
    for cx in clf_xs:
        for ex in esx:
            line(ax, cx, y_clf + NODE_H / 2, ex, y_es - NODE_H / 2)

    # Match Level
    total_ml = 2 * nw_ev + gap_ev
    cluster(ax, CX, (c_ml_top + c_ml_bot) / 2,
            total_ml + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    mlx = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    for x, lab in zip(mlx, ['subject_match', 'visit_match']):
        node(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'], fs=FS_XS)
    for ex in esx:
        for mx in mlx:
            line(ax, ex, y_es + NODE_H / 2, mx, y_ml - NODE_H / 2)

    # Eval Unit
    total_eu = 2 * nw_ev + gap_ev
    cluster(ax, CX, (c_eu_top + c_eu_bot) / 2,
            total_eu + 0.9, c_eu_bot - c_eu_top, C_SA['bg'])
    eux = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    for x, lab in zip(eux, ['eval_by_subject', 'eval_by_visit']):
        node(ax, x, y_eu, nw_ev, NODE_H, lab, C_SA['nd'], fs=FS_XS)
    for mx in mlx:
        for eu in eux:
            line(ax, mx, y_ml + NODE_H / 2, eu, y_eu - NODE_H / 2)

    # Match Strategy
    nw_ms = 2.4
    gap_ms = 0.65
    total_ms = 3 * nw_ms + 2 * gap_ms
    cluster(ax, CX, (c_ms_top + c_ms_bot) / 2,
            total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX - (nw_ms + gap_ms), CX, CX + (nw_ms + gap_ms)]
    for x, lab in zip(msx, ['match_randomly', 'match_acs_first', 'match_nad_first']):
        node(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'], fs=FS_XS)
    for eu in eux:
        for ms in msx:
            line(ax, eu, y_eu + NODE_H / 2, ms, y_ms - NODE_H / 2)

    # Partitions
    nw_pt = 1.65
    gap_pt = 0.2
    total_pt = 5 * nw_pt + 4 * gap_pt
    cluster(ax, CX, (c_pt_top + c_pt_bot) / 2,
            total_pt + 0.7, c_pt_bot - c_pt_top, C_PT['bg'])
    ptx = [CX + (i - 2) * (nw_pt + gap_pt) for i in range(5)]
    for x, lab in zip(ptx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS', 'MMSE hi/lo', 'CASI hi/lo']):
        node(ax, x, y_pt, nw_pt, NODE_H, lab, C_PT['nd'], fs=FS_XS)
    for ms in msx:
        for px in ptx:
            line(ax, ms, y_ms + NODE_H / 2, px, y_pt - NODE_H / 2)

    # ── Save ──
    out_path = OUT / "meta_pipeline_mpl.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


def build_show():
    """Highlighted-path version: only active features keep color, rest gray."""
    G = dict(bg='#E0E0E0', nd='#C0C0C0')

    # Model row positions (equal model-level gaps ~0.8)
    xm_emb = 5.2
    xm_dem = 11.4
    xm_age = 18.5
    xm_emo = 31.6

    # Feature row positions (equal feature-group gaps ~1.0)
    xf_emb = 6.0
    xf_dem = 13.85
    xf_age = 17.45
    xf_emo = 30.5

    mnw = 1.7
    mgap = 0.15

    y = 0.5

    c_mod_top = y
    y_mod = y + SP + NODE_H / 2
    c_mod_bot = y_mod + NODE_H / 2 + SP
    y = c_mod_bot

    c_feat_top = y + SP
    y_feat = c_feat_top + SP + NODE_H / 2
    c_feat_bot = y_feat + NODE_H / 2 + SP
    y = c_feat_bot

    c_cv_top = y + SP
    y_cv = c_cv_top + SP + NODE_H / 2
    c_cv_bot = y_cv + NODE_H / 2 + SP
    y = c_cv_bot

    c_clf_top = y + SP
    y_clf = c_clf_top + SP + NODE_H / 2
    c_clf_bot = y_clf + NODE_H / 2 + SP
    y = c_clf_bot

    c_es_top = y + SP
    y_es = c_es_top + SP + NODE_H / 2
    c_es_bot = y_es + NODE_H / 2 + SP
    y = c_es_bot

    c_ml_top = y + SP
    y_ml = c_ml_top + SP + NODE_H / 2
    c_ml_bot = y_ml + NODE_H / 2 + SP
    y = c_ml_bot

    c_eu_top = y + SP
    y_eu = c_eu_top + SP + NODE_H / 2
    c_eu_bot = y_eu + NODE_H / 2 + SP
    y = c_eu_bot

    c_ms_top = y + SP
    y_ms = c_ms_top + SP + NODE_H / 2
    c_ms_bot = y_ms + NODE_H / 2 + SP
    y = c_ms_bot

    c_pt_top = y + SP
    y_pt = c_pt_top + SP + NODE_H / 2
    c_pt_bot = y_pt + NODE_H / 2 + SP
    y = c_pt_bot

    fig_h = y + 0.5
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(-0.5, FIG_W + 0.5)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def _n(ax, cx, cy, w, h, text, fc_on, on=True, fs=FS_XS):
        node(ax, cx, cy, w, h, text, fc_on if on else G['nd'], fs=fs)

    def _cl(ax, cx, cy, w, h, fc_on, on=True):
        cluster(ax, cx, cy, w, h, fc_on if on else G['bg'])

    # ═══════════════════════════════════
    # Row 1: Model clusters + Demographic
    # ═══════════════════════════════════

    # Embedding models (4) — only ArcFace ACTIVE
    emb_labels = ['dlib', 'ArcFace', 'TopoFR', 'VGGFace']
    emb_on = [False, True, False, False]
    emb_xs = _lay(xm_emb, len(emb_labels), mnw, mgap)
    _cl(ax, xm_emb, (c_mod_top + c_mod_bot) / 2,
        _cluster_w(len(emb_labels), mnw, mgap),
        c_mod_bot - c_mod_top, C_EMB['bg'])
    for x, lab, on in zip(emb_xs, emb_labels, emb_on):
        _n(ax, x, y_mod, mnw, NODE_H, lab, C_EMB['nd'], on=on)

    # Demographic — ACTIVE
    dem_w = 2.5
    _cl(ax, xm_dem, (c_mod_top + c_mod_bot) / 2,
        dem_w + 0.4, c_mod_bot - c_mod_top, C_DEM['bg'])
    _n(ax, xm_dem, y_mod, dem_w, NODE_H, 'Demographic', C_DEM['nd'])

    # Age prediction models (5) — only MiVOLO ACTIVE
    age_labels = ['MiVOLO', 'Insight\nFace', 'Deep\nFace', 'Fair\nFace', 'OpenCV\nDNN']
    age_on = [True, False, False, False, False]
    age_xs = _lay(xm_age, len(age_labels), mnw, mgap)
    _cl(ax, xm_age, (c_mod_top + c_mod_bot) / 2,
        _cluster_w(len(age_labels), mnw, mgap),
        c_mod_bot - c_mod_top, C_AGE['bg'])
    for x, lab, on in zip(age_xs, age_labels, age_on):
        _n(ax, x, y_mod, mnw, NODE_H, lab, C_AGE['nd'], on=on)

    # Emotion models (9) — GRAY
    emo_labels = ['EmoNet', 'Open\nFace', 'FER', 'HS\nEmotion', 'Libre\nFace',
                  'DAN', 'POSTER\n++', 'Py-Feat', 'ViT']
    mnw_emo = 1.5
    mgap_emo = 0.2
    emo_xs = _lay(xm_emo, len(emo_labels), mnw_emo, mgap_emo)
    _cl(ax, xm_emo, (c_mod_top + c_mod_bot) / 2,
        _cluster_w(len(emo_labels), mnw_emo, mgap_emo),
        c_mod_bot - c_mod_top, C_EMO['bg'], on=False)
    for x, lab in zip(emo_xs, emo_labels):
        _n(ax, x, y_mod, mnw_emo, NODE_H, lab, C_EMO['nd'], on=False)

    idx_emonet = 0
    idx_openface = 5

    # ═══════════════════════════════════
    # Row 2: Features
    # ═══════════════════════════════════
    feat_nw = 2.0
    feat_gap = 0.15
    feat_h = c_feat_bot - c_feat_top

    # Cluster A1: original — ACTIVE
    asym_labels = ['diff', '|diff|', 'rel_diff', '|rel_diff|']
    asym_on = [True, True, True, True]
    asym_nw = 1.7
    asym_gap = 0.15
    w_orig = feat_nw + 0.6
    w_asym = _cluster_w(len(asym_labels), asym_nw, asym_gap)
    sub_gap = 0.6
    total_emb_feat = w_orig + sub_gap + w_asym
    emb_feat_left = xf_emb - total_emb_feat / 2
    x_orig = emb_feat_left + w_orig / 2
    x_asym = emb_feat_left + w_orig + sub_gap + w_asym / 2

    _cl(ax, x_orig, (c_feat_top + c_feat_bot) / 2, w_orig, feat_h, C_EMB['bg'])
    _n(ax, x_orig, y_feat, feat_nw, NODE_H, 'original', C_EMB['nd'])
    orig_feat_xs = [x_orig]

    # Cluster A2: asymmetry — |rel_diff| highlighted
    asym_feat_xs = _lay(x_asym, len(asym_labels), asym_nw, asym_gap)
    _cl(ax, x_asym, (c_feat_top + c_feat_bot) / 2, w_asym, feat_h, C_EMB['bg'])
    for x, lab, on in zip(asym_feat_xs, asym_labels, asym_on):
        _n(ax, x, y_feat, asym_nw, NODE_H, lab, C_EMB['nd'], on=on)

    emb_feat_xs = orig_feat_xs + asym_feat_xs

    # Cluster B: Demographic feature — ACTIVE
    dem_feat_xs = [xf_dem]
    _cl(ax, xf_dem, (c_feat_top + c_feat_bot) / 2,
        feat_nw + 0.6, feat_h, C_DEM['bg'])
    _n(ax, xf_dem, y_feat, feat_nw, NODE_H, 'real_age', C_DEM['nd'])

    # Cluster C: Age feature — ACTIVE
    age_feat_xs = [xf_age]
    _cl(ax, xf_age, (c_feat_top + c_feat_bot) / 2,
        feat_nw + 0.6, feat_h, C_AGE['bg'])
    _n(ax, xf_age, y_feat, feat_nw, NODE_H, 'age_error', C_AGE['nd'])

    # Emotion feature sub-clusters — GRAY
    emo_fnw = 1.7
    emo_fgap = 0.15
    emo_sub_gap = 0.8

    va_labels = ['valence', 'arousal']
    shared_emo_labels = ['anger', 'disgust', 'fear', 'happiness',
                         'sadness', 'surprise', 'neutral']
    w_va = _cluster_w(len(va_labels), emo_fnw, emo_fgap)
    w_cont = emo_fnw + 0.6
    w_shared = _cluster_w(len(shared_emo_labels), emo_fnw, emo_fgap)
    total_emo_feat = w_va + emo_sub_gap + w_cont + emo_sub_gap + w_shared
    emo_feat_left = xf_emo - total_emo_feat / 2
    x_va = emo_feat_left + w_va / 2
    x_contempt = emo_feat_left + w_va + emo_sub_gap + w_cont / 2
    x_shared = emo_feat_left + w_va + emo_sub_gap + w_cont + emo_sub_gap + w_shared / 2

    va_feat_xs = _lay(x_va, len(va_labels), emo_fnw, emo_fgap)
    _cl(ax, x_va, (c_feat_top + c_feat_bot) / 2, w_va, feat_h, C_EMO['bg'], on=False)
    for x, lab in zip(va_feat_xs, va_labels):
        _n(ax, x, y_feat, emo_fnw, NODE_H, lab, C_EMO['nd'], on=False)

    contempt_feat_xs = [x_contempt]
    _cl(ax, x_contempt, (c_feat_top + c_feat_bot) / 2, w_cont, feat_h, C_EMO['bg'], on=False)
    _n(ax, x_contempt, y_feat, emo_fnw, NODE_H, 'contempt', C_EMO['nd'], on=False)

    shared_emo_feat_xs = _lay(x_shared, len(shared_emo_labels), emo_fnw, emo_fgap)
    _cl(ax, x_shared, (c_feat_top + c_feat_bot) / 2, w_shared, feat_h, C_EMO['bg'], on=False)
    for x, lab in zip(shared_emo_feat_xs, shared_emo_labels):
        _n(ax, x, y_feat, emo_fnw, NODE_H, lab, C_EMO['nd'], on=False)

    feat_xs = (emb_feat_xs + dem_feat_xs + age_feat_xs
               + va_feat_xs + contempt_feat_xs + shared_emo_feat_xs)

    # ── Connections ──

    for ex in emb_xs:
        for fx in emb_feat_xs:
            line(ax, ex, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    line(ax, xm_dem, y_mod + NODE_H / 2, xf_dem, y_feat - NODE_H / 2)

    for ax_ in age_xs:
        line(ax, ax_, y_mod + NODE_H / 2, xf_age, y_feat - NODE_H / 2)

    # EmoNet → all 10
    for fx in va_feat_xs:
        line(ax, emo_xs[idx_emonet], y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)
    line(ax, emo_xs[idx_emonet], y_mod + NODE_H / 2, x_contempt, y_feat - NODE_H / 2)
    for fx in shared_emo_feat_xs:
        line(ax, emo_xs[idx_emonet], y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # OpenFace → contempt + 7
    line(ax, emo_xs[idx_openface], y_mod + NODE_H / 2, x_contempt, y_feat - NODE_H / 2)
    for fx in shared_emo_feat_xs:
        line(ax, emo_xs[idx_openface], y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # Other 7 → 7 shared only
    for i, ex in enumerate(emo_xs):
        if i in (idx_emonet, idx_openface):
            continue
        for fx in shared_emo_feat_xs:
            line(ax, ex, y_mod + NODE_H / 2, fx, y_feat - NODE_H / 2)

    # ═══════════════════════════════════
    # Row 3: K-Fold CV — ACTIVE
    # ═══════════════════════════════════
    cv_w = 12.0
    cluster(ax, CX, (c_cv_top + c_cv_bot) / 2,
            cv_w + 0.8, c_cv_bot - c_cv_top, C_TRN['bg'])
    node(ax, CX, y_cv, cv_w, NODE_H,
         "Fold-aligned K-Fold CV\n(test = fold K, train = all others)",
         C_TRN['nd'], fs=FS_SM)

    for fx in feat_xs:
        line(ax, fx, y_feat + NODE_H / 2, CX, y_cv - NODE_H / 2)

    # ═══════════════════════════════════
    # Row 4: Classifiers — TabPFN ACTIVE, others GRAY
    # ═══════════════════════════════════
    C_CLF = dict(bg='#ECF0E4', nd='#C5D6A8')
    clf_labels = ['TabPFN', 'Logistic\nRegression', 'XGBoost']
    clf_on = [True, True, True]
    clf_nw = 2.8
    clf_gap = 0.4
    clf_xs = _lay(CX, len(clf_labels), clf_nw, clf_gap)
    cluster(ax, CX, (c_clf_top + c_clf_bot) / 2,
            _cluster_w(len(clf_labels), clf_nw, clf_gap),
            c_clf_bot - c_clf_top, C_CLF['bg'])
    for x, lab, on in zip(clf_xs, clf_labels, clf_on):
        _n(ax, x, y_clf, clf_nw, NODE_H, lab, C_CLF['nd'], on=on)

    for cx in clf_xs:
        line(ax, CX, y_cv + NODE_H / 2, cx, y_clf - NODE_H / 2)

    # ═══════════════════════════════════
    # Row 5–9: Eval chain (selective highlight)
    # ═══════════════════════════════════
    C_ES = dict(bg='#F8ECE4', nd='#E0C0A8')
    C_ML = dict(bg='#EDF0E8', nd='#C0D0A8')
    C_SA = dict(bg='#E8ECF8', nd='#B0B8E0')
    C_MS = dict(bg='#F5F0E0', nd='#D8C890')
    C_PT = dict(bg='#FBF5E0', nd='#F0D870')
    nw_ev = 2.4
    gap_ev = 0.65

    # Eval Strategy: 1by1matched=ON, caliper_group=OFF
    total_es = 2 * nw_ev + gap_ev
    _cl(ax, CX, (c_es_top + c_es_bot) / 2, total_es + 0.9, c_es_bot - c_es_top, C_ES['bg'])
    esx = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    es_on = [True, False]
    for x, lab, on in zip(esx, ['1by1matched', 'caliper_group'], es_on):
        _n(ax, x, y_es, nw_ev, NODE_H, lab, C_ES['nd'], on=on)
    for cx in clf_xs:
        for ex in esx:
            line(ax, cx, y_clf + NODE_H / 2, ex, y_es - NODE_H / 2)

    # Match Level: subject_match=ON, visit_match=OFF
    total_ml = 2 * nw_ev + gap_ev
    _cl(ax, CX, (c_ml_top + c_ml_bot) / 2, total_ml + 0.9, c_ml_bot - c_ml_top, C_ML['bg'])
    mlx = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    ml_on = [True, False]
    for x, lab, on in zip(mlx, ['subject_match', 'visit_match'], ml_on):
        _n(ax, x, y_ml, nw_ev, NODE_H, lab, C_ML['nd'], on=on)
    for ex in esx:
        for mx in mlx:
            line(ax, ex, y_es + NODE_H / 2, mx, y_ml - NODE_H / 2)

    # Eval Unit: eval_by_subject=OFF, eval_by_visit=ON
    total_eu = 2 * nw_ev + gap_ev
    _cl(ax, CX, (c_eu_top + c_eu_bot) / 2, total_eu + 0.9, c_eu_bot - c_eu_top, C_SA['bg'])
    eux = [CX - (nw_ev + gap_ev) / 2, CX + (nw_ev + gap_ev) / 2]
    eu_on = [False, True]
    for x, lab, on in zip(eux, ['eval_by_subject', 'eval_by_visit'], eu_on):
        _n(ax, x, y_eu, nw_ev, NODE_H, lab, C_SA['nd'], on=on)
    for mx in mlx:
        for eu in eux:
            line(ax, mx, y_ml + NODE_H / 2, eu, y_eu - NODE_H / 2)

    # Match Strategy: match_acs_first=ON, others=OFF
    nw_ms = 2.4
    gap_ms = 0.65
    total_ms = 3 * nw_ms + 2 * gap_ms
    _cl(ax, CX, (c_ms_top + c_ms_bot) / 2, total_ms + 0.9, c_ms_bot - c_ms_top, C_MS['bg'])
    msx = [CX - (nw_ms + gap_ms), CX, CX + (nw_ms + gap_ms)]
    ms_on = [False, True, False]
    for x, lab, on in zip(msx, ['match_randomly', 'match_acs_first', 'match_nad_first'], ms_on):
        _n(ax, x, y_ms, nw_ms, NODE_H, lab, C_MS['nd'], on=on)
    for eu in eux:
        for ms in msx:
            line(ax, eu, y_eu + NODE_H / 2, ms, y_ms - NODE_H / 2)

    # Partitions: AD vs HC/NAD/ACS=ON, MMSE/CASI=OFF
    nw_pt = 1.65
    gap_pt = 0.2
    total_pt = 5 * nw_pt + 4 * gap_pt
    _cl(ax, CX, (c_pt_top + c_pt_bot) / 2, total_pt + 0.7, c_pt_bot - c_pt_top, C_PT['bg'])
    ptx = [CX + (i - 2) * (nw_pt + gap_pt) for i in range(5)]
    pt_on = [True, True, True, False, False]
    for x, lab, on in zip(ptx, ['AD vs HC', 'AD vs NAD', 'AD vs ACS', 'MMSE hi/lo', 'CASI hi/lo'], pt_on):
        _n(ax, x, y_pt, nw_pt, NODE_H, lab, C_PT['nd'], on=on)
    for ms in msx:
        for px in ptx:
            line(ax, ms, y_ms + NODE_H / 2, px, y_pt - NODE_H / 2)

    # ── Save ──
    out_path = OUT / "meta_pipeline_mpl_show.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
    build_show()
