"""scripts/overview/literature_monitor_pipeline.py
Flowchart of the literature_monitor system, drawn in the V2 drawer style
(draw_pipeline_v2_common: indexed palette, content-driven node sizing).

Two lanes:
  • daily sweep  (scripts/.../run.py -> runner.run_slot):
        schedule -> search(4 sources) -> dedup -> save -> pool -> digest -> push
  • pool curation (the other 4 thin CLIs act on waiting_review/):
        enrich_abstracts -> cleanup_keywords (4-stage) -> download_pdfs
        -> extract_text -> accepted references

Output:
  workspace_refactor/overview/literature_monitor_pipeline.png
"""
import matplotlib.pyplot as plt
from draw_pipeline_v2_common import *


def _place(cx, y_top, labels, nd, bg, gap=GAP):
    """Compute (no drawing) a stage row centred on cx below y_top."""
    h = max(hgt(l) for l in labels)
    cy = y_top + SP + h / 2
    widths = [node_w(l) for l in labels]
    tot = sum(widths) + (len(widths) - 1) * gap
    xs, x = [], cx - tot / 2
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    return dict(cx=cx, cy=cy, h=h, xs=xs, widths=widths, tot=tot,
                labels=labels, nd=nd, bg=bg, top=cy - h / 2, bot=cy + h / 2)


def build():
    XL, XR = 9.0, 27.0          # sweep lane / curation lane centres
    VGAP = SP + 0.7             # vertical gap between stage rows

    # ── lane A: daily sweep ──────────────────────────────────────────────
    y = 1.6                      # leave room for the title
    A = []

    def add_a(labels, nd, bg):
        nonlocal y
        b = _place(XL, y, labels, nd, bg)
        A.append(b)
        y = b['bot'] + VGAP
        return b

    a_entry = add_a(['run_slot()\nscripts/.../run.py -> runner.py'], C_PRE['nd'], C_PRE['bg'])
    a_sched = add_a(['SLOT_PLAN\n10 slots / day',
                     '5 topics\nembedding · asymmetry\nemotion · age · bmi',
                     'query_for(topic, idx)\n-> query'], C1['nd'], C1['bg'])
    a_src = add_a(['arXiv', 'Semantic\nScholar', 'PubMed', 'OpenAlex'],
                  C2['nd'], C2['bg'])
    a_rec = add_a(['search(source, q, offset)\n-> PaperRecord[]'], C2['nd'], None)
    a_dedup = add_a(['cursor\n(get / advance)',
                     'is_seen\n(seen_ids · aliases)',
                     'is_existing_reference\n(_indexed.json)'], C_AGE['nd'], C_AGE['bg'])
    a_save = add_a(['save_record',
                    'PDF chain:\narXiv -> pdf_url ->\nUnpaywall -> OpenAlex',
                    'JSON sidecar', 'mark_seen'], C4['nd'], C4['bg'])
    a_pool = add_a(['waiting_review /\n<topic> / <date> /\n{json, pdf}',
                    '_state.json'], C_ES['nd'], C_ES['bg'])
    a_dig = add_a(['append_slot_digest\n-> <date>.md',
                   'write_daily_summary\n(slot 9)'], C_PA['nd'], C_PA['bg'])
    a_push = add_a(['auto_push:\ngit add -> commit ->\nfetch -> rebase -> push'],
                   C_ASY['nd'], C_ASY['bg'])

    # ── lane B: pool curation (the other 4 CLIs, on waiting_review/) ──────
    y = a_pool['top']              # start the curation lane at the pool level
    B = []

    def add_b(labels, nd, bg, cx=XR):
        nonlocal y
        b = _place(cx, y, labels, nd, bg)
        B.append(b)
        y = b['bot'] + VGAP
        return b

    b_enr = add_b(['enrich_abstracts\n-> fill_missing_abstracts\n(fetch_*_abstract)'],
                  C_EMO['nd'], C_EMO['bg'])
    b_cln = add_b(['cleanup_keywords\n-> curate.pipeline'], C6['nd'], C6['bg'])
    b_stg = add_b(['Stage 1\ntitle block', 'Stage 2\nabstract positive',
                   'Stage 3\nenrich', 'Stage 4\nre-positive'], C6['nd'], C6['bg'])
    b_arc = add_b(['_archive_rejected /\n<date>/  (reversible)'], C3['nd'], C3['bg'])
    b_dl = add_b(['download_pdfs\n-> missing_pdf_targets\n-> download_missing'],
                 C_FT['nd'], C_FT['bg'])
    b_txt = add_b(['extract_text\n-> PDF -> .txt (pymupdf)'], C_FT['nd'], C_FT['bg'])
    b_ref = add_b(['references / <topic> /\n(accepted)'], C_ES['nd'], C_ES['bg'])

    # ── canvas ───────────────────────────────────────────────────────────
    fig_h = max(b['bot'] for b in A + B) + 0.6
    x_left, x_right = -1.0, 35.0
    fig, ax = plt.subplots(figsize=(x_right - x_left, fig_h))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(fig_h + 0.1, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text((XL + XR) / 2, 0.7, 'Literature Monitor — pipeline',
            ha='center', va='center', fontsize=FS + 4, fontfamily=FONT,
            color=TEXT_COLOR, fontweight='bold')

    # draw every band: cluster behind, then content-sized nodes
    for b in A + B:
        if b['bg'] is not None:
            cluster(ax, b['cx'], b['cy'], b['tot'] + PADDING, b['h'] + 2 * SP, b['bg'])
        for x, w, lab in zip(b['xs'], b['widths'], b['labels']):
            node(ax, x, b['cy'], w, hgt(lab), lab, b['nd'])

    def vlink(b1, b2):
        line(ax, b1['cx'], b1['bot'], b2['cx'], b2['top'])

    def fan(src_xs, src_y, dst_xs, dst_y):
        for sx in src_xs:
            for dx in dst_xs:
                line(ax, sx, src_y, dx, dst_y)

    # lane A wiring
    vlink(a_entry, a_sched)
    fan([a_sched['xs'][2]], a_sched['bot'], a_src['xs'], a_src['top'])   # query -> 4 sources
    fan(a_src['xs'], a_src['bot'], [a_rec['cx']], a_rec['top'])          # sources -> records
    vlink(a_rec, a_dedup)
    fan([a_dedup['cx']], a_dedup['bot'], a_save['xs'], a_save['top'])    # dedup -> save row
    fan(a_save['xs'], a_save['bot'], a_pool['xs'], a_pool['top'])        # save -> pool
    fan(a_pool['xs'], a_pool['bot'], a_dig['xs'], a_dig['top'])          # pool -> digest
    fan(a_dig['xs'], a_dig['bot'], [a_push['cx']], a_push['top'])        # digest -> push

    # pool -> curation lane (horizontal hand-off)
    line(ax, a_pool['cx'] + a_pool['tot'] / 2 + PADDING / 2, a_pool['cy'],
         b_enr['cx'] - b_enr['tot'] / 2 - PADDING / 2, b_enr['cy'])

    # lane B wiring
    vlink(b_enr, b_cln)
    fan([b_cln['cx']], b_cln['bot'], b_stg['xs'], b_stg['top'])          # cleanup -> 4 stages
    line(ax, b_stg['xs'][1], b_stg['bot'], b_arc['cx'], b_arc['top'])    # rejects -> archive
    line(ax, b_stg['xs'][-1], b_stg['bot'], b_dl['cx'], b_dl['top'])     # final survivors -> download
    vlink(b_dl, b_txt)
    vlink(b_txt, b_ref)

    out = OUT / "literature_monitor_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
