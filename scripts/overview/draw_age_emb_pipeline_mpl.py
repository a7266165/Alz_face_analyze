"""Combined runner for all pipeline diagram versions.

Usage:
    python scripts/overview/draw_age_emb_pipeline_mpl.py
"""
from draw_age_emb_pipeline_common import OUT
from draw_age_emb_pipeline_v1 import build, build_show
from draw_age_emb_pipeline_v2 import build_v2, build_v2_show
from draw_age_emb_pipeline_v3 import build_v3, build_v3_show
from draw_age_emb_pipeline_v4 import build_v4, build_v4_show, build_v4_refactor

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
    build_show()
    build_v2()
    build_v2_show()
    build_v3()
    build_v3_show()
    build_v4()
    build_v4_show()
    build_v4_refactor()
