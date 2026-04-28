"""Topic-level keyword queries for the literature monitor.

Each topic has multiple query strings — slot scheduling rotates through them
so that a single day's 10 runs cover broad + narrow angles.

Conventions:
- Boolean syntax follows Semantic Scholar / OpenAlex style (AND / OR / quoted phrases).
  arXiv and PubMed clients translate to their native syntax in sources.py.
- Keep each list 6-10 entries. Order: broad -> narrow.
"""
from __future__ import annotations

TOPICS = ("embedding", "asymmetry", "emotion", "age")

TOPIC_QUERIES: dict[str, list[str]] = {
    "embedding": [
        '("face embedding" OR "facial embedding") AND ("Alzheimer" OR "dementia" OR "MCI" OR "cognitive decline")',
        '("face recognition" OR "ArcFace" OR "TopoFR" OR "FaceNet") AND ("Alzheimer" OR "dementia")',
        '"deep face" "Alzheimer" biomarker',
        '"longitudinal face" ("Alzheimer" OR "dementia")',
        '"face descriptor" ("Alzheimer" OR "dementia" OR "MCI")',
        '"facial biometric" ("Alzheimer" OR "dementia")',
        '"facial phenotype" "Alzheimer"',
        '"face representation" "cognitive impairment"',
    ],
    "asymmetry": [
        '"facial asymmetry" AND ("Alzheimer" OR "dementia" OR "MCI" OR "cognitive decline")',
        '"face symmetry" ("Alzheimer" OR "dementia" OR "cognitive")',
        '"facial landmark" "asymmetry" ("Alzheimer" OR "dementia")',
        '"hemifacial" ("Alzheimer" OR "dementia" OR "cognitive")',
        '"left-right facial" ("Alzheimer" OR "dementia")',
        '"facial fluctuating asymmetry" cognitive',
        '"face midline" ("Alzheimer" OR "dementia")',
        '"facial asymmetry" "neurodegeneration"',
    ],
    "emotion": [
        '("facial expression" OR "emotion recognition" OR "FER") AND ("Alzheimer" OR "dementia")',
        '"facial affect" ("Alzheimer" OR "dementia" OR "MCI")',
        '("OpenFace" OR "Py-Feat" OR "POSTER" OR "HSEmotion") AND ("Alzheimer" OR "dementia")',
        '"action unit" ("Alzheimer" OR "dementia" OR "cognitive impairment")',
        '"facial micro-expression" ("Alzheimer" OR "dementia")',
        '"emotional expressivity" "Alzheimer"',
        '"valence arousal" ("Alzheimer" OR "dementia")',
        '"emotion expression" "neurodegeneration"',
    ],
    "age": [
        '("apparent age" OR "facial age" OR "age estimation") AND ("Alzheimer" OR "dementia" OR "MCI")',
        '"age error" ("cognitive decline" OR "Alzheimer" OR "dementia")',
        '("MiVOLO" OR "DEX" OR "SSR-Net") AND ("Alzheimer" OR "dementia")',
        '"perceived age" ("Alzheimer" OR "dementia" OR "cognitive")',
        '"biological age" face ("Alzheimer" OR "dementia")',
        '"facial aging" ("Alzheimer" OR "dementia" OR "MCI")',
        '"age gap" face ("Alzheimer" OR "dementia")',
        '"brain age" face',
    ],
}


# ---------------------------------------------------------------------------
# Slot -> (topic, sources, query_index) mapping
# ---------------------------------------------------------------------------
# 10 slots / day. Each slot picks one topic + a subset of sources + a query.
# Rotation strategy:
#   slot 0..3  -> each topic, arxiv + s2,    broad query (idx 0)
#   slot 4..7  -> each topic, pubmed + openalex, broad query (idx 1)
#   slot 8     -> all topics x narrow query (idx 2)
#   slot 9     -> digest only (no fetch)
SLOT_PLAN: dict[int, dict] = {
    0: {"topics": ["embedding"], "sources": ["arxiv", "s2"], "query_idx": 0},
    1: {"topics": ["asymmetry"], "sources": ["arxiv", "s2"], "query_idx": 0},
    2: {"topics": ["emotion"],   "sources": ["arxiv", "s2"], "query_idx": 0},
    3: {"topics": ["age"],       "sources": ["arxiv", "s2"], "query_idx": 0},
    4: {"topics": ["embedding"], "sources": ["pubmed", "openalex"], "query_idx": 1},
    5: {"topics": ["asymmetry"], "sources": ["pubmed", "openalex"], "query_idx": 1},
    6: {"topics": ["emotion"],   "sources": ["pubmed", "openalex"], "query_idx": 1},
    7: {"topics": ["age"],       "sources": ["pubmed", "openalex"], "query_idx": 1},
    8: {"topics": list(TOPICS),  "sources": ["arxiv", "s2", "pubmed", "openalex"], "query_idx": 2},
    9: {"topics": [],            "sources": [],                                     "query_idx": -1, "digest_only": True},
}


def query_for(topic: str, idx: int) -> str:
    """Return the query string at index `idx` for `topic`, with safe wrap-around."""
    qs = TOPIC_QUERIES[topic]
    return qs[idx % len(qs)]
