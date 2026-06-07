"""各主題的關鍵字查詢與 slot 排程。

每主題多條 query（broad→narrow），slot 輪流取用；Boolean 用 S2/OpenAlex 風格，
arXiv/PubMed 由 sources.py 轉成各自語法。
"""
from __future__ import annotations

TOPICS = ("embedding", "asymmetry", "emotion", "age", "bmi", "facedisease")

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
        # 2026-05-05 additions: face-shape / photograph-based AD detection
        '"face shape" ("Alzheimer" OR "dementia")',
        '"facial photograph" ("dementia" OR "Alzheimer") detection',
        '"image-based phenotype" ("Alzheimer" OR "dementia")',
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
        # 2026-05-05 additions: Chien-style anchor (3D face + AD biomarker)
        '"3D face" ("Alzheimer" OR "dementia" OR "MCI")',
        '"facial morphometry" ("Alzheimer" OR "dementia")',
        '"image-based" "facial" ("Alzheimer" OR "dementia")',
        '"Procrustes" ("Alzheimer" OR "dementia") face',
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
        # 2026-05-05 additions: tighter AD-biomarker framing
        '"facial biomarker" ("Alzheimer" OR "dementia" OR "MCI")',
        '"face image" "Alzheimer" diagnosis',
        '"facial features" "AD" detection',
    ],
    "bmi": [
        '("body mass index" OR "BMI") AND ("Alzheimer" OR "dementia" OR "MCI" OR "cognitive decline")',
        '"obesity" ("Alzheimer" OR "dementia") ("risk" OR "protective" OR "paradox")',
        '"underweight" ("Alzheimer" OR "dementia" OR "cognitive impairment")',
        '"midlife obesity" ("Alzheimer" OR "dementia")',
        '"late-life BMI" ("Alzheimer" OR "dementia" OR "cognitive")',
        '"weight loss" ("Alzheimer" OR "dementia") preclinical',
        '"anthropometric" ("Alzheimer" OR "dementia") ("body mass" OR "BMI")',
        '"obesity paradox" ("Alzheimer" OR "dementia" OR "cognitive")',
    ],
    "age": [
        # Face-image-explicit only. AVOID: "age estimation" alone, "brain age",
        # "biological age" alone, "age error" alone — all of these pull in brain
        # MRI / epigenetic / blood biomarker papers we do NOT want.
        '("facial age estimation" OR "age from face" OR "age from photograph" OR "age from facial image") AND ("Alzheimer" OR "dementia" OR "MCI")',
        '"apparent age" ("face" OR "facial" OR "photograph") ("Alzheimer" OR "dementia" OR "cognitive")',
        '("MiVOLO" OR "DEX" OR "SSR-Net" OR "FaceAge") AND ("Alzheimer" OR "dementia" OR "aging")',
        '"perceived age" ("face" OR "facial" OR "photograph") ("Alzheimer" OR "dementia" OR "cognitive")',
        '"facial aging" ("Alzheimer" OR "dementia" OR "MCI" OR "cognitive decline")',
        '("face age gap" OR "facial age gap") ("Alzheimer" OR "dementia" OR "cognitive")',
        '"chronological age" ("face image" OR "facial photograph") ("Alzheimer" OR "dementia")',
        '"face-based age" ("Alzheimer" OR "dementia" OR "cognitive")',
    ],
    # facedisease 刻意「不」限縮 AD：從人臉影像偵測/關聯各種疾病（與 AD-from-face 同套方法論）。
    "facedisease": [
        '"facial image" ("disease" OR "diagnosis" OR "biomarker")',
        '"facial photograph" ("disease" OR "diagnosis" OR "screening")',
        '"deep learning" facial ("disease detection" OR "diagnosis")',
        '"facial phenotype" ("rare disease" OR "syndrome")',
        '("Face2Gene" OR "DeepGestalt" OR "GestaltMatcher") diagnosis',
        '"face image" ("diabetes" OR "cardiovascular" OR "endocrine" OR "thyroid")',
        '"facial features" ("disease risk" OR "health outcome" OR "diagnosis")',
        '"facial recognition" ("medical" OR "clinical diagnosis")',
        '"photo" face ("Down syndrome" OR "acromegaly" OR "genetic syndrome")',
        '"computer vision" facial ("disease" OR "clinical")',
    ],
}


# ---------------------------------------------------------------------------
# slot → (topics, sources, query_index)：每天 14 slot
#   0..5 各主題 arxiv+s2 broad；6..11 各主題 pubmed+openalex deep；12 全主題 narrow；13 只出 digest
# ---------------------------------------------------------------------------
SLOT_PLAN: dict[int, dict] = {
    0:  {"topics": ["embedding"], "sources": ["arxiv", "s2"], "query_idx": 0},
    1:  {"topics": ["asymmetry"], "sources": ["arxiv", "s2"], "query_idx": 0},
    2:  {"topics": ["emotion"],   "sources": ["arxiv", "s2"], "query_idx": 0},
    3:  {"topics": ["age"],       "sources": ["arxiv", "s2"], "query_idx": 0},
    4:  {"topics": ["bmi"],       "sources": ["arxiv", "s2"], "query_idx": 0},
    5:  {"topics": ["facedisease"], "sources": ["arxiv", "s2"], "query_idx": 0},
    6:  {"topics": ["embedding"], "sources": ["pubmed", "openalex"], "query_idx": 1},
    7:  {"topics": ["asymmetry"], "sources": ["pubmed", "openalex"], "query_idx": 1},
    8:  {"topics": ["emotion"],   "sources": ["pubmed", "openalex"], "query_idx": 1},
    9:  {"topics": ["age"],       "sources": ["pubmed", "openalex"], "query_idx": 1},
    10: {"topics": ["bmi"],       "sources": ["pubmed", "openalex"], "query_idx": 1},
    11: {"topics": ["facedisease"], "sources": ["pubmed", "openalex"], "query_idx": 1},
    12: {"topics": list(TOPICS),  "sources": ["arxiv", "s2", "pubmed", "openalex"], "query_idx": 2},
    13: {"topics": [],            "sources": [],                                     "query_idx": -1, "digest_only": True},
}


def query_for(topic: str, idx: int) -> str:
    """Return the query string at index `idx` for `topic`, with safe wrap-around."""
    qs = TOPIC_QUERIES[topic]
    return qs[idx % len(qs)]
