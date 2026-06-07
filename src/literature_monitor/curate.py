"""下載後的池子整理：補抓 abstract + 關鍵字噪音過濾。

fill_missing_abstracts：依來源補空 abstract（enrich）。
pipeline：多階段關鍵字過濾，把判定噪音的論文歸檔到 _archive_rejected/。
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .sources import (
    DEFAULT_USER_AGENT,
    _pubmed_fetch_abstracts,
    fetch_arxiv_abstract,
    fetch_openalex_abstract,
    fetch_s2_abstract,
)
from .state import iter_sidecars, write_meta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 補抓 abstract（enrich）
# ---------------------------------------------------------------------------
def fill_missing_abstracts(targets: list[tuple[Path, dict]]) -> int:
    """對 abstract 為空的 (path, meta) 依來源補抓並寫回，回填好的篇數。"""
    by_source: dict[str, list[tuple[Path, dict]]] = {}
    for path, meta in targets:
        by_source.setdefault(meta.get("source", "?"), []).append((path, meta))

    filled = 0

    def _set(path: Path, meta: dict, abst: str) -> None:
        nonlocal filled
        if abst:
            meta["abstract"] = abst
            write_meta(path, meta)
            filled += 1

    # PubMed：efetch 批次（每批 100）
    pubmed = by_source.get("pubmed", [])
    for i in range(0, len(pubmed), 100):
        batch = pubmed[i:i + 100]
        pmids = [m.get("extra", {}).get("pmid") for _, m in batch if m.get("extra", {}).get("pmid")]
        if not pmids:
            continue
        abst_map = _pubmed_fetch_abstracts(pmids, headers={"User-Agent": DEFAULT_USER_AGENT})
        for path, meta in batch:
            _set(path, meta, abst_map.get(meta.get("extra", {}).get("pmid"), ""))

    # 其餘來源逐篇
    for path, meta in by_source.get("openalex", []):
        _set(path, meta, fetch_openalex_abstract((meta.get("extra") or {}).get("openalex_id")))
    for path, meta in by_source.get("arxiv", []):
        _set(path, meta, fetch_arxiv_abstract(meta.get("arxiv_id")))
    for path, meta in by_source.get("s2", []):
        _set(path, meta, fetch_s2_abstract(
            meta.get("doi"), meta.get("arxiv_id"), (meta.get("extra") or {}).get("s2_paperId")))

    return filled


# ---------------------------------------------------------------------------
# 過濾規則
# ---------------------------------------------------------------------------
# 標題命中任一即刪（Stage 1，最便宜的訊號）
BLOCK_TITLE_PATTERNS = [
    # 臨床照護 / 訓練 / 政策 / 試驗
    r"\bcaregiv", r"\bcare plan", r"\bcare team",
    r"\btraining\b.{0,30}\b(staff|nurs)", r"\bnursing home",
    r"\beducat\w+\b.{0,30}(staff|caregivers|migrant|nurs)",
    r"\bcompetence education",
    r"\btrial protocol\b", r"\brandomized clinical",
    r"\brandomi[sz]ed controlled trial", r"\bRCT\b",
    r"\bcrossover.{0,10}randomi",
    r"\bspirituality", r"\bend.of.life", r"\bbereavement",
    r"\brace and prevalence", r"\brural.urban",
    r"\bdisparit", r"\bequity.{0,30}aging",
    r"\bfamily resilience", r"\bworkforce",
    r"\bcommunity dwell", r"\bcommunity.based participatory",
    r"\bday care center", r"\bday respite",
    r"\bAboriginal Women", r"\beHealth Intervention",
    r"\bGrowth and Development",

    # 腦影像（要臉影像，不要腦影像）
    r"brain[\s\-]age", r"brain aging", r"brain abnormality", r"brain morpholog",
    r"brain imaging", r"brain network", r"brain MRI", r"brain volume",
    r"neuroimaging", r"neuro.imaging",
    r"cerebrovascular", r"cerebellum", r"cerebellar",
    r"\bMRI\b", r"\bfMRI\b", r"\bsMRI\b", r"magnetic resonance",
    r"voxel", r"T1.?weighted", r"T2.?weighted", r"\bDTI\b", r"diffusion tensor",
    r"white matter", r"gr[ae]y matter", r"cortical thickness", r"\bcortex\b",
    r"hippocamp", r"amygdal", r"thalam",
    r"\batrophy\b", r"\batrophic\b", r"ventricul",
    r"\bPET\b", r"positron emission", r"\bSPECT\b",
    r"amyloid", r"\btau\b", r"\bAβ\b",
    r"\bEEG\b", r"electroencephalo", r"magnetoencephalograph",
    r"functional connectivity",
    r"diffusion.?weighted",

    # 急性臨床 / 腦傷 / 口顎面 / 牙（非臉影像疾病訊號）
    r"neuro.Behcet",
    r"war.related", r"traumatic brain injury",
    r"mitochondrial dysfunction.*autism",
    r"ALSL.{0,5}seizure",
    r"\bdental\b", r"\bocclusal\b", r"\bteeth\b",
    r"craniofacial osteoma", r"condilar hyperplasia",
    r"temporomandibular",
    r"COVID.{0,10}brain", r"COVID.{0,10}perfusion",

    # 純語音/音訊情緒
    r"\bspeech\b.*\bemotion", r"\bvoice\b.*emotion",
    r"audio.only", r"verbal reaction",
    r"\bspeech features\b",
    r"paralinguistic",

    # 無臉脈絡的泛 ML / 工程論文
    r"^TRKM:", r"Twin restricted kernel", r"^Trans.ResNet\b",
    r"hyper.connectivity network",
    r"Quantum Wavelet Neural Network",
    r"Twin restricted",
    r"\bSpeechFormer\b",

    # Pareidolia（臉錯覺，相關但離題）
    r"\bpareidolia\b", r"face pareidolia",

    # 穿戴 / 睡眠 / 感測
    r"\bsleep\b.{0,30}(health|track)",
    r"wearable sensor", r"physiological signal.*wearable",
    r"smartphone.{0,30}eye.tracking",

    r"social engagement",

    # 主題雜項
    r"\bcontrastive learning of continuous",
    r"\bPathology\b.{0,5}\b(blood|fluid)",
    r"\bcortisol\b",
    r"face pareidolia in chimpanzees",

    # 疼痛模型 / 姿態生成（有臉但與 Alz 無關）
    r"\bpain control", r"^Paincontrol",
    r"synthetic faces for.*pain",
    r"automated pain", r"pain assessment.*automate",

    # 法醫 / 牙齒 / 牙齒年齡推估
    r"forensic medicine", r"first molar images",
    r"DNA methylation.*age",
    r"\bELOVL2\b",
    r"epigenetic clock",
]

# 跨疾病 / 遺傳症候群 block：只套用在 AD 主題。facedisease（臉影像↔疾病）要豁免，
# 否則正好把它想抓的糖尿病/中風/遺傳症候群論文掃掉。
CROSS_DISEASE_BLOCK = [
    r"Sturge.Weber", r"\bMTHFR\b", r"thalidomid", r"\bHuntington",
    r"Lafora", r"epilepsy syndrome", r"transthyretin", r"\bamyloidosis\b",
    r"genetic.*mutation", r"\bgenotype.phenotype\b", r"compound heterozyg",
    r"Phelan.McDermid",
    r"type 2 diabetes",
    r"\bstroke\b.*case", r"acute ischemic stroke", r"\bischemic stroke\b",
    r"\bmidlife\b.{0,30}stroke",
    r"infectious disease",
]
# 這些主題談「臉影像↔疾病」，不套用 CROSS_DISEASE_BLOCK。
CROSS_DISEASE_EXEMPT_TOPICS = {"facedisease"}

# 標題或 abstract 命中任一才留（Stage 2/4）。
# 移除裸 \bface\b：會誤中 dementia 文獻的 "face challenges" 動詞用法，改用複合詞。
POSITIVE_PATTERNS = [
    r"\bfacial\b",
    r"\bFER\b",
    r"\bfacial expression", r"\bfacial emotion", r"\bfacial affect",
    r"\bfacial recognition", r"\bfacial landmark", r"\bfacial photograph",
    r"\bfacial image", r"\bfacial video",
    r"\bfacial age", r"\bfacial aging",
    r"\bfacial asymmetry", r"\bface symmetry",
    r"\bfacial soft.?tissue", r"\bfacial morpholog",
    r"\bhemifac", r"\bhalf.face\b",
    r"\bface image", r"\bface photograph", r"\bface video",
    r"\bface recognition", r"\bface detection", r"\bface verification",
    r"\bface identification", r"\bface classification",
    r"\bface embedding", r"\bface representation",
    r"\bface analy", r"\bface processing\b",
    r"\bface mesh\b", r"\bface keypoint\b", r"\bface landmark\b",
    r"\bface biometric", r"\bface attribute",
    r"\bface aging\b", r"\bface age\b",
    r"\bphotograph\b.{0,30}\b(face|patient|subject|expression|emotion|age)",
    r"\bportrait\b.{0,30}\b(photo|image|patient)", r"\bselfie\b",
    r"\bArcFace\b", r"\bFaceNet\b", r"\bTopoFR\b",
    r"\bVGG.?Face\b", r"\bDlib\b",
    r"\bMiVOLO\b", r"\bFaceAge\b", r"\bDeepFace\b",
    r"\baction unit\b", r"\bAU intensity\b", r"\bAU detection\b",
    r"\bopen.?face\b", r"\bpy.?feat\b", r"\blibreface\b",
    r"\bpyramid cross.?fusion", r"\bPOSTER\b",
    r"\bHSEmotion\b", r"\bAffectNet\b", r"\bFER2013\b", r"\bRAF.?DB\b",
    r"\bMediaPipe\b.*face",
    r"\bMicro.?expression\b",
    r"\bvalence.{0,5}arousal\b",
]

COMPILED_BLOCK = [re.compile(p, re.IGNORECASE) for p in BLOCK_TITLE_PATTERNS]
COMPILED_CROSS_DISEASE = [re.compile(p, re.IGNORECASE) for p in CROSS_DISEASE_BLOCK]
COMPILED_POSITIVE = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS]


@dataclass
class Paper:
    json_path: Path
    pdf_path: Path | None
    meta: dict
    title: str
    abstract: str
    source: str
    topic: str

    @property
    def has_pdf(self) -> bool:
        return self.pdf_path is not None and self.pdf_path.exists()


def load_papers(waiting_review_dir: Path) -> list[Paper]:
    out: list[Paper] = []
    for json_path, meta in iter_sidecars(waiting_review_dir):
        pdf_path = json_path.with_suffix(".pdf")
        out.append(Paper(
            json_path=json_path,
            pdf_path=pdf_path if pdf_path.exists() else None,
            meta=meta,
            title=(meta.get("title") or "").strip(),
            abstract=(meta.get("abstract") or "").strip(),
            source=meta.get("source", ""),
            topic=json_path.parent.parent.name,
        ))
    return out


def _block_match(text: str, topic: str = "") -> str | None:
    patterns = COMPILED_BLOCK
    if topic not in CROSS_DISEASE_EXEMPT_TOPICS:
        patterns = COMPILED_BLOCK + COMPILED_CROSS_DISEASE
    for p in patterns:
        if p.search(text):
            return p.pattern
    return None


def _positive_match(text: str) -> bool:
    return any(p.search(text) for p in COMPILED_POSITIVE)


def _archive_paper(p: Paper, archive_root: Path, reason: str) -> None:
    """把論文（pdf+json）移到 _archive_rejected/<topic>/，json 標記原因。可逆。"""
    target_dir = archive_root / p.topic
    target_dir.mkdir(parents=True, exist_ok=True)
    if p.json_path.exists():
        target = target_dir / p.json_path.name
        try:
            p.meta["_rejected_reason"] = reason
            write_meta(target, p.meta)
            p.json_path.unlink()
        except Exception:
            p.json_path.replace(target)
    if p.pdf_path and p.pdf_path.exists():
        p.pdf_path.replace(target_dir / p.pdf_path.name)


# ---------------------------------------------------------------------------
# 多階段 pipeline
# ---------------------------------------------------------------------------
def pipeline(waiting_review_dir: Path, apply: bool) -> dict:
    papers = load_papers(waiting_review_dir)
    initial = len(papers)
    print(f"\nInitial: {initial} papers loaded")

    # Stage 1：標題 block
    deletes_1, survivors = [], []
    for p in papers:
        (deletes_1 if _block_match(p.title, p.topic) else survivors).append(p)
    print(f"Stage 1 (title block) deletes: {len(deletes_1)} (survivors: {len(survivors)})")

    # Stage 2：用現有 abstract 做 positive
    deletes_2, survivors_2, needs_enrich = [], [], []
    for p in survivors:
        if _positive_match(f"{p.title} {p.abstract}".lower()):
            survivors_2.append(p)
        elif not p.abstract:
            needs_enrich.append(p)
        else:
            deletes_2.append(p)
    print(f"Stage 2 deletes: {len(deletes_2)}; surviving: {len(survivors_2)}; "
          f"need enrich: {len(needs_enrich)}")

    # Stage 3：補抓缺 abstract
    if apply and needs_enrich:
        filled = fill_missing_abstracts([(p.json_path, p.meta) for p in needs_enrich])
        for p in needs_enrich:
            p.abstract = (p.meta.get("abstract") or "").strip()
        print(f"Stage 3 filled abstracts: {filled}/{len(needs_enrich)}")
    elif needs_enrich:
        print(f"Stage 3 SKIPPED (dry-run; would enrich {len(needs_enrich)})")

    # Stage 4：補抓後再 positive
    deletes_4, final = [], list(survivors_2)
    for p in needs_enrich:
        (final if _positive_match(f"{p.title} {p.abstract}".lower()) else deletes_4).append(p)
    print(f"Stage 4 deletes: {len(deletes_4)} (final survivors: {len(final)})")

    total_deletes = len(deletes_1) + len(deletes_2) + len(deletes_4)
    print(f"\nTotal to delete: {total_deletes}; final survivors: {len(final)}")

    if apply:
        archive_root = waiting_review_dir / "_archive_rejected" / time.strftime("%Y%m%d")
        for p in deletes_1:
            _archive_paper(p, archive_root, "title_block")
        for p in deletes_2:
            _archive_paper(p, archive_root, "no_face_in_existing_abstract")
        for p in deletes_4:
            _archive_paper(p, archive_root, "no_face_in_enriched_abstract")
        print(f"archived: {total_deletes} → {archive_root}")

    return {
        "initial": initial,
        "stage1_deletes": len(deletes_1),
        "stage2_deletes": len(deletes_2),
        "stage4_deletes": len(deletes_4),
        "final": len(final),
    }
