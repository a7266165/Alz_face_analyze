"""Multi-stage keyword-based noise cleanup for waiting_review.

Pipeline (each stage only inspects survivors of the previous):

  Stage 1  Title-only BLOCK filter
           Drops obvious off-topic papers (clinical care/training/policy,
           brain imaging, dental, syndromes, etc.). Cheapest signal.

  Stage 2  Title+abstract POSITIVE filter (using existing abstract)
           Survivors must mention face / facial / photograph / FER / etc.
           somewhere in title or already-stored abstract.

  Stage 3  Enrich missing abstracts
           For Stage 2 survivors with empty abstract, refetch via
           the appropriate API (OpenAlex inverted-index decode, PubMed
           efetch, S2 GET, arXiv GET).

  Stage 4  Repeat POSITIVE filter with enriched abstracts.

  Stage 5  Report (dry-run) or apply deletions.

Deletions remove disk files (.pdf + .json) but PRESERVE _state.json
entries so re-fetch in future sweeps is suppressed.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.literature_monitor.enrich_abstracts import (
    fetch_arxiv_abstract,
    fetch_openalex_abstract,
    fetch_s2_abstract,
)
from scripts.literature_monitor.sources import (
    DEFAULT_USER_AGENT,
    _pubmed_fetch_abstracts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter rules
# ---------------------------------------------------------------------------
# Title-block: if title matches ANY of these, delete immediately (Stage 1).
BLOCK_TITLE_PATTERNS = [
    # Clinical care / training / policy / trial protocol
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

    # Brain imaging (any topic — we want face image not brain image)
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

    # Syndromes / acute clinical / unrelated genetics
    r"Sturge.Weber", r"\bMTHFR\b", r"thalidomid",
    r"neuro.Behcet", r"\bHuntington",
    r"Lafora", r"epilepsy syndrome",
    r"war.related", r"traumatic brain injury",
    r"transthyretin", r"\bamyloidosis\b",
    r"\bstroke\b.*case", r"acute ischemic stroke",
    r"genetic.*mutation", r"\bgenotype.phenotype\b",
    r"Phelan.McDermid", r"mitochondrial dysfunction.*autism",
    r"ALSL.{0,5}seizure",
    r"\bdental\b", r"\bocclusal\b", r"\bteeth\b",
    r"craniofacial osteoma", r"condilar hyperplasia",
    r"temporomandibular",
    r"COVID.{0,10}brain", r"COVID.{0,10}perfusion",

    # Speech/audio-only emotion
    r"\bspeech\b.*\bemotion", r"\bvoice\b.*emotion",
    r"audio.only", r"verbal reaction",
    r"\bspeech features\b",
    r"paralinguistic",

    # Generic ML / engineering papers without face context
    r"^TRKM:", r"Twin restricted kernel", r"^Trans.ResNet\b",
    r"hyper.connectivity network",
    r"Quantum Wavelet Neural Network",
    r"Twin restricted",
    r"\bSpeechFormer\b",

    # Pareidolia (face-illusion in dementia is interesting but tangential)
    r"\bpareidolia\b", r"face pareidolia",

    # Wearables / sleep / sensors
    r"\bsleep\b.{0,30}(health|track)",
    r"wearable sensor", r"physiological signal.*wearable",
    r"smartphone.{0,30}eye.tracking",

    # Diabetes / non-AD chronic disease
    r"type 2 diabetes",
    r"social engagement",
    r"\bmidlife\b.{0,30}stroke",

    # Topic-specific extras
    r"\bcontrastive learning of continuous",
    r"compound heterozyg",
    r"\bischemic stroke\b",
    r"infectious disease",
    r"\bPathology\b.{0,5}\b(blood|fluid)",
    r"\bcortisol\b",
    r"face pareidolia in chimpanzees",  # this one's just funny

    # Pain models / pose generation (face but not Alz-relevant)
    r"\bpain control", r"^Paincontrol",
    r"synthetic faces for.*pain",
    r"automated pain", r"pain assessment.*automate",

    # Forensic / dental / age estimation from teeth
    r"forensic medicine", r"first molar images",
    r"DNA methylation.*age",
    r"\bELOVL2\b",
    r"epigenetic clock",
]

# Positive: keep if title or abstract matches ANY of these (Stage 2/4).
# Bare \bface\b / \bfaces\b removed — they match "face challenges" / "faces difficulties"
# in dementia literature (verb usage). Use compound patterns instead.
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
    # face + image/photo/recognition compounds (replaces bare \bface\b)
    r"\bface image", r"\bface photograph", r"\bface video",
    r"\bface recognition", r"\bface detection", r"\bface verification",
    r"\bface identification", r"\bface classification",
    r"\bface embedding", r"\bface representation",
    r"\bface analy", r"\bface processing\b",
    r"\bface mesh\b", r"\bface keypoint\b", r"\bface landmark\b",
    r"\bface biometric", r"\bface attribute",
    r"\bface aging\b", r"\bface age\b",
    # photograph-based studies
    r"\bphotograph\b.{0,30}\b(face|patient|subject|expression|emotion|age)",
    r"\bportrait\b.{0,30}\b(photo|image|patient)", r"\bselfie\b",
    # named models / toolkits
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
COMPILED_POSITIVE = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _load_papers(base: Path) -> list[Paper]:
    out: list[Paper] = []
    for json_path in base.glob("*/*/*.json"):
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        topic = json_path.parent.parent.name
        pdf_path = json_path.with_suffix(".pdf")
        out.append(Paper(
            json_path=json_path,
            pdf_path=pdf_path if pdf_path.exists() else None,
            meta=meta,
            title=(meta.get("title") or "").strip(),
            abstract=(meta.get("abstract") or "").strip(),
            source=meta.get("source", ""),
            topic=topic,
        ))
    return out


def _block_match(text: str) -> str | None:
    for p in COMPILED_BLOCK:
        if p.search(text):
            return p.pattern
    return None


def _positive_match(text: str) -> bool:
    for p in COMPILED_POSITIVE:
        if p.search(text):
            return True
    return False


def _archive_paper(p: Paper, archive_root: Path, reason: str) -> None:
    """Move paper to _archive_rejected/<DATE>/<topic>/ for safe-keeping.

    Preserves both .pdf and .json. Reversible (user can git mv back).
    """
    target_dir = archive_root / p.topic
    target_dir.mkdir(parents=True, exist_ok=True)
    if p.json_path.exists():
        target = target_dir / p.json_path.name
        # Stamp a reason into the JSON for traceability
        try:
            meta = json.loads(p.json_path.read_text(encoding="utf-8"))
            meta["_rejected_reason"] = reason
            target.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                              encoding="utf-8")
            p.json_path.unlink()
        except Exception:
            p.json_path.replace(target)
    if p.pdf_path and p.pdf_path.exists():
        p.pdf_path.replace(target_dir / p.pdf_path.name)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def pipeline(base: Path, apply: bool) -> dict:
    papers = _load_papers(base)
    initial = len(papers)
    print(f"\nInitial: {initial} papers loaded")
    by_topic = {}
    for p in papers:
        by_topic.setdefault(p.topic, []).append(p)
    for t, ps in by_topic.items():
        print(f"  {t}: {len(ps)}")

    deletes_stage1: list[Paper] = []
    survivors: list[Paper] = []

    # ---- Stage 1: title block ----
    print("\n=== Stage 1: title-block filter ===")
    for p in papers:
        m = _block_match(p.title)
        if m:
            deletes_stage1.append(p)
        else:
            survivors.append(p)
    print(f"Stage 1 deletes: {len(deletes_stage1)} (survivors: {len(survivors)})")

    # ---- Stage 2: positive filter using existing abstract ----
    print("\n=== Stage 2: title+existing-abstract positive filter ===")
    deletes_stage2: list[Paper] = []
    survivors_after_2: list[Paper] = []
    needs_enrich: list[Paper] = []
    for p in survivors:
        text = f"{p.title} {p.abstract}".lower()
        if _positive_match(text):
            survivors_after_2.append(p)
        elif not p.abstract:
            # No abstract yet — defer to enrichment
            needs_enrich.append(p)
        else:
            # Has abstract but no positive match → delete
            deletes_stage2.append(p)
    print(
        f"Stage 2 deletes: {len(deletes_stage2)}; "
        f"already-surviving: {len(survivors_after_2)}; "
        f"need enrichment: {len(needs_enrich)}"
    )

    # ---- Stage 3: enrich missing abstracts ----
    if not apply:
        print("\n=== Stage 3: SKIPPED (dry-run, would enrich {} papers) ===".format(
            len(needs_enrich)))
    else:
        print(f"\n=== Stage 3: enrich {len(needs_enrich)} missing abstracts ===")
        # Group by source
        by_source: dict[str, list[Paper]] = {}
        for p in needs_enrich:
            by_source.setdefault(p.source, []).append(p)

        # PubMed batch
        pm_papers = by_source.get("pubmed", [])
        if pm_papers:
            pmids = [p.meta.get("extra", {}).get("pmid") for p in pm_papers]
            pmids = [x for x in pmids if x]
            print(f"  pubmed: efetch {len(pmids)} ids (batch 100)...")
            for i in range(0, len(pmids), 100):
                chunk_ids = pmids[i:i + 100]
                abst_map = _pubmed_fetch_abstracts(
                    chunk_ids, headers={"User-Agent": DEFAULT_USER_AGENT}
                )
                for p in pm_papers:
                    pmid = p.meta.get("extra", {}).get("pmid")
                    if pmid in abst_map and abst_map[pmid]:
                        p.abstract = abst_map[pmid]
                        p.meta["abstract"] = p.abstract
                        p.json_path.write_text(
                            json.dumps(p.meta, indent=2, ensure_ascii=False),
                            encoding="utf-8")

        # OpenAlex one-by-one
        oa_papers = by_source.get("openalex", [])
        if oa_papers:
            print(f"  openalex: GET {len(oa_papers)} works...")
            for i, p in enumerate(oa_papers, 1):
                oa_id = (p.meta.get("extra") or {}).get("openalex_id")
                abst = fetch_openalex_abstract(oa_id)
                if abst:
                    p.abstract = abst
                    p.meta["abstract"] = abst
                    p.json_path.write_text(
                        json.dumps(p.meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")
                if i % 25 == 0:
                    print(f"    {i}/{len(oa_papers)}")

        # arXiv one-by-one
        ax_papers = by_source.get("arxiv", [])
        if ax_papers:
            print(f"  arxiv: GET {len(ax_papers)} ids...")
            for p in ax_papers:
                abst = fetch_arxiv_abstract(p.meta.get("arxiv_id"))
                if abst:
                    p.abstract = abst
                    p.meta["abstract"] = abst
                    p.json_path.write_text(
                        json.dumps(p.meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

        # S2 one-by-one
        s2_papers = by_source.get("s2", [])
        if s2_papers:
            print(f"  s2: GET {len(s2_papers)} papers...")
            for p in s2_papers:
                abst = fetch_s2_abstract(
                    p.meta.get("doi"),
                    p.meta.get("arxiv_id"),
                    (p.meta.get("extra") or {}).get("s2_paperId"),
                )
                if abst:
                    p.abstract = abst
                    p.meta["abstract"] = abst
                    p.json_path.write_text(
                        json.dumps(p.meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

        filled = sum(1 for p in needs_enrich if p.abstract)
        print(f"  filled abstracts: {filled}/{len(needs_enrich)}")

    # ---- Stage 4: re-apply positive filter on enriched data ----
    print("\n=== Stage 4: positive filter post-enrichment ===")
    deletes_stage4: list[Paper] = []
    final_survivors: list[Paper] = list(survivors_after_2)
    for p in needs_enrich:
        text = f"{p.title} {p.abstract}".lower()
        if _positive_match(text):
            final_survivors.append(p)
        else:
            deletes_stage4.append(p)
    print(f"Stage 4 deletes: {len(deletes_stage4)} (final survivors: {len(final_survivors)})")

    # ---- Report ----
    print("\n=== SUMMARY ===")
    total_deletes = len(deletes_stage1) + len(deletes_stage2) + len(deletes_stage4)
    print(f"Initial: {initial}")
    print(f"  Stage 1 (title block): {len(deletes_stage1)}")
    print(f"  Stage 2 (existing abstract no-face): {len(deletes_stage2)}")
    print(f"  Stage 4 (enriched abstract no-face): {len(deletes_stage4)}")
    print(f"Total to delete: {total_deletes}")
    print(f"Final survivors: {len(final_survivors)}")

    # By topic
    print("\n--- final survivors per topic ---")
    surv_by_topic: dict[str, list[Paper]] = {}
    for p in final_survivors:
        surv_by_topic.setdefault(p.topic, []).append(p)
    for topic, ps in sorted(surv_by_topic.items()):
        print(f"  {topic}: {len(ps)}")

    # Show samples
    print("\n--- Stage 1 deletes (sample 8) ---")
    for p in deletes_stage1[:8]:
        m = _block_match(p.title)
        print(f"  [{p.topic}] [{m[:30]}] {p.title[:90]}")
    print("\n--- Stage 2 deletes (sample 8) ---")
    for p in deletes_stage2[:8]:
        print(f"  [{p.topic}] {p.title[:90]}")
    print("\n--- Stage 4 deletes (sample 8) ---")
    for p in deletes_stage4[:8]:
        print(f"  [{p.topic}] {p.title[:90]}")

    if apply:
        archive_root = base / "_archive_rejected" / time.strftime("%Y%m%d")
        archive_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Archiving rejects to {archive_root.relative_to(base.parent)} ===")
        n = 0
        for p in deletes_stage1:
            _archive_paper(p, archive_root, reason="title_block")
            n += 1
        for p in deletes_stage2:
            _archive_paper(p, archive_root, reason="no_face_in_existing_abstract")
            n += 1
        for p in deletes_stage4:
            _archive_paper(p, archive_root, reason="no_face_in_enriched_abstract")
            n += 1
        print(f"archived: {n}")

    return {
        "initial": initial,
        "stage1_deletes": len(deletes_stage1),
        "stage2_deletes": len(deletes_stage2),
        "stage4_deletes": len(deletes_stage4),
        "final": len(final_survivors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="actually enrich abstracts and delete files (default is dry-run)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    repo = Path(__file__).resolve().parents[2]
    base = repo / "references" / "waiting_review"
    pipeline(base, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
