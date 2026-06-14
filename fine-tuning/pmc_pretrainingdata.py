#!/usr/bin/env python3
"""
pmc_pretrainingdata_comprehensive.py

Goal:
  Given a disease name, fetch PMC open-access candidate papers and save ONLY the
  top 5 papers that provide the most comprehensive disease-level clinical coverage.

  Designed for the NepMedJP pipeline.  The output Markdown files are consumed
  directly by orchestration.py — no changes to that script are required.

Selection logic (three-stage funnel):
  Stage 1 – PMC search: collect candidate IDs via targeted queries.
  Stage 2 – Abstract screening: score title + esummary abstract; discard hard
            negatives before spending any full-text HTTP calls.
  Stage 3 – Full-text scoring: fetch XML, flatten all sections (including
            nested sub-sections), score clinical topic coverage across 8 axes,
            and pick the top 5 by combined score.

Scoring axes (no LLM calls):
  1. Clinical Definition / Overview
  2. Etiology / Cause
  3. Transmission / Pathogenesis / Mechanism
  4. Signs & Symptoms
  5. Types / Classification
  6. Risk Factors / Complications
  7. Diagnosis
  8. Treatment / Management
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# NCBI credentials (optional but strongly recommended for rate-limit headroom)
# ---------------------------------------------------------------------------
NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "your@email.com")

# ---------------------------------------------------------------------------
# Run-time defaults (all overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_RETMAX_PER_QUERY: int = 100
DEFAULT_TARGET_PAPERS: int = 5
DEFAULT_MAX_ABSTRACT_CANDIDATES: int = 200   # maximum IDs entering Stage 2
DEFAULT_MAX_FULLTEXT_FETCH: int = 40         # maximum full-text HTTP calls
DEFAULT_MIN_TOPICS_COVERED: int = 4          # strict-mode gate (out of 8)
DEFAULT_MIN_DATE: str = "1995"               # filter out very old papers
DEFAULT_STRICT: bool = False                 # whether to enforce min_topics gate


# ---------------------------------------------------------------------------
# Publication type filter applied at query time
# ---------------------------------------------------------------------------
BROAD_PUBTYPES: List[str] = [
    "review[pt]",
    "systematic review[pt]",
    "practice guideline[pt]",
    "guideline[pt]",
]

# Used in fallback queries — no pubtype filter, but still title-focused
_BROAD_CLINICAL_TERMS = (
    "overview[Title/Abstract] OR "
    "review[Title/Abstract] OR "
    "clinical features[Title/Abstract] OR "
    "clinical presentation[Title/Abstract] OR "
    "diagnosis[Title/Abstract] OR "
    "treatment[Title/Abstract] OR "
    "management[Title/Abstract] OR "
    "pathogenesis[Title/Abstract] OR "
    "etiology[Title/Abstract] OR "
    "epidemiology[Title/Abstract] OR "
    "prevention[Title/Abstract] OR "
    "guideline[Title/Abstract]"
)


# ---------------------------------------------------------------------------
# Title scoring — broad signals (positive) and narrow signals (negative)
# ---------------------------------------------------------------------------
BROAD_TITLE_TERMS: List[str] = [
    "overview",
    "clinical review",
    "narrative review",
    "state of the art",
    "clinical features",
    "clinical presentation",
    "clinical manifestations",
    "diagnosis and treatment",
    "diagnosis and management",
    "pathogenesis",
    "etiology",
    "aetiology",
    "epidemiology",
    "prevention and control",
    "guideline",
    "guidelines",
    "consensus",
    "practice",
    "management of",
    "treatment of",
    "update on",
    "current concepts",
    "advances in",
    "a review",
    "review of",
    "approach to",
]

# Each hit subtracts NARROW_PENALTY points
NARROW_TITLE_TERMS: List[str] = [
    "association",
    "associations",
    "predictor",
    "predictors",
    "prediction model",
    "risk score",
    "cohort study",
    "cross-sectional study",
    "case report",
    "case series",
    "case-control",
    "single-centre",
    "single-center",
    "multicentre",              # can be fine, but usually not a review
    "retrospective",
    "prospective",
    "randomized controlled trial",
    "randomised controlled trial",
    "randomized trial",
    "randomised trial",
    "clinical trial",
    "protocol",
    "efficacy of",
    "safety of",
    "high-dose",
    "low-dose",
    "dose-response",
    "dosage",
    "drug resistance",
    "antimicrobial resistance",
    "mutation",
    "mutations",
    "genomic",
    "genome-wide",
    "whole genome",
    "transcriptomic",
    "proteomic",
    "metabolomic",
    "biomarker",
    "biomarkers",
    "molecular pathway",
    "signaling pathway",
    "in vitro",
    "in vivo",
    "mouse model",
    "animal model",
    "ct scan",
    "mri",
    "radiological",
    "radiographic",
    "rehabilitation",
    "quality of life",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "neural network",
    "cost-effectiveness",
    "economic analysis",
    "meta-analysis",
    "pooled analysis",
    "network meta-analysis",
    # Procedure / organ-specific titles
    "bronchoscopy",
    "endobronchial",
    "endoscopic",
    "laparoscopic",
    "surgical technique",
    "imaging findings",
    "ct findings",
    "mri findings",
    "radiological findings",
    "pathological findings",
    "histological findings",
    "biopsy",
    # Single-drug / single-intervention papers
    "rifampicin",
    "isoniazid",
    "metformin",
    "statin",
    "insulin therapy",
    # Rehabilitation / quality-of-life
    "physical therapy",
    "exercise training",
    "pulmonary rehabilitation",
]

# These title signals are hard disqualifiers regardless of other score
HARD_NEGATIVE_TITLE_TERMS: List[str] = [
    "editorial",
    "letter to the editor",
    "author's reply",
    "erratum",
    "corrigendum",
    "correction to",
    "conference abstract",
    "study protocol",
    "reply to",
    "response to",
]

NARROW_PENALTY: int = 3
BROAD_BONUS: int = 2
DISEASE_IN_TITLE_BONUS: int = 6
DISEASE_ABSENT_PENALTY: int = 3


# ---------------------------------------------------------------------------
# Abstract screening — terms that strongly suggest narrow scope
# ---------------------------------------------------------------------------
ABSTRACT_NARROW_PHRASES: List[str] = [
    "we report a case",
    "case report",
    "we present a case",
    "this case",
    "patient presented with",
    "retrospective study",
    "prospective study",
    "randomized controlled",
    "we performed a meta-analysis",
    "systematic search",
    "prisma",
    "we enrolled",
    "participants were randomized",
    "we investigated the association",
    "the aim of this study was to investigate",
    "in vitro",
    "in vivo",
    "mouse model",
    "murine model",
    "cell line",
    "mrna expression",
    "protein expression",
    "snp",
    "single nucleotide",
    "genome-wide",
    "whole genome",
    "radiographic findings",
    "ct findings",
    "mri findings",
]

ABSTRACT_BROAD_PHRASES: List[str] = [
    "clinical features",
    "clinical presentation",
    "clinical manifestations",
    "signs and symptoms",
    "diagnosis and",
    "treatment and",
    "management of",
    "this review",
    "we review",
    "we summarize",
    "we describe",
    "overview of",
    "pathogenesis of",
    "etiology of",
    "epidemiology of",
    "risk factors",
    "complications",
    "classification",
    "guidelines",
]


# ---------------------------------------------------------------------------
# Full-text clinical topic profiles
# ---------------------------------------------------------------------------
TOPIC_PROFILES: Dict[str, Dict[str, List[str]]] = {
    "src_01_Clinical_Definition_Overview": {
        "section_terms": [
            "introduction",
            "background",
            "overview",
            "definition",
            "epidemiology",
            "burden",
        ],
        "body_terms": [
            "definition",
            "defined as",
            "is a disease",
            "is an infection",
            "refers to",
            "characterized by",
            "overview",
            "epidemiology",
            "prevalence",
            "incidence",
            "global burden",
            "public health",
            "morbidity",
        ],
    },
    "src_02_Etiology_Cause": {
        "section_terms": [
            "etiology",
            "aetiology",
            "cause",
            "causes",
            "causative agent",
        ],
        "body_terms": [
            "etiology",
            "aetiology",
            "caused by",
            "causative",
            "pathogen",
            "causative agent",
            "organism",
            "bacterium",
            "virus",
            "parasite",
            "fungus",
            "genetic cause",
            "autoimmune",
            "environmental cause",
            "multifactorial",
        ],
    },
    "src_03_Transmission_Pathogenesis_Mechanism": {
        "section_terms": [
            "pathogenesis",
            "mechanism",
            "transmission",
            "pathophysiology",
            "spread",
        ],
        "body_terms": [
            "pathogenesis",
            "pathophysiology",
            "mechanism",
            "transmission",
            "route of",
            "spread",
            "progression",
            "immune response",
            "inflammation",
            "host response",
            "infection",
            "replication",
        ],
    },
    "src_04_Signs_Symptoms": {
        "section_terms": [
            "symptoms",
            "signs",
            "clinical presentation",
            "clinical features",
            "clinical manifestations",
        ],
        "body_terms": [
            "symptom",
            "symptoms",
            "sign",
            "signs",
            "clinical presentation",
            "clinical features",
            "clinical manifestations",
            "presents with",
            "complains of",
            "fever",
            "pain",
            "cough",
            "fatigue",
            "dyspnea",
            "shortness of breath",
            "weight loss",
            "night sweats",
        ],
    },
    "src_05_Types_Classification": {
        "section_terms": [
            "classification",
            "types",
            "subtypes",
            "forms",
            "staging",
        ],
        "body_terms": [
            "classification",
            "classified",
            "type",
            "types",
            "subtype",
            "subtypes",
            "acute",
            "chronic",
            "primary",
            "secondary",
            "mild",
            "moderate",
            "severe",
            "stage",
            "staging",
        ],
    },
    "src_06_Risk_Factors_Complications": {
        "section_terms": [
            "risk factors",
            "complications",
            "prognosis",
            "sequelae",
            "outcomes",
        ],
        "body_terms": [
            "risk factor",
            "risk factors",
            "predispose",
            "predisposed",
            "complication",
            "complications",
            "sequelae",
            "prognosis",
            "outcome",
            "outcomes",
            "mortality",
            "comorbidity",
            "comorbidities",
        ],
    },
    "src_07_Diagnosis": {
        "section_terms": [
            "diagnosis",
            "diagnostic",
            "testing",
            "laboratory diagnosis",
            "imaging",
        ],
        "body_terms": [
            "diagnosis",
            "diagnostic criteria",
            "diagnosed",
            "laboratory",
            "culture",
            "pcr",
            "biopsy",
            "serology",
            "blood test",
            "radiograph",
            "imaging",
            "screening",
            "diagnostic test",
        ],
    },
    "src_08_Treatment_Management": {
        "section_terms": [
            "treatment",
            "management",
            "therapy",
            "prevention",
            "guidelines",
        ],
        "body_terms": [
            "treatment",
            "management",
            "therapy",
            "therapeutic",
            "medication",
            "drug",
            "antibiotic",
            "antiviral",
            "surgery",
            "prevention",
            "vaccine",
            "regimen",
            "guideline",
            "recommended",
            "first-line",
            "second-line",
        ],
    },
}

# Section titles that indicate non-clinical boilerplate — filtered before scoring
BOILERPLATE_SECTION_TERMS: List[str] = [
    "methods",
    "materials and methods",
    "statistical analysis",
    "data extraction",
    "search strategy",
    "eligibility criteria",
    "inclusion criteria",
    "exclusion criteria",
    "prisma",
    "author contributions",
    "funding",
    "acknowledgements",
    "acknowledgments",
    "conflict of interest",
    "conflicts of interest",
    "references",
    "supplementary",
    "supporting information",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateMeta:
    """Metadata about a PMC candidate fetched from esummary."""
    pmc_id: str
    title: str = ""
    journal: str = ""
    pubdate: str = ""
    abstract: str = ""          # esummary does not always return abstract;
                                # populated during abstract pre-screen from efetch
    title_score: int = 0
    title_reasons: List[str] = field(default_factory=list)
    abstract_score: int = 0
    final_score: int = 0
    covered_topic_count: int = 0
    section_titles: List[str] = field(default_factory=list)
    disease_mentions: int = 0
    decision: str = "PENDING"
    decision_reason: str = ""


@dataclass
class ScoredArticle:
    """Full article with all scores."""
    meta: CandidateMeta
    title_text: str = ""
    abstract_text: str = ""
    body_md: str = ""           # Markdown-formatted body for saving
    covered_topics: Dict[str, Dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NCBI helpers
# ---------------------------------------------------------------------------

def _ncbi_params() -> Dict[str, str]:
    p: Dict[str, str] = {}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    return p


def _sleep() -> None:
    """Polite NCBI sleep: 0.12 s with key, 0.37 s without."""
    time.sleep(0.12 if NCBI_API_KEY else 0.37)


def _safe_get(url: str, params: Dict, timeout: int = 40) -> requests.Response:
    """Perform GET with up to 2 retries on transient errors."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            print(f"    [RETRY {attempt + 1}] {exc}")
            time.sleep(2)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# PMC search
# ---------------------------------------------------------------------------

def build_queries(disease: str, open_access: bool, min_date: str) -> List[Tuple[str, str]]:
    """
    Return a list of (label, query) pairs in descending quality order.

    The strategy uses four tiers:
      T1 – disease in *title* + review publication type   → very precise
      T2 – disease in title + any broad pubtype            → broader but still title-anchored
      T3 – disease in title + broad clinical terms         → no pubtype filter; last resort for
           diseases with few formal reviews
      T4 – disease anywhere + broad pubtype                → wide fallback
    Each tier adds IDs not already seen.
    """
    dn = f'"{disease}"'
    dt = f'{dn}[Title]'
    dta = f'{dn}[Title/Abstract]'
    oa = ' AND "open access"[filter]' if open_access else ""
    date = f" AND {min_date}:3000[pdat]" if min_date else ""
    bct = f"({_BROAD_CLINICAL_TERMS})"
    pubtypes = " OR ".join(BROAD_PUBTYPES)

    return [
        ("T1_title_review",
         f"{dt} AND review[pt]{oa}{date}"),

        ("T2_title_broad_pubtype",
         f"{dt} AND ({pubtypes}){oa}{date}"),

        ("T3_title_broad_clinical",
         f"{dt} AND {bct}{oa}{date}"),

        ("T4_anywhere_broad_pubtype",
         f"{dta} AND ({pubtypes}){oa}{date}"),

        ("T5_anywhere_broad_clinical",
         f"{dta} AND {bct}{oa}{date}"),
    ]


def search_pmc(query: str, retmax: int) -> List[str]:
    r = _safe_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pmc",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            **_ncbi_params(),
        },
    )
    return r.json().get("esearchresult", {}).get("idlist", [])


def collect_candidate_ids(
    disease: str,
    retmax: int,
    open_access: bool,
    max_candidates: int,
    min_date: str,
) -> List[str]:
    """Run all query tiers and return deduplicated IDs, capped at max_candidates."""
    seen: set = set()
    ids: List[str] = []

    print("\nStage 1 — PMC search queries:")
    for label, query in build_queries(disease, open_access, min_date):
        if len(ids) >= max_candidates:
            break
        print(f"  [{label}] {query[:120]}")
        try:
            found = search_pmc(query, retmax=retmax)
            new = [x for x in found if x not in seen]
            ids.extend(new[:max_candidates - len(ids)])
            seen.update(new)
            print(f"    → {len(found)} returned, {len(new)} new (total={len(ids)})")
            _sleep()
        except Exception as exc:
            print(f"    [SEARCH ERROR] {exc}")

    return ids


# ---------------------------------------------------------------------------
# Esummary metadata fetch
# ---------------------------------------------------------------------------

def fetch_summaries(pmc_ids: List[str]) -> Dict[str, Dict]:
    """Return a {pmc_id: summary_dict} mapping."""
    result: Dict[str, Dict] = {}
    for i in range(0, len(pmc_ids), 100):
        batch = pmc_ids[i: i + 100]
        try:
            r = _safe_get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={
                    "db": "pmc",
                    "id": ",".join(batch),
                    "retmode": "json",
                    **_ncbi_params(),
                },
            )
            data = r.json().get("result", {})
            for uid in data.get("uids", []):
                result[uid] = data.get(uid, {})
            _sleep()
        except Exception as exc:
            print(f"  [ESUMMARY ERROR] batch starting {batch[0]}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Title scoring
# ---------------------------------------------------------------------------

def score_title(title: str, disease: str) -> Tuple[int, List[str]]:
    """
    Return (score, reasons).
    Hard negative terms return a heavily penalised fixed score.
    """
    t = (title or "").lower()
    d = disease.lower()
    reasons: List[str] = []
    score = 0

    if not t:
        return -30, ["empty title"]

    # Hard negatives — drop immediately
    for term in HARD_NEGATIVE_TITLE_TERMS:
        if term in t:
            return -50, [f"hard negative: {term!r}"]

    # Disease presence
    if d in t:
        score += DISEASE_IN_TITLE_BONUS
        reasons.append("disease in title")
    else:
        score -= DISEASE_ABSENT_PENALTY
        reasons.append("disease not in title")

    # Broad clinical terms
    for term in BROAD_TITLE_TERMS:
        if term in t:
            score += BROAD_BONUS
            reasons.append(f"broad: {term!r}")

    # Narrow/specific terms
    for term in NARROW_TITLE_TERMS:
        if term in t:
            score -= NARROW_PENALTY
            reasons.append(f"narrow: {term!r}")

    # Short, disease-focused title is a good sign — only if no narrow terms fired
    word_count = len(t.split())
    narrow_hit_count = sum(1 for r in reasons if r.startswith("narrow:"))
    if d in t and word_count <= 10 and narrow_hit_count == 0:
        score += 4
        reasons.append("short disease-focused title")
    elif word_count >= 20:
        score -= 2
        reasons.append("very long title")

    return score, reasons


# ---------------------------------------------------------------------------
# Abstract pre-screening (Stage 2)
# ---------------------------------------------------------------------------

def score_abstract(abstract: str, disease: str) -> Tuple[int, str]:
    """
    Light scoring of the abstract to detect narrow-scope papers before
    spending a full-text HTTP call on them.

    Returns (score, reason).  Negative scores are soft rejects.
    """
    a = (abstract or "").lower()
    d = disease.lower()

    if not a:
        return 0, "no abstract available"

    narrow_hits = sum(1 for p in ABSTRACT_NARROW_PHRASES if p in a)
    broad_hits = sum(1 for p in ABSTRACT_BROAD_PHRASES if p in a)
    disease_hits = a.count(d)

    score = broad_hits * 2 - narrow_hits * 3

    # A single narrow phrase is not fatal; but 3+ is a red flag
    if narrow_hits >= 3:
        reason = f"abstract: {narrow_hits} narrow phrases vs {broad_hits} broad"
        return score, reason

    reason = f"abstract: broad={broad_hits}, narrow={narrow_hits}, disease_mentions={disease_hits}"
    return score, reason


# ---------------------------------------------------------------------------
# Full-text XML fetch and extraction
# ---------------------------------------------------------------------------

def fetch_fulltext_xml(pmc_id: str) -> str:
    r = _safe_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db": "pmc",
            "id": pmc_id,
            "rettype": "full",
            "retmode": "xml",
            **_ncbi_params(),
        },
        timeout=70,
    )
    return r.text


@dataclass
class Section:
    title: str
    text: str
    depth: int = 0


def _flatten_sections(sec_el: ET.Element, depth: int = 0) -> List[Section]:
    """
    Recursively flatten a <sec> element and all its nested sub-sections.
    This ensures sub-sections like <sec><title>Diagnosis</title>...</sec>
    inside a parent <sec><title>Clinical Features</title>...</sec> are captured.
    """
    sections: List[Section] = []
    title_el = sec_el.find("title")
    title = _norm(" ".join(title_el.itertext())) if title_el is not None else "Untitled"

    # Collect direct <p> children of this section (not of sub-sections)
    paras: List[str] = []
    for child in sec_el:
        if child.tag == "p":
            text = _norm(" ".join(child.itertext()))
            if text:
                paras.append(text)

    if paras:
        sections.append(Section(title=title, text="\n\n".join(paras), depth=depth))

    # Recurse into nested <sec> children
    for child in sec_el:
        if child.tag == "sec":
            sections.extend(_flatten_sections(child, depth + 1))

    return sections


def _norm(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_article_content(xml_string: str) -> Dict:
    """
    Parse PMC XML and return a dict with:
      title, abstract, sections (list of Section), body_md (Markdown string)
    """
    root = ET.fromstring(xml_string)
    result: Dict = {
        "title": None,
        "abstract": None,
        "sections": [],
        "body_md": None,
    }

    title_el = root.find(".//article-title")
    if title_el is not None:
        result["title"] = _norm(" ".join(title_el.itertext()))

    abstract_el = root.find(".//abstract")
    if abstract_el is not None:
        paras = [_norm(" ".join(p.itertext())) for p in abstract_el.iter("p")]
        result["abstract"] = "\n\n".join(p for p in paras if p) or None

    body_el = root.find(".//body")
    if body_el is not None:
        sections: List[Section] = []
        for top_sec in body_el.findall("sec"):
            sections.extend(_flatten_sections(top_sec, depth=0))
        result["sections"] = sections

        md_parts: List[str] = []
        for sec in sections:
            heading = "###" if sec.depth > 0 else "##"
            md_parts.append(f"{heading} {sec.title}\n\n{sec.text}")
        result["body_md"] = "\n\n".join(md_parts) if md_parts else None

    return result


def has_usable_body(article: Dict) -> bool:
    return bool(article.get("body_md")) and bool(article.get("sections"))


# ---------------------------------------------------------------------------
# Full-text clinical topic coverage scoring
# ---------------------------------------------------------------------------

def _count_hits(text: str, terms: List[str]) -> int:
    t = text.lower()
    return sum(1 for term in terms if term.lower() in t)


def score_full_text(article: Dict, disease: str) -> Tuple[int, Dict]:
    """
    Score an article for coverage of the 8 NepMedJP clinical topics.

    Returns (score, debug_info).
    """
    title: str = article.get("title") or ""
    abstract: str = article.get("abstract") or ""
    sections: List[Section] = article.get("sections") or []

    # Filter out boilerplate sections before scoring
    clinical_sections = [
        s for s in sections
        if not any(bt in s.title.lower() for bt in BOILERPLATE_SECTION_TERMS)
    ]

    section_titles_text = " ".join(s.title for s in clinical_sections)

    # Use up to 12 000 chars per section to avoid cutting off late sections
    body_text = " ".join(s.text[:12_000] for s in clinical_sections)

    combined = " ".join([title, abstract, section_titles_text, body_text]).lower()
    section_titles_l = section_titles_text.lower()

    covered_topics: Dict[str, Dict] = {}
    score = 0

    for topic_name, profile in TOPIC_PROFILES.items():
        body_hits = _count_hits(combined, profile["body_terms"])
        section_hits = _count_hits(section_titles_l, profile["section_terms"])

        # A topic is "covered" if there's a dedicated section heading OR
        # at least 2 matching body terms
        covered = section_hits >= 1 or body_hits >= 2

        covered_topics[topic_name] = {
            "covered": covered,
            "body_hits": body_hits,
            "section_hits": section_hits,
        }

        if covered:
            score += 5                        # base coverage reward

        if section_hits >= 1:
            score += 3                        # explicit section heading is strong evidence

        score += min(body_hits, 6)            # body evidence, capped per topic

    # Disease mention density (normalised by body length)
    d_lower = disease.lower()
    body_len = max(len(body_text), 1)
    disease_count = combined.count(d_lower)
    density = disease_count / (body_len / 1000)   # mentions per 1000 chars

    if density >= 1.5:
        score += 6
    elif density >= 0.8:
        score += 3
    elif density >= 0.2:
        score += 1
    elif disease_count < 2:
        score -= 8    # paper barely mentions the disease at all

    # Penalty for very few clinical sections
    if len(clinical_sections) < 3:
        score -= 5

    covered_count = sum(1 for v in covered_topics.values() if v["covered"])

    debug = {
        "covered_topic_count": covered_count,
        "covered_topics": covered_topics,
        "disease_mentions": disease_count,
        "disease_density_per1k": round(density, 3),
        "clinical_section_count": len(clinical_sections),
        "section_titles": [s.title for s in clinical_sections[:10]],
    }

    return score, debug


def score_candidate(article: Dict, meta: CandidateMeta, disease: str) -> ScoredArticle:
    """
    Combine title score + abstract score + full-text coverage score.
    Populates meta.final_score and meta.covered_topic_count.
    """
    ft_score, debug = score_full_text(article, disease)

    meta.final_score = ft_score + meta.title_score + meta.abstract_score
    meta.covered_topic_count = debug["covered_topic_count"]
    meta.section_titles = debug.get("section_titles", [])
    meta.disease_mentions = debug.get("disease_mentions", 0)

    return ScoredArticle(
        meta=meta,
        title_text=article.get("title") or "",
        abstract_text=article.get("abstract") or "",
        body_md=article.get("body_md") or "",
        covered_topics=debug.get("covered_topics", {}),
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def safe_dir(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", text.strip(), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "disease"


def save_markdown(scored: ScoredArticle, pmc_id: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"PMC{pmc_id}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {scored.title_text or 'Untitled'}\n\n")
        if scored.abstract_text:
            f.write("## Abstract\n\n")
            f.write(scored.abstract_text)
            f.write("\n\n")
        if scored.body_md:
            f.write(scored.body_md)
            f.write("\n")
    return path


def save_report(
    candidates: List[CandidateMeta],
    selected_ids: List[str],
    out_dir: str,
    disease: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{safe_dir(disease)}_selection_report.json")

    report_entries = []
    for cm in candidates:
        entry = asdict(cm)
        entry["selected"] = cm.pmc_id in selected_ids
        entry["title_reasons_summary"] = ", ".join(cm.title_reasons[:6])
        # Drop full title_reasons list to keep JSON compact
        entry.pop("title_reasons", None)
        report_entries.append(entry)

    # Sort: selected first, then by final_score
    report_entries.sort(
        key=lambda x: (not x["selected"], -x.get("final_score", 0))
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"disease": disease, "total_candidates": len(candidates), "papers": report_entries},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and rank PMC articles for NepMedJP pre-training data."
    )
    parser.add_argument(
        "--disease", type=str, default=None,
        help="Disease name, e.g. 'tuberculosis' or 'type 2 diabetes mellitus'",
    )
    parser.add_argument(
        "--target_papers", type=int, default=DEFAULT_TARGET_PAPERS,
        help=f"Number of papers to save (default: {DEFAULT_TARGET_PAPERS})",
    )
    parser.add_argument(
        "--retmax", type=int, default=DEFAULT_RETMAX_PER_QUERY,
        help=f"Max results per search query (default: {DEFAULT_RETMAX_PER_QUERY})",
    )
    parser.add_argument(
        "--max_candidates", type=int, default=DEFAULT_MAX_ABSTRACT_CANDIDATES,
        help=f"Max IDs entering abstract screening (default: {DEFAULT_MAX_ABSTRACT_CANDIDATES})",
    )
    parser.add_argument(
        "--max_fetch", type=int, default=DEFAULT_MAX_FULLTEXT_FETCH,
        help=f"Max full-text HTTP calls (default: {DEFAULT_MAX_FULLTEXT_FETCH})",
    )
    parser.add_argument(
        "--min_topics", type=int, default=DEFAULT_MIN_TOPICS_COVERED,
        help=f"Min topics covered in strict mode (default: {DEFAULT_MIN_TOPICS_COVERED})",
    )
    parser.add_argument(
        "--min_date", type=str, default=DEFAULT_MIN_DATE,
        help=f"Minimum publication year (default: {DEFAULT_MIN_DATE})",
    )
    parser.add_argument(
        "--strict", action="store_true", default=DEFAULT_STRICT,
        help="Reject papers covering fewer topics than --min_topics",
    )
    parser.add_argument(
        "--no_open_access_filter", action="store_true",
        help="Disable the open-access PMC filter",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: ./pmc_data/<disease>)",
    )
    args = parser.parse_args()

    disease = args.disease or input("Disease name: ").strip()
    if not disease:
        print("Error: disease name cannot be empty.")
        return

    out_dir = args.out_dir or f"./pmc_data/{safe_dir(disease)}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  NepMedJP PMC data collector")
    print(f"  Disease      : {disease}")
    print(f"  Target papers: {args.target_papers}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Strict mode  : {args.strict}")
    print(f"  Min date     : {args.min_date}")
    print(f"  OA filter    : {not args.no_open_access_filter}")
    print(f"{'=' * 70}")

    # -----------------------------------------------------------------------
    # Stage 1 — collect candidate IDs
    # -----------------------------------------------------------------------
    ids = collect_candidate_ids(
        disease=disease,
        retmax=args.retmax,
        open_access=not args.no_open_access_filter,
        max_candidates=args.max_candidates,
        min_date=args.min_date,
    )
    print(f"\nTotal unique candidate IDs after Stage 1: {len(ids)}")
    if not ids:
        print("No PMC candidates found. Try removing --no_open_access_filter or broadening the disease name.")
        return

    # -----------------------------------------------------------------------
    # Stage 2 — abstract screening
    # -----------------------------------------------------------------------
    print(f"\nStage 2 — Abstract screening ({len(ids)} candidates)...")
    summaries = fetch_summaries(ids)

    all_meta: List[CandidateMeta] = []
    for pmc_id in ids:
        info = summaries.get(pmc_id, {})
        title = _norm(info.get("title", ""))
        journal = _norm(info.get("fulljournalname", ""))
        pubdate = _norm(info.get("pubdate", ""))

        t_score, t_reasons = score_title(title, disease)
        a_score, a_reason = score_abstract(info.get("summary", ""), disease)

        meta = CandidateMeta(
            pmc_id=pmc_id,
            title=title,
            journal=journal,
            pubdate=pubdate,
            title_score=t_score,
            title_reasons=t_reasons,
            abstract_score=a_score,
        )

        combined_pre = t_score + a_score
        if t_score <= -50:
            meta.decision = "SKIP_HARD_NEGATIVE"
            meta.decision_reason = f"Hard negative title. {t_reasons}"
        elif combined_pre < -8:
            meta.decision = "SKIP_LOW_PRE_SCORE"
            meta.decision_reason = f"Pre-score {combined_pre}: {a_reason}"
        else:
            meta.decision = "PENDING_FULLTEXT"
            meta.decision_reason = a_reason

        all_meta.append(meta)

    pending = [m for m in all_meta if m.decision == "PENDING_FULLTEXT"]
    pending.sort(key=lambda x: x.title_score + x.abstract_score, reverse=True)
    to_fetch = pending[: args.max_fetch]

    skipped = len(all_meta) - len(pending)
    print(f"  Skipped (hard negatives / low pre-score): {skipped}")
    print(f"  Proceeding to full-text fetch:            {len(to_fetch)}")

    # -----------------------------------------------------------------------
    # Stage 3 — full-text fetch and scoring
    # -----------------------------------------------------------------------
    print(f"\nStage 3 — Full-text scoring ({len(to_fetch)} papers)...")
    print("-" * 70)

    all_scored: List[ScoredArticle] = []

    for meta in to_fetch:
        pmc_id = meta.pmc_id
        print(f"  PMC{pmc_id} | title_score={meta.title_score:+d} | {meta.title[:80]}")

        try:
            xml_text = fetch_fulltext_xml(pmc_id)
            article = extract_article_content(xml_text)

            if not has_usable_body(article):
                print(f"    [SKIP] no body text")
                meta.decision = "SKIP_NO_BODY"
                meta.decision_reason = "No full-text body found in XML"
                _sleep()
                continue

            scored = score_candidate(article, meta, disease)

            print(
                f"    topics={scored.meta.covered_topic_count}/8  "
                f"final_score={scored.meta.final_score:+d}  "
                f"sections={scored.meta.section_titles[:4]}"
            )

            if args.strict and scored.meta.covered_topic_count < args.min_topics:
                meta.decision = "SKIP_LOW_COVERAGE"
                meta.decision_reason = (
                    f"Covered only {scored.meta.covered_topic_count}/{args.min_topics} topics"
                )
                print(f"    [SKIP] strict mode: insufficient topic coverage")
            else:
                meta.decision = "CANDIDATE"
                all_scored.append(scored)

            _sleep()

        except ET.ParseError as exc:
            print(f"    [SKIP] XML parse error: {exc}")
            meta.decision = "SKIP_XML_ERROR"
            meta.decision_reason = str(exc)
            _sleep()
        except Exception as exc:
            print(f"    [SKIP] error: {exc}")
            meta.decision = "SKIP_ERROR"
            meta.decision_reason = str(exc)
            _sleep()

    # -----------------------------------------------------------------------
    # Final selection — top N by (covered_topic_count, final_score)
    # -----------------------------------------------------------------------
    all_scored.sort(
        key=lambda x: (
            x.meta.covered_topic_count,
            x.meta.final_score,
        ),
        reverse=True,
    )
    selected = all_scored[: args.target_papers]

    print(f"\n{'=' * 70}")
    print(f"FINAL SELECTION — top {len(selected)} of {len(all_scored)} candidates")
    print(f"{'=' * 70}")

    saved_ids: List[str] = []
    for rank, sc in enumerate(selected, start=1):
        md_path = save_markdown(sc, sc.meta.pmc_id, out_dir)
        sc.meta.decision = "SELECTED"
        saved_ids.append(sc.meta.pmc_id)
        print(
            f"  [{rank}] PMC{sc.meta.pmc_id}  "
            f"topics={sc.meta.covered_topic_count}/8  "
            f"score={sc.meta.final_score:+d}"
        )
        print(f"       Title  : {sc.meta.title[:90]}")
        print(f"       Journal: {sc.meta.journal}")
        print(f"       Saved  : {md_path}")

    # -----------------------------------------------------------------------
    # Save JSON report
    # -----------------------------------------------------------------------
    # Combine all_meta (which includes SKIP decisions) and scored candidates
    meta_map = {m.pmc_id: m for m in all_meta}
    # Update meta for scored candidates (they carry richer information)
    for sc in all_scored:
        meta_map[sc.meta.pmc_id] = sc.meta

    report_path = save_report(
        candidates=list(meta_map.values()),
        selected_ids=saved_ids,
        out_dir=out_dir,
        disease=disease,
    )

    print(f"\n{'=' * 70}")
    print(f"Done. {len(selected)} paper(s) saved to: {out_dir}")
    print(f"Selection report : {report_path}")
    print(f"{'=' * 70}")

    if len(selected) < args.target_papers:
        shortfall = args.target_papers - len(selected)
        print(f"\nWarning: {shortfall} fewer paper(s) than requested.")
        print("Options to improve coverage:")
        print("  --retmax 150 --max_fetch 60")
        print("  --no_open_access_filter")
        print("  --min_date 1990")
        if args.strict:
            print("  Remove --strict to allow lower-coverage papers")


if __name__ == "__main__":
    main()