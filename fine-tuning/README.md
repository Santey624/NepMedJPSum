# Japanese-Nepali Medical Summarization — Fine-tuning Data Creation

## Overview

This document describes the plan for constructing a silver-standard fine-tuning dataset for Japanese-Nepali medical summarization. The dataset will be used to fine-tune a continued-pretrained mT5-XL model for the downstream task. Evaluation is performed against a separate, gold-standard doctor-annotated benchmark (not described here).

**Target:** ~50,000 training pairs of (Japanese medical text, Nepali summary).

## Project Context

- **Task:** Japanese → Nepali cross-lingual medical summarization
- **Base model:** mT5-XL, continued-pretrained on mixed English/Japanese/Nepali medical and general data
- **Fine-tuning objective:** Adapt the pretrained model to produce Nepali clinical summaries given Japanese medical input
- **Evaluation:** Doctor-annotated gold benchmark covering 200+ diseases across 5 clinical specialties (ENT, ophthalmology, cardiovascular, infectious diseases, neurology/psychiatry) and 8 clinical topics (Clinical Definition, Etiology, Transmission, Signs & Symptoms, Types, Risk Factors, Diagnosis, Treatment)

## Design Principles

1. **Authentic English source material.** All training content is grounded in real medical literature. For this fine-tuning project, we will be using PMC.
2. **Parallel generation, not sequential chaining.** Japanese input and Nepali target are produced independently from a shared English anchor, avoiding error compounding through the pipeline.
3. **Multi-LLM generation.** Multiple frontier LLMs are used across the corpus to diversify training signal and reduce single-teacher stylistic overfitting.
4. **Train/evaluation alignment.** Training corpus matches the benchmark's 5-specialty, 8-topic structure to ensure fair evaluation.
5. **Quality-controlled at every stage.** Automatic filters plus doctor spot-checks validate the pipeline before full-scale generation.

## Pipeline Architecture

```
[English medical corpus]
        |
        |  (curated by disease x topic)
        v
[English source passage]
        |
        +--------------------------+
        |                          |
        v                          v
 [Translation LLM]          [Summarization LLM]
   (EN -> JA)                  (EN -> NE summary)
        |                          |
        v                          v
  [Japanese input]           [Nepali target]
        |                          |
        +----------+---------------+
                   |
                   v
            [Quality filters]
                   |
                   v
         [Doctor spot-check sample]
                   |
                   v
           [Training pair]
```

### Stage 1: English source retrieval

- **Source:**
  - PMC full-text sections
- **Curation:** Disease × topic database matching benchmark structure. Target ~500 diseases across 5 specialties, with all 8 topics covered per disease.
- **Specialty distribution:** Approximately 10,000 pairs per specialty (ENT, ophthalmology, cardiovascular, infectious, neuro/psychiatry).
- **Topic coverage:** All 8 topics represented, with MeSH subheading queries (`/Clinical_definition`, `/etiology`, `/transmission`, `/diagnosis`, `/therapy`, etc.) used to target topic-specific content from PubMed/PMC.

### Stage 2: English → Japanese translation

- **Translation models:** 2–3 frontier LLMs (to be selected via pilot), e.g., GPT-4, Claude, Gemini, or DeepL
- **Selection pilot:** 20 medical passages × 3 candidate systems, rated by Japanese physicians for medical accuracy and natural clinical register
- **Per-pair assignment:** Random across the corpus; LLM identity logged per pair
- **Prompt requirements:**
  - Specify medical domain and clinical target audience
  - Preserve dosages, drug names, numbers, negations, and units
  - Japanese medical register appropriate to source type
  - Iterated with Japanese physicians during pilot

### Stage 3: English → Nepali summarization

- **Summarization models:** 2–3 frontier LLMs (to be selected via pilot)
- **Selection pilot:** 20 medical passages × 3 candidate systems, rated by Nepali physicians for medical Nepali quality
- **Generation from English source, not from Japanese MT.** This is a deliberate design choice to break error chains.
- **Per-pair assignment:** Random across corpus; LLM identity logged
- **Prompt requirements:**
  - Target length based on gold benchmark Nepali summary distribution
  - Loanword handling aligned with doctor annotation style guide
  - Few-shot demonstrations drawn from gold benchmark
  - Instructions to preserve clinical facts without fabrication

## Quality Control

### Automatic filters

**Input side (Japanese MT quality):**
- Length ratio (Japanese / English character count within 0.3–1.5)
- Character composition (predominantly Japanese characters, not English fallback)
- Back-translation consistency (MT Japanese back to English, BERTScore similarity to original English above threshold)
- Minimum length threshold

**Target side (Nepali summary quality):**
- Devanagari character ratio (predominantly Devanagari, minimal English bleed-through)
- Length appropriateness (summary is 20–60% of source length)
- Non-trivial content

**Cross-check (input-target consistency):**
- Named entity extraction from both Japanese and Nepali
- Flag pairs where Nepali target references entities absent from Japanese input (hallucination risk)

### Doctor spot-check

- Stratified random sample of 300 pairs after automatic filtering
- 150 Japanese outputs reviewed by Japanese physicians
- 150 Nepali outputs reviewed by Nepali physicians
- 3-point rating: acceptable / borderline / unacceptable
- Target: ≥75% acceptance rate before scaling to full 50K
- If below threshold, iterate on prompts / LLM selection before proceeding

## Pilot Phase (Required Before Scaling)

Before generating the full 50K, a pilot on 300 pairs validates the pipeline end-to-end.

**Pilot deliverables:**
1. Finalized translation LLM selection (with doctor rationale)
2. Finalized summarization LLM selection (with doctor rationale)
3. Calibrated filter thresholds (based on observed quality distributions)
4. Measured rejection rate (how much over-collection is needed)
5. Doctor acceptance rate per LLM and overall
6. Documented generation prompts

**Exit criteria for pilot:**
- Overall doctor acceptance ≥75%
- Filter rejection rate ≤40%
- No systematic failure modes identified
- Per-LLM consistency within acceptable range

## Provenance and Metadata

Each training pair is stored with full metadata:

- `pair_id`: unique identifier
- `english_source_id`: identifier of source passage
- `english_source_url`: citation/URL
- `disease`: disease name
- `specialty`: one of {ENT, ophthalmology, cardiovascular, infectious, neuro_psych}
- `topic`: one of the 8 clinical topics
- `translation_llm`: which LLM produced the Japanese
- `summarization_llm`: which LLM produced the Nepali
- `filter_scores`: back-translation similarity, length ratios, entity-consistency score
- `spot_check_rating`: if applicable
- `japanese_text`: the training input
- `nepali_summary`: the training target

## Data Splits

- **Train:** ~45,000 pairs
- **Dev:** ~2,500 pairs (held out for hyperparameter tuning and early stopping)
- **Silver test:** ~2,500 pairs (for distribution-matched evaluation)
- **Gold evaluation:** separate 200+ disease benchmark (not part of this 50K; see benchmark documentation)

Splits are stratified by specialty and topic to maintain balanced distribution.

## Framing for Publication

The pipeline is described in publications as:

> A silver-standard training dataset constructed by parallel LLM-based generation from authentic English medical sources. Japanese inputs are produced by LLM translation; Nepali targets are produced by LLM summarization of the same English source. Multi-LLM generation reduces single-teacher stylistic overfitting. Quality is validated through automatic filtering and physician spot-checks.

The resulting model is positioned as a distillation of frontier-LLM medical summarization capability into an open, deployable model suitable for low-resource clinical settings where API-dependent LLMs are impractical.

## Known Limitations

- Training inputs (Japanese) carry MT quality limitations, unlike the gold benchmark where Japanese is doctor-post-edited.
- Training targets (Nepali) inherit LLM-generation limitations; this is the primary motivation for the doctor-annotated gold evaluation.
- Disease coverage is restricted to the 5 specialties represented in the gold benchmark; generalization beyond these specialties is not claimed.
- The 8-topic structure is adapted from standard medical textbook convention and may not fully represent all clinical content organization.

## Dependencies

- **Compute:** GPU access for continued pretraining and fine-tuning (A100 80GB or equivalent)
- **API budget:** Frontier LLM API access for translation and summarization at scale
- **Human resources:** Japanese physicians (for translation pilots and Japanese spot-checks), Nepali physicians (for summarization pilots and Nepali spot-checks)

## Open Decisions

The following are intentionally not yet finalized and will be determined during the pilot phase:

- Specific LLM selections for translation and summarization
- Final source mix proportions (may adjust based on accessibility and quality)
- Filter threshold values (calibrated from pilot quality distributions)
- Specific disease list (initial candidate: diseases from gold benchmark plus extensions within the same 5 specialties)

## Status

- [ ] Disease list finalized
- [ ] Source access verified (PubMed API, PMC API, StatPearls)
- [ ] Translation LLM pilot complete
- [ ] Summarization LLM pilot complete
- [ ] 300-pair end-to-end pilot complete
- [ ] Filter thresholds calibrated
- [ ] Full 50K generation complete
- [ ] Quality validation complete
- [ ] Dataset ready for fine-tuning