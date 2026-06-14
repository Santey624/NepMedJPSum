# NepMedJP: Japanese–Nepali Medical Summarization

NepMedJP is a research pipeline for cross-lingual medical summarization from Japanese to Nepali. It covers the full lifecycle: multilingual pre-training corpus construction, silver-standard fine-tuning dataset generation, and model training using mT5-XL.

---

## Task

Given a Japanese clinical text, generate a clinically accurate Nepali summary. The system targets five medical specialties and eight clinical topics aligned with a doctor-annotated gold benchmark.

**Languages:** Japanese (input) → Nepali (output)  
**Base model:** mT5-XL, continued pre-trained on mixed EN/JA/NE medical data  
**Fine-tuning target:** ~50,000 parallel (Japanese input, Nepali summary) training pairs

---

## Pipeline Overview

```
[English medical corpus — PMC open access]
              |
              | curated by disease × clinical topic
              v
     [English source passage]
              |
      ┌───────┴────────┐
      ▼                ▼
[LLM Translation]  [LLM Summarization]
  (EN → Japanese)    (EN → Nepali summary)
      |                |
      ▼                ▼
[Japanese input]  [Nepali target]
      |                |
      └───────┬────────┘
              ▼
      [Quality filters]
        - Length ratio
        - Back-translation consistency
        - Devanagari character ratio
        - Named entity cross-check
              |
              ▼
    [Doctor spot-check sample]
        (300 pairs, ≥75% acceptance)
              |
              ▼
       [Training pair]
```

---

## Repository Structure

```
NepMedJPSum/
│
├── pre-training/                        # mT5 continued pre-training
│   ├── pretraining_script.py            # main training loop
│   ├── data_pipeline_japanese/          # Japanese corpus cleaning
│   ├── data_pipeline_nepali/            # Nepali corpus cleaning
│   ├── data_preprocessing_nepali/       # Nepali normalization & splitting
│   ├── japanese_data_processing/        # Additional Japanese processing
│   ├── download_pubmed/                 # PMC/PubMed download scripts
│   └── data_download/                   # HuggingFace & database downloads
│
├── fine-tuning/                         # Fine-tuning data creation pipeline
│   ├── README.md                        # Detailed fine-tuning spec
│   ├── diseases/                        # Disease taxonomy (3,637 diseases × 6 domains)
│   │   ├── cardiovascular/data.jsonl
│   │   ├── ent/data.jsonl
│   │   ├── eye/data.jsonl
│   │   ├── infectious/data.jsonl
│   │   ├── psychiatry/data.jsonl
│   │   └── surgery/data.jsonl
│   ├── pipeline/
│   │   └── orchestration.py             # End-to-end pair generation
│   ├── pmc_collector.py                 # Fetch & rank PMC articles per disease
│   ├── pmc_coverage_checker.py          # Check PMC coverage for all diseases
│   └── generalize_disease_names.py      # Broaden disease names with no PMC hits
│
├── data_preprocessing/                  # Shared data cleaning utilities
│   └── Nepalidatacleaning/
│       ├── cleaning.py
│       └── deduplication.py
│
├── data_juicer/                         # Japanese data quality filtering
│
├── LICENSE                              # Apache 2.0
├── requirements.txt
└── .gitignore
```

---

## Disease Coverage

The fine-tuning corpus is structured around **3,637 diseases** across six clinical domains:

| Domain | Diseases |
|---|---|
| Cardiovascular | 592 |
| ENT | 542 |
| Ophthalmology | 600 |
| Infectious Disease | 704 |
| Psychiatry | 599 |
| Surgery | 600 |

Each disease maps to up to 5 PMC open-access review articles, collected and scored by `pmc_collector.py` across 8 clinical topic axes (Definition, Etiology, Pathogenesis, Signs & Symptoms, Classification, Risk Factors, Diagnosis, Treatment).

---

## Setup

```bash
git clone https://github.com/Santey624/NepMedJPSum.git
cd NepMedJPSum
pip install -r requirements.txt
```

Copy the environment template and add your API keys:

```bash
cp fine-tuning/.env.example fine-tuning/.env
# edit fine-tuning/.env with your OpenAI / Anthropic / Gemini keys
```

---

## Usage

### 1. Check PMC coverage for all diseases
```bash
cd fine-tuning
python pmc_coverage_checker.py              # all domains
python pmc_coverage_checker.py --domain ent # single domain
```
Results saved to `fine-tuning/pmc_coverage.jsonl`.

### 2. Generalize disease names with no PMC hits
```bash
python generalize_disease_names.py
```
Reads `pmc_coverage.jsonl`, tries progressive name simplification, saves to `pmc_coverage_generalized.jsonl`.

### 3. Collect PMC articles for a single disease
```bash
python pmc_collector.py --disease "tuberculosis"
python pmc_collector.py --disease "atrial fibrillation" --target_papers 5
```

### 4. Run the full fine-tuning data pipeline
```bash
python fine-tuning/pipeline/orchestration.py
```

### 5. Pre-training
```bash
python pre-training/pretraining_script.py
```

---

## Evaluation

Evaluation is performed against a **gold benchmark** of 200+ diseases annotated by Japanese and Nepali physicians across 5 specialties and 8 clinical topics. The benchmark is not included in this repository.

Automatic metrics used: BLEU, BERTScore, chrF.  
Human evaluation: 3-point physician rating (acceptable / borderline / unacceptable), target ≥75% acceptance.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{nepmedjp2025,
  title  = {NepMedJP: Cross-lingual Japanese–Nepali Medical Summarization},
  author = {Gaire, Santosh},
  year   = {2025},
  url    = {https://github.com/Santey624/NepMedJPSum}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
