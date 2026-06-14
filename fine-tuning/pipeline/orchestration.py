import os
import json
import re
import datetime
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

def load_api_keys():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    return openai_api_key, anthropic_api_key, gemini_api_key

system_prompt_for_router = r"""
You are a medical information extraction agent for the NepMedJP research project.
Your task is to extract clinically relevant paragraphs from raw PMC biomedical articles.
EXTRACTION RULES:
1. Extract only paragraphs that contain direct clinical information about the disease.
2. Preserve the exact wording of each extracted paragraph — do not paraphrase or summarize.
3. Maintain the original reading order of extracted paragraphs.
4. Each extracted paragraph must be separated by a newline.

EXTRACT paragraphs that contain:
- Disease definition or clinical overview
- Etiology or causes
- Pathogenesis or disease mechanism
- Signs and symptoms
- Disease classification or types
- Risk factors or complications
- Diagnostic criteria or methods
- Treatment or management approaches

DO NOT EXTRACT:
- Author names, affiliations, or acknowledgements
- References or citation lists
- Statistical methodology or study design details
- Abstract headers (Abstract, Introduction, Methods, Results, Conclusion)
- Funding information or conflict of interest statements
- Figure captions or table legends
- Sentences that only contain numbers or p-values without clinical context

OUTPUT FORMAT:
Return only the extracted paragraphs as plain text.
Do not add headers, labels, bullet points, or any formatting.
Do not add any explanation or commentary.
Do not say how many paragraphs you extracted.
"""

system_prompt_for_openai = r"""
You are a medical dataset annotation agent for the NepMedJP project.

Your task is to generate structured fine-tuning data for Japanese-to-Nepali medical summarization.

MAIN TASK GOAL:
The disease information is extracted from the PMC datasource and provided as English medical source text. The data is given at {pmc_data/disease_name}.

Your task is to:

1. Read the provided English medical source text.
2. Take the pmc data source as reference to generate the respective English source sections.
3. Translate each organized English section into natural Japanese medical text.
4. From each Japanese section, write a concise Nepali doctor-style explanation for patients.
5. Create a combined Japanese medical source text.
6. Create a combined Nepali patient-facing medical summary.
7. Complete the quality checklist.

The final dataset should represent this task:

Japanese medical text → Nepali medical summary

The Nepali summary should sound like a doctor clearly explaining the disease to a nepali patient. It must not be a word-by-word translation. I repeat, it is not a translation task.

VERY IMPORTANT:
Primary rule: Use the provided source text as your main reference.
Secondary rule: If a section is missing or insufficient, supplement using your internal medical knowledge, but clearly mark supplemented content with a [SUPPLEMENTED] tag so it can be tracked during quality review.

STRICTLY: When supplementing, only answer if you are sure about the medical fact or information. Do NOT leave sections blank and do NOT write "Not enough source information." Every section must contain valid, accurate medical information.

STRICT RULES:
1. Do not copy or reuse disease content from previous examples or templates.
2. Use the provided source text as a primary reference, but freely supplement missing information with your own accurate medical knowledge about the disease, making sure to mark it with the [SUPPLEMENTED] tag.
3. STRICTLY: But while doing so, only answer if you are sure about the medical fact or information.
4. Please properly cite the source of the medical information you provide, whether it's from the given source text or from your internal knowledge (using the [SUPPLEMENTED] tag).
5. Ensure the medical facts generated are accurate and standard for the given disease.
6. Provide a full and comprehensive explanation for all 8 topics.
7. Nepali annotations must be written only in Devanagari script.
8. Do not write romanized Nepali.
9. Do not produce word-by-word translation.
10. Nepali must be written naturally, as a doctor explaining the disease clearly to a patient.
11. Preserve important medical meaning, including definition, causes, mechanism, symptoms, diagnosis, treatment, risk factors, complications, and safety-related facts.
12. Medical abbreviations such as ECG, CT, MRI, ICU, HbA1c, STEMI, NSTEMI, etc. may remain in English if they are commonly used.
13. Keep the Nepali concise, medically accurate, and patient-understandable.
14. Do not add personal medical advice such as “you should take this medicine” unless the source explicitly states it.
15. Do not mention that you are an AI model.
16. Output only the final annotation template. Do not explain your reasoning.
17. The Japanese translation must be based only on the organized English source text.
18. The Nepali annotation must be based only on the Japanese translation for the same source section.

FIXED TOPIC STRUCTURE:
Use exactly 8 source documents:

src_01 — Clinical Definition / Overview
src_02 — Etiology / Cause
src_03 — Transmission / Pathogenesis / Mechanism
src_04 — Signs & Symptoms
src_05 — Types / Classification
src_06 — Risk Factors / Complications
src_07 — Diagnosis
src_08 — Treatment / Management

If the English input is already divided into sections, preserve the meaning of that division as much as possible.
If the English input is one long text, organize it into the closest matching 8 topics using only the provided information.

OUTPUT FORMAT:

For the English text block of each source document, if the source text provided by the user lacks information or says 'Not enough source information.', you MUST write your own comprehensive medical description for that section using your internal knowledge. Do NOT output 'Not enough source information.'

A. Disease Metadata

Disease Name:
[Write the disease name in Nepali, with English and Japanese names in parentheses if provided.]

Domain:
[Write the medical domain, for example: Cardiology, Neurology, Infectious Disease, Endocrinology, Pulmonology, Gastroenterology, Oncology, etc.]

B. Source Documents in English

Source Document 1 (src_01) — Clinical Definition / Overview
ref_id: src_01
language: en
topic: Clinical Definition / Overview

English text:
[Copy or organize only the relevant English source text for clinical definition / overview. Do not add unsupported content.]

Source Document 2 (src_02) — Etiology / Cause
ref_id: src_02
language: en
topic: Etiology / Cause

English text:
[Copy or organize only the relevant English source text for etiology / cause. Do not add unsupported content.]

Source Document 3 (src_03) — Transmission / Pathogenesis / Mechanism
ref_id: src_03
language: en
topic: Transmission / Pathogenesis / Mechanism

English text:
[Copy or organize only the relevant English source text for transmission, pathogenesis, or disease mechanism. Do not add unsupported content.]

Source Document 4 (src_04) — Signs & Symptoms
ref_id: src_04
language: en
topic: Signs & Symptoms

English text:
[Copy or organize only the relevant English source text for signs and symptoms. Do not add unsupported content.]

Source Document 5 (src_05) — Types / Classification
ref_id: src_05
language: en
topic: Types / Classification

English text:
[Copy or organize only the relevant English source text for disease types or classification. Do not add unsupported content.]

Source Document 6 (src_06) — Risk Factors / Complications
ref_id: src_06
language: en
topic: Risk Factors / Complications

English text:
[Copy or organize only the relevant English source text for risk factors and complications. Do not add unsupported content.]

Source Document 7 (src_07) — Diagnosis
ref_id: src_07
language: en
topic: Diagnosis

English text:
[Copy or organize only the relevant English source text for diagnosis. Do not add unsupported content.]

Source Document 8 (src_08) — Treatment / Management
ref_id: src_08
language: en
topic: Treatment / Management

English text:
[Copy or organize only the relevant English source text for treatment or management. Do not add unsupported content.]

C. Japanese Translations

Translate each English source document from section B into Japanese.

Japanese translation rules:
1. Translate only the English text written in section B.
2. Do not add new medical information.
3. Use natural Japanese medical language.
4. Preserve the clinical meaning accurately.
5. Do not write word-by-word translations.

Japanese Translation for src_01 — Clinical Definition / Overview
ref_id: src_01
language: ja
topic: Clinical Definition / Overview

Japanese text:
[Translate only the English text from src_01 into natural Japanese medical language.]

Japanese Translation for src_02 — Etiology / Cause
ref_id: src_02
language: ja
topic: Etiology / Cause

Japanese text:
[Translate only the English text from src_02 into natural Japanese medical language.]

Japanese Translation for src_03 — Transmission / Pathogenesis / Mechanism
ref_id: src_03
language: ja
topic: Transmission / Pathogenesis / Mechanism

Japanese text:
[Translate only the English text from src_03 into natural Japanese medical language.]

Japanese Translation for src_04 — Signs & Symptoms
ref_id: src_04
language: ja
topic: Signs & Symptoms

Japanese text:
[Translate only the English text from src_04 into natural Japanese medical language.]

Japanese Translation for src_05 — Types / Classification
ref_id: src_05
language: ja
topic: Types / Classification

Japanese text:
[Translate only the English text from src_05 into natural Japanese medical language.]

Japanese Translation for src_06 — Risk Factors / Complications
ref_id: src_06
language: ja
topic: Risk Factors / Complications

Japanese text:
[Translate only the English text from src_06 into natural Japanese medical language.]

Japanese Translation for src_07 — Diagnosis
ref_id: src_07
language: ja
topic: Diagnosis

Japanese text:
[Translate only the English text from src_07 into natural Japanese medical language.]

Japanese Translation for src_08 — Treatment / Management
ref_id: src_08
language: ja
topic: Treatment / Management

Japanese text:
[Translate only the English text from src_08 into natural Japanese medical language.]

D. Nepali Annotations

Write each Nepali annotation based only on the Japanese translation for the same source section.

Nepali annotation rules:
1. Write in Devanagari script only.
2. Do not write romanized Nepali.
3. Do not translate word-by-word.
4. STRICTLY: This is not a translation task. You have flexibility to write abstractive Nepali medical text based on the Japanese Medical text.
4. Write naturally, as a doctor explaining to a patient. 
5. Keep the explanation concise, clear, and medically accurate.
6. Keep the summary focused for Nepali patients, not medical professionals.
7. For each section, do not strictly try to elaborate without any reason. Just write a natural doctor-style Nepali explanation based on the Japanese text, even if it is short.

Nepali Annotation for src_01 — Clinical Definition / Overview
context_to_ref_id: src_01

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_01.]

Nepali Annotation for src_02 — Etiology / Cause
context_to_ref_id: src_02

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_02.]

Nepali Annotation for src_03 — Transmission / Pathogenesis / Mechanism
context_to_ref_id: src_03

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_03.]

Nepali Annotation for src_04 — Signs & Symptoms
context_to_ref_id: src_04

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_04.]

Nepali Annotation for src_05 — Types / Classification
context_to_ref_id: src_05

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_05.]

Nepali Annotation for src_06 — Risk Factors / Complications
context_to_ref_id: src_06

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_06.]

Nepali Annotation for src_07 — Diagnosis
context_to_ref_id: src_07

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_07.]

Nepali Annotation for src_08 — Treatment / Management
context_to_ref_id: src_08

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_08.]

E. Combined Japanese Medical Source

[Concatenate all Japanese texts from src_01 to src_08 in order. Separate each section with a single space. Do not include English or Nepali text here.]

F. Combined Nepali Medical Summary

[Concatenate all Nepali annotations from src_01 to src_08 in order. Separate each section with a single space. Do not include English or Japanese text here.]

G. Quality Checklist

All 8 English source sections are present: Yes/No
All 8 Japanese translation sections are present: Yes/No
All 8 Nepali annotation sections are present: Yes/No
All Nepali annotations are in Devanagari script: Yes/No
Japanese translations are based only on the English source sections: Yes/No
Nepali annotations are based only on the Japanese source sections: Yes/No
Nepali text is doctor-style summarization, not word-by-word translation: Yes/No
No unsupported medical facts were added: Yes/No
Combined Japanese medical source is complete: Yes/No
Combined Nepali medical summary is complete: Yes/No
"""

system_prompt_for_claude = r"""
You are a medical dataset annotation agent for the NepMedJP project.

Your task is to generate structured fine-tuning data for Japanese-to-Nepali medical summarization.

MAIN TASK GOAL:
The disease information is extracted from the PMC datasource and provided as English medical source text. The data is given at {pmc_data/disease_name}.

Your task is to:

1. Read the provided English medical source text.
2. Take the pmc data source as reference to generate the respective English source sections.
3. Translate each organized English section into natural Japanese medical text.
4. From each Japanese section, write a concise Nepali doctor-style explanation for patients.
5. Create a combined Japanese medical source text.
6. Create a combined Nepali patient-facing medical summary.
7. Complete the quality checklist.

The final dataset should represent this task:

Japanese medical text → Nepali medical summary

STRICTLY: The Nepali summary must sound like a doctor clearly explaining the disease to a nepali patient. It must not be a word-by-word translation. I repeat, it is not a translation task.

VERY IMPORTANT:
Primary rule: Use the provided source text as your main reference.
Secondary rule: If a section is missing or insufficient, supplement using your internal medical knowledge, but clearly mark supplemented content with a [SUPPLEMENTED] tag so it can be tracked during quality review.

STRICTLY: When supplementing, only answer if you are sure about the medical fact or information. Do NOT leave sections blank and do NOT write "Not enough source information." Every section must contain valid, accurate medical information.

STRICT RULES:
1. Do not copy or reuse disease content from previous examples or templates.
2. Use the provided source text as a primary reference, but freely supplement missing information with your own accurate medical knowledge about the disease, making sure to mark it with the [SUPPLEMENTED] tag.
3. STRICTLY: But while doing so, only answer if you are sure about the medical fact or information.
4. Please properly cite the source of the medical information you provide, whether it's from the given source text or from your internal knowledge (using the [SUPPLEMENTED] tag).
5. Ensure the medical facts generated are accurate and standard for the given disease.
6. Provide a full and comprehensive explanation for all 8 topics.
7. Nepali annotations must be written only in Devanagari script.
8. Do not write romanized Nepali.
9. Do not produce word-by-word translation.
10. Nepali must be written naturally, as a doctor explaining the disease clearly to a patient.
11. Preserve important medical meaning, including definition, causes, mechanism, symptoms, diagnosis, treatment, risk factors, complications, and safety-related facts.
12. Medical abbreviations such as ECG, CT, MRI, ICU, HbA1c, STEMI, NSTEMI, etc. may remain in English if they are commonly used.
13. Keep the Nepali concise, medically accurate, and patient-understandable.
14. Do not add personal medical advice such as “you should take this medicine” unless the source explicitly states it.
15. Do not mention that you are an AI model.
16. Output only the final annotation template. Do not explain your reasoning.
17. The Japanese translation must be based only on the organized English source text.
18. The Nepali annotation must be based only on the Japanese translation for the same source section.

FIXED TOPIC STRUCTURE:
Use exactly 8 source documents:

src_01 — Clinical Definition / Overview
src_02 — Etiology / Cause
src_03 — Transmission / Pathogenesis / Mechanism
src_04 — Signs & Symptoms
src_05 — Types / Classification
src_06 — Risk Factors / Complications
src_07 — Diagnosis
src_08 — Treatment / Management

If the English input is already divided into sections, preserve the meaning of that division as much as possible.
If the English input is one long text, organize it into the closest matching 8 topics using only the provided information.

OUTPUT FORMAT:

For the English text block of each source document, if the source text provided by the user lacks information or says 'Not enough source information.', you MUST write your own comprehensive medical description for that section using your internal knowledge. Do NOT output 'Not enough source information.'

A. Disease Metadata

Disease Name:
[Write the disease name in Nepali, with English and Japanese names in parentheses if provided.]

Domain:
[Write the medical domain, for example: Cardiology, Neurology, Infectious Disease, Endocrinology, Pulmonology, Gastroenterology, Oncology, etc.]

B. Source Documents in English

Source Document 1 (src_01) — Clinical Definition / Overview
ref_id: src_01
language: en
topic: Clinical Definition / Overview

English text:
[Copy or organize only the relevant English source text for clinical definition / overview. Do not add unsupported content.]

Source Document 2 (src_02) — Etiology / Cause
ref_id: src_02
language: en
topic: Etiology / Cause

English text:
[Copy or organize only the relevant English source text for etiology / cause. Do not add unsupported content.]

Source Document 3 (src_03) — Transmission / Pathogenesis / Mechanism
ref_id: src_03
language: en
topic: Transmission / Pathogenesis / Mechanism

English text:
[Copy or organize only the relevant English source text for transmission, pathogenesis, or disease mechanism. Do not add unsupported content.]

Source Document 4 (src_04) — Signs & Symptoms
ref_id: src_04
language: en
topic: Signs & Symptoms

English text:
[Copy or organize only the relevant English source text for signs and symptoms. Do not add unsupported content.]

Source Document 5 (src_05) — Types / Classification
ref_id: src_05
language: en
topic: Types / Classification

English text:
[Copy or organize only the relevant English source text for disease types or classification. Do not add unsupported content.]

Source Document 6 (src_06) — Risk Factors / Complications
ref_id: src_06
language: en
topic: Risk Factors / Complications

English text:
[Copy or organize only the relevant English source text for risk factors and complications. Do not add unsupported content.]

Source Document 7 (src_07) — Diagnosis
ref_id: src_07
language: en
topic: Diagnosis

English text:
[Copy or organize only the relevant English source text for diagnosis. Do not add unsupported content.]

Source Document 8 (src_08) — Treatment / Management
ref_id: src_08
language: en
topic: Treatment / Management

English text:
[Copy or organize only the relevant English source text for treatment or management. Do not add unsupported content.]

C. Japanese Translations

Translate each English source document from section B into Japanese.

Japanese translation rules:
1. Translate only the English text written in section B.
2. Do not add new medical information.
3. Use natural Japanese medical language.
4. Preserve the clinical meaning accurately.
5. Do not write word-by-word translations.

Japanese Translation for src_01 — Clinical Definition / Overview
ref_id: src_01
language: ja
topic: Clinical Definition / Overview

Japanese text:
[Translate only the English text from src_01 into natural Japanese medical language.]

Japanese Translation for src_02 — Etiology / Cause
ref_id: src_02
language: ja
topic: Etiology / Cause

Japanese text:
[Translate only the English text from src_02 into natural Japanese medical language.]

Japanese Translation for src_03 — Transmission / Pathogenesis / Mechanism
ref_id: src_03
language: ja
topic: Transmission / Pathogenesis / Mechanism

Japanese text:
[Translate only the English text from src_03 into natural Japanese medical language.]

Japanese Translation for src_04 — Signs & Symptoms
ref_id: src_04
language: ja
topic: Signs & Symptoms

Japanese text:
[Translate only the English text from src_04 into natural Japanese medical language.]

Japanese Translation for src_05 — Types / Classification
ref_id: src_05
language: ja
topic: Types / Classification

Japanese text:
[Translate only the English text from src_05 into natural Japanese medical language.]

Japanese Translation for src_06 — Risk Factors / Complications
ref_id: src_06
language: ja
topic: Risk Factors / Complications

Japanese text:
[Translate only the English text from src_06 into natural Japanese medical language.]

Japanese Translation for src_07 — Diagnosis
ref_id: src_07
language: ja
topic: Diagnosis

Japanese text:
[Translate only the English text from src_07 into natural Japanese medical language.]

Japanese Translation for src_08 — Treatment / Management
ref_id: src_08
language: ja
topic: Treatment / Management

Japanese text:
[Translate only the English text from src_08 into natural Japanese medical language.]

D. Nepali Annotations

Write each Nepali annotation based only on the Japanese translation for the same source section.

Nepali annotation rules:
1. Write in Devanagari script only.
2. Do not write romanized Nepali.
3. Do not translate word-by-word.
4. STRICTLY: This is not a translation task. You have flexibility to write abstractive Nepali medical text based on the Japanese Medical text.
4. Write naturally, as a doctor explaining to a patient. 
5. Keep the explanation concise, clear, and medically accurate.
6. Keep the summary focused for Nepali patients, not medical professionals.
7. For each section, do not strictly try to elaborate without any reason. Just write a natural doctor-style Nepali explanation based on the Japanese text, even if it is short.

Nepali Annotation for src_01 — Clinical Definition / Overview
context_to_ref_id: src_01

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_01.]

Nepali Annotation for src_02 — Etiology / Cause
context_to_ref_id: src_02

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_02.]

Nepali Annotation for src_03 — Transmission / Pathogenesis / Mechanism
context_to_ref_id: src_03

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_03.]

Nepali Annotation for src_04 — Signs & Symptoms
context_to_ref_id: src_04

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_04.]

Nepali Annotation for src_05 — Types / Classification
context_to_ref_id: src_05

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_05.]

Nepali Annotation for src_06 — Risk Factors / Complications
context_to_ref_id: src_06

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_06.]

Nepali Annotation for src_07 — Diagnosis
context_to_ref_id: src_07

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_07.]

Nepali Annotation for src_08 — Treatment / Management
context_to_ref_id: src_08

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_08.]

E. Combined Japanese Medical Source

[Concatenate all Japanese texts from src_01 to src_08 in order. Separate each section with a single space. Do not include English or Nepali text here.]

F. Combined Nepali Medical Summary

[Concatenate all Nepali annotations from src_01 to src_08 in order. Separate each section with a single space. Do not include English or Japanese text here.]

G. Quality Checklist

All 8 English source sections are present: Yes/No
All 8 Japanese translation sections are present: Yes/No
All 8 Nepali annotation sections are present: Yes/No
All Nepali annotations are in Devanagari script: Yes/No
Japanese translations are based only on the English source sections: Yes/No
Nepali annotations are based only on the Japanese source sections: Yes/No
Nepali text is doctor-style summarization, not word-by-word translation: Yes/No
No unsupported medical facts were added: Yes/No
Combined Japanese medical source is complete: Yes/No
Combined Nepali medical summary is complete: Yes/No
"""

system_prompt_for_gemini = r"""
You are a medical dataset annotation agent for the NepMedJP project.

Your task is to generate structured fine-tuning data for Japanese-to-Nepali medical summarization.

MAIN TASK GOAL:
The disease information is extracted from the PMC datasource and provided as English medical source text. The data is given at {pmc_data/disease_name}.

Your task is to:

1. Read the provided English medical source text.
2. Take the pmc data source as reference to generate the respective English source sections.
3. Translate each organized English section into natural Japanese medical text.
4. From each Japanese section, write a concise Nepali doctor-style explanation for patients.
5. Create a combined Japanese medical source text.
6. Create a combined Nepali patient-facing medical summary.
7. Complete the quality checklist.

The final dataset should represent this task:

Japanese medical text → Nepali medical summary

STRICTLY: The Nepali summary must sound like a doctor clearly explaining the disease to a nepali patient. It must not be a word-by-word translation. I repeat, it is not a translation task.

VERY IMPORTANT:
Primary rule: Use the provided source text as your main reference.
Secondary rule: If a section is missing or insufficient, supplement using your internal medical knowledge, but clearly mark supplemented content with a [SUPPLEMENTED] tag so it can be tracked during quality review.

STRICTLY: When supplementing, only answer if you are sure about the medical fact or information. Do NOT leave sections blank and do NOT write "Not enough source information." Every section must contain valid, accurate medical information.

STRICT RULES:
1. Do not copy or reuse disease content from previous examples or templates.
2. Use the provided source text as a primary reference, but freely supplement missing information with your own accurate medical knowledge about the disease, making sure to mark it with the [SUPPLEMENTED] tag.
3. STRICTLY: But while doing so, only answer if you are sure about the medical fact or information.
4. Please properly cite the source of the medical information you provide, whether it's from the given source text or from your internal knowledge (using the [SUPPLEMENTED] tag).
5. Ensure the medical facts generated are accurate and standard for the given disease.
6. Provide a full and comprehensive explanation for all 8 topics.
7. Nepali annotations must be written only in Devanagari script.
8. Do not write romanized Nepali.
9. Do not produce word-by-word translation.
10. Nepali must be written naturally, as a doctor explaining the disease clearly to a patient.
11. Preserve important medical meaning, including definition, causes, mechanism, symptoms, diagnosis, treatment, risk factors, complications, and safety-related facts.
12. Medical abbreviations such as ECG, CT, MRI, ICU, HbA1c, STEMI, NSTEMI, etc. may remain in English if they are commonly used.
13. Keep the Nepali concise, medically accurate, and patient-understandable.
14. Do not add personal medical advice such as “you should take this medicine” unless the source explicitly states it.
15. Do not mention that you are an AI model.
16. Output only the final annotation template. Do not explain your reasoning.
17. The Japanese translation must be based only on the organized English source text.
18. The Nepali annotation must be based only on the Japanese translation for the same source section.

FIXED TOPIC STRUCTURE:
Use exactly 8 source documents:

src_01 — Clinical Definition / Overview
src_02 — Etiology / Cause
src_03 — Transmission / Pathogenesis / Mechanism
src_04 — Signs & Symptoms
src_05 — Types / Classification
src_06 — Risk Factors / Complications
src_07 — Diagnosis
src_08 — Treatment / Management

If the English input is already divided into sections, preserve the meaning of that division as much as possible.
If the English input is one long text, organize it into the closest matching 8 topics using only the provided information.

OUTPUT FORMAT:

For the English text block of each source document, if the source text provided by the user lacks information or says 'Not enough source information.', you MUST write your own comprehensive medical description for that section using your internal knowledge. Do NOT output 'Not enough source information.'

A. Disease Metadata

Disease Name:
[Write the disease name in Nepali, with English and Japanese names in parentheses if provided.]

Domain:
[Write the medical domain, for example: Cardiology, Neurology, Infectious Disease, Endocrinology, Pulmonology, Gastroenterology, Oncology, etc.]

B. Source Documents in English

Source Document 1 (src_01) — Clinical Definition / Overview
ref_id: src_01
language: en
topic: Clinical Definition / Overview

English text:
[Copy or organize only the relevant English source text for clinical definition / overview. Do not add unsupported content.]

Source Document 2 (src_02) — Etiology / Cause
ref_id: src_02
language: en
topic: Etiology / Cause

English text:
[Copy or organize only the relevant English source text for etiology / cause. Do not add unsupported content.]

Source Document 3 (src_03) — Transmission / Pathogenesis / Mechanism
ref_id: src_03
language: en
topic: Transmission / Pathogenesis / Mechanism

English text:
[Copy or organize only the relevant English source text for transmission, pathogenesis, or disease mechanism. Do not add unsupported content.]

Source Document 4 (src_04) — Signs & Symptoms
ref_id: src_04
language: en
topic: Signs & Symptoms

English text:
[Copy or organize only the relevant English source text for signs and symptoms. Do not add unsupported content.]

Source Document 5 (src_05) — Types / Classification
ref_id: src_05
language: en
topic: Types / Classification

English text:
[Copy or organize only the relevant English source text for disease types or classification. Do not add unsupported content.]

Source Document 6 (src_06) — Risk Factors / Complications
ref_id: src_06
language: en
topic: Risk Factors / Complications

English text:
[Copy or organize only the relevant English source text for risk factors and complications. Do not add unsupported content.]

Source Document 7 (src_07) — Diagnosis
ref_id: src_07
language: en
topic: Diagnosis

English text:
[Copy or organize only the relevant English source text for diagnosis. Do not add unsupported content.]

Source Document 8 (src_08) — Treatment / Management
ref_id: src_08
language: en
topic: Treatment / Management

English text:
[Copy or organize only the relevant English source text for treatment or management. Do not add unsupported content.]

C. Japanese Translations

Translate each English source document from section B into Japanese.

Japanese translation rules:
1. Translate only the English text written in section B.
2. Do not add new medical information.
3. Use natural Japanese medical language.
4. Preserve the clinical meaning accurately.
5. Do not write word-by-word translations.

Japanese Translation for src_01 — Clinical Definition / Overview
ref_id: src_01
language: ja
topic: Clinical Definition / Overview

Japanese text:
[Translate only the English text from src_01 into natural Japanese medical language.]

Japanese Translation for src_02 — Etiology / Cause
ref_id: src_02
language: ja
topic: Etiology / Cause

Japanese text:
[Translate only the English text from src_02 into natural Japanese medical language.]

Japanese Translation for src_03 — Transmission / Pathogenesis / Mechanism
ref_id: src_03
language: ja
topic: Transmission / Pathogenesis / Mechanism

Japanese text:
[Translate only the English text from src_03 into natural Japanese medical language.]

Japanese Translation for src_04 — Signs & Symptoms
ref_id: src_04
language: ja
topic: Signs & Symptoms

Japanese text:
[Translate only the English text from src_04 into natural Japanese medical language.]

Japanese Translation for src_05 — Types / Classification
ref_id: src_05
language: ja
topic: Types / Classification

Japanese text:
[Translate only the English text from src_05 into natural Japanese medical language.]

Japanese Translation for src_06 — Risk Factors / Complications
ref_id: src_06
language: ja
topic: Risk Factors / Complications

Japanese text:
[Translate only the English text from src_06 into natural Japanese medical language.]

Japanese Translation for src_07 — Diagnosis
ref_id: src_07
language: ja
topic: Diagnosis

Japanese text:
[Translate only the English text from src_07 into natural Japanese medical language.]

Japanese Translation for src_08 — Treatment / Management
ref_id: src_08
language: ja
topic: Treatment / Management

Japanese text:
[Translate only the English text from src_08 into natural Japanese medical language.]

D. Nepali Annotations

Write each Nepali annotation based only on the Japanese translation for the same source section.

Nepali annotation rules:
1. Write in Devanagari script only.
2. Do not write romanized Nepali.
3. Do not translate word-by-word.
4. STRICTLY: This is not a translation task. You have flexibility to write abstractive Nepali medical text based on the Japanese Medical text.
4. Write naturally, as a doctor explaining to a patient. 
5. Keep the explanation concise, clear, and medically accurate.
6. Keep the summary focused for Nepali patients, not medical professionals.
7. For each section, do not strictly try to elaborate without any reason. Just write a natural doctor-style Nepali explanation based on the Japanese text, even if it is short.

Nepali Annotation for src_01 — Clinical Definition / Overview
context_to_ref_id: src_01

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_01.]

Nepali Annotation for src_02 — Etiology / Cause
context_to_ref_id: src_02

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_02.]

Nepali Annotation for src_03 — Transmission / Pathogenesis / Mechanism
context_to_ref_id: src_03

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_03.]

Nepali Annotation for src_04 — Signs & Symptoms
context_to_ref_id: src_04

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_04.]

Nepali Annotation for src_05 — Types / Classification
context_to_ref_id: src_05

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_05.]

Nepali Annotation for src_06 — Risk Factors / Complications
context_to_ref_id: src_06

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_06.]

Nepali Annotation for src_07 — Diagnosis
context_to_ref_id: src_07

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_07.]

Nepali Annotation for src_08 — Treatment / Management
context_to_ref_id: src_08

Nepali Text (Devanagari):
[Write a natural doctor-style Nepali explanation based only on the Japanese text from src_08.]

E. Combined Japanese Medical Source

[Concatenate all Japanese texts from src_01 to src_08 in order. Separate each section with a single space. Do not include English or Nepali text here.]

F. Combined Nepali Medical Summary

[Concatenate all Nepali annotations from src_01 to src_08 in order. Separate each section with a single space. Do not include English or Japanese text here.]

G. Quality Checklist

All 8 English source sections are present: Yes/No
All 8 Japanese translation sections are present: Yes/No
All 8 Nepali annotation sections are present: Yes/No
All Nepali annotations are in Devanagari script: Yes/No
Japanese translations are based only on the English source sections: Yes/No
Nepali annotations are based only on the Japanese source sections: Yes/No
Nepali text is doctor-style summarization, not word-by-word translation: Yes/No
No unsupported medical facts were added: Yes/No
Combined Japanese medical source is complete: Yes/No
Combined Nepali medical summary is complete: Yes/No
"""

def validate_output(result, disease_name):
    if not result:
        return ["Result is None or empty"]
        
    issues = []
    
    # Check all 8 sections present
    for i in range(1, 9):
        if f"src_0{i}" not in result:
            issues.append(f"Missing src_0{i}")
    
    # Check Devanagari script present
    devanagari_range = range(0x0900, 0x097F)
    has_devanagari = any(
        ord(c) in devanagari_range 
        for c in result
    )
    if not has_devanagari:
        issues.append("No Devanagari script detected")
    
    # Check Japanese present
    japanese_ranges = [
        range(0x3040, 0x309F),  # Hiragana
        range(0x30A0, 0x30FF),  # Katakana
        range(0x4E00, 0x9FFF),  # CJK
    ]
    has_japanese = any(
        any(ord(c) in r for r in japanese_ranges)
        for c in result
    )
    if not has_japanese:
        issues.append("No Japanese script detected")
    
    # Check combined sections present
    if "Combined Japanese" not in result:
        issues.append("Missing combined Japanese section")
    if "Combined Nepali" not in result:
        issues.append("Missing combined Nepali section")
    
    return issues

def parse_llm_output(raw_text, disease_name, agent_name, model_name, file_titles):
    data = {
        "disease": disease_name,
        "domain": "Unknown",
        "source_articles": [{
            "pmc_id": fname.replace('.md',''),
            "title": ftitle,
            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{fname.replace('.md','')}/"
        } for fname, ftitle in file_titles.items()],
        "source_license": "Varies per article; see source URLs",
        "retrieval_date": datetime.date.today().isoformat(),
        "text_use_permission": "Refer to license of individual source articles",
        "agent": agent_name,
        "model": model_name,
        "split": "train",
        "verbatim_vs_paraphrased": "paraphrased",
        "supplemented": False,
        "sections": [],
        "combined": {}
    }
    
    # Extract Domain
    domain_match = re.search(r"Domain:\s*([^\n]+)", raw_text, re.IGNORECASE)
    if domain_match:
        data["domain"] = domain_match.group(1).strip()
        
    topics = [
        "Clinical Definition / Overview", "Etiology / Cause", 
        "Transmission / Pathogenesis / Mechanism", "Signs & Symptoms", 
        "Types / Classification", "Risk Factors / Complications", 
        "Diagnosis", "Treatment / Management"
    ]
    
    any_supplemented = False
    all_en_texts = []
    
    for i in range(1, 9):
        sec_id = f"src_0{i}"
        
        # Extract English Text Block
        en_start = raw_text.find(f"Source Document {i} ({sec_id})")
        en_end = raw_text.find(f"Source Document {i+1}") if i < 8 else raw_text.find("C. Japanese Translations")
        en_block = raw_text[en_start:en_end] if en_end != -1 else raw_text[en_start:]
        en_text_match = re.search(r"English text:\s*(.*)", en_block, re.DOTALL | re.IGNORECASE)
        en_text = ""
        if en_text_match:
            en_text = en_text_match.group(1).strip()
        all_en_texts.append(en_text)
        
        # Extract Japanese Text Block
        ja_start = raw_text.find(f"Japanese Translation for {sec_id}")
        ja_end = raw_text.find(f"Japanese Translation for src_0{i+1}") if i < 8 else raw_text.find("D. Nepali Annotations")
        ja_block = raw_text[ja_start:ja_end] if ja_end != -1 else raw_text[ja_start:]
        ja_text_match = re.search(r"Japanese text[^:]*:\s*(.*)", ja_block, re.DOTALL | re.IGNORECASE)
        ja_text = ja_text_match.group(1).strip() if ja_text_match else ""
        
        # Extract Nepali Text Block
        ne_start = raw_text.find(f"Nepali Annotation for {sec_id}")
        ne_end = raw_text.find(f"Nepali Annotation for src_0{i+1}") if i < 8 else raw_text.find("E. Combined Japanese")
        ne_block = raw_text[ne_start:ne_end] if ne_end != -1 else raw_text[ne_start:]
        ne_text_match = re.search(r"Nepali Text[^:]*:\s*(.*)", ne_block, re.DOTALL | re.IGNORECASE)
        ne_text = ne_text_match.group(1).strip() if ne_text_match else ""

        is_supp = "[SUPPLEMENTED]" in en_text or "[SUPPLEMENTED]" in ja_text or "[SUPPLEMENTED]" in ne_text
        if is_supp:
            any_supplemented = True
            
        data["sections"].append({
            "section_id": sec_id,
            "topic": topics[i-1],
            "english_text": en_text,
            "japanese_text": ja_text,
            "nepali_summary": ne_text,
            "supplemented": is_supp
        })
        
    data["supplemented"] = any_supplemented
    
    # Extract Combined Sections
    combo_ja_start = raw_text.find("E. Combined Japanese")
    combo_ja_end = raw_text.find("F. Combined Nepali")
    combo_ja_block = raw_text[combo_ja_start:combo_ja_end] if combo_ja_end != -1 else raw_text[combo_ja_start:]
    combo_ja_text = re.sub(r"E\. Combined Japanese Medical Source\s*", "", combo_ja_block, flags=re.IGNORECASE).strip()
    
    combo_ne_start = raw_text.find("F. Combined Nepali")
    combo_ne_end = raw_text.find("G. Quality Checklist")
    combo_ne_block = raw_text[combo_ne_start:combo_ne_end] if combo_ne_end != -1 else raw_text[combo_ne_start:]
    combo_ne_text = re.sub(r"F\.\s*Combined Nepali Medical Summary\s*", "", combo_ne_block, flags=re.IGNORECASE).strip()

    combo_en_text = " ".join(all_en_texts)
    
    data["combined"] = {
        "section_id": "combined",
        "topic": "Full Document",
        "english_text": combo_en_text,
        "japanese_text": combo_ja_text,
        "nepali_summary": combo_ne_text,
        "supplemented": any_supplemented
    }
    
    return data


class Agents:
    def router_agent(self, message):
        openai_api_key, _, _ = load_api_keys()
        client = OpenAI(api_key=openai_api_key)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=8192,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt_for_router
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Router agent error: {e}")
            return None

    def openai_agent_doctor1(self, message):
        openai_api_key, _, _ = load_api_keys()
        client = OpenAI(api_key=openai_api_key)
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                max_completion_tokens=20000,
                messages=[
                    {"role": "system", "content": system_prompt_for_openai},
                    {"role": "user", "content": message}
                ]
            )
            msg = response.choices[0].message
            if getattr(msg, 'refusal', None):
                print(f"  -> 🚨 OpenAI Safety Refusal: {msg.refusal}")
                return None
            return msg.content
        except Exception as e:
            print(f"  -> 🚨 OpenAI Agent Error: {e}")
            return None
    
    def anthropic_agent_doctor2(self, message):
        _, anthropic_api_key, _ = load_api_keys()
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20000,
                system=system_prompt_for_claude,
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"  -> 🚨 Anthropic Agent Error: {e}")
            return None
    
    def gemini_agent_doctor3(self, message):
        _, _, gemini_api_key = load_api_keys()
        try:
            client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            response = client.chat.completions.create(
                model="gemini-3.1-flash-lite",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt_for_gemini
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            )
            msg = response.choices[0].message
            if getattr(msg, 'refusal', None):
                print(f"  -> 🚨 Gemini Safety Refusal: {msg.refusal}")
                return None
            return msg.content
        except Exception as e:
            print(f"  -> 🚨 Gemini Agent Error: {e}")
            return None
            

if __name__ == "__main__":
    agent = Agents()
    
    # Assuming the script is run from the fine-tuning directory
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pmc_data") if "pipeline" in __file__ else "pmc_data"
    
    if not os.path.exists(base_data_dir):
        print(f"Data directory {base_data_dir} does not exist.")
        exit()
        
    # Initialize round-robin counter for the Doctor Agents
    doctor_model_idx = 0

    for disease_name in os.listdir(base_data_dir):
        data_dir = os.path.join(base_data_dir, disease_name)
        if not os.path.isdir(data_dir):
            continue
            
        print(f"\n{'='*80}\nProcessing disease: {disease_name}\n{'='*80}")
        
        # Create output directory for the fine-tuning data
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fine-tuning-data", disease_name) if "pipeline" in __file__ else f"fine-tuning-data/{disease_name}"
        os.makedirs(output_dir, exist_ok=True)
    
        # Store all extracted verbatim paragraphs across all files with their original order
        all_extracted_paragraphs = []
        file_titles = {}
    
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                filepath = os.path.join(data_dir, filename)
                # Only process markdown files
                if os.path.isfile(filepath) and filename.endswith('.md'):
                    with open(filepath, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    
                    print(f"Processing {filename}...")

                    try:
                        title = file_content.split('\n', 1)[0].lstrip('# ').strip()
                        file_titles[filename] = title
                    except IndexError:
                        file_titles[filename] = "Title not found"

                    # Rough token estimate before sending
                    estimated_tokens = len(file_content.split()) * 1.3
                    if estimated_tokens > 100000:
                        print(f"  -> WARNING: {filename} may exceed context limit ({estimated_tokens:.0f} estimated tokens)")
                        # Truncate to safe limit
                        file_content = " ".join(file_content.split()[:75000])
                    
                    try:
                        # 1. Router Agent extracts and reduces the context
                        print("  -> Extracting exact verbatim quotes (Router Agent)...")
                        router_prompt = f"""Extract all clinically relevant paragraphs about {disease_name} from the following PMC article.
                        Focus on paragraphs covering: definition, causes, mechanism, symptoms, classification, risk factors, diagnosis, and treatment.
                        PMC ARTICLE: {file_content}"""
                        extracted_context = agent.router_agent(router_prompt)
                        
                        # Split the output into paragraphs to prepare for deduplication
                        paragraphs = [p.strip() for p in extracted_context.split('\n') if len(p.strip()) > 20]
                        
                        for i, p in enumerate(paragraphs):
                            all_extracted_paragraphs.append({
                                "text": p,
                                "filename": filename,
                                "index": i
                            })
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
    
            if all_extracted_paragraphs:
                # 2. RAG Approach using ChromaDB
                print(f"\nTotal extracted paragraphs across all files: {len(all_extracted_paragraphs)}")
                print("-> Setting up ChromaDB for Retrieval-Augmented Generation (RAG)...")
                
                # Initialize in-memory ChromaDB client
                chroma_client = chromadb.Client()
                # Use a unique collection name per disease
                collection_name = f"rag_collection_{disease_name.replace(' ', '_').replace('-', '_')}"
                
                try:
                    chroma_client.delete_collection(name=collection_name)
                except Exception:
                    pass
                    
                collection = chroma_client.create_collection(name=collection_name)
                
                documents = []
                metadatas = []
                ids = []
                for global_idx, item in enumerate(all_extracted_paragraphs):
                    documents.append(item["text"])
                    metadatas.append({"filename": item["filename"], "index": item["index"]})
                    ids.append(f"doc_{global_idx}")
                    
                # Add exact extracted paragraphs to Vector DB
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
                
                # The 8 specific topics we want the Doctor Agent to focus on
                topics = [
                    "Clinical Definition / Overview",
                    "Etiology / Cause",
                    "Transmission / Pathogenesis / Mechanism",
                    "Signs & Symptoms",
                    "Types / Classification",
                    "Risk Factors / Complications",
                    "Diagnosis",
                    "Treatment / Management"
                ]
                
                retrieved_context = []
                for topic in topics:
                    # Retrieve the top 5 most relevant EXACT paragraphs for each specific medical topic
                    results = collection.query(
                        query_texts=[f"{disease_name} {topic}"],
                        n_results=min(5, len(documents))
                    )
                    
                    # Combine the retrieved docs with their metadata so we can sort them back into their original chronological order
                    retrieved_chunks = []
                    for i in range(len(results['documents'][0])):
                        retrieved_chunks.append({
                            "text": results['documents'][0][i],
                            "filename": results['metadatas'][0][i]["filename"],
                            "index": results['metadatas'][0][i]["index"]
                        })
                        
                    # Sort by filename and then by their original index in the text to restore the original reading flow!
                    retrieved_chunks.sort(key=lambda x: (x["filename"], x["index"]))
                    
                    retrieved_context.append(f"### {topic} ###")
                    for chunk in retrieved_chunks:
                        retrieved_context.append(chunk["text"])
                    retrieved_context.append("\n")
                    
                final_master_context = "\n".join(retrieved_context)
                print(f"RAG Retrieval complete. Combined context size: {len(final_master_context)} characters.")
    
                # 3. Doctor Agent generates ONE definitive source text for the disease
                print("\n-> Generating definitive annotations (Doctor Agent) from retrieved RAG context...")
                prompt = f"For {disease_name}, we have retrieved the exact, most relevant quotes from a vector database using RAG. Can you create ONE definitive finetuning record about {disease_name}?\n\n### RETRIEVED ENGLISH MEDICAL CONTEXT ###\n{final_master_context}"
    
                # Round-robin agent selection
                agent_choice = doctor_model_idx % 3
                if agent_choice == 0:
                    print("  -> Using OpenAI Agent")
                    agent_name, model_name = "openai", "gpt-5-mini"
                    result = agent.openai_agent_doctor1(prompt)
                elif agent_choice == 1:
                    print("  -> Using Anthropic Agent")
                    agent_name, model_name = "anthropic", "claude-haiku-4-5-20251001"
                    result = agent.anthropic_agent_doctor2(prompt)
                else:
                    print("  -> Using Gemini Agent")
                    agent_name, model_name = "gemini", "gemini-3.1-flash-lite"
                    result = agent.gemini_agent_doctor3(prompt)
                
                doctor_model_idx += 1

                issues = validate_output(result, disease_name)
                if issues:
                    print(f"  -> WARNING: Validation failed for {disease_name}: {issues}")
                    # Save to a rejection log
                    rejected_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rejected_records.log") if "pipeline" in __file__ else "rejected_records.log"
                    with open(rejected_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{disease_name}: {issues}\n")
                else:
                    # Parse the raw LLM output into the highly structured JSON format
                    parsed_data = parse_llm_output(result, disease_name, agent_name, model_name, file_titles)
                    
                    # Save the unified result
                    output_filepath = os.path.join(output_dir, f"{disease_name.replace(' ', '_')}_master.jsonl")
                    with open(output_filepath, "w", encoding="utf-8") as f:
                        f.write(json.dumps(parsed_data, ensure_ascii=False) + "\n")
                    print(f"-> Saved parsed JSONL annotation to {output_filepath}")
                    
                    # Save a human-readable Markdown file for easy visual inspection
                    readable_filepath = os.path.join(output_dir, f"{disease_name.replace(' ', '_')}_master.md")
                    with open(readable_filepath, "w", encoding="utf-8") as f:
                        f.write(result)
                    print(f"-> Saved human-readable version to {readable_filepath}")

                # Clean up ChromaDB collection to free memory
                try:
                    chroma_client.delete_collection(name=collection_name)
                    print(f"-> Cleaned up ChromaDB collection: {collection_name}")
                except Exception as e:
                    print(f"  -> Failed to clean up ChromaDB collection: {e}")

                print("=" * 80)