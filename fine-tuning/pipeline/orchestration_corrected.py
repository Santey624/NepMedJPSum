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
14. Do not add personal medical advice such as "you should take this medicine" unless the source explicitly states it.
15. Do not mention that you are an AI model.
16. Output only the final annotation template. Do not explain your reasoning.
17. The Japanese translation must be based only on the organized English source text.
18. The Nepali annotation must be based only on the Japanese translation for the same source section.
19. CRITICAL ALPHABET RESTRICTION: The Nepali section must be written 100% in pure Devanagari script. You are strictly forbidden from leaving Japanese Kanji (e.g., 脱水, 疑い例, 確定例, 流行) or Kana anywhere in the Nepali text. If you encounter a Japanese medical term, you must completely express its meaning using valid Nepali medical/patient phrasing (e.g., translate 脱水 to 'जलवियोजन' or 'शरीरमा पानीको कमी').

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
14. Do not add personal medical advice such as "you should take this medicine" unless the source explicitly states it.
15. Do not mention that you are an AI model.
16. Output only the final annotation template. Do not explain your reasoning.
17. The Japanese translation must be based only on the organized English source text.
18. The Nepali annotation must be based only on the Japanese translation for the same source section.
19. CRITICAL ALPHABET RESTRICTION: The Nepali section must be written 100% in pure Devanagari script. You are strictly forbidden from leaving Japanese Kanji (e.g., 脱水, 疑い例, 確定例, 流行) or Kana anywhere in the Nepali text. If you encounter a Japanese medical term, you must completely express its meaning using valid Nepali medical/patient phrasing (e.g., translate 脱水 to 'जलवियोजन' or 'शरीरमा पानीको कमी').

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
14. Do not add personal medical advice such as "you should take this medicine" unless the source explicitly states it.
15. Do not mention that you are an AI model.
16. Output only the final annotation template. Do not explain your reasoning.
17. The Japanese translation must be based only on the organized English source text.
18. The Nepali annotation must be based only on the Japanese translation for the same source section.
19. CRITICAL ALPHABET RESTRICTION: The Nepali section must be written 100% in pure Devanagari script. You are strictly forbidden from leaving Japanese Kanji (e.g., 脱水, 疑い例, 確定例, 流行) or Kana anywhere in the Nepali text. If you encounter a Japanese medical term, you must completely express its meaning using valid Nepali medical/patient phrasing (e.g., translate 脱水 to 'जलवियोजन' or 'शरीरमा पानीको कमी').

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

# ──────────────────────────────────────────────────────────────────────────────
# Final-Checking Agent system prompt
# Evaluates the full LLM-generated annotation for:
#   1. Japanese grammar / naturalness
#   2. Nepali grammar / naturalness (Devanagari)
#   3. Translation fidelity between each EN→JA and JA→NE pair
# ──────────────────────────────────────────────────────────────────────────────
system_prompt_for_finalchecking = r"""
You are a strict quality-control and correction agent for the NepMedJP medical dataset project.

Your task is to evaluate a fully-generated annotation record that contains:
- English source text (sections src_01 to src_08)
- Japanese translation of each English section
- Nepali (Devanagari) summary of each Japanese section

2. NEPALI GRAMMAR & NATURALNESS (Devanagari)
   - Identify sections where the Nepali is grammatically incorrect or unnatural.
   - Flag wrong verb forms, missing postpositions, or awkward phrasing.
   - Note if the text sounds like a word-for-word translation rather than a natural doctor-style explanation.
   - STRICTLY CONFIRM: The Nepali text must contain absolutely ZERO Japanese Kanji, Hiragana, or Katakana characters (e.g., terms like 脱水, 確定例, 疑い例, 米 must be completely translated into their proper Nepali equivalents like जलवियोजन, प्रमाणित केस, शंकास्पद केस, चामल). If any Japanese characters are present, mark the section as false and provide the fully cleaned text in 'corrected_nepali'.

EVALUATION & CORRECTION SCOPE — check ONLY the following:

1. JAPANESE GRAMMAR & NATURALNESS
   - Identify sections where the Japanese is grammatically broken, unnatural, or reads like a word-for-word mechanical translation.
   - Flag missing particles (は、が、を、に etc.), wrong verb conjugations, or unnatural sentence endings.
   - Note if honorific / medical register is inconsistent.

2. NEPALI GRAMMAR & NATURALNESS (Devanagari)
   - Identify sections where the Nepali is grammatically incorrect or unnatural.
   - Flag wrong verb forms, missing postpositions, or awkward phrasing.
   - Note if the text sounds like a word-for-word translation rather than a natural doctor-style explanation.
   - Confirm all Nepali text is in Devanagari script.

3. TRANSLATION FIDELITY
   - Check whether the Japanese faithfully conveys the meaning of the English source.
   - Check whether the Nepali faithfully summarises the Japanese.
   - Flag any section where a key medical concept was dropped, mistranslated, or significantly distorted.

4. REQUIRED CORRECTIONS
   - If ANY of the above checks fail for a section, you MUST provide the fully corrected text for that specific language in the output JSON.
   - The corrected text must resolve the flagged issues while maintaining strict medical accuracy.

OUTPUT FORMAT — respond ONLY with a structured JSON object. Do not include any text outside the JSON.

{
  "overall_pass": true/false,
  "sections": {
    "src_01": {
      "japanese_grammar_ok": true/false,
      "japanese_issues": "<describe issue or 'None'>",
      "corrected_japanese": "<provide corrected Japanese text here if issues found, otherwise null>",
      "nepali_grammar_ok": true/false,
      "nepali_issues": "<describe issue or 'None'>",
      "corrected_nepali": "<provide corrected Nepali Devanagari text here if issues found, otherwise null>",
      "translation_fidelity_ok": true/false,
      "fidelity_issues": "<describe issue or 'None'>"
    },
    "src_02": { ... },
    "src_03": { ... },
    "src_04": { ... },
    "src_05": { ... },
    "src_06": { ... },
    "src_07": { ... },
    "src_08": { ... }
  },
  "combined_japanese_ok": true/false,
  "combined_nepali_ok": true/false,
  "combined_issues": "<describe any combined-section issues or 'None'>",
  "summary": "<2-3 sentence plain English summary of the main quality issues found, or 'All checks passed.'>"
}

RULES:
- overall_pass is true only if ALL per-section booleans are true AND combined sections are ok.
- Be strict: a single broken sentence in a section is enough to set that section's boolean to false.
- Never output anything outside the JSON object.
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
    devanagari_range = range(0x0900, 0x0980)  # U+0900–U+097F inclusive
    has_devanagari = any(ord(c) in devanagari_range for c in result)
    if not has_devanagari:
        issues.append("No Devanagari script detected")
    
    # Check Japanese present
    japanese_ranges = [
        range(0x3040, 0x30A0),  # Hiragana U+3040–U+309F inclusive
        range(0x30A0, 0x3100),  # Katakana U+30A0–U+30FF inclusive
        range(0x4E00, 0xA000),  # CJK Unified Ideographs U+4E00–U+9FFF inclusive
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
        if en_start == -1:
            en_block = ""
        elif en_end == -1:
            en_block = raw_text[en_start:]
        else:
            en_block = raw_text[en_start:en_end]
        en_text_match = re.search(r"English text:\s*(.*)", en_block, re.DOTALL | re.IGNORECASE)
        en_text = ""
        if en_text_match:
            en_text = en_text_match.group(1).strip()
        all_en_texts.append(en_text)
        
        # Extract Japanese Text Block
        ja_start = raw_text.find(f"Japanese Translation for {sec_id}")
        ja_end = raw_text.find(f"Japanese Translation for src_0{i+1}") if i < 8 else raw_text.find("D. Nepali Annotations")
        if ja_start == -1:
            ja_block = ""
        elif ja_end == -1:
            ja_block = raw_text[ja_start:]
        else:
            ja_block = raw_text[ja_start:ja_end]
        ja_text_match = re.search(r"Japanese text[^:]*:\s*(.*)", ja_block, re.DOTALL | re.IGNORECASE)
        ja_text = ja_text_match.group(1).strip() if ja_text_match else ""
        
        # Extract Nepali Text Block
        ne_start = raw_text.find(f"Nepali Annotation for {sec_id}")
        ne_end = raw_text.find(f"Nepali Annotation for src_0{i+1}") if i < 8 else raw_text.find("E. Combined Japanese")
        if ne_start == -1:
            ne_block = ""
        elif ne_end == -1:
            ne_block = raw_text[ne_start:]
        else:
            ne_block = raw_text[ne_start:ne_end]
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
    if combo_ja_start == -1:
        combo_ja_block = ""
    elif combo_ja_end == -1:
        combo_ja_block = raw_text[combo_ja_start:]
    else:
        combo_ja_block = raw_text[combo_ja_start:combo_ja_end]
    combo_ja_text = re.sub(r"E\. Combined Japanese Medical Source\s*", "", combo_ja_block, flags=re.IGNORECASE).strip()

    combo_ne_start = raw_text.find("F. Combined Nepali")
    combo_ne_end = raw_text.find("G. Quality Checklist")
    if combo_ne_start == -1:
        combo_ne_block = ""
    elif combo_ne_end == -1:
        combo_ne_block = raw_text[combo_ne_start:]
    else:
        combo_ne_block = raw_text[combo_ne_start:combo_ne_end]
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
                    {"role": "system", "content": system_prompt_for_router},
                    {"role": "user", "content": message}
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
                model="gpt-5-mini",       # update to valid model string as needed
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
                model="gemini-3.1-flash-lite",    # update to valid model string as needed
                messages=[
                    {"role": "system", "content": system_prompt_for_gemini},
                    {"role": "user", "content": message}
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

    # ──────────────────────────────────────────────────────────────────────────
    # Final-checking agent
    # ──────────────────────────────────────────────────────────────────────────
    def finalchecking(self, generated_text: str) -> dict | None:
        """
        Sends the full LLM-generated annotation to for quality review.

        Checks:
          - Japanese grammar and naturalness per section
          - Nepali grammar and naturalness per section
          - Translation fidelity (EN→JA and JA→NE) per section
          - Combined-section integrity

        Returns:
          A parsed dict matching the JSON schema in system_prompt_for_finalchecking,
          or None if the API call fails or the response cannot be parsed.
        """
        openai_api_key, _, _ = load_api_keys()
        client = OpenAI(api_key=openai_api_key)

        prompt = (
            "Please review the following NepMedJP annotation record "
            "and return your evaluation as a JSON object.\n\n"
            f"{generated_text}"
        )

        raw_json = ""
        try:
            response = client.chat.completions.create(
                model="gpt-5.4-mini",
                temperature=0,
                max_completion_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt_for_finalchecking},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_json = response.choices[0].message.content

            # Strip accidental markdown fences (```json ... ```) if present
            raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json.strip(), flags=re.IGNORECASE)
            raw_json = re.sub(r"\s*```$", "", raw_json.strip())

            return json.loads(raw_json)

        except json.JSONDecodeError as e:
            print(f"  -> 🚨 Final-Checking Agent: JSON parse error: {e}")
            print(f"     Raw response was: {raw_json[:500]}")
            return None
        except Exception as e:
            print(f"  -> 🚨 Final-Checking Agent Error: {e}")
            return None


if __name__ == "__main__":
    agent = Agents()

    # Assuming the script is run from the fine-tuning directory
    base_data_dir = (
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "pmc_data")
        if "pipeline" in __file__
        else "pmc_data"
    )

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
        output_dir = (
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "fine-tuning-data", disease_name)
            if "pipeline" in __file__
            else f"fine-tuning-data/{disease_name}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Store all extracted verbatim paragraphs across all files
        all_extracted_paragraphs = []
        file_titles = {}

        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                filepath = os.path.join(data_dir, filename)
                if not (os.path.isfile(filepath) and filename.endswith('.md')):
                    continue

                with open(filepath, "r", encoding="utf-8") as f:
                    file_content = f.read()

                print(f"Processing {filename}...")

                try:
                    file_titles[filename] = file_content.split('\n', 1)[0].lstrip('# ').strip()
                except IndexError:
                    file_titles[filename] = "Title not found"

                # Rough token estimate — truncate if too large
                estimated_tokens = len(file_content.split()) * 1.3
                if estimated_tokens > 100000:
                    print(f"  -> WARNING: {filename} may exceed context limit ({estimated_tokens:.0f} estimated tokens)")
                    file_content = " ".join(file_content.split()[:75000])

                try:
                    print("  -> Extracting exact verbatim quotes (Router Agent)...")
                    router_prompt = (
                        f"Extract all clinically relevant paragraphs about {disease_name} "
                        f"from the following PMC article.\n"
                        f"Focus on paragraphs covering: definition, causes, mechanism, symptoms, "
                        f"classification, risk factors, diagnosis, and treatment.\n"
                        f"PMC ARTICLE: {file_content}"
                    )
                    extracted_context = agent.router_agent(router_prompt)

                    if not extracted_context:
                        print(f"  -> WARNING: Router agent returned nothing for {filename}, skipping.")
                        continue

                    paragraphs = [p.strip() for p in extracted_context.split('\n') if len(p.strip()) > 20]
                    for i, p in enumerate(paragraphs):
                        all_extracted_paragraphs.append({
                            "text": p,
                            "filename": filename,
                            "index": i
                        })
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

            if not all_extracted_paragraphs:
                continue

            # ── RAG retrieval via ChromaDB ────────────────────────────────────
            print(f"\nTotal extracted paragraphs across all files: {len(all_extracted_paragraphs)}")
            print("-> Setting up ChromaDB for RAG...")

            chroma_client = chromadb.Client()
            collection_name = f"rag_collection_{disease_name.replace(' ', '_').replace('-', '_')}"

            try:
                chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass

            collection = chroma_client.create_collection(name=collection_name)

            documents, metadatas, ids = [], [], []
            for global_idx, item in enumerate(all_extracted_paragraphs):
                documents.append(item["text"])
                metadatas.append({"filename": item["filename"], "index": item["index"]})
                ids.append(f"doc_{global_idx}")

            collection.add(documents=documents, metadatas=metadatas, ids=ids)

            topics = [
                "Clinical Definition / Overview",
                "Etiology / Cause",
                "Transmission / Pathogenesis / Mechanism",
                "Signs & Symptoms",
                "Types / Classification",
                "Risk Factors / Complications",
                "Diagnosis",
                "Treatment / Management",
            ]

            retrieved_context = []
            for topic in topics:
                results = collection.query(
                    query_texts=[f"{disease_name} {topic}"],
                    n_results=min(5, len(documents))
                )
                retrieved_chunks = []
                for i in range(len(results['documents'][0])):
                    retrieved_chunks.append({
                        "text": results['documents'][0][i],
                        "filename": results['metadatas'][0][i]["filename"],
                        "index": results['metadatas'][0][i]["index"],
                    })
                retrieved_chunks.sort(key=lambda x: (x["filename"], x["index"]))
                retrieved_context.append(f"### {topic} ###")
                for chunk in retrieved_chunks:
                    retrieved_context.append(chunk["text"])
                retrieved_context.append("\n")

            final_master_context = "\n".join(retrieved_context)
            print(f"RAG Retrieval complete. Combined context size: {len(final_master_context)} characters.")

            # ── Doctor Agent (round-robin) ────────────────────────────────────
            print("\n-> Generating definitive annotations (Doctor Agent)...")
            prompt = (
                f"For {disease_name}, we have retrieved the exact, most relevant quotes from a "
                f"vector database using RAG. Can you create ONE definitive finetuning record "
                f"about {disease_name}?\n\n"
                f"### RETRIEVED ENGLISH MEDICAL CONTEXT ###\n{final_master_context}"
            )

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

            # Only advance the counter when the agent actually returned something,
            # so a failed call does not silently skew the round-robin sequence.
            if result is not None:
                doctor_model_idx += 1

            # ── Structural validation ─────────────────────────────────────────
            issues = validate_output(result, disease_name)
            if issues:
                print(f"  -> WARNING: Structural validation failed for {disease_name}: {issues}")
                rejected_log_path = (
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "rejected_records.log")
                    if "pipeline" in __file__
                    else "rejected_records.log"
                )
                with open(rejected_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{disease_name} [structural]: {issues}\n")
                # Clean up and skip to the next disease
                try:
                    chroma_client.delete_collection(name=collection_name)
                except Exception:
                    pass
                continue

            parsed_data = parse_llm_output(result, disease_name, agent_name, model_name, file_titles)

            # ── Final-checking & Correction agent ─────────────────────────────
            print("  -> Running Final-Checking & Correction Agent...")
            qc_report = agent.finalchecking(result)

            if qc_report is None:
                print("  -> WARNING: Final-Checking Agent returned no report; skipping QC gate.")
                qc_passed = True   # degrade gracefully — don't block on agent failure
            else:
                qc_passed = qc_report.get("overall_pass", False)
                summary = qc_report.get("summary", "No summary.")
                print(f"  -> QC overall_pass={qc_passed}  |  {summary}")

                if not qc_passed:
                    print(f"  -> 🛠️ QC failed. Applying auto-corrections from the agent...")
                    for i in range(1, 9):
                        sec_id = f"src_0{i}"
                        sec_info = qc_report.get("sections", {}).get(sec_id, {})
                        
                        # Patch Japanese corrections
                        corrected_ja = sec_info.get("corrected_japanese")
                        if corrected_ja and corrected_ja.lower() != "null":
                            old_ja = parsed_data["sections"][i-1]["japanese_text"]
                            parsed_data["sections"][i-1]["japanese_text"] = corrected_ja
                            # Safeguard against empty string corruption
                            if old_ja.strip():
                                result = result.replace(old_ja, corrected_ja)
                            
                        # Patch Nepali corrections
                        corrected_ne = sec_info.get("corrected_nepali")
                        if corrected_ne and corrected_ne.lower() != "null":
                            old_ne = parsed_data["sections"][i-1]["nepali_summary"]
                            parsed_data["sections"][i-1]["nepali_summary"] = corrected_ne
                            # Safeguard against empty string corruption
                            if old_ne.strip():
                                result = result.replace(old_ne, corrected_ne)

                    # Rebuild combined text blocks in the parsed data
                    parsed_data["combined"]["japanese_text"] = " ".join([s["japanese_text"] for s in parsed_data["sections"]])
                    parsed_data["combined"]["nepali_summary"] = " ".join([s["nepali_summary"] for s in parsed_data["sections"]])
                    
                    # Update the combined sections using lambda functions to prevent regex escape errors
                    result = re.sub(
                        r"(E\. Combined Japanese Medical Source\s*)(.*?)(?=F\. Combined Nepali Medical Summary)",
                        lambda m: f"{m.group(1)}\n\n{parsed_data['combined']['japanese_text']}\n\n",
                        result,
                        flags=re.DOTALL
                    )
                    result = re.sub(
                        r"(F\. Combined Nepali Medical Summary\s*)(.*?)(?=G\. Quality Checklist|$)",
                        lambda m: f"{m.group(1)}\n\n{parsed_data['combined']['nepali_summary']}\n\n",
                        result,
                        flags=re.DOTALL
                    )

            output_filepath = os.path.join(output_dir, f"{disease_name.replace(' ', '_')}_master.jsonl")
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps(parsed_data, ensure_ascii=False) + "\n")
            print(f"-> Saved (and corrected) parsed JSONL annotation to {output_filepath}")

            readable_filepath = os.path.join(output_dir, f"{disease_name.replace(' ', '_')}_master.md")
            with open(readable_filepath, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"-> Saved (and corrected) human-readable version to {readable_filepath}")

            # ── Clean up ChromaDB ─────────────────────────────────────────────
            try:
                chroma_client.delete_collection(name=collection_name)
                print(f"-> Cleaned up ChromaDB collection: {collection_name}")
            except Exception as e:
                print(f"  -> Failed to clean up ChromaDB collection: {e}")

        print("=" * 80)