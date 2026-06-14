"""
Nepali Biomedical Dataset Sanitization Pipeline (vLLM edition)
Output: {"text": "..."} — only records with score >= 6 after cleaning

Expected JSONL input format (one JSON object per line):
  {"text": "..."}
"""

import gc
import json
import re
import argparse
import logging
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-14B-AWQ"          # ← AWQ variant
INPUT_FILE  = "/workspace/Nepmedjp/part0nepalitranslated.jsonl"
OUTPUT_FILE = "/workspace/Nepmedjp/part0nepalitranslated_cleaned.jsonl"
TEXT_FIELD  = "text"

MAX_NEW_TOKENS = 2048
TEMPERATURE    = 0.2
TOP_P          = 0.9
BATCH_SIZE     = 24   # vLLM handles memory efficiently so we can go much larger

# Only keep records where the model assigns a score >= this after cleaning
QUALITY_THRESHOLD = 7

# Log every N batches
GPU_LOG_INTERVAL = 10

# ─── CHECKPOINT ──────────────────────────────────────────────────────────────
CHECKPOINT_FILE = "processed_ids.txt"

# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """

Nepali Biomedical Abstract Cleaner + Pretraining Suitability Judge (JSON-only, vLLM)

You will receive ONE Nepali biomedical text snippet. It is a machine translation from English PubMed abstracts (NLLB) and may contain MT artifacts, encoding noise, duplicated fragments, word/character repetitions, partial Hindi contamination, stray symbols, and formatting issues.

Your task has exactly TWO stages:
1) CLEAN the text with minimal, safe edits (no meaning change).
2) SCORE the cleaned text (1–10) for pretraining suitability.

IMPORTANT DOWNSTREAM RULE (for your awareness):
- Records are kept ONLY if score >= 7 which can be used for Nepali medical pretraining.
- If score < 7, the record will be deleted.
- STRICTLY only output the data you think it can be used for nepali medical pretraining.
- STRICTLY solve these issues if there are like (एमआरटीबीम वाइट्सिसिसिसिया) word repititon issue or (रगत रगत परीक्षण) word repititon issue or (गरिएको गरिएको छ) word repititon issue or (गर्छछछ) character repititon issue or (।।।।) character repititon issue or (,,, ) character repititon issue.


You MUST follow every constraint below.

════════════════════════════════════════════
STAGE 1 — CLEAN (minimal + meaning-preserving)
════════════════════════════════════════════

Goal: produce the cleanest usable Nepali biomedical abstract WITHOUT changing the scientific meaning.

ALLOWED edits (ONLY when unambiguous):

A) Normalize Unicode + whitespace + punctuation
- Remove control characters, BOM, zero-width junk, and obvious encoding noise.
- Fix spacing around punctuation (no extra spaces before commas/periods/danda).
- Use Nepali sentence punctuation "।" for sentence endings where appropriate.
- Keep "." when it is part of:
  - decimals (e.g., ३.१४),
  - abbreviations (e.g., E. coli),
  - references like "Fig. 1" if present.
- Normalize quotes and dashes to consistent plain forms.
- Normalize common scientific formatting WITHOUT changing values:
  - "p = 0.05" → "p=0.05"
  - "p < 0.05" → "p<0.05"
  - "95 %" → "95%"
  - "37 ° C" → "37°C"
  - "10 m g" → "10 mg" (only when it is clearly a split unit token)
- Normalize CD markers only in the pattern CD + digits:
  - "CD 4", "CD-4", "CD 4+" → "CD4", "CD4+" (do NOT touch other hyphenated terms)
- Fix MT-introduced intra-word spaces (e.g., "ग र्छ" → "गर्छ", "भ यो" → "भयो") ONLY when the correct Devanagari word is unambiguous.

B) Remove junk / boilerplate / artifacts
- Remove obvious non-abstract junk: "अनुवाद:", "Translated by", UI/website fragments, navigation text, repeated headers/footers, stray HTML tags/entities, broken bullets, random section separators.
- Remove duplicate sentences/lines caused by MT repetition; keep ONE best copy.
- Remove isolated corrupted fragments (random symbol runs, mojibake chunks) if they are clearly separable.

C) Fix word-level and character-level repetitions (IMPORTANT)
- Remove consecutive word repetitions caused by MT artifacts:
  - e.g., "रगत रगत परीक्षण" → "रगत परीक्षण"
  - e.g., "गरिएको गरिएको छ" → "गरिएको छ"
  - Remove any word that is immediately repeated 2 or more times consecutively.
- Remove consecutive character/token repetitions:
  - e.g., "गर्छछछ" → "गर्छ"
  - e.g., "।।।।" → "।"
  - e.g., ",,," → ","
- Remove phrase-level repetitions where the same clause appears 2+ times in immediate succession due to MT beam artifact.
- Do NOT collapse intentional Nepali reduplication (e.g., "बिस्तारै बिस्तारै" meaning "very slowly") — only remove repetitions that are clearly MT artifacts based on context.

D) Terminology rules (keep biomedical signal)
- Preserve all biomedical entities exactly: diseases, symptoms, procedures, genes/proteins, organisms, drug names, trial acronyms, and proper nouns.
- Keep standard English/Latin scientific tokens (e.g., COVID-19, SARS-CoV-2, DNA, RNA, PCR, WHO, RCT, BRCA1, E. coli, metformin). DO NOT translate them.
- You MAY replace a small amount of Hindi with obvious Nepali equivalents ONLY if the meaning is crystal clear and the replacement is standard (no guesswork). If Hindi dominates or replacements would require inference, output empty text.

E) Digit normalization (IMPORTANT)
- Convert all Western/ASCII digits (0–9) to Nepali Devanagari digits (०–९) throughout the text.
  - e.g., "95%" → "९५%", "p=0.05" → "p=०.०५", "37°C" → "३७°C", "2024" → "२०२४"
- This applies to ALL numbers: statistics, dosages, sample sizes, dates, percentages, ranges, p-values, CIs, etc.
- Exception: Do NOT convert digits that are embedded inside pure ASCII/Latin scientific tokens where converting would break the token (e.g., "COVID-19", "SARS-CoV-2", "BRCA1", "CD4", "IL-6"). Keep those exactly as-is.
- After conversion, ensure all digits in the text are consistently Devanagari.

F) Numbers, units, statistics (must be exact)
- Do NOT change any numeric values, decimals, negatives, ranges, inequalities (>, <, ≥, ≤), denominators, sample sizes, dates, dosages, or units — only change the script of the digits per rule E above.
- Preserve statistical expressions: OR/RR/HR, ९५% CI, SD/SE, p-values.
- Do NOT convert units or rewrite to different units.

G) Very light grammar fixes ONLY
- Fix only obvious MT grammar/spacing issues that do NOT alter meaning (postpositions, agreement, token splits).
- Do NOT paraphrase for style. Do NOT rewrite sentences beyond necessary repairs.

H) PII handling (strict + minimal)
- Remove ONLY explicit identifiers:
  - phone numbers, emails, full street addresses,
  - patient IDs/MRN,
  - a patient-tied full person name (when clearly an identifier).
- Do NOT remove author names, institutions, or place names if they are part of normal abstract metadata unless clearly patient-identifying.
- If removing PII makes the remaining text meaningless or mostly PII, output empty text.

HARD PROHIBITIONS (NEVER):
- Do NOT add new facts, explanations, or context.
- Do NOT change study direction, effect direction, comparisons, negations, or significance.
- Do NOT expand abbreviations unless the expansion already exists in the snippet.
- Do NOT invent missing words/drugs/diagnoses/conclusions.
- Do NOT restructure into a "better abstract" or add headings not already present.

════════════════════════════════════════════
STAGE 1.5 — HARD REJECTION GATES (output empty)
════════════════════════════════════════════

After cleaning, you MUST output EMPTY text ("text": "") if ANY condition below is true:

1) Contains Unicode replacement char: "?"
2) Dominated by unreadable garbage / mojibake / symbol runs that prevent understanding.
3) Contains substantial non-Devanagari scripts (CJK/Arabic/etc.) that are not standard scientific tokens.
4) Cleaned text is too short to be meaningful as an abstract fragment:
   - fewer than 25 Nepali characters total (excluding spaces/punctuation), OR
   - fewer than 2 complete sentences.
5) The cleaned text does NOT end with a complete sentence:
   - After trimming whitespace, final character must be one of: "।" "?" "!"
   - Optionally allow a closing quote immediately after punctuation.

If you return empty text, you MUST return score=1.

════════════════════════════════════════════
STAGE 2 — SCORE (1–10) on CLEANED TEXT ONLY
════════════════════════════════════════════

Score rubric:
- 9–10: Clean, coherent Nepali biomedical text; meaning fully intact; Devanagari digits used consistently; essentially no artifacts.
- 8: Very good; tiny issues but fully usable.
- 6–7: Minor issues remain (slight awkwardness or small leftover artifacts) but meaning is clear and usable.
- 4–5: Significant problems; meaning partially recoverable; risky for pretraining.
- 1–3: Heavily corrupted, Hindi-dominated, incoherent, or empty.

Reminder:
- Records are kept ONLY if score >= 7. If score < 7, they will be deleted.

════════════════════════════════════════════
OUTPUT FORMAT (ABSOLUTELY STRICT)
════════════════════════════════════════════

Return ONLY valid JSON, exactly two keys, no extra text, no markdown:
{"text":"...","score":<integer 1-10>}

If gated empty:
{"text":"","score":1}
"""

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─── GPU UTILS ───────────────────────────────────────────────────────────────
def log_gpu_memory():
    """Log current VRAM usage across all visible GPUs."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved  = torch.cuda.memory_reserved(i)  / 1024 ** 3
        log.info(f"  GPU {i} — allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def extract_json(raw: str) -> dict | None:
    """Robustly extract the first valid JSON object from model output."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def build_prompts(tokenizer, texts: list[str]) -> list[str]:
    """Build chat-formatted prompts for a list of texts."""
    prompts = []
    for text in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # Disable Qwen3 thinking mode at template level
            )
        )
    return prompts


def run_model_batch(tokenizer, model, sampling_params, texts: list[str]) -> list[str]:
    """
    Run inference on a batch of texts using vLLM.
    vLLM handles batching, padding, and memory internally — no OOM retry needed.
    Returns one raw output string per input.
    """
    prompts = build_prompts(tokenizer, texts)
    outputs = model.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


# ─── PIPELINE ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=INPUT_FILE)
    parser.add_argument("--output",     default=OUTPUT_FILE)
    parser.add_argument("--model",      default=MODEL_ID)
    parser.add_argument("--threshold",  default=QUALITY_THRESHOLD, type=int,
                        help="Min score (1-10) to keep a record after cleaning (default: 6)")
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int,
                        help="Inference batch size (default: 24)")
    parser.add_argument("--reset",      action="store_true",
                        help="Ignore existing checkpoint and start from scratch")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint ───────────────────────────────────────────────────────────
    id_log = Path(CHECKPOINT_FILE)
    processed_ids: set[str] = set()
    if not args.reset and id_log.exists():
        processed_ids = set(id_log.read_text(encoding="utf-8").splitlines())
        log.info(f"Checkpoint found — resuming, skipping {len(processed_ids)} already-processed lines")
    elif args.reset:
        log.info("--reset passed — starting from scratch, checkpoint ignored")
    else:
        log.info("No checkpoint found — starting fresh")

    id_log_fh = open(id_log, "a", encoding="utf-8")
    output_fh = open(args.output, "a", encoding="utf-8")

    stats = {"kept": 0, "rejected": 0, "parse_failed": 0, "skipped": 0}

    # ── Load tokenizer ────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── Load model with vLLM (AWQ) ────────────────────────────────────────────
    log.info(f"Loading model with vLLM: {args.model}")
    model_obj = LLM(
        model=args.model,
        quantization="awq_marlin",               # ← AWQ: static kernels, CUDA graphs work
        dtype="auto",                            # Auto-selects FP16/FP8 based on GPU
        max_model_len=8192,
        gpu_memory_utilization=0.88,
        max_num_seqs=128,
        trust_remote_code=True,
        enable_chunked_prefill=True,             # Reduces latency for long prompts
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
    )
    log.info("Model loaded. Initial GPU state:")
    log_gpu_memory()

    # ── Read and parse input ──────────────────────────────────────────────────
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    log.info(f"Total lines: {len(lines)}")

    # ── Collect pending (index, rec_id, text) tuples ─────────────────────────
    pending = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            log.warning(f"Line {i}: invalid JSON, skipping.")
            stats["skipped"] += 1
            continue
        rec_id = str(record.get("id", i))
        if rec_id in processed_ids:
            stats["skipped"] += 1
            continue
        text = record.get(TEXT_FIELD, "").strip()
        if not text:
            log.warning(f"Line {i}: empty text, skipping.")
            stats["skipped"] += 1
            continue
        pending.append((i, rec_id, text))

    log.info(f"Pending lines to process: {len(pending)}")

    # Sort by text length to minimize padding within batches
    pending.sort(key=lambda x: len(x[2]))
    log.info("Pending lines sorted by length for efficient batching.")

    # ── Process in batches ───────────────────────────────────────────────────
    for batch_idx, batch_start in enumerate(
        tqdm(range(0, len(pending), args.batch_size), desc="Cleaning")
    ):
        batch = pending[batch_start : batch_start + args.batch_size]
        indices, rec_ids, texts = zip(*batch)

        try:
            raw_outputs = run_model_batch(
                tokenizer, model_obj, sampling_params, list(texts)
            )
        except Exception as e:
            log.error(f"Batch {batch_idx} failed with error: {e} — skipping batch.")
            for rec_id in rec_ids:
                stats["skipped"] += 1
                id_log_fh.write(rec_id + "\n")
            id_log_fh.flush()
            continue

        for i, rec_id, raw_output in zip(indices, rec_ids, raw_outputs):
            parsed = extract_json(raw_output)

            if parsed is None:
                log.warning(f"Line {i}: could not parse model JSON output.")
                stats["parse_failed"] += 1

            else:
                cleaned_text = parsed.get("text", "").strip()
                score        = parsed.get("score", 0)

                if cleaned_text and score >= args.threshold:
                    output_fh.write(
                        json.dumps({"text": cleaned_text}, ensure_ascii=False) + "\n"
                    )
                    output_fh.flush()
                    stats["kept"] += 1
                    log.debug(f"Line {i}: KEPT (score={score})")
                else:
                    stats["rejected"] += 1
                    log.info(f"Line {i}: REMOVED (score={score})")

            id_log_fh.write(rec_id + "\n")
            id_log_fh.flush()

        # Periodic stats + GPU memory log
        if batch_idx % GPU_LOG_INTERVAL == 0:
            total_processed = batch_start + len(batch)
            log.info(
                f"[{total_processed}/{len(pending)}] kept={stats['kept']} "
                f"rejected={stats['rejected']} parse_failed={stats['parse_failed']}"
            )
            log_gpu_memory()

    output_fh.close()
    id_log_fh.close()

    log.info("=" * 60)
    log.info("Done.")
    log.info(f"  kept         : {stats['kept']}")
    log.info(f"  rejected     : {stats['rejected']}")
    log.info(f"  parse_failed : {stats['parse_failed']}")
    log.info(f"  skipped      : {stats['skipped']}")
    log.info(f"Output → {Path(args.output).resolve()}")
    log.info("Final GPU state:")
    log_gpu_memory()


if __name__ == "__main__":
    main()