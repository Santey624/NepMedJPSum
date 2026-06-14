"""
Deduplication pipeline for translated PubMed Nepali medical data (JSONL).

Pipeline:
    1. Stream source JSONL (no full load into memory)
    2. Exact deduplication on the target field (xxhash + whitespace normalisation)
    3. MinHash near-duplicate deduplication on surviving records
       — skips MinHash for texts shorter than MIN_TEXT_LEN chars (reduces false positives)
    4. Save cleaned output to JSONL

Changes from v2:
    - Default --field changed to "text"  (was "nepali_translation")
    - Internal whitespace / CRLF normalised before hashing  (re.sub)
    - NFKC unicode normalisation applied before shingling
    - MinHash skipped for texts < 200 chars  (avoids false positives on short entries)
    - Default --num-perm raised to 256  (tighter threshold precision)
    - Records streamed through both stages; no full list kept in RAM

Dependencies:
    pip install datasketch tqdm xxhash
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Generator

import xxhash
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Minimum text length to apply MinHash (shorter texts → too many false positives)
MIN_TEXT_LEN = 200


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Generator[dict[str, Any], None, None]:
    """Yield records from a JSONL file one at a time (streaming)."""
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", lineno, exc)


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """
    Normalise text before hashing or shingling:
      1. NFKC unicode normalisation — superset of NFC; additionally collapses
         fullwidth digits, compatibility punctuation, and legacy Devanagari
         compatibility codepoints. Safe for PubMed medical text.
      2. Strip Zero Width Joiner (U+200D) and Zero Width Non-Joiner (U+200C) —
         invisible characters that break hash equality in Devanagari text where
         conjunct formation is forced/prevented differently across sources.
      3. Collapse all internal whitespace / CRLF to a single space.
      4. Strip leading / trailing whitespace.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200d", "").replace("\u200c", "")  # strip ZWJ / ZWNJ
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# MinHash builder
# ---------------------------------------------------------------------------

def _build_minhash(text: str, num_perm: int) -> MinHash:
    """
    Build a MinHash signature using character-level trigrams.
    Text must already be NFKC-normalised before calling this.
    """
    mh = MinHash(num_perm=num_perm)
    shingles = {text[i : i + 3].encode("utf-8") for i in range(len(text) - 2)}
    if not shingles:
        shingles = {text.encode("utf-8")}
    for shingle in shingles:
        mh.update(shingle)
    return mh


# ---------------------------------------------------------------------------
# Combined streaming pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    output_path: Path,
    field: str,
    minhash_threshold: float,
    num_perm: int,
    min_text_len: int = MIN_TEXT_LEN,
) -> None:
    """
    Stream records through exact → MinHash dedup in a single pass.
    Only unique records are accumulated in memory for final write.
    """
    exact_seen: set[str] = set()
    lsh = MinHashLSH(threshold=minhash_threshold, num_perm=num_perm)
    unique: list[dict[str, Any]] = []

    exact_removed = 0
    minhash_removed = 0
    short_kept = 0      # texts too short for MinHash — kept unconditionally
    total = 0

    log.info("Streaming %s …", input_path)

    for idx, rec in enumerate(tqdm(iter_jsonl(input_path), desc="Processing", unit="rec")):
        total += 1
        text: str = rec.get(field, "") or ""
        text_norm = normalise(text)

        # ── Keep empty / missing fields without dedup ──────────────────────
        if not text_norm:
            unique.append(rec)
            continue

        # ── Exact deduplication ────────────────────────────────────────────
        digest = xxhash.xxh64(text_norm.encode("utf-8")).hexdigest()
        if digest in exact_seen:
            exact_removed += 1
            continue
        exact_seen.add(digest)

        # ── MinHash near-duplicate deduplication ───────────────────────────
        if len(text_norm) < min_text_len:
            # Too short — skip MinHash to avoid false positives
            unique.append(rec)
            short_kept += 1
            continue

        mh = _build_minhash(text_norm, num_perm)
        key = str(idx)

        if lsh.query(mh):
            minhash_removed += 1
        else:
            lsh.insert(key, mh)
            unique.append(rec)

    log.info("Total records read   : %d", total)
    log.info("Exact dupes removed  : %d", exact_removed)
    log.info("Near-dupes removed   : %d", minhash_removed)
    log.info("Short texts kept     : %d  (len < %d, MinHash skipped)", short_kept, min_text_len)
    log.info("Records kept         : %d", len(unique))

    log.info("Saving to %s …", output_path)
    save_jsonl(unique, output_path)
    log.info("Done.")

    print("\n─── Deduplication summary ───────────────────────────")
    print(f"  Total records read         : {total}")
    print(f"  Exact duplicates removed   : {exact_removed}")
    print(f"  Near-duplicates removed    : {minhash_removed}")
    print(f"  Short texts (MinHash skip) : {short_kept}")
    print(f"  Records saved to output    : {len(unique)}")
    print(f"  Output file                : {output_path}")
    print("─────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Default file paths
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH  = Path("/workspace/Nepmedjp/datapreporcessing/Nepalidatacleaning/translatedpubmeddatacleaning/Part10output_01.jsonl")
DEFAULT_OUTPUT_PATH = Path("/workspace/Nepmedjp/datapreporcessing/Nepalidatacleaning/translatedpubmeddatacleaning/Part10output_01_deduped.jsonl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate translated PubMed Nepali JSONL data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the deduplicated output JSONL file.",
    )
    parser.add_argument(
        "--field", "-f",
        default="text",
        help="Name of the JSONL key containing the text to deduplicate.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.85,
        help=(
            "Jaccard similarity threshold for MinHash near-duplicate detection. "
            "Range: 0.0–1.0. Higher = stricter (fewer removals)."
        ),
    )
    parser.add_argument(
        "--num-perm", "-n",
        type=int,
        default=256,
        help=(
            "Number of MinHash permutations. "
            "Higher = more accurate but slower. Default raised to 256 for better threshold precision."
        ),
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=MIN_TEXT_LEN,
        help=(
            "Minimum text length (chars) to apply MinHash. "
            "Shorter texts are kept unconditionally to avoid false positives."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    if not (0.0 < args.threshold <= 1.0):
        log.error("--threshold must be in the range (0, 1]. Got: %s", args.threshold)
        sys.exit(1)

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        field=args.field,
        minhash_threshold=args.threshold,
        num_perm=args.num_perm,
        min_text_len=args.min_len,
    )
