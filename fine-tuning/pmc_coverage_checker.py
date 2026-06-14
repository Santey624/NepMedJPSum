#!/usr/bin/env python3
"""
PMC coverage checker for all diseases in the NepMedJP pipeline.

For each disease across all 6 domain files, runs a lightweight PMC search
(Stage 1 only) and records how many open-access articles were found.
Does NOT download or store any paper content.

Results are appended to pmc_coverage.jsonl — one line per disease.
Already-checked diseases are skipped on re-run (resume-safe).

Run from the fine-tuning/ directory:
    python3 run_all_pmc.py [--domain cardiovascular] [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import List, Optional, Tuple

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASE_FILES = {
    "cardiovascular": os.path.join(BASE_DIR, "diseases/cardiovascular/data.jsonl"),
    "ent":            os.path.join(BASE_DIR, "diseases/ent/data.jsonl"),
    "eye":            os.path.join(BASE_DIR, "diseases/eye/data.jsonl"),
    "infectious":     os.path.join(BASE_DIR, "diseases/infectious/data.jsonl"),
    "psychiatry":     os.path.join(BASE_DIR, "diseases/psychiatry/data.jsonl"),
    "surgery":        os.path.join(BASE_DIR, "diseases/surgery/data.jsonl"),
}
RESULTS_FILE = os.path.join(BASE_DIR, "pmc_coverage.jsonl")
LOG_FILE = os.path.join(BASE_DIR, "run_all_pmc.log")

NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "gairesantosh509@gmail.com")
DEFAULT_MIN_DATE = "1995"


# ---------------------------------------------------------------------------
# PMC search helpers (Stage 1 only — no content download)
# ---------------------------------------------------------------------------

def _sleep() -> None:
    time.sleep(0.12 if NCBI_API_KEY else 0.40)


def _ncbi_params() -> dict:
    p: dict = {}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    return p


def _safe_get(url: str, params: dict, timeout: int = 30) -> requests.Response:
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _build_queries(disease: str) -> List[Tuple[str, str]]:
    dn = f'"{disease}"'
    dt = f'{dn}[Title]'
    dta = f'{dn}[Title/Abstract]'
    oa = ' AND "open access"[filter]'
    date = f" AND {DEFAULT_MIN_DATE}:3000[pdat]"
    return [
        ("T1_title_review",        f"{dt} AND review[pt]{oa}{date}"),
        ("T2_title_broad_pubtype", f"{dt} AND (review[pt] OR practice guideline[pt] OR guideline[pt]){oa}{date}"),
        ("T3_title_broad_clinical",f"{dt} AND (overview[Title/Abstract] OR diagnosis[Title/Abstract] OR treatment[Title/Abstract]){oa}{date}"),
        ("T4_abstract_review",     f"{dta} AND review[pt]{oa}{date}"),
        ("T5_abstract_broad",      f"{dta} AND (overview[Title/Abstract] OR diagnosis[Title/Abstract] OR treatment[Title/Abstract]){oa}{date}"),
    ]


def count_pmc_results(disease: str) -> Tuple[int, dict]:
    """
    Run Stage 1 queries for a disease and return (total_unique_ids, tier_counts).
    Does not fetch abstracts or full text.
    """
    seen: set = set()
    tier_counts: dict = {}

    for label, query in _build_queries(disease):
        try:
            r = _safe_get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pmc",
                    "term": query,
                    "retmode": "json",
                    "retmax": 100,
                    **_ncbi_params(),
                },
            )
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            new = [x for x in ids if x not in seen]
            seen.update(new)
            tier_counts[label] = len(ids)
            _sleep()
        except Exception as exc:
            tier_counts[label] = f"ERROR: {exc}"
            _sleep()

    return len(seen), tier_counts


# ---------------------------------------------------------------------------
# Disease loading
# ---------------------------------------------------------------------------

def load_diseases(domain_filter: Optional[str]) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    files = {k: v for k, v in DISEASE_FILES.items() if not domain_filter or k == domain_filter}
    for domain, path in files.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    name = row.get("disease_name", "").strip()
                    if name:
                        result.append((domain, name))
                except json.JSONDecodeError:
                    pass
    return result


def load_already_checked() -> set:
    """Return set of disease names already in the results file."""
    done: set = set()
    if not os.path.exists(RESULTS_FILE):
        return done
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["disease_name"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Check PMC coverage for all NepMedJP diseases.")
    parser.add_argument("--domain", type=str, default=None,
                        choices=list(DISEASE_FILES.keys()),
                        help="Only check this domain")
    parser.add_argument("--dry_run", action="store_true",
                        help="List diseases to check, do not run")
    args = parser.parse_args()

    setup_logging()

    diseases = load_diseases(args.domain)
    already_checked = load_already_checked()

    pending = [(dom, d) for dom, d in diseases if d not in already_checked]

    logging.info(f"Total diseases: {len(diseases)} | Already checked: {len(already_checked)} | Pending: {len(pending)}")

    if args.dry_run:
        for i, (dom, d) in enumerate(pending, 1):
            print(f"  {i:4d}. [{dom:15s}] {d}")
        return

    start_time = time.time()
    results_fh = open(RESULTS_FILE, "a", encoding="utf-8")

    try:
        for i, (domain, disease) in enumerate(pending, 1):
            elapsed = time.time() - start_time
            avg = elapsed / i if i > 1 else 0
            eta_h = avg * (len(pending) - i) / 3600 if avg > 0 else 0

            logging.info(f"[{i}/{len(pending)}] [{domain}] {disease} | ETA {eta_h:.1f}h")

            try:
                total_ids, tier_counts = count_pmc_results(disease)
                status = "found" if total_ids > 0 else "not_found"
                record = {
                    "domain": domain,
                    "disease_name": disease,
                    "status": status,
                    "unique_ids_found": total_ids,
                    "tier_counts": tier_counts,
                }
                logging.info(f"  → {status.upper()} ({total_ids} unique IDs)")
            except Exception as exc:
                record = {
                    "domain": domain,
                    "disease_name": disease,
                    "status": "error",
                    "unique_ids_found": 0,
                    "error": str(exc),
                }
                logging.warning(f"  → ERROR: {exc}")

            results_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_fh.flush()

    finally:
        results_fh.close()

    total_elapsed = time.time() - start_time
    # Print summary from results file
    results = []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    found = sum(1 for r in results if r.get("status") == "found")
    not_found = sum(1 for r in results if r.get("status") == "not_found")
    errors = sum(1 for r in results if r.get("status") == "error")

    logging.info(f"\n{'='*60}")
    logging.info(f"Summary: found={found}, not_found={not_found}, errors={errors}")
    logging.info(f"Total time: {total_elapsed/3600:.2f}h")
    logging.info(f"Results saved to: {RESULTS_FILE}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
