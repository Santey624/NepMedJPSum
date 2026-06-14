#!/usr/bin/env python3
"""
Generalize the not-found diseases from pmc_coverage.jsonl and re-check PMC.

For each disease whose status was "not_found", produce a less-specific name
by stripping specificity qualifiers, then re-run the PMC search to confirm
resources now exist. Two tiers are tried in order (light, then heavy) so the
name stays as specific as possible while still findable.

Originals are NOT modified. Results go to pmc_coverage_generalized.jsonl,
one line per not-found disease:
    {original_name, generalized_name, tier, new_status, unique_ids_found, ...}

Resume-safe: re-running skips originals already in the output file.

Run from the fine-tuning/ directory:
    python3 generalize_notfound.py [--dry_run] [--domain eye]
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

from run_all_pmc import count_pmc_results  # reuse Stage-1 PMC search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COVERAGE_FILE = os.path.join(BASE_DIR, "pmc_coverage.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "pmc_coverage_generalized.jsonl")
LOG_FILE = os.path.join(BASE_DIR, "generalize_notfound.log")

# Connector phrases — everything from the first match onward is dropped (Tier B)
CONNECTORS = [
    " due to ", " secondary to ", " caused by ", " resulting from ",
    " associated with ", " related to ", " complicating ", " complicated by ",
    " in the setting of ", " in the context of ", " following ", " after ",
    " status post ", " s/p ", " from ", " without ", " with ",
]
_CONNECTOR_RE = re.compile("|".join(re.escape(c) for c in CONNECTORS), re.IGNORECASE)

# Anatomical tail — "X of the Y" / "X of Y" (Tier B)
_ANATOMICAL_RE = re.compile(r"\s+of(?:\s+the)?\s+.+$", re.IGNORECASE)

# Trailing junk left after a cut
_TRAILING_JUNK_RE = re.compile(r"[\s,;:\-/]+$")


def _clean(name: str) -> str:
    name = _TRAILING_JUNK_RE.sub("", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def tier_a(name: str) -> str:
    """Light strip: parentheticals, slash-alternatives, trailing comma-qualifiers."""
    s = re.sub(r"\s*\([^)]*\)", "", name)        # remove "(...)"
    s = s.split(" / ")[0]                          # "A / B" -> "A"
    s = s.split("/")[0] if "/" in s and " " in s.split("/")[0] else s
    s = s.split(",")[0]                            # "X, persecutory type" -> "X"
    return _clean(s)


def tier_b(name: str) -> str:
    """Medium strip: Tier A + connector phrases (keeps anatomical 'of X')."""
    s = tier_a(name)
    m = _CONNECTOR_RE.search(s)
    if m:
        s = s[: m.start()]
    return _clean(s)


def tier_c(name: str) -> str:
    """Heavy strip: Tier B + anatomical 'of [the] X' tail."""
    s = _ANATOMICAL_RE.sub("", tier_b(name))
    return _clean(s)


def load_notfound(domain_filter: Optional[str]) -> List[Tuple[str, str]]:
    """Return [(domain, disease_name)] for status == not_found."""
    out: List[Tuple[str, str]] = []
    with open(COVERAGE_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "not_found":
                continue
            if domain_filter and r.get("domain") != domain_filter:
                continue
            out.append((r["domain"], r["disease_name"]))
    return out


def load_already_done() -> set:
    done: set = set()
    if not os.path.exists(OUTPUT_FILE):
        return done
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["original_name"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def process(domain: str, original: str) -> dict:
    """Try Tier A then Tier B, re-checking PMC. Return a result record."""
    candidates: List[Tuple[str, str]] = []
    seen_lower = {original.lower()}

    # Build the ordered, de-duplicated list of distinct candidates to test,
    # least-aggressive first so we keep the name as specific as possible.
    for tier_name, fn in (("A", tier_a), ("B", tier_b), ("C", tier_c)):
        cand = fn(original)
        if cand and cand.lower() not in seen_lower:
            candidates.append((tier_name, cand))
            seen_lower.add(cand.lower())

    if not candidates:
        # Nothing to strip — same query as before, already known not_found
        return {
            "domain": domain,
            "original_name": original,
            "generalized_name": original,
            "tier": "none",
            "new_status": "still_not_found",
            "unique_ids_found": 0,
            "note": "no qualifier to strip — needs manual review",
        }

    for tier, cand in candidates:
        total, _ = count_pmc_results(cand)
        if total > 0:
            return {
                "domain": domain,
                "original_name": original,
                "generalized_name": cand,
                "tier": tier,
                "new_status": "found",
                "unique_ids_found": total,
            }

    # Both tiers came up empty — report the most-general attempt
    last_tier, last_cand = candidates[-1]
    return {
        "domain": domain,
        "original_name": original,
        "generalized_name": last_cand,
        "tier": last_tier,
        "new_status": "still_not_found",
        "unique_ids_found": 0,
        "note": "generalized but still no PMC results — needs manual review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalize not-found diseases and re-check PMC.")
    parser.add_argument("--domain", type=str, default=None, help="Only process this domain")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show original -> Tier A / Tier B rewrites without hitting PMC")
    args = parser.parse_args()

    setup_logging()

    notfound = load_notfound(args.domain)
    done = load_already_done()
    pending = [(dom, d) for dom, d in notfound if d not in done]

    logging.info(f"Not-found: {len(notfound)} | Already processed: {len(done)} | Pending: {len(pending)}")

    if args.dry_run:
        for dom, d in pending:
            a, b, c = tier_a(d), tier_b(d), tier_c(d)
            print(f"[{dom}] {d}")
            seen = {d.lower()}
            any_change = False
            for label, cand in (("A", a), ("B", b), ("C", c)):
                if cand.lower() not in seen:
                    print(f"     {label}: {cand}")
                    seen.add(cand.lower())
                    any_change = True
            if not any_change:
                print(f"     (no qualifier to strip — would be flagged)")
        return

    out = open(OUTPUT_FILE, "a", encoding="utf-8")
    start = time.time()
    counts = {"found": 0, "still_not_found": 0}

    try:
        for i, (domain, original) in enumerate(pending, 1):
            rec = process(domain, original)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            counts[rec["new_status"]] = counts.get(rec["new_status"], 0) + 1

            eta_h = (time.time() - start) / i * (len(pending) - i) / 3600 if i > 1 else 0
            arrow = f"-> [{rec['tier']}] {rec['generalized_name']}" if rec["tier"] != "none" else "(no strip)"
            logging.info(
                f"[{i}/{len(pending)}] {rec['new_status'].upper():16s} "
                f"({rec['unique_ids_found']:>3} IDs) | {original}  {arrow} | ETA {eta_h:.1f}h"
            )
    finally:
        out.close()

    logging.info(f"\n{'='*60}")
    logging.info(f"Recovered (now found): {counts.get('found', 0)}")
    logging.info(f"Still not found:       {counts.get('still_not_found', 0)}")
    logging.info(f"Output: {OUTPUT_FILE}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
