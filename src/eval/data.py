"""Load TREC Clinical Trials topics and qrels.

Trap #1 of this dataset: BOTH years number topics starting from 1. Topic 1 of
2021 is a 45-year-old with sarcoma; topic 1 of 2022 is a 19-year-old asking
about sexual health. Loading both into one dict keyed by a raw integer would
SILENTLY merge two unrelated patients.

Every id here is therefore year-prefixed ("2021_1"), and a raw number never
leaves this module.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

RAWDATA = "rawdata"

# Label meaning, verbatim from NIST:
#   "Judgment of 0 is non-relevant, 1 is excluded, and 2 is eligible."
# Reads backwards from most TREC collections, hence named instead of raw numbers.
NOT_RELEVANT = 0
EXCLUDED = 1      # medically relevant BUT excluded by criteria -> the key case
ELIGIBLE = 2

DEV_YEAR = 2021   # 75 topics
TEST_YEAR = 2022  # 50 topics — scored EXACTLY ONCE, at Phase 11

Qrels = dict[str, dict[str, int]]


def topic_id(year: int, number: str | int) -> str:
    return f"{year}_{number}"


def load_topics(year: int, rawdata: str = RAWDATA) -> dict[str, str]:
    path = os.path.join(rawdata, f"topics{year}.xml")
    root = ET.parse(path).getroot()
    out: dict[str, str] = {}
    for t in root.findall("topic"):
        num = t.get("number")
        if num is None:
            continue
        out[topic_id(year, num)] = " ".join((t.text or "").split())
    return out


def load_qrels(year: int, rawdata: str = RAWDATA) -> Qrels:
    """Read a TREC-format qrels file: `topic 0 nct_id label`."""
    path = os.path.join(rawdata, f"qrels{year}.txt")
    out: Qrels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            tid, _, nct, rel = parts
            out.setdefault(topic_id(year, tid), {})[nct] = int(rel)
    return out


def eligible_only(qrels: Qrels) -> Qrels:
    """Remap labels for the 'eligibility-aware' metric family.

    ELIGIBLE(2) -> 1,  EXCLUDED(1) -> 0,  NOT_RELEVANT(0) -> 0.

    This is half of Phase 2's real contribution. On TREC's OFFICIAL scale, a
    trial that's "medically relevant but excluded" gains positive credit — the
    exact failure this project exists to filter out. Judging the headline
    claim by official nDCG alone would make a working eligibility filter look
    like a regression. Always report both families, side by side.
    """
    return {t: {d: (1 if r == ELIGIBLE else 0) for d, r in docs.items()}
            for t, docs in qrels.items()}


def label_counts(qrels: Qrels) -> dict[int, int]:
    out = {NOT_RELEVANT: 0, EXCLUDED: 0, ELIGIBLE: 0}
    for docs in qrels.values():
        for r in docs.values():
            out[r] = out.get(r, 0) + 1
    return out
