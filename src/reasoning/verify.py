"""Phase 8 step 3 — turns invariant 3 from a promise into a measurement.

A decision that can't be quoted gets dropped. `criterion_quote` is checked
against the ORIGINAL criterion text (via store.verify_quote, offsets into the
raw XML), never a text that could have been edited elsewhere.
`patient_evidence` is checked separately against the narrative — a model can
fabricate either side while quoting the other correctly. The rejection rate
is reported, not silently filtered — it measures the model's groundedness.
Exception: `unverifiable` may have empty `patient_evidence`, since forcing a
quote there would force a fabrication.
"""

from __future__ import annotations

import re

from src.corpus import store
from src.reasoning.schema import LABELS

# Words that carry no clinical content, so overlapping on them means nothing.
_STOP = set("""the a an of or and to in for with without any all other than more less at
least most within prior previous current currently including include such as who are is
was be been have has had not no non will may must should can subject subject's subjects
patient patients study trial screening baseline visit day days week weeks month months
year years old age aged during from this that these those their there if by on per e g
i e etc""".split())

# A stated sex or age legitimately settles a criterion the patient cannot meet by
# definition ("A 47-year-old man" vs "Pregnant or breastfeeding"), with zero word
# overlap. Without this carve-out a relevance check throws those away too.
_DEMOGRAPHIC_CRIT = re.compile(
    r"pregnan|breast.?feed|lactating|postmenopausal|menarche|vaginal bleed|"
    r"endometri|uterus|women|female|male|\bmen\b|years? of age|aged?\b", re.I)
_DEMOGRAPHIC_EV = re.compile(
    r"\b\d{1,3}[- ]?(year|month|week)s?[- ]?old\b|\b(man|woman|male|female|boy|girl)\b", re.I)


def norm(s: str) -> str:
    """Normalize exactly like store.verify_quote(): collapse whitespace, lowercase."""
    return " ".join((s or "").split()).lower()


def grounded_in(quote: str, source: str) -> bool:
    q = norm(quote)
    return bool(q) and q in norm(source)


def _content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", (s or "").lower()) if w not in _STOP}


def supports(evidence: str, criterion_quote: str, exempt_demographic: bool = True) -> bool:
    """Does the cited patient evidence bear on this criterion at all?

    `grounded_in` only asks whether a quote is VERBATIM. Phase 11's error
    analysis found that is a weaker guarantee than invariant 3 claims: 25% of
    contaminated top-10 trials cleared an exclusion criterion with a real but
    entirely unrelated span (one trial cleared 12 criteria — "multitrauma",
    "open fracture", "substance abuse" — with "A 19-year-old girl comes to the
    clinic due to a left wrist mass"). Every one passed verification.

    NOT WIRED INTO `check()` — measured and rejected. On dev 2021 it drops 4,439
    more decisions and makes every metric worse: Macro F1 0.6222 -> 0.5993,
    accuracy 0.6982 -> 0.6078, and contamination@10 0.2813 -> 0.3000, i.e. the
    opposite of its purpose. The reason is the instrument, not the tuning: word
    overlap cannot tell "cleared an exclusion with an unrelated quote" (what we
    wanted to catch) from "flagged a real violation in different words" (what we
    must keep), so it destroys correct disqualifications too — and paraphrase is
    exactly why this system carries a dense retrieval leg at all. A working
    version of this check has to be semantic, not lexical.

    Kept as the measured artifact behind that claim; see
    `scripts/phase11_relevance_check.py` to reproduce the numbers.
    """
    if exempt_demographic and _DEMOGRAPHIC_CRIT.search(criterion_quote or "") \
            and _DEMOGRAPHIC_EV.search(evidence or ""):
        return True
    return bool(_content(evidence) & _content(criterion_quote))


def check(conn, decision: dict, nct_id: str, idx: int, narrative: str
          ) -> tuple[bool, str | None]:
    """(is_valid, rejection_reason). Reason is kept for analysis, not just counted."""
    label = decision.get("label")
    if label not in LABELS:
        return False, "label_invalid"

    cq = decision.get("criterion_quote") or ""
    if not store.verify_quote(conn, nct_id, idx, cq):
        return False, "criterion_quote_not_in_criterion"

    pe = decision.get("patient_evidence") or ""
    if label == "unverifiable":
        # Deliberately allowed empty: nothing to quote for something never mentioned.
        if pe and not grounded_in(pe, narrative):
            return False, "patient_evidence_not_in_narrative"
        return True, None

    if not pe.strip():
        return False, "patient_evidence_empty"
    if not grounded_in(pe, narrative):
        return False, "patient_evidence_not_in_narrative"
    return True, None


def summarize(rejections: dict[str, int], total: int) -> str:
    if not total:
        return "  (khong co quyet dinh nao)"
    bad = sum(rejections.values())
    lines = [f"  {total:,} quyet dinh, {bad:,} bi vut vi khong trich dan duoc "
             f"({bad/total*100:.1f}%)"]
    for reason, n in sorted(rejections.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {reason:38s} {n:6,}  ({n/total*100:.1f}%)")
    return "\n".join(lines)
