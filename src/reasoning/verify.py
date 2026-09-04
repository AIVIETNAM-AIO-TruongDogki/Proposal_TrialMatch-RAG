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

from src.corpus import store
from src.reasoning.schema import LABELS


def norm(s: str) -> str:
    """Normalize exactly like store.verify_quote(): collapse whitespace, lowercase."""
    return " ".join((s or "").split()).lower()


def grounded_in(quote: str, source: str) -> bool:
    q = norm(quote)
    return bool(q) and q in norm(source)


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
