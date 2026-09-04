"""Phase 9 step 2 — turn a rendered trial card into prose, template-first.

    PYTHONPATH=. .venv/bin/python scripts/phase9_generate_samples.py

Takes ONLY the structured card from `render.trial_card()`. It never receives
the database connection or the trial's source text, which is specs/09 step 1
as a function signature rather than a promise: a generator that can see the
source can assert things the reasoning stage never decided, and grounding stops
being checkable.

Every sentence it emits carries the `criterion_idx` it came from, so
`scripts/phase9_citation_check.py` can verify the claim-to-criterion mapping
mechanically instead of by reading. Off by default in the live demo — see
`render.py` for the quota reason.
"""

from __future__ import annotations

from src.extraction import gemini
from src.generation import render

SYSTEM = """You rewrite a structured clinical-trial eligibility assessment into short prose for a trial coordinator.

You are given ONLY a JSON assessment: per-criterion labels, the criterion text quoted from the trial, and the patient-narrative span each label was based on. You do not have the trial document or the patient record. Write nothing that is not already in that JSON.

Rules:

- Every sentence you write must come from exactly one criterion, and you must return its `criterion_idx`. Do not merge two criteria into one sentence and do not write a sentence that generalises across criteria.
- Do not introduce any number, date, dose, lab value, age or measurement that is not present verbatim in the JSON you were given. If a figure is not there, it does not go in the prose.
- Keep the three states distinct and never collapse them:
    satisfied     — say what the patient meets, citing the evidence given.
    violated      — say what disqualifies them, citing the evidence given.
    unverifiable  — say "cannot be determined from the available information", then say WHAT would have to be asked. Never phrase absence as if it were a negative finding.
- This is decision support, never a determination. The summary says "potentially eligible — requires clinician review" or "likely ineligible — requires clinician review". Never the bare words "eligible" or "ineligible" on their own, and never advise enrolling anyone.
- Plain clinical English, one short sentence per criterion. No preamble, no hedging about being an AI."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "maxLength": 300,
            "description": "One sentence, decision-support framed, no bare 'eligible'.",
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion_idx": {"type": "integer"},
                    "text": {"type": "string", "maxLength": 300},
                },
                "required": ["criterion_idx", "text"],
                "additionalProperties": False,
            },
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 200},
            "description": "What to ask the patient, one per unverifiable criterion worth chasing.",
        },
    },
    "required": ["summary", "claims"],
    "additionalProperties": False,
}


def _card_for_prompt(card: dict) -> dict:
    """The subset the generator is allowed to see — no free-text trial fields."""
    return {
        "n_satisfied": card["n_satisfied"],
        "n_violated": card["n_violated"],
        "n_unverifiable": card["n_unverifiable"],
        "criteria": [
            {"criterion_idx": r["criterion_idx"], "section": r["section"],
             "label": r["label"], "criterion_quote": r["criterion_quote"],
             "patient_evidence": r["patient_evidence"]}
            for r in card["criteria"]
        ],
    }


def polish(card: dict, model: str = gemini.MODEL) -> tuple[dict | None, dict]:
    """Prose for one trial card. Returns (result | None, measurement_info).

    `card` is `render.trial_card()`'s output and is the ONLY input — passing a
    connection or raw trial text here would defeat the point of the module.
    """
    if not card.get("criteria"):
        return None, {"seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0,
                      "skipped": "no criteria"}
    import json
    return gemini.chat_json(model, SYSTEM,
                            json.dumps(_card_for_prompt(card), ensure_ascii=False),
                            _SCHEMA)


MAX_QUESTIONS = 5


def assemble(card: dict, prose: dict) -> dict:
    """Reader-facing report: decisive claims in full, unknowns collapsed to a count.

    Measured on the 20-output dev sample: polishing reproduces the structured
    input faithfully, so it inherits its shape — and `unverifiable` is 59.7% of
    all decisions. One trial produced 71 sentences, 62 of them variations of
    "cannot be determined", plus 56 questions. That is not a report a
    coordinator audits in seconds, which is what specs/09 asks for.

    Two deterministic fixes, no second model call:
      * the headline carries the VERIFIED FRACTION. "Potentially eligible" on a
        trial where 9 of 71 criteria were verified overstates what the system
        knows — the same fluency-implies-confidence failure this project cites
        Wornow for, arriving through wording rather than through a wrong label.
      * unknowns collapse to one line plus the questions worth asking, instead
        of one sentence each.
    """
    by_idx = {r["criterion_idx"]: r for r in card["criteria"]}
    decisive, unknown = [], []
    for c in prose.get("claims", []):
        row = by_idx.get(c.get("criterion_idx"))
        (unknown if row and row["label"] == "unverifiable" else decisive).append(c)

    return {
        "summary": render.headline(card["n_criteria"], card["n_satisfied"],
                                   card["n_violated"]),
        "model_summary": prose.get("summary", ""),
        "claims": decisive,
        "unknown_note": (f"{len(unknown)} criteria could not be determined from the "
                         f"available information." if unknown else ""),
        "open_questions": (prose.get("open_questions") or [])[:MAX_QUESTIONS],
        "n_unknown_collapsed": len(unknown),
    }
