"""Phase 4 step 1 — patient-profile schema, and the JSON Schema constraining Gemini's output.

Three states, not two: a fact is `present`, `negated` (the narrative explicitly denies it), or
absent from the output (never mentioned). Collapsing `negated` into `absent` throws away a real
clinical fact needed by Phase 8's eligibility reasoning (invariant 2). `null` is never emitted —
it would blur "read and found nothing" with "never read".
"""

from __future__ import annotations

# Extracted entity groups; this order is reused in query.py and report tables.
LIST_FIELDS = (
    "conditions",        # diagnoses, primary conditions
    "biomarkers",        # EGFR, HER2, PD-L1, mutations...
    "prior_treatments",  # drugs / surgery / radiation received
    "labs",              # lab results with values
    "comorbidities",     # comorbidities
)

SCALAR_FIELDS = ("age", "sex")

# Fields allowed into the Phase 4 BM25 query. `labs` excluded — numeric values
# ("WBC 11.2") are poor query terms; `comorbidities` kept as topical signal.
QUERY_FIELDS = ("conditions", "biomarkers", "prior_treatments", "comorbidities")

_EVIDENCE = {
    "type": "string",
    # Measured on a 3B model returning evidence="" yet still schema-valid,
    # forcing verify.py to discard an otherwise-good value — enforce it here.
    "minLength": 3,
    "description": "Trich NGUYEN VAN tu benh an. Phai la chuoi con dung tung "
                   "chu cua benh an goc, khong dien dat lai, khong tom tat.",
}

_ENTITY = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["present", "negated"],
            "description": "present = benh an noi CO. negated = benh an noi "
                           "KHONG co. Neu benh an khong nhac toi, DUNG dua "
                           "muc nay vao danh sach.",
        },
        "evidence": _EVIDENCE,
    },
    "required": ["name", "status", "evidence"],
    "additionalProperties": False,
}

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "age": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string", "enum": ["years", "months", "weeks", "days"]},
                "evidence": _EVIDENCE,
            },
            "required": ["value", "unit", "evidence"],
            "additionalProperties": False,
        },
        "sex": {
            "type": "object",
            "properties": {
                "value": {"type": "string", "enum": ["male", "female"]},
                "evidence": _EVIDENCE,
            },
            "required": ["value", "evidence"],
            "additionalProperties": False,
        },
        **{f: {"type": "array", "items": _ENTITY} for f in LIST_FIELDS},
    },
    # Only the lists are required (can be empty). age/sex may be absent —
    # that's how the model says "not stated", a valid answer, not a failure.
    "required": list(LIST_FIELDS),
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You extract structured clinical facts from a patient narrative for clinical trial matching. You are a careful medical information extractor, not a diagnostician.

Rules, in order of importance:

1. NEVER infer, impute, or guess. Extract only what the narrative literally states. If the narrative does not mention something, leave it out entirely. Omission is a correct answer; guessing is not.

2. Every extracted value MUST carry an `evidence` field containing a VERBATIM substring of the narrative. Copy the exact characters. Do not paraphrase, normalize, expand abbreviations, or fix typos inside `evidence`. If you cannot quote it exactly, do not extract it.

3. Distinguish two different things:
   - status "present": the narrative says the patient HAS it.
   - status "negated": the narrative says the patient does NOT have it ("no history of diabetes", "denies chest pain", "ruled out for sepsis").
   A negation is a clinical fact worth recording. Record it as "negated" with the negating phrase as evidence. Never record it as "present", and never drop it silently.

4. If the narrative does not mention a condition at all, it does not belong in the output in any form.

5. `name` should be the clinical concept in its normal form (e.g. "type 2 diabetes"). Only `evidence` must be verbatim."""

# --- Batched calls -----------------------------------------------------------
# extract.py sends N narratives per call to cut request count under the free
# tier. Each result carries its own `index` for matching back — never rely on
# array order, since a mismatch would attribute one patient's facts to another.
_EXTRACTION_BATCH_ADDENDUM = """You will receive MULTIPLE independent patient narratives in this one request, each labeled with an index number. Extract one profile per patient, completely independently of the others.

Under no circumstance let a detail from one patient's narrative appear in another patient's profile. If you are unsure which patient a fact belongs to, leave it out rather than guess.

Return a JSON object with a single field "profiles": an array with exactly one entry per patient given, each entry carrying its own "index" field (matching the label below) in addition to the profile fields described above."""

_EXTRACTION_BATCH_HEADER = "Extract a structured profile for each of the {n} independent patient narratives below."

_EXTRACTION_BATCH_ITEM = """--- Patient {index} ---

{narrative}"""

BATCH_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + _EXTRACTION_BATCH_ADDENDUM

_BATCH_ITEM_SCHEMA = {
    "type": "object",
    "properties": {"index": {"type": "integer"}, **PROFILE_SCHEMA["properties"]},
    "required": ["index", *PROFILE_SCHEMA["required"]],
    "additionalProperties": False,
}


def batch_schema(n: int) -> dict:
    """Schema for one call covering n profiles — PROFILE_SCHEMA plus an `index` field."""
    return {
        "type": "object",
        "properties": {
            "profiles": {"type": "array", "items": _BATCH_ITEM_SCHEMA,
                        "minItems": n, "maxItems": n},
        },
        "required": ["profiles"],
        "additionalProperties": False,
    }


def batch_user_prompt(narratives: list[str]) -> str:
    header = _EXTRACTION_BATCH_HEADER.format(n=len(narratives))
    items = "\n\n".join(_EXTRACTION_BATCH_ITEM.format(index=i, narrative=narr)
                        for i, narr in enumerate(narratives))
    return f"{header}\n\n{items}"


def prompt_hash() -> str:
    """Fingerprint of prompt + schema, used as part of the cache key.

    Changing either must invalidate the cache — otherwise a result from one
    prompt gets compared against another as if it were the same experiment.
    """
    import hashlib
    import json

    blob = (BATCH_SYSTEM_PROMPT + _EXTRACTION_BATCH_HEADER + _EXTRACTION_BATCH_ITEM +
            json.dumps(PROFILE_SCHEMA, sort_keys=True))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
