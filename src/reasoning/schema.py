"""Phase 8 step 1 — CRITERION-level decision schema, three states.

    satisfied     the narrative states something that meets the criterion
    violated      the narrative states something that fails the criterion
    unverifiable  the narrative says nothing about what the criterion asks

For an EXCLUSION criterion the direction inverts: "satisfied" means the
patient does NOT have the excluding condition. Collapsing `unverifiable` into
either other state would destroy invariant 1 — `FORCED_SCHEMA` drops it
deliberately, only to run the ablation testing whether three states help.

Two separate quotes, checked independently: a model can fabricate either
`criterion_quote` or `patient_evidence` while quoting the other correctly.
"""

from __future__ import annotations

import hashlib
import json

import prompts

LABELS = ("satisfied", "violated", "unverifiable")
FORCED_LABELS = ("satisfied", "violated")   # forced-choice ablation

_DECISION_PROPS = {
    "label": {
        "type": "string",
        "enum": list(LABELS),
        "description": "unverifiable la MAC DINH khi benh an khong noi gi.",
    },
    "criterion_quote": {
        "type": "string",
        "minLength": 3,
        "description": "Trich NGUYEN VAN tu van ban tieu chi, khong dien dat lai.",
    },
    "patient_evidence": {
        "type": "string",
        "description": "Trich NGUYEN VAN tu benh an. De RONG neu unverifiable.",
    },
    "reasoning": {"type": "string", "maxLength": 300},
}


def decision_schema(forced: bool = False) -> dict:
    """Schema for one decision (per-criterion call mode)."""
    props = json.loads(json.dumps(_DECISION_PROPS))
    if forced:
        props["label"]["enum"] = list(FORCED_LABELS)
        props["label"]["description"] = ("CHI hai lua chon — bat buoc chon mot "
                                          "ben ke ca khi benh an khong du thong tin.")
    return {"type": "object", "properties": props,
            "required": ["label", "criterion_quote", "patient_evidence"],
            "additionalProperties": False}


def batch_schema(n: int, forced: bool = False) -> dict:
    """Schema for one whole-trial call — each item carries `criterion_idx` to match back.

    Matched by idx, never by array order: a mismatch would attribute one
    criterion's decision to another, which invariant 3 forbids.
    """
    item = decision_schema(forced)
    item["properties"] = {"criterion_idx": {"type": "integer"}, **item["properties"]}
    item["required"] = ["criterion_idx", *item["required"]]
    return {"type": "object",
            "properties": {"decisions": {"type": "array", "items": item,
                                          "minItems": n, "maxItems": n}},
            "required": ["decisions"], "additionalProperties": False}


SYSTEM = prompts.load("eligibility_system")
BATCH_SYSTEM = SYSTEM + "\n\n" + prompts.load("eligibility_batch_addendum")


def user_prompt(narrative: str, nct_id: str, section: str, criterion: str) -> str:
    return prompts.load("eligibility_user").format(
        narrative=narrative, nct_id=nct_id, section=section, criterion=criterion)


def batch_user_prompt(narrative: str, nct_id: str, criteria: list[dict]) -> str:
    tpl = prompts.load("eligibility_batch_item")
    body = "\n".join(tpl.format(idx=c["idx"], section=c["section"], text=c["text"])
                     for c in criteria)
    return prompts.load("eligibility_batch_user").format(
        narrative=narrative, nct_id=nct_id, n=len(criteria), criteria=body)


def prompt_hash(forced: bool = False, batched: bool = True) -> str:
    """Fingerprint of prompt + schema, part of the cache key — changing the prompt must invalidate it."""
    blob = ((BATCH_SYSTEM if batched else SYSTEM)
            + prompts.load("eligibility_batch_item")
            + prompts.load("eligibility_batch_user" if batched else "eligibility_user")
            + json.dumps(decision_schema(forced), sort_keys=True))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
