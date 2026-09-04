"""Phase 9 (spec) / Phase 10 (actually used) — render WITHOUT calling the LLM.

`specs/09-evidence-grounded-generation.md` specifies an LLM prose-polishing
step. This live demo deliberately SKIPS it: every request already costs 1
extraction + up to LIVE_TOP_N reasoning Gemini calls, and adding one more
call per trial for wording would double the quota cost of a feature that's
already quota-tight — see "Scope" in the plan.

The rest of specs/09 is STRUCTURE, not prose, and applies as a template:
  - step 3: `unverifiable` must show as an explanatory row, not an empty
    badge (invariant 1)
  - step 4: every claim carries its criterion citation (invariant 3)
  - step 5: language frames this as "decision support", never "eligible"
    (invariant 4)

Functions here only reshape data ALREADY produced by Phase 8
(`aggregate`/`reason`) and Phase 1 (`store`) into a dict for JSON over SSE —
no guessing, no text not already present in the source data.
"""

from __future__ import annotations

UNVERIFIABLE_NOTE = "cannot be determined from the available information"

LABEL_COPY = {
    "satisfied": "satisfied",
    "violated": "violated",
    "unverifiable": f"unverifiable — {UNVERIFIABLE_NOTE}",
}

DISCLAIMER = (
    "Decision support only — not a clinical eligibility determination. "
    "Review with a clinician or trial coordinator before acting on this output."
)


def criterion_row(d: dict) -> dict:
    """One criterion-table row, in the format already validated by `verify.check()` server-side."""
    return {
        "criterion_idx": d["criterion_idx"],
        "section": d.get("section", "unknown"),
        "label": d["label"],
        "label_display": LABEL_COPY.get(d["label"], d["label"]),
        "criterion_quote": d.get("criterion_quote", ""),
        "patient_evidence": d.get("patient_evidence", ""),
        "reasoning": d.get("reasoning", ""),
    }


def headline(n_criteria: int, n_satisfied: int, n_violated: int) -> str:
    """Verdict WITH the verified fraction — never a verdict on its own.

    Measured on the Phase 9 dev sample: a trial where 9 of 71 criteria were
    verified still summarised as "potentially eligible", which reads as more
    confidence than the system has. Under the three-state design that verdict is
    correct (silence never disqualifies), so nothing here is a wrong label — the
    overstatement lives entirely in the wording. Carrying the count is the only
    fix that survives an automated check, since there is no error to detect.
    """
    verdict = "Likely ineligible" if n_violated else "Potentially eligible"
    return (f"{verdict} — {n_satisfied + n_violated} of {n_criteria} criteria "
            f"verified — requires clinician review.")


def trial_card(trial: dict, decisions: list[dict], score: float) -> dict:
    """One ranked trial card, ready to JSON-serialize over SSE.

    `trial` is a dict from `store.get_trial()`. `decisions` is the verified
    output of `reason.run_batch_trial()`. No Gemini call happens here.
    """
    rows = [criterion_row(d) for d in sorted(decisions, key=lambda x: x["criterion_idx"])]
    n_vio = sum(1 for r in rows if r["label"] == "violated")
    n_unv = sum(1 for r in rows if r["label"] == "unverifiable")
    n_sat = sum(1 for r in rows if r["label"] == "satisfied")
    return {
        "nct_id": trial["nct_id"],
        "title": trial.get("title") or "",
        "phase": trial.get("phase") or "",
        "status": trial.get("status") or "",
        "score": score,
        "headline": headline(len(rows), n_sat, n_vio),
        "n_criteria": len(rows),
        "n_satisfied": n_sat,
        "n_violated": n_vio,
        "n_unverifiable": n_unv,
        "criteria": rows,
    }
