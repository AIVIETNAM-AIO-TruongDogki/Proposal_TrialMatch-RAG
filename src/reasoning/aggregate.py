"""Phase 8 step 5 — aggregate criterion-level decisions into a TRIAL-level decision.

The aggregation rule is a named, swappable parameter, not a hidden implementation
detail — it directly shapes the headline number, and Phase 11 ablates it. Treat
`unverifiable` as neither a pass nor a violation: it demotes a trial's rank but
never disqualifies it (collapsing it into `violated` would be a binary rule
wearing three-state clothing).
"""

from __future__ import annotations

import re

RULES = ("strict", "lenient", "count")

DISQUALIFIED = 0.0    # disqualified — a value no other trial's score reaches
_MIN = 1e-6           # floor for a trial that is NOT disqualified

# A criterion opening with a negation is functionally exclusionary even when
# the source has no "Exclusion Criteria:" header (older NCI-style trials list
# both under one header, e.g. "PATIENT CHARACTERISTICS:"). The parser tags
# these `unknown`, which is structurally correct but semantically wrong for
# `lenient`, which then misses real exclusion criteria.
_NEG_LEAD_RE = re.compile(r"^\s*(?:no|not|none|without|absence of|free of)\b", re.I)


def effective_section(d: dict) -> str | None:
    """Section used for aggregation — infers exclusion from negation when the parser tags `unknown`.

    Kept separate from the DB's `section` (a structural fact, not a semantic
    guess) so Phase 11 can ablate this inference via
    `trial_score(..., infer_section=False)`. Reads `criterion_quote` rather
    than the raw criterion text so this stays DB-free.
    """
    sec = d.get("section")
    if sec in ("inclusion", "exclusion"):
        return sec
    if _NEG_LEAD_RE.match(d.get("criterion_quote") or ""):
        return "exclusion"
    return sec


def _spread(raw: float, lo: float, hi: float) -> float:
    """Linearly rescale `raw` into [_MIN, 1] — monotonic, never touches 0.

    The old `max(base - penalty, 1e-6)` went negative whenever unv > 2*sat,
    collapsing most trials to the same floor score and erasing the ranking.
    """
    z = (raw - lo) / (hi - lo)
    return _MIN + (1.0 - _MIN) * min(max(z, 0.0), 1.0)


def trial_score(decisions: list[dict], rule: str = "strict",
                infer_section: bool = True) -> float:
    """Ranking score for one trial from its verified criterion-level decisions.

    strict  — any `violated` decision (any section) disqualifies the trial;
              otherwise rank by more `satisfied`, fewer `unverifiable`.
    lenient — only an EXCLUSION-criterion `violated` disqualifies; violating an
              inclusion criterion only costs points (matches clinical practice).
    count   — no disqualification, ranks purely by satisfied ratio; the
              no-penalty baseline for measuring what the other rules add.

    `infer_section=False` disables the negation-based section inference, to
    ablate how much of `lenient`'s behavior it accounts for.
    """
    if not decisions:
        return DISQUALIFIED
    n = len(decisions)
    sat = sum(1 for d in decisions if d["label"] == "satisfied")
    unv = sum(1 for d in decisions if d["label"] == "unverifiable")
    vio_any = sum(1 for d in decisions if d["label"] == "violated")
    sec_of = effective_section if infer_section else (lambda d: d.get("section"))
    vio_exc = sum(1 for d in decisions
                  if d["label"] == "violated" and sec_of(d) == "exclusion")

    if rule == "strict" and vio_any:
        return DISQUALIFIED
    if rule == "lenient" and vio_exc:
        return DISQUALIFIED

    if rule == "count":
        return _spread(sat / n, 0.0, 1.0)

    # More-verified trials rank ahead of mostly-unverifiable ones; `unverifiable`
    # only demotes rank, never disqualifies — see module docstring.
    raw = (sat - 0.5 * unv) / n
    lo = -0.5
    if rule == "lenient":
        raw -= 0.3 * ((vio_any - vio_exc) / n)
        lo = -0.8
    return _spread(raw, lo, 1.0)


def rerank_by_eligibility(base_run: dict, decisions_by: dict, rule: str = "strict",
                          keep_unjudged: bool = True) -> dict:
    """Re-rank a run by eligibility score, keeping retrieval score as tie-break.

    Trials outside the reasoned top-N keep their original relative order rather
    than being dropped — dropping would artificially deflate recall and flatter rung 5.
    """
    out: dict[str, dict[str, float]] = {}
    for tid, docs in base_run.items():
        ranked = sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))
        n = len(ranked)
        new: dict[str, float] = {}
        for i, (nct, ret_score) in enumerate(ranked):
            key = (tid, nct)
            if key in decisions_by:
                elig = trial_score(decisions_by[key], rule)
                # A reasoned trial always outranks an unreasoned one; retrieval
                # order breaks ties within each group.
                new[nct] = 1000.0 + elig * 100.0 + (n - i) / (n + 1)
            elif keep_unjudged:
                new[nct] = (n - i) / (n + 1)
        out[tid] = new
    return out
