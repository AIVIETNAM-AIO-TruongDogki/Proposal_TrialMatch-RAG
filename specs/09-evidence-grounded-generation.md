[← specs index](README.md) · Phase 9 of 11 · prev: [8 — Eligibility reasoning](08-eligibility-reasoning.md) · next: [10 — End-to-end system](10-end-to-end-system.md)

# Phase 9 — Evidence-grounded generation

**Goal.** A per-trial explanation that a clinician can audit in seconds.

**Steps.**
1. Generate strictly from the Phase 8 structured output. **Do not re-read the trial in this phase** — if
   the generator can see the source text, it can introduce claims the reasoning stage never made, and
   grounding becomes unverifiable again.
2. Template-first: render the structured labels into readable prose with a template, then let the LLM
   polish. This bounds hallucination far better than free generation.
3. Surface the three states distinctly in the output. Anything `unverifiable` becomes an explicit
   *"cannot be determined from the available information: …"* line. Uncertainty is a first-class output,
   not a footnote.
4. Every claim renders with its criterion citation.
5. Frame the output as decision support (invariant 4): the UI language is "potentially eligible —
   requires clinician review", never "eligible".

**Decide.** Template vs. free generation (recommended: hybrid).

**Deliverable.** `src/generation/`, sample outputs for 10 topics.

**Exit criterion.** Automated check passes on a 20-output sample: every claim traces to a criterion index,
and no factual claim appears that is absent from the structured input.

**Reading.** ALCE [G1] for how to evaluate citation quality automatically — the citation-precision /
citation-recall framing transfers directly to criterion-grounded explanation. Full citations in
[reading-list.md](reading-list.md).

---
[← specs index](README.md) · prev: [8 — Eligibility reasoning](08-eligibility-reasoning.md) · next: [10 — End-to-end system](10-end-to-end-system.md)
