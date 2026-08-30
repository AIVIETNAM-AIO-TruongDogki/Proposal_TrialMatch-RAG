[← specs index](README.md)

# Research edge — where a contribution can actually live

TrialGPT [E1] already does zero-shot criterion-level matching, and does it well. The project needs to be
clear-eyed about that: "use an LLM to check eligibility criteria" is no longer novel. Four directions
where this design still has genuine room, roughly ordered by tractability within 8 weeks:

**1 — Uncertainty as a first-class, measured output.** Most systems, including strong ones, collapse to
a binary decision or treat abstention as a side effect. This project's invariants make `unverifiable` a
primary output. The contribution is not asserting three states — it is **measuring** them: abstention
rate against accuracy-on-answered, risk–coverage curves, and the forced-choice ablation from
[Phase 8](08-eligibility-reasoning.md). The question *"does explicit abstention buy anything a two-state
system cannot get?"* is open, cheap to run given the harness, and the answer is publishable in either
direction. This is the strongest edge available and it is largely already in your design.

**2 — The evaluation critique.** The observation in [Phase 2](02-evaluation-harness.md) — that TREC's
official gains *reward* retrieving excluded trials, which is precisely the failure this task should
penalize — is a real methodological point about the benchmark, not just an implementation detail. A
careful demonstration that ladder rungs reorder under eligibility-aware measures versus official measures
is a contribution independent of whether your system wins. Cheap to produce, hard to argue with.

**3 — Grounding verification rather than grounding claims.** Wornow et al. [E2] found clinicians rated
justifications coherent in 75% of *incorrect* decisions — fluent explanations mask errors. Mechanical
span verification ([Phase 8](08-eligibility-reasoning.md), step 3) turns grounding into a measurable rate
rather than a design promise. Reporting grounding-violation rates per model, and correlating them with
correctness, addresses a real weakness in the current literature.

**4 — Neuro-symbolic constraint handling.** SatIR [E4] and related work argue that eligibility is
fundamentally a constraint-satisfaction problem and that LLMs should translate to constraints rather than
adjudicate them directly. This is the most interesting direction and the most likely to overrun the
timeline. Treat it as the "later extensions" slot the proposal already reserves — but knowing it exists
should shape how Phase 8 structures its output, because a well-typed criterion representation is the
bridge to it.

**What to avoid.** Do not position the contribution as "RAG applied to clinical trials" — the proposal
already says as much, and the survey [S1] shows the space is crowded. The defensible framing is the one
the invariants imply: *eligibility as a three-state, grounded, measurable decision, evaluated with metrics
that penalize the exclusion failure the standard benchmark rewards.*

Full citations for [E1]–[E4], [S1] in [reading-list.md](reading-list.md).

---
[← specs index](README.md)
