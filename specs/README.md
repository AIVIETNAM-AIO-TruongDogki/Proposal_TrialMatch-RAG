# TrialMatch-RAG — Implementation Plan

Companion to `documents/TrialMatch-RAG Proposal.pdf`. The proposal states **what** the system is and
**why** it matters. This directory states **in what order to build it**, **what must be decided at each
step**, and **how you know a step is finished**.

Nothing here changes the scope or the design invariants in the proposal. It turns them into a build order.

This is the split-up form of the plan — one file per phase, plus shared reference docs. It supersedes
`IMPLEMENTATION_PLAN.md` at the repo root, which is kept only as an unsplit historical copy.

---

## How to use these documents

Every phase file has the same six fields:

| Field | Meaning |
|---|---|
| **Goal** | The one thing this phase exists to produce. |
| **Steps** | Concrete work, in order. |
| **Decide** | Choices that must be made *and recorded* in this phase. Write the decision and the reason into `docs/decisions/`. |
| **Deliverable** | The artefact that exists at the end. |
| **Exit criterion** | A measurable condition. Do not start the next phase until it holds. |
| **Reading** | Papers that make this phase easier. Full citations in [reading-list.md](reading-list.md). |

> `docs/decisions/` is a **local-only** working record and is not published in this repository.
> References to it throughout `specs/` and `src/` name where a decision was written down, not a
> file you can open from a clone. The published record of every decision is the phase spec itself
> plus the numbers in [`README.md`](../README.md).

Two rules that make the plan work:

1. **No phase is "done" because the code runs.** It is done when its exit criterion is measured and
   written into the results log.
2. **Each phase adds exactly one rung of the ablation ladder.** If a phase changes two things at once,
   the ablation cannot attribute the difference and the experiment is wasted.

## Three principles that shape the build order

**1 — Evaluation before models.** The harness ([02](02-evaluation-harness.md)) is built before the first
retriever. A retrieval project without a scoring script from day one accumulates weeks of un-comparable
results. This is why BM25 ([03](03-lexical-retrieval.md)) comes after the harness even though it is the
easiest component to write.

**2 — The ablation ladder is the spine.** The proposal commits to
`lexical < dense < hybrid < +reranking < +eligibility reasoning`. Each rung must be measurable against
the one below. Phases 3, 5, 6, 7, 8 *are* those rungs. Everything else is scaffolding for them.

**3 — The three-state invariant propagates all the way down, including to hard filters.** It is tempting
to treat `eligibility/minimum_age` as a cheap deterministic filter. But if the patient narrative never
states an age, dropping the trial silently converts *unverifiable* into *violated* and breaks invariant 2
before the LLM is ever called. Every filter in this system must have three outcomes, not two.

## Timeline at a glance (≈8 weeks)

| Week | Phase | Ladder rung produced |
|---|---|---|
| 1 | [0 — Ground truth & environment](00-ground-truth-environment.md) | — |
| 1–2 | [1 — Corpus construction & criteria segmentation](01-corpus-construction.md) | — |
| 2 | [2 — Evaluation harness](02-evaluation-harness.md) | baseline scoring |
| 2 | [3 — Lexical retrieval](03-lexical-retrieval.md) | **rung 1: lexical** |
| 3 | [4 — Patient profile extraction](04-patient-profile-extraction.md) | query construction |
| 3–4 | [5 — Dense retrieval & embedding benchmark](05-dense-retrieval.md) | **rung 2: dense** |
| 4 | [6 — Hybrid fusion](06-hybrid-fusion.md) | **rung 3: hybrid** |
| 5 | [7 — Reranking](07-reranking.md) | **rung 4: +reranking** |
| 6 | [8 — Eligibility reasoning](08-eligibility-reasoning.md) | **rung 5: +eligibility** |
| 7 | [9 — Evidence-grounded generation](09-evidence-grounded-generation.md) | — |
| 7 | [10 — End-to-end system & API](10-end-to-end-system.md) | — |
| 8 | [11 — Ablation study, error analysis, write-up](11-ablation-writeup.md) | final results |

Weeks 6 and 8 are the two that historically overrun. Protect them by keeping Phases 9 and 10 deliberately
small — they are engineering, not research.

## Phases

- [Phase 0 — Ground truth and environment](00-ground-truth-environment.md)
- [Phase 1 — Corpus construction and criteria segmentation](01-corpus-construction.md)
- [Phase 2 — Evaluation harness](02-evaluation-harness.md)
- [Phase 3 — Lexical retrieval (ladder rung 1)](03-lexical-retrieval.md)
- [Phase 4 — Patient profile extraction](04-patient-profile-extraction.md)
- [Phase 5 — Dense retrieval and embedding benchmark (ladder rung 2)](05-dense-retrieval.md)
- [Phase 6 — Hybrid retrieval and rank fusion (ladder rung 3)](06-hybrid-fusion.md)
- [Phase 7 — Reranking (ladder rung 4)](07-reranking.md)
- [Phase 8 — Eligibility reasoning (ladder rung 5 — the contribution)](08-eligibility-reasoning.md)
- [Phase 9 — Evidence-grounded generation](09-evidence-grounded-generation.md)
- [Phase 10 — End-to-end system](10-end-to-end-system.md)
- [Phase 11 — Ablation study, error analysis, write-up](11-ablation-writeup.md)

## Reference

- [Target repository layout](repo-layout.md)
- [Risk register](risk-register.md)
- [What "done" looks like](definition-of-done.md)
- [Reading list](reading-list.md)
- [Research edge — where a contribution can actually live](research-edge.md)
