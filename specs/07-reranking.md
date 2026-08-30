[← specs index](README.md) · Phase 7 of 11 · prev: [6 — Hybrid fusion](06-hybrid-fusion.md) · next: [8 — Eligibility reasoning](08-eligibility-reasoning.md)

# Phase 7 — Reranking (ladder rung 4)

**Goal.** Rung 4. Reorder the top ~100 candidates with a model that sees query and document jointly.

**Steps.**
1. Fix the candidate depth (recommended: top 100 from fusion) and hold it constant across rerankers.
2. Compare at least three families, as the proposal requires: a cross-encoder (MedCPT's paired reranker
   [M2] is a natural fit since it is trained jointly with its retriever), a general cross-encoder
   (monoT5 / bge-reranker class) [K1], and an LLM-as-reranker (listwise, RankGPT-style) [K2].
3. Report **latency alongside quality**. An LLM reranker that adds 0.01 nDCG for 40× the latency is a
   negative result for a system meant to be usable, and stating that clearly is a contribution.
4. Note the ceiling: reranking cannot recover what retrieval never returned. Report Recall@100 as the
   hard bound on what any reranker can achieve.

**Decide.** Reranker and candidate depth.

**Deliverable.** Reranker comparison table (quality × latency), `runs/hybrid_rerank.*.txt`.

**Exit criterion.** Best reranker improves dev nDCG@10 over Phase 6 by a margin that survives the
significance test from [Phase 2](02-evaluation-harness.md).

**Reading.** monoT5 [K1]; RankGPT [K2]; MedCPT [M2]. Full citations in [reading-list.md](reading-list.md).

---
[← specs index](README.md) · prev: [6 — Hybrid fusion](06-hybrid-fusion.md) · next: [8 — Eligibility reasoning](08-eligibility-reasoning.md)
