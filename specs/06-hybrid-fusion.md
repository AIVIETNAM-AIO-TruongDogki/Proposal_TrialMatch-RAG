[← specs index](README.md) · Phase 6 of 11 · prev: [5 — Dense retrieval](05-dense-retrieval.md) · next: [7 — Reranking](07-reranking.md)

# Phase 6 — Hybrid retrieval and rank fusion (ladder rung 3)

**Goal.** Rung 3. Combine the two legs the proposal insists are both load-bearing.

**Steps.**
1. Start with **Reciprocal Rank Fusion** [F1]: `score(d) = Σ_i 1/(k + rank_i(d))`, `k = 60`. It is one
   line, needs no score normalization, and is a strong baseline that weighted score fusion often fails to
   beat.
2. Then try weighted score fusion with min-max or z-score normalization, and tune the weight on dev.
   Report both.
3. Ablate the legs: lexical-only, dense-only, fused. This directly tests the proposal's claim that both
   legs carry distinct signal.
4. **Run a complementarity analysis**, not just a score comparison: how many relevant trials are found by
   dense but missed by lexical, and vice versa? A large overlap means fusion is buying little and the
   claim needs qualifying. This analysis is more informative than the fused nDCG number and belongs in
   the final report.

**Decide.** Fusion method and `k` / weights.

**Deliverable.** `src/retrieval/fusion.py`, fused runs, the complementarity table.

**Exit criterion.** Hybrid ≥ max(lexical, dense) on dev nDCG@10, **and** the complementarity table exists.

**Reading.** RRF [F1]. Full citations in [reading-list.md](reading-list.md).

---
[← specs index](README.md) · prev: [5 — Dense retrieval](05-dense-retrieval.md) · next: [7 — Reranking](07-reranking.md)
