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

## Status: COMPLETE

Both criteria met on the full 375,580-trial corpus, dev 2021. Run: `runs/hybrid.dev.txt` (RRF,
`w_lex=0.5`, k=60). Decision log: [`docs/decisions/phase6-fusion.md`](../docs/decisions/phase6-fusion.md).

| Rung | Configuration | elig nDCG@10 | Recall@1000 | contam@10 |
|---|---|---|---|---|
| 1 | Lexical — BM25 `crit_fields`, query `prof_narr_lite` | 0.2782 | 0.5307 | 0.3240 |
| 2 | Dense — qwen3 `base` | 0.2918 | 0.6050 | 0.2547 |
| **3** | **Hybrid — RRF, `w_lex=0.5`** | **0.3501** | **0.6490** | 0.3573 |

Hybrid beats `max(rung 1, rung 2) = 0.2918` by **+0.0583 (p=0.0087)**, and beats rung 1 by
**+0.0719 (p=0.0015)**. Recall@1000 improves at every rung, all significant.

**The complementarity table — the deliverable that mattered more than the fused score:**

```
5,570 eligible trials
  both legs found        0.3594
  ONLY lexical found     0.1352
  ONLY dense found       0.1749   ← 26.1% of the union, far above the 5% "redundant" threshold
  NEITHER leg found      0.3305   ← hard ceiling for rungs 4–5
  union recall           0.6695
```

The proposal's claim that both legs are load-bearing **holds and did not need softening**. The fourth
row is the one to carry forward: 33.1% of eligible trials are unreachable at top-1000 by either leg,
and no reranker or eligibility reasoner downstream can recover them.

**Two decisions recorded, both measured rather than defaulted:**

1. **Lexical leg = `prof_narr_lite`, not `bm25_best`.** Fusing both candidates against the same dense
   leg: `prof_narr_lite` wins elig nDCG@10 (+0.0237, p=0.0015), official nDCG@10 (+0.0260, p=0.0014),
   and recall@1000 (+0.0294, p=0.0000). Cost: normalized contamination 0.4038 → 0.4149 — the same
   relevance-vs-eligibility trade already measured at three earlier layers.
2. **Rung 1 is therefore re-declared as `prof_narr_lite` (0.2782), not `bm25_best` (0.2399).** Reporting
   rung 1 with a weaker configuration than the one actually fused would inflate rung 3's apparent gain
   by folding in the query-variant improvement. All ladder tables in `paper/` use 0.2782.

**Both fusion methods reported, as step 2 requires.** Weight swept 0→1 in steps of 0.1 for each:

| Method | best `w_lex` | elig@10 | off@10 | rec@1k |
|---|---|---|---|---|
| **RRF (k=60)** — selected | 0.5 | 0.3501 | **0.5309** | 0.6490 |
| `wsum` + min-max | 0.5 | **0.3526** | 0.5275 | **0.6494** |
| `wsum` + z-score | 0.4 | 0.3414 | 0.5095 | 0.6276 |

`wsum`+min-max edges RRF on elig nDCG@10 by +0.0025, but **none of the three margins is significant**
(p=0.77 / 0.68 / 0.81). RRF is kept: same performance, no score normalization to get wrong, and one
fewer tuned knob. This matches the module docstring's claim that RRF is a baseline weighted fusion
"often fails to beat" — here it did not beat it.

z-score normalization is clearly worse than min-max on every measure, which is the expected direction:
BM25 returns unbounded Lucene scores while dense returns cosine in [-1, 1], and z-score preserves that
asymmetry in the tails more than min-max does.

Both legs peak at roughly equal weight (0.4–0.5), which is itself informative — neither leg dominates,
consistent with the complementarity table above.

**Reading.** RRF [F1]. Full citations in [reading-list.md](reading-list.md).

---
[← specs index](README.md) · prev: [5 — Dense retrieval](05-dense-retrieval.md) · next: [7 — Reranking](07-reranking.md)
