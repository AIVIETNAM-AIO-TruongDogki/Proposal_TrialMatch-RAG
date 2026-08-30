[← specs index](README.md) · Phase 3 of 11 · prev: [2 — Evaluation harness](02-evaluation-harness.md) · next: [4 — Patient profile extraction](04-patient-profile-extraction.md)

# Phase 3 — Lexical retrieval (ladder rung 1)

**Goal.** The floor of the ladder, and a strong one. BM25 is not a straw man on this task.

**Steps.**
1. Index the relevance store with BM25 (Pyserini/Lucene, or OpenSearch if you want one engine for the
   final system).
2. Run with the **raw patient narrative as the query** first. This is rung 1 and it must be the honest
   version — no query cleaning, no extraction yet.
3. Tune `k1` and `b` on the 2021 dev topics only.
4. Try field weighting (title / conditions / summary) as a *separate* run, not a replacement.

**Decide.** Lexical engine — this choice propagates to the final infrastructure, so decide it here.

**Deliverable.** `runs/bm25.dev.txt`, `runs/bm25.test.txt`, scored.

**Exit criterion.** BM25 numbers recorded on dev, with tuned parameters. **Every later rung is measured
against this line.** A neural component that does not beat a properly tuned BM25 has not earned its place
in the pipeline.

**Reading.** Robertson & Zaragoza [R1] for BM25 as it actually behaves; the TREC 2021/2022 participant
papers [T2] for what lexical baselines scored on this exact collection. Full citations in
[reading-list.md](reading-list.md).

## Status: BUILT AND PASSING

`src/retrieval/` — `export_corpus.py` (store → JSONL, bulk `GROUP_CONCAT`, 4s not 2.3min),
`build_index.py`, `bm25.py` (raw-narrative queries, Lucene escaping only), `tune.py` (grid search).
Engine decision: **Pyserini/Lucene**. Indexing 375,580 docs takes 11–28s, not the 5–10 min estimated.

Winning configuration: index `crit_fields`, **k1 = 1.8, b = 1.0**. Full record in
`docs/decisions/phase3-lexical.md`.

| Configuration | elig nDCG@10 | official nDCG@10 | contam@10 |
|---|---|---|---|
| Lucene defaults (k1=0.9 b=0.4), `base` index | 0.1112 | 0.1848 | 0.1400 |
| + tuned (k1=4.0 b=0.75) | 0.1600 | 0.2534 | 0.1840 |
| + criteria folded into the index | 0.2070 | 0.3386 | 0.2507 |
| **+ retuned (k1=1.8 b=1.0)** | **0.2399** | **0.3859** | **0.2840** |

Recall@1000 = **0.4176**, judged@10 = 0.8467.

**Three findings worth carrying forward.**

1. **Folding criteria into the index helps** (+0.047, p=0.03) — contradicting the assumption written
   into `store.retrieval_text()`'s own docstring. The comment was left in place and marked as
   disproven rather than quietly edited, because dense retrieval faces different constraints and the
   question reopens at Phase 5.
2. **Contamination rose 0.1400 → 0.2840 as retrieval improved.** The project's thesis, measured at
   rung 1: getting better at *relevance* actively makes the *eligibility* problem worse. Every later
   rung inherits this and Phase 8 exists to reverse it.
3. **The two metric families select different parameters** — eligible-only picks k1=1.8, official
   picks k1=1.2. Reported rather than reconciled.

Baseline correction recorded: the original exit criterion compared against `results/_random.dev.json`,
which is random *within the judged pool* (Recall@1000 = 1.000 by construction) and therefore not a
full-corpus baseline at all. A true random-over-375,580 run scores **0.0000 on every measure**
(Recall@1000 = 0.0019). The corrected baseline is in `runs/_random_full.dev.txt`.

**Unsolved at this rung, by design: negation.** `"no history of cardiovascular disease"` still matches
a patient who has it. No value of `b` fixes that — it needs [Phase 8](08-eligibility-reasoning.md).

---
[← specs index](README.md) · prev: [2 — Evaluation harness](02-evaluation-harness.md) · next: [4 — Patient profile extraction](04-patient-profile-extraction.md)
