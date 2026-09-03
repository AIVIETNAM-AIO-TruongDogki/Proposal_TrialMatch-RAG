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

## Status: COMPLETE — exit criterion NOT met, and that is the finding

All three families were benchmarked on `runs/hybrid.dev.txt` (rung 3), depth 100, full corpus, dev 2021.
**No reranker improved over rung 3. All three degraded official nDCG@10 significantly.**
Decision log: [`docs/decisions/phase7-rerank.md`](../docs/decisions/phase7-rerank.md).

| Model | family | elig@10 | official@10 | contam/judged | s/topic |
|---|---|---|---|---|---|
| *rung 3 input* | *(RRF hybrid)* | ***0.3501*** | ***0.5309*** | *0.4149* | — |
| `MedCPT-Cross-Encoder` | biomedical cross-encoder | 0.3356 | 0.4813 | 0.3684 | **0.62** |
| `bge-reranker-v2-m3` | general cross-encoder | 0.2780 | 0.4120 | 0.3792 | 1.85 |
| `Qwen3-Reranker-0.6B` | LLM (logP yes−no) | 0.3281 | 0.4593 | 0.3774 | 8.31 |

Paired bootstrap vs rung 3, 75 dev topics:

```
medcpt   elig −0.0146 (p=0.5119, ns)    official −0.0496 (p=0.0109)  significant HARM
bge      elig −0.0722 (p=0.0186)  HARM  official −0.1189 (p=0.0001)  significant HARM
qwen3    elig −0.0220 (p=0.3828, ns)    official −0.0716 (p=0.0010)  significant HARM
```

### The competing hypothesis was tested and ruled out

The obvious objection is that depth 100 is too shallow — Recall@100 (macro) is only 0.3096 against rung
3's 0.6490 at depth 1000, so more than half the reachable eligible trials are discarded before the
reranker runs. If depth were the binding constraint, a deeper pool should help. **It does the opposite:**

| depth | ceiling (macro Recall@k) | elig nDCG@10 | Δ vs rung 3 | p |
|---|---|---|---|---|
| 100 | 0.3096 | 0.3356 | −0.0146 | 0.5119 |
| 300 | 0.4798 | 0.3161 | −0.0340 | 0.1367 |
| 500 | 0.5549 | 0.3113 | −0.0389 | 0.0965 |

The ceiling nearly doubles while quality falls monotonically. More candidates give the reranker more
opportunity to disturb an RRF ordering that was already better than what the cross-encoder produces.
The depth-is-too-shallow explanation is therefore **rejected**, and the finding stands on its own:
**on this task, these rerankers score worse than reciprocal-rank fusion does.**

### Latency, reported as a first-class result

`Qwen3-Reranker-0.6B` costs **13.4× MedCPT's latency** (8.31 vs 0.62 s/topic) and still degrades
quality. This is exactly the negative result step 3 of this spec asks to be stated plainly rather than
buried. *Caveat:* the GPU was power-capped at 55 W of a 90 W maximum throughout (see
`docs/decisions/phase5-dense.md`), so absolute latencies are inflated; all three were measured on the
same throttled hardware, so the relative ordering is sound.

### A real bug found and fixed

`Qwen3-Reranker` OOM'd on the first run. The cause was not insufficient VRAM: `_QwenReranker.score()`
called the model without `logits_to_keep`, so transformers computed logits for **every** sequence
position and the code then discarded all but the last — 16 × 2048 × 151,669 × 2 bytes = **9.94 GB**
where **4.9 MB** was needed. Fixed by passing `logits_to_keep=1`. The `--self-test` scores are
byte-identical before and after (`qwen3 rel=+1.531 unrel=−8.902`), confirming the change removes waste
without altering behaviour.

### Contamination fell, but read it with judged@k

All three rerankers *lowered* normalized contamination (0.4149 → 0.368–0.379). Part of that is real and
part is the pool artifact seen in Phase 5: `judged@10` also fell (0.8613 → 0.67–0.73), so the rerankers
push unjudged documents into the top-10, and unjudged documents count as non-contaminating by
construction.

### Decision

**No reranker is carried forward.** Rung 4 of the ablation ladder is reported as *attempted and
negative*. Rung 3 (`runs/hybrid.dev.txt`) remains the input to Phase 8. Keeping a stage that costs
latency and significantly reduces official nDCG@10 would be indefensible; the honest ladder has a gap
at rung 4, and the write-up says so.

**Reading.** monoT5 [K1]; RankGPT [K2]; MedCPT [M2]. Full citations in [reading-list.md](reading-list.md).

---
[← specs index](README.md) · prev: [6 — Hybrid fusion](06-hybrid-fusion.md) · next: [8 — Eligibility reasoning](08-eligibility-reasoning.md)
