# Phase 5 decision log — dense retrieval

**Date.** 2026-08-30. Status: benchmark in progress.

## Encoder recipes: one per author's card, not one shared guess

| Model | Pooling / prompt | Normalize | max_len |
|---|---|---|---|
| `BAAI/bge-m3` | sentence-transformers `encode_query`/`encode_document` | cosine | model default |
| `Qwen/Qwen3-Embedding-0.6B` | same — ST reads the model's own `config_sentence_transformers.json` | cosine | model default |
| `ncbi/MedCPT-*-Encoder` | **manual CLS** (`last_hidden_state[:,0,:]`) | **none** | **512** |

Using `sentence-transformers` for MedCPT would apply mean pooling and normalization — both wrong per
its card, both silent, and both would depress its score toward the false conclusion *"the biomedical
model loses to general"*, which is the exact assumption this phase exists to test.

**MedCPT asymmetry is asserted in code**, not left to care: `Query-Encoder` for narratives,
`Article-Encoder` for trials, with `assert q_name != d_name` and a `--self-test` line confirming the two
loaded models are distinct objects.

**Deliberate deviation from the MedCPT card:** it specifies `max_length=64` on the query side because
PubMed queries are short. Our narratives run ~200 tokens, so 512 (the model's positional limit) is used
instead. Recorded rather than silently applied.

## Self-test gate — all four green before anything was encoded

Paraphrase pair must score above an unrelated pair. A pooling or prompt error raises no exception; it
only makes a model look bad.

| encoder | dim | sim(para) | sim(unrel) |
|---|---|---|---|
| bge-m3 | 1024 | +0.9385 | +0.4255 |
| qwen3 | 1024 | +0.8760 | +0.2788 |
| medcpt | 768 | +0.7173 | +0.3884 |
| gemini | 3072 | +0.8395 | +0.5314 |

## The Gemini embedding API: correct, but 30× too slow for this corpus

`gemini-embedding-001` was added as a **fourth candidate** — not a replacement. The proposal names
BGE-M3 and MedCPT specifically, and the question `specs/05` asks is *"does biomedical beat general?"*;
dropping those two would delete the question rather than answer it.

**It was then removed from the default roster on measured evidence.** The quota reads
`EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier = 100`, which looks like 100 API calls
per minute. It is not: **it counts individual texts.** The API's own 400 error says so —
`BatchEmbedContentsRequest.requests: at most 100 requests` — it calls each text a "request". Confirmed
empirically: six concurrent calls of 100 texts each → five immediate 429s, one success.

Real throughput is therefore **100 texts/minute**:

| | API | GPU (measured) |
|---|---|---|
| subsample (49,652 chunks) | 8.3 h | 16 min |
| full corpus (~400k chunks) | **67 h** | ~2 h |

Two wrong estimates were made before this was measured, and both are worth naming. First: "~38 min for
the full corpus", computed by treating the quota as the constraint. Second, after observing 23 s per
100-text call: "latency is the constraint, so parallelize". Both were wrong for the same underlying
reason — the quota unit had not been checked, only assumed. The 23 s was itself throttling, not latency.

The code is kept (`--models gemini`) because it is correct and becomes viable immediately on a paid
tier. Encoding the **75 queries** through it is fine at any tier; only corpus encoding is blocked.

## Two API quotas that are not comparable

This is the fact that explains why the embedding work runs freely while text generation is metered.

**Correction, 2026-08-31.** An earlier version of this note used the `20/day` row to conclude Phase 8
was infeasible on the free tier. That was wrong: the value is **per model**, and 20 was measured on
`gemini-3.6-flash`, not on `gemini-3.5-flash-lite` (the model actually selected). This repo's own
artifacts disprove it — on 2026-08-30 `flash-lite` served 15 extraction calls at 23:05 and 15 HyDE
calls at 23:14, 30 in one day, all successful. See `specs/risk-register.md` for the reprojection
(1,665 calls $\to$ ~3.3 days, not 83).

| API | Quota ID | Limit | Unit |
|---|---|---|---|
| `generateContent` (`3.6-flash`) | `GenerateRequestsPerDayPerProjectPerModel-FreeTier` | 20 | per **day**, per call |
| `generateContent` (`3.5-flash-lite`) | same quota ID, **different value** | >30 (reported 500) | per **day**, per call |
| `embedContent` | `EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier` | 100 | per **minute**, per **text** |

Also confirmed: **there is no dedicated rerank API.** The full set of supported actions is
`embedContent`, `generateContent`, `batchGenerateContent`, `asyncBatchEmbedContent`, `generateAnswer`,
`bidiGenerateContent`, `countTokens`, `createCachedContent`, `predictLongRunning` — nothing rank-shaped.
MedCPT-Cross-Encoder and bge-reranker-v2-m3 therefore have no API substitute and must run on GPU.

## Bug caught before it could bite: OOM at full-corpus scale only

`search.py` originally called `Vt.float()` **inside the per-query loop**. At 3072 dims × 375,580 chunks
that allocates a 4.6 GB float32 copy per query on an 8 GB GPU — guaranteed OOM. It would have passed
every subsample test and failed only on the final full-corpus run. Now the matmul stays in fp16 and
normalization is done in 200k-row slices.

## Benchmark results — 8 runs, subsample of 46,162 docs

**Subsample scores are inflated and must never be placed beside Phase 3's 0.2399.** Judged trials are
57% of the subsample but only 7% of the real corpus. These numbers rank candidates *against each
other*; the BM25 row is an internal reference built over the identical 46,162 docs.

| Model | Variant | off@10 | elig@10 | rec@1k | contam@10 | judged@10 | **contam/judged** | only-dense | only-bm25 | **union** | s/doc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 | `fields` | 0.5149 | 0.3584 | **0.8421** | 0.3053 | 0.7973 | 0.3829 | **0.2890** | 0.0555 | **0.8743** | 0.0407 |
| **qwen3** | **`base`** | **0.5220** | **0.3655** | 0.8352 | 0.3027 | 0.8133 | 0.3721 | 0.2844 | 0.0605 | 0.8697 | 0.0339 |
| qwen3 | `crit` | 0.5130 | 0.3536 | 0.7921 | 0.3107 | 0.8013 | 0.3877 | 0.2452 | 0.0580 | 0.8305 | 0.0766 |
| qwen3 | `crit_fields` | 0.4986 | 0.3394 | 0.7922 | 0.3080 | 0.7893 | 0.3902 | 0.2420 | 0.0571 | 0.8273 | 0.0848 |
| bge-m3 | `base` | 0.4499 | 0.3113 | 0.6628 | 0.2733 | 0.7227 | 0.3782 | 0.2126 | 0.1474 | 0.7978 | 0.0211 |
| bge-m3 | `crit` | 0.4364 | 0.2827 | 0.6137 | 0.3107 | 0.6987 | 0.4447 | 0.1697 | 0.1420 | 0.7549 | 0.0400 |
| medcpt | `base` | 0.2689 | 0.1825 | 0.4951 | *0.1587* | 0.4720 | 0.3362 | 0.1230 | 0.2596 | 0.7083 | 0.0051 |
| medcpt | `crit` | 0.1535 | 0.0938 | 0.3215 | *0.1213* | 0.2947 | 0.4118 | 0.0587 | 0.3481 | 0.6440 | 0.0094 |
| *BM25* | *`crit_fields`* | 0.3934 | 0.2402 | 0.5964 | 0.3013 | 0.9027 | **0.3338** | — | — | — | — |

## The three open questions, answered

### 1. Do criteria help or hurt dense? — **They hurt. Confirmed on all three models.**

The answer inverts Phase 3's lexical finding (+0.047, p=0.03). Paired bootstrap, 75 dev topics:

| | Δ elig nDCG@10 | | Δ **recall@1000** | |
|---|---|---|---|---|
| qwen3 | −0.0119 | p=0.4126 *ns* | **−0.0431** | **p=0.0000** |
| bge-m3 | −0.0286 | p=0.0834 *ns* | **−0.0491** | **p=0.0000** |
| medcpt | −0.0887 | **p=0.0000** | **−0.1736** | **p=0.0000** |

**Report the recall column, not the nDCG column.** On nDCG@10 the direction is consistent but only
significant for the weakest model — stating "criteria hurt nDCG on all three" would overclaim. On
recall@1000 the effect is unambiguous and significant everywhere, and recall@1000 is the metric that
matters: it is the hard ceiling that Phases 7–8 inherit.

Magnitude scales inversely with model strength (−0.043 / −0.049 / −0.174), which is consistent with a
noise-dilution mechanism: stronger encoders tolerate more boilerplate before the signal degrades.

**Mechanism.** Criteria blocks are long and near-identical across trials (*"informed consent"*,
*"age 18 or older"*, *"adequate organ function"*). One vector per chunk means that boilerplate dilutes
the discriminative content. Chunks go from 1.08 to 1.70 per doc, and since scoring is **max over
chunks**, every added boilerplate chunk is another chance for a spurious maximum. And it costs double:
`crit` runs 0.0766 s/doc against `base`'s 0.0339 — the worse variant is also 2.3× more expensive.

### 2. Does biomedical beat general? — **No. MedCPT loses to everything.**

MedCPT (0.1825) loses to bge-m3 (0.3113), qwen3 (0.3655), **and** BM25 (0.2402). The assumption
`specs/05` set out to test rather than accept is false on this task.

The cause is unlikely to be size. Cost tracks the *transformer body* — the embedding matrix is a lookup
table and costs almost no FLOPs:

| | layers × hidden | vocab | embedding matrix | body | s/doc |
|---|---|---|---|---|---|
| MedCPT-Article | 12 × 768 | 30,522 | 23.4M (21.6%) | **84.9M** | 0.0051 |
| bge-m3 | 24 × 1024 | 250,002 | 256.0M (45.9%) | **302.0M** | 0.0211 |
| Qwen3-Emb-0.6B | 28 × 1024 | 151,669 | 155.3M (28.9%) | **381.7M** | 0.0339 |

Cost ≈ layers × hidden², and the prediction matches: 12/24 × (768/1024)² = 0.28 ≈ 84.9/302.0.

**The likelier cause is task mismatch, not capacity.** MedCPT is trained on *short PubMed query ↔
article* pairs — its own card uses `max_length=64` on the query side. Our queries are ~200-token
patient narratives, far outside that distribution. Right domain, wrong task shape. This is a
hypothesis, not a measured cause; testing it would need a short-query arm (see "Untested axis" below).

**`s/doc` is not an architecture comparison.** qwen3 ran at `BATCH=8` to avoid OOM while the other two
ran at `BATCH=32`. The column is honest as *cost on this hardware* — which is what decides the
full-corpus run — but no architectural conclusion may be drawn from it.

### 3. Selection by union-recall — and a recorded deviation from it

`specs/05` pre-registers union-recall with BM25 as the decision column, not nDCG. It picks **`fields`**:

```
union          0.8743  vs  0.8697   (+0.0046)
recall@1000    +0.0070   p = 0.0016   CI [0.0029, 0.0111]   SIGNIFICANT
elig nDCG@10   −0.0071   p = 0.3043   not significant
official@10    −0.0071   p = 0.2742   not significant
```

`fields` gives up nothing measurable on nDCG and gains a small but significant amount of recall.

**We encoded `base` anyway.** The full-corpus decision was made on cost (212 min vs 255 min) and on the
judgement that a +0.8% relative recall gain measured on an 8×-inflated subsample may not survive the
real corpus. **This is a deviation from the pre-registered rule and is recorded as one** — changing the
decision rule after seeing the data is exactly what invalidates a benchmark, so it is stated here
rather than quietly absorbed. If the full-corpus run disappoints, `fields` is the first thing to retry.

## The 2×2: a hypothesis raised, then refuted

`crit_fields` was run to test a specific mechanism. `base` (120 words) and `fields` (150 words) both
fit in a single 320-word chunk, so repetition there can only change the token mix. But `crit` (321
words) spans ~1.7 chunks, so the 90 repeated words should push criteria into chunk 1 and leave chunk 0
as a concentrated disease/intervention chunk — and under max-over-chunks the scorer could then *ignore*
the boilerplate. That predicted a **positive interaction**.

|  | no boost | boost |
|---|---|---|
| **no criteria** | `base` 0.3655 | `fields` 0.3584 |
| **criteria** | `crit` 0.3536 | `crit_fields` 0.3394 |

```
criteria effect      −0.0119
boost effect         −0.0071
additive prediction   0.3465
observed              0.3394
INTERACTION          −0.0072      ← negative
```

**The hypothesis is refuted.** The two factors compound slightly worse than additively; boost does not
rescue criteria. Recorded with the hypothesis intact, because a refuted prediction is a result — and
because the reasoning behind it (chunk composition under max-pooling) is sound enough that someone will
propose it again.

Incidental finding: `data/jsonl/crit_fields` and `data/jsonl/crit_x3` are **byte-identical** (same
md5). What looked like five index variants is four.

## The contamination trap — never report contamination@k alone

MedCPT posts the lowest raw contamination in the table (0.1587, and 0.1213 with criteria). Both are
artifacts. `contamination_at_k` divides by `k`, not by the number of *judged* documents, so unjudged
documents count as non-contaminating and a system that retrieves outside the pool looks clean:

| | contam@10 raw | judged@10 | normalized |
|---|---|---|---|
| BM25 | 0.3013 | 0.9027 | **0.3338** |
| qwen3 `base` | 0.3027 | 0.8133 | 0.3721 |
| bge-m3 `base` | 0.2733 | 0.7227 | 0.3782 |
| medcpt `base` | *0.1587* | 0.4720 | 0.3362 |
| medcpt `crit` | *0.1213* | **0.2947** | 0.4118 |

After normalization every system lands in 0.33–0.38. The same trap fires again on the criteria
comparison: medcpt's contamination "improves" significantly with criteria (−0.0373, p=0.0229) purely
because its judged@10 collapses to 0.2947.

**Rule: `contamination@k` is never reported without `judged@k` beside it.**

## The headline finding: better retrieval does not reduce contamination

qwen3 `base` beats BM25 decisively on the same subsample — every margin significant:

```
elig nDCG@10    +0.1253   p = 0.0003
official@10     +0.1286   p = 0.0007
recall@1000     +0.2388   p = 0.0000
```

And normalized contamination goes **0.3338 → 0.3721**: slightly worse, certainly not better. Concretely,
qwen3's average top-10 holds 3.5 eligible trials and 3.0 relevant-but-excluded ones.

This is the third independent layer at which the project's thesis has now been measured — after the
index layer (Phase 3: contamination 0.1400 → 0.2840 as relevance improved) and the query layer
(Phase 4: 0.2840 → 0.3467, p=0.0008). Three different interventions, three mechanisms, one direction.
**Retrieval cannot separate "relevant" from "eligible", and getting better at retrieval does not help.**
That is the gap Phase 8 exists to close, and it is now a measurement rather than a proposal claim.

## Full-corpus encoding — COMPLETE. This is the Phase 5 exit number.

Finished 2026-08-31 22:35. All 4 shards encoded (`indexes/dense/qwen3.base.npz`, 375,580 docs,
395,147 chunks, 816 MB), merged, searched, and scored against `bm25_best`.

| | off@10 | elig@10 | **rec@1k** | contam raw | judged | normalized |
|---|---|---|---|---|---|---|
| BM25 rung 1 (`bm25_best`) | 0.3859 | 0.2399 | 0.4176 | 0.2840 | 0.8467 | 0.3354 |
| **Dense rung 2 (qwen3 `base`)** | 0.4245 | 0.2918 | **0.6050** | 0.2547 | 0.6307 | 0.4038 |

Paired bootstrap, 75 dev topics:

```
eligible/ndcg_cut_10   +0.0519   p=0.1154   not significant
official/ndcg_cut_10   +0.0386   p=0.2923   not significant
elig/recall_1000       +0.1875   p=0.0000   significant
```

**This reverses part of the subsample picture.** On the 46,162-doc subsample dense beat BM25 on all
three metrics with strong significance (elig nDCG p=0.0003). At full scale, only recall@1000 keeps its
edge — the two nDCG margins are no longer distinguishable from noise. Not a bug: `specs/05`'s exit
criterion states this explicitly — *"dense losing to BM25 remains a legitimate finding."* The subsample
is 12.3% of the corpus with 8× fewer distractors; a model that separates signal from a small, curated
distractor pool does not automatically separate it from 375,580 real trials, where exact tokens
(biomarkers, drug names) that BM25 matches losslessly and embeddings blur become harder to beat.

**recall@1000 is still the number that matters most in this table.** It is the hard ceiling Phases 6–8
inherit, and dense wins it decisively — 0.6050 vs 0.4176, +18.75 points absolute, p<0.0001. A model
that ties on nDCG@10 but surfaces 45% more eligible trials into the top-1000 is worth carrying into
fusion regardless of the nDCG result, because Phase 6 consumes the complement, not the score.

### How this run recovered from a mid-run stop

The run was deliberately interrupted after shard 2 (79.9% of corpus) on user request, scored as a
preliminary checkpoint (see below), then resumed: shards 0–2 were skipped (already saved), only shard 3
(75,580 docs) was encoded, and the merge ran once over all 4. No shard was re-encoded. Total encoding
time across both sessions: ~3,400s for shard 3 alone at the throttled 23 chunk/s rate; full-corpus
throughput was GPU-power-capped at 55 W of a 90 W maximum throughout (see below) rather than limited by
the sharding approach itself.

## Preliminary checkpoint at 3/4 shards (superseded — kept for the record)

Full-corpus encoding (`scripts/encode_full_corpus.sh`) started 2026-08-31 19:50. Two things happened
worth recording, neither of which is a code defect:

**Throughput dropped from the benchmarked 52 chunk/s to 24 chunk/s, and it was not memory
fragmentation.** `nvidia-smi -q` showed `SW Power Cap: Active`, current power limit 55 W against a
90 W hardware max, graphics clock 1920–2280 MHz against a 3105 MHz boost ceiling — the laptop's power
management throttling the GPU under sustained load on AC power. Adding
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the fix for the `[W] memory allocation failed`
warning seen during shard 0) did **not** restore the rate — confirming the warning was a red herring
and power capping was the real constraint. Raising the cap requires `sudo nvidia-smi -pl 90`, not
available in this environment, and laptop OEMs frequently lock TGP at the BIOS level regardless.

**Run stopped by user request after shard 2 (300,000/375,580 docs, 79.9% of corpus), not because of a
failure.** Shards 0–2 are saved (`indexes/dense/qwen3.base.shard0{0,1,2}.npz`, kept on disk); shard 3
(75,580 docs, the remaining 20.1%) was never started. Resuming later only requires running shard 3 and
re-merging — the three completed shards are not repeated.

**Coverage impact, measured before drawing any conclusion from this partial run:**

```
26,162 judged trials (any label)  →  5,315 (20.3%) fall in the unencoded shard 3
 5,124 ELIGIBLE trials            →  1,115 (21.8%) fall in the unencoded shard 3
```

21.8% of eligible trials cannot be retrieved by this run at any rank — not a ranking weakness, an
absence from the search space entirely. `recall@1000` on this run is therefore a **valid lower bound**,
not the number that belongs in the final ablation table.

| | off@10 | elig@10 | rec@1k | contam raw | judged | normalized |
|---|---|---|---|---|---|---|
| BM25 rung 1 (full corpus, `bm25_best`) | 0.3859 | 0.2399 | 0.4176 | 0.2840 | 0.8467 | 0.3354 |
| **Dense rung 2 (qwen3 `base`, 3/4 shards only)** | 0.4233 | **0.3061** | **0.5137** | 0.2227 | 0.6000 | 0.3711 |

Paired bootstrap, 75 dev topics:

```
eligible/ndcg_cut_10   +0.0662   p=0.0498   significant (borderline)
official/ndcg_cut_10   +0.0374   p=0.3083   not significant
elig/recall_1000       +0.0961   p=0.0008   significant
```

Dense beats BM25 on recall@1000 and (marginally) on eligible nDCG@10 even missing a fifth of the
corpus — consistent with the subsample benchmark's finding. **This is not the Phase 5 exit number.**
It is recorded here as a checkpoint; the table that belongs in the ablation ladder and in `paper/` is
the one from the completed 4/4-shard run (`runs/dense.dev.txt`), not `runs/dense.partial3of4.dev.txt`.

## Untested axis, recorded for later

Every dense run here uses the **raw patient narrative** as the query. The query-side variants built in
Phase 4 (`prof`, 19 words; `prof_narr`; `hyde`) were only ever evaluated against BM25. Re-querying an
existing vector index costs seconds — 75 queries, no re-encoding of documents — so this is nearly free
to measure and was simply out of scope for the encoder benchmark. Phase 4 explicitly flagged short
queries as valuable for dense and reranking stages, so the combination is worth one run.
