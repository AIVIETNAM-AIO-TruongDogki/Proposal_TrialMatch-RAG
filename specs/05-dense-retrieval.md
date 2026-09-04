[← specs index](README.md) · Phase 5 of 11 · prev: [4 — Patient profile extraction](04-patient-profile-extraction.md) · next: [6 — Hybrid fusion](06-hybrid-fusion.md)

# Phase 5 — Dense retrieval and embedding benchmark (ladder rung 2)

**Goal.** Rung 2, plus an evidence-based embedding choice rather than a default one.

**Steps.**
1. Benchmark embedding models **in isolation on the dev topics before building the index for all 375k
   trials.** Encoding the full corpus with the wrong model costs days of GPU time. Sub-sample: index the
   union of all judged trials plus a random 20k distractors, benchmark candidates there, then commit.
2. Candidate set: BGE-M3 [M1] and MedCPT [M2] as the proposal's named baselines, plus whichever
   general-purpose retrieval models lead the MTEB/BEIR boards at implementation time. Include at least
   one general-purpose and one biomedical model — the biomedical-beats-general assumption is frequently
   false on retrieval benchmarks and is worth testing rather than assuming.
3. Handle length: trial documents exceed most encoders' context. Either chunk-and-max-pool or use a
   long-context encoder (BGE-M3 handles 8192 tokens). Whichever you pick, apply it identically across all
   candidate models or the comparison is invalid.
4. Only then build the full index (Qdrant or FAISS). Record encoding time and index size — these are
   real constraints for the final system and belong in the write-up.

**Decide.** Final embedding model, chunking strategy, vector store. Record the benchmark table that
justified each.

**Deliverable.** An embedding benchmark table, a full-corpus vector index, `runs/dense.*.txt`.

**Exit criterion.** Dense retrieval is scored on dev and compared to Phase 3. **If dense loses to BM25,
that is a legitimate finding on this task, not a bug — record it and continue.** Lexical retrieval is
strong here precisely because biomarkers and drug names are exact tokens.

**Reading.** BGE-M3 [M1]; MedCPT [M2]; Trial2Vec [M3] for trial-specific document representation. Full
citations in [reading-list.md](reading-list.md).

## Status: COMPLETE

All 8 subsample runs are scored (3 encoders × `base`/`crit`, plus `fields`/`crit_fields` for the
winner), and the winner is encoded over the full 375,580-trial corpus and scored against Phase 3's
`bm25_best`. Full record with significance tests: `docs/decisions/phase5-dense.md`.

| # | Exit criterion | Status |
|---|---|---|
| 1 | `--self-test` green before any encoding | ✅ all four encoders |
| 2 | 3 × 2 table, all metric families + union-recall + s/doc | ✅ 8 rows, both metric families, contamination normalized by `judged@k` |
| 3 | *Do criteria help or hurt dense?* | ✅ **Hurt** — recall@1000 −0.043 / −0.049 / −0.174, all p=0.0000. Inverts the lexical finding |
| 4 | *Does biomedical beat general?* | ✅ **No** — MedCPT 0.1825 loses to bge-m3, qwen3 *and* BM25 |
| 5 | Winner encoded over full corpus, time + index size recorded | ✅ 375,580 docs, 395,147 chunks, 816 MB, `indexes/dense/qwen3.base.npz` |
| 6 | Dense losing to BM25 is a legitimate finding | ✅ **Applies at full scale.** Dense wins recall@1000 decisively (0.6050 vs 0.4176, p=0.0000) but the nDCG@10 margins that were significant on the subsample (+0.1253, p=0.0003) shrink to non-significant at full corpus (+0.0519, p=0.1154) — BM25's exact-token matching regains ground once the distractor pool is the real 375,580 trials rather than a curated 46,162 |

**Full-corpus result (the Phase 5 exit number):**

| | off@10 | elig@10 | **rec@1k** | contam raw | judged | normalized |
|---|---|---|---|---|---|---|
| BM25 rung 1 (`bm25_best`) | 0.3859 | 0.2399 | 0.4176 | 0.2840 | 0.8467 | 0.3354 |
| **Dense rung 2 (qwen3 `base`)** | 0.4245 | 0.2918 | **0.6050** | 0.2547 | 0.6307 | 0.4038 |

recall@1000 is the column that decides Phase 6: dense surfaces 45% more eligible trials into the
top-1000 than BM25, which is what fusion consumes regardless of the nDCG tie.

**Selected:** `Qwen/Qwen3-Embedding-0.6B`, variant `base`, word chunking 320/40, exact matmul search.
Union-recall — the pre-registered decision column — actually picked `fields` (0.8743 vs 0.8697;
recall@1000 +0.0070, p=0.0016). `base` was chosen anyway on cost grounds; **the deviation is recorded**
in the decision log rather than absorbed silently.

**A hypothesis was raised and refuted**, not quietly dropped: `crit_fields` was predicted to show a
*positive* interaction (repetition segregating boilerplate criteria into their own chunk, which
max-pooling could then ignore). Measured interaction is **−0.0072**. Both the prediction and its
refutation are in the decision log.

### Original plan (kept for the record)

## Status at planning time: MODELS DOWNLOADED, CODE NOT WRITTEN

All candidates are on disk (9.4 GB in the HF cache), plus the Phase 7 rerankers — pulled in one pass
because network access here has been intermittent. `sentence-transformers` 6.0.0 installed with no
change to `torch` or `transformers`.

### What Phase 5 is actually for

Phase 3 reached **Recall@1000 = 0.4176**. That is a **hard ceiling** on everything downstream —
reranking cannot recover what retrieval never returned, and neither can eligibility reasoning. So the
job here is not to beat BM25 on nDCG@10. It is to **find the trials BM25 missed.**

This changes the selection criterion:

| Measure | Why it matters here |
|---|---|
| `eligible/ndcg_cut_10` | consistency with Phase 3 — standalone quality |
| `recall_1000` | the hard ceiling for Phases 7–8 |
| **union-recall with BM25** | **the column that actually decides this** |
| seconds/doc encoded | extrapolate to 375,580 before committing |

A model that loses to BM25 on nDCG@10 but surfaces 15% of relevant trials BM25 never returned is
**worth more** than one that ties on score while returning the same documents — Phase 6 consumes the
complement, not the score. If the two columns pick different models, report both, as Phase 3 did with
k1=1.8 vs k1=1.2.

### Three encoders, three different recipes

Verified against each model's own card and config, not recalled:

| Model | Arch | Dim | Ctx | Pooling | Normalize | Prefix |
|---|---|---|---|---|---|---|
| `BAAI/bge-m3` | XLM-R 568M | 1024 | 8194 | via sentence-transformers | cosine | — |
| `Qwen/Qwen3-Embedding-0.6B` | Qwen3 0.6B | 1024 | 32768 | via sentence-transformers | cosine | `Instruct: …\nQuery:` **query side only** |
| `ncbi/MedCPT-*-Encoder` | BERT 109M ×2 | 768 | **512** | **manual CLS** (`last_hidden_state[:,0,:]`) | **none** | — |

Satisfies the "at least one general and one biomedical" requirement, with Qwen3-Embedding as the
current-generation general model the step above asks for.

`sentence-transformers` 6.0.0 exposes `encode_query()` and `encode_document()`, which read each model's
own `config_sentence_transformers.json` and apply the right prompt automatically. Use those rather than
passing `prompt_name` by hand — the principle is **follow each author's recipe**, and the model's own
config is the authority, not our guess.

**MedCPT is asymmetric — assert it in code, do not rely on care.** It is *two* models: `Query-Encoder`
for narratives, `Article-Encoder` for trials. Using the article encoder on both sides is an easy
mistake that would depress its score and lead straight to the false conclusion *"the biomedical model
loses to general"* — the exact assumption this phase is supposed to test rather than accept.

One deliberate deviation from the MedCPT card: it uses `max_length=64` on the query side because PubMed
queries are short. Our narratives run ~200 tokens, so use **512** (the model's positional limit).
Record the deviation in `docs/decisions/phase5-dense.md`.

**Guard before anything is encoded:** `--self-test` encodes one paraphrase pair and one unrelated pair
per encoder and asserts `sim(paraphrase) > sim(unrelated)`. A pooling error raises no exception — it
just makes that model look bad. Two seconds here beats six wasted encoding runs.

### Two text variants, applied identically to all three

- **A — `base`** (`data/jsonl/base`, already on disk): 124 words ≈ 175 tokens. **No model truncates**,
  so score differences are attributable to model quality alone. This is the clean comparison.
- **B — `crit`** (`data/jsonl/crit`, already on disk): tests whether Phase 3's finding (*criteria help
  BM25, +0.047, p=0.03*) reproduces for dense. Doubtful — an embedding compresses the whole document
  into one vector and negation dissolves — but Phase 3 already disproved one such hunch, so measure it.

**Chunk by words, not model tokens: 320 words, 40 overlap, score = max over chunks.** The three
encoders have three different tokenizers (XLM-R, BERT, Qwen); chunking by each model's own tokens
produces three different chunk sets, and the comparison then runs on three different datasets. Word
boundaries are identical across models *by construction*. 320 words ≈ 420 tokens for clinical text,
safely under MedCPT's 512.

**Do not use `crit_x3`.** Tripling title+conditions is a BM25 term-frequency trick; for an embedding it
only consumes context. Carrying it over would silently import a rung-1 bias into rung 2.

### Search: exact matmul, no FAISS

```
375,580 × 1024 dims × fp16 = 769 MB   →  fits on the 8 GB GPU with room to spare
75 queries × full corpus              →  torch.topk, seconds
```

Brute force is **exact**. FAISS/HNSW introduces approximation error that would mix into the
model-vs-model deltas on the ablation ladder — you would not know whether rung 2 differs from rung 1
because of the model or the index parameters. This drops a dependency *and* removes a confound.
Qdrant/FAISS is a Phase 10 deployment question, not a Phase 5 research question.

### The subsample, and the trap inside it

```
26,162 judged trials (2021) + 20,000 random distractors = 46,162 docs   (12.3% of corpus, 8.1× cheaper)
```

**Subsample scores are inflated and must never be compared to Phase 3's 0.2399.** Judged trials are 57%
of the subsample but only 7% of the real corpus — 8× fewer distractors lifts every measure. Subsample
numbers rank candidates *against each other*, nothing more. Build a BM25 run over the **same 46,162
docs** as the internal reference point (seconds to index). Comparison against Phase 3 becomes valid only
after the winner is encoded over the full corpus.

### VRAM contention

8 GB does not hold `qwen3:8b` (5.2 GB) and BGE-M3 at once. Run `ollama stop` before encoding, or torch
will OOM partway through and lose the batch. `extract.py --unload` releases the model via
`keep_alive: 0` for this reason.

**Revised exit criterion.** `--self-test` green for all three encoders *before* anything is encoded;
the 3 × 2 table with all three metric families plus union-recall and seconds/doc;
`docs/decisions/phase5-dense.md` answering two questions explicitly — *do criteria help or hurt dense?*
and *does biomedical beat general?*; only the winner encoded over the full corpus, with encoding time
and index size recorded. Dense losing to BM25 remains a legitimate finding.

---
[← specs index](README.md) · prev: [4 — Patient profile extraction](04-patient-profile-extraction.md) · next: [6 — Hybrid fusion](06-hybrid-fusion.md)
