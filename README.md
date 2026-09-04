# TrialMatch-RAG

Eligibility-aware retrieval and reasoning for patient-to-clinical-trial matching, evaluated on the
TREC Clinical Trials 2021/2022 benchmark. Core idea: medical relevance does not imply eligibility —
a trial can be a near-perfect topical match and still exclude the patient on one criterion.

Full design, invariants, and phase-by-phase build log: [`specs/`](specs/README.md).

## Method

```
patient narrative → extraction → hybrid retrieval (BM25 ∥ dense, RRF) → per-criterion
eligibility reasoning (satisfied / violated / unverifiable, grounded) → ranked trials
```

Reranking was benchmarked and dropped (negative result, see Results). Every eligibility decision
cites the exact criterion text and patient-narrative span it's based on; missing information is
reported as `unverifiable`, never guessed.

## Results

Held-out test set (TREC 2022, 50 narratives), full 375,580-trial corpus, scored **exactly once**
with the configuration frozen on dev-2021. Significance: paired bootstrap, 10k resamples.

| Rung | Configuration | elig nDCG@10 | Recall@1000 | Contamination@10 ↓ |
|---|---|---|---|---|
| 1 | Lexical (BM25) | 0.3191 | 0.5669 | 0.2100 |
| 2 | Dense (`qwen3`) | 0.3624 | 0.6120 | 0.1820 |
| 3 | Hybrid (RRF) | 0.3997 | **0.6912** | 0.2460 |
| 4 | + reranking — *not adopted* | 0.4103 | n/a † | 0.1880 |
| 5 | **+ eligibility reasoning** | **0.4641** | **0.6912** | **0.1600** |

† Reranking only reorders the top-N and truncates, so its Recall@1000 isn't comparable.

**Central finding.** Three independent tiers of the pipeline (indexing, querying, model choice) all
showed the same pattern on dev: improving medical relevance *increased* contamination. On test that
pattern is significant too — going dense → hybrid raises official nDCG@10 ($p=0.0019$) while making
contamination *worse* ($+0.0640$, $p=0.0010$). Rung 5 reverses it: contamination@10 drops
$0.2460\to0.1600$ ($p<0.0001$) and elig nDCG@10 rises $0.3997\to0.4641$ ($p=0.0048$), at **no** cost
to official nDCG@10 ($p=0.51$) and with recall untouched.

**Honest negatives.** On 50 test topics, dense beats lexical on no metric at significance ($p\ge0.10$
across all five) — the dev ladder's monotone look doesn't survive the smaller sample. Reranking
improves no relevance metric over hybrid ($p\ge0.37$), confirming on unseen data the decision not to
adopt it ([`specs/07`](specs/07-reranking.md)).

**Where it breaks.** Error analysis of all 80 contaminated top-10 trials: only 40% are genuine
reasoning errors. 31% are trials the reasoner *correctly disqualified* that the ranking policy kept
visible; 25% are the three-state design working as designed (the narrative is silent, and silence
never disqualifies). The grounding check turns out to verify that a quote is *verbatim*, not that it
is *relevant* — 25% of cases cleared an exclusion criterion with a real but unrelated quote, and all
of them passed verification.

**On the three-state design:** a forced-choice ablation (removing `unverifiable`, dev-2021) shows its
contribution is *not* uniform — it wins Macro F1 (0.6222 vs 0.5851) but loses raw accuracy (0.6982 vs
0.7134), because forcing a binary choice wrongly disqualifies more truly-eligible trials. F1, the
metric this project treats as primary, favors three states.
Details: [`specs/08`](specs/08-eligibility-reasoning.md).

Dev-2021 numbers and per-phase detail: [`specs/03`](specs/03-lexical-retrieval.md)–[`specs/08`](specs/08-eligibility-reasoning.md).

## Setup

`rawdata/`, `data/`, `indexes/`, `runs/`, `results/`, `.venv/`, `.env` are gitignored and must be
rebuilt per machine.

```bash
cp .env.example .env                # fill in GEMINI_API_KEY_1 (+ _2, _3, ... optional), HF_DATASET_REPO
python -m src.fetch_data             # pulls rawdata/ from HF_DATASET_REPO
uv sync --inexact                    # never bare `uv sync` — see docs/decisions/data-fetch-recovery.md
PYTHONPATH=. .venv/bin/python -m src.corpus.build_db
PYTHONPATH=. .venv/bin/python -m src.retrieval.export_corpus
PYTHONPATH=. .venv/bin/python -m src.retrieval.build_index
```

Dense index (needed for hybrid/reasoning/demo, ~3.5h on an 8GB GPU, sharded/checkpointed):
```bash
PYTHONPATH=. .venv/bin/python -m src.dense.encode --self-test
for n in 0 1 2 3; do PYTHONPATH=. .venv/bin/python -m src.dense.encode \
    --model qwen3 --input data/jsonl/base --out indexes/dense/qwen3.base.npz --shard $n; done
PYTHONPATH=. .venv/bin/python -m src.dense.encode --model qwen3 --out indexes/dense/qwen3.base.npz --merge
```

Smoke test: `PYTHONPATH=. .venv/bin/python -m src.retrieval.bm25 --index indexes/bm25-base --year 2021 --out runs/bm25.dev.txt && PYTHONPATH=. .venv/bin/python -m src.eval.score runs/bm25.dev.txt --year 2021`

## Live demo

Backend (`src/api/`) and frontend (`frontend/`) are separate deployables — split for independent
hosting, since the backend needs a GPU-capable host with `data/`/`indexes/` on disk while the
frontend is a plain static site that runs anywhere.

```bash
# backend — needs the DB + both indexes built and GEMINI_API_KEY_* set
PYTHONPATH=. .venv/bin/python -m uvicorn src.api.app:app --port 8000

# frontend — any static file server; edit window.TRIALMATCH_API_BASE in
# frontend/index.html first if the backend isn't on http://localhost:8000
python -m http.server 8080 --directory frontend
```

Runs the real pipeline live for any typed-in narrative, streaming progress over SSE.
Rate-limited/budget-capped (`src/api/quota.py`) — it's a real quota consumer. CORS is open by
default (`DEMO_CORS_ORIGINS` env var to restrict it). Decision support only, never a bare "eligible".

`Dockerfile` (repo root) builds the backend for deployment elsewhere — see its header comment for
the `docker build`/`docker run` commands and the volumes it expects (`data/`, `indexes/`).
