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

Dev set (TREC 2021, 75 narratives), full 375,580-trial corpus. Full write-up:
[`paper/trialmatch-rag.tex`](paper/trialmatch-rag.tex); detail + significance tests:
[`specs/05`](specs/05-dense-retrieval.md)–[`specs/08`](specs/08-eligibility-reasoning.md).

| Rung | Configuration | elig nDCG@10 | Recall@1000 | Contamination@10 ↓ |
|---|---|---|---|---|
| 1 | Lexical (BM25) | 0.2782 | 0.5307 | 0.3240 |
| 2 | Dense (`qwen3`) | 0.2918 | 0.6050 | 0.2547 |
| 3 | Hybrid (RRF) | 0.3501 | **0.6490** | 0.3573 |
| 4 | + reranking — *not adopted* | 0.3356 | n/a | see [specs/07](specs/07-reranking.md) |
| 5 | **+ eligibility reasoning** | **0.4182** | **0.6490** | **0.2813** |

**Central finding.** Three independent tiers of the pipeline (indexing, querying, model choice) all
showed the same pattern: improving medical relevance *increased* contamination. Rung 5 reverses that,
with significance — contamination@10 drops $0.3573\to0.2813$ ($p<0.0001$), at **no** cost to official
nDCG@10 ($0.5309\to0.5472$, $p=0.31$).

**On the three-state design specifically:** a forced-choice ablation (removing `unverifiable`) shows
its contribution is *not* uniform — it wins Macro F1 (0.6222 vs 0.5851) but loses raw accuracy (0.6982
vs 0.7134), because forcing a binary choice wrongly disqualifies more truly-eligible trials. F1, the
metric this project treats as primary, favors three states. Details:
[`specs/08`](specs/08-eligibility-reasoning.md).

**Scope.** Dev-2021 only. The 2022 test set is scored exactly once, at Phase 11 (not yet run).

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

```bash
PYTHONPATH=. .venv/bin/python -m uvicorn src.api.app:app --port 8000
```
Runs the real pipeline live for any typed-in narrative, streaming progress over SSE. Needs the DB +
both indexes built and `GEMINI_API_KEY_*` set. Rate-limited/budget-capped (`src/api/quota.py`) —
it's a real quota consumer. Decision support only, never a bare "eligible".
