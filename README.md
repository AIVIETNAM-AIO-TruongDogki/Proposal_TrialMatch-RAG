# TrialMatch-RAG

Eligibility-aware retrieval and reasoning for patient-to-clinical-trial matching, evaluated on the
TREC Clinical Trials 2021/2022 benchmark. The core idea: medical relevance does not imply
eligibility — a trial can be a near-perfect topical match for a patient and still exclude them on a
single criterion. This project measures whether explicit, three-state (`satisfied` / `violated` /
`unverifiable`), grounded eligibility reasoning improves trial matching over lexical or dense
retrieval alone.

Full design, invariants, and the phase-by-phase build plan live in [`specs/`](specs/README.md) —
start there, not here. This file only covers getting the project running.

## Setup

Almost nothing except source code is committed to git — `rawdata/` (the corpus), `data/`,
`indexes/`, `runs/`, `results/`, `.venv/`, and `.env` are all gitignored (~25 GB combined) and must
be rebuilt or fetched on every new machine.

1. **Clone**, then copy the env template and fill it in:
   ```
   cp .env.example .env
   ```
   - `GEMINI_API_KEY_1/2/3` — three Gemini API keys, rotated round-robin by
     [`src/extraction/gemini.py`](src/extraction/gemini.py) for Phase 4 extraction.
   - `HF_DATASET_REPO` — the Hugging Face dataset repo holding `rawdata.tar.gz` (see below).
     `HF_TOKEN` is only needed if that repo is private.

2. **Get `rawdata/`** (8.9 GB — the April 2021 ClinicalTrials.gov snapshot + TREC 2021/2022
   topics/qrels):
   ```
   python -m src.fetch_data
   ```
   This downloads `rawdata.tar.gz` from `HF_DATASET_REPO` and extracts it. If `rawdata/` isn't on
   Hugging Face yet, produce it once from a machine that already has it:
   ```
   tar -czf rawdata.tar.gz rawdata/      # from the project root, not from inside rawdata/
   hf auth login
   hf upload <your-repo> rawdata.tar.gz --repo-type=dataset
   ```
   (Do **not** run `hf upload <repo> .` from the project root — `hf upload` has no built-in
   gitignore-style exclusion and would try to upload `.env`, `.venv/`, `.git/`, and everything else
   in the tree.) See [`docs/decisions/data-fetch-recovery.md`](docs/decisions/data-fetch-recovery.md)
   for the full story, including a from-scratch fallback if the Hugging Face copy is ever lost.

3. **Install dependencies** — either:
   ```
   uv sync --inexact
   ```
   or, without `uv`:
   ```
   python -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
   **Use `--inexact`, never bare `uv sync`.** `pyproject.toml`/`uv.lock` only lock the packages
   from `requirements.txt` — retrieval/embedding work (Phases 3/5) needs `pyserini`,
   `sentence-transformers`, `torch`, `transformers`, none of which are declared there yet (a
   pre-existing gap, not new). Bare `uv sync` prunes the venv to *exactly* match the lockfile and
   will silently uninstall all of those. `--inexact` locks/installs what's declared without removing
   anything else already present. See
   [`docs/decisions/data-fetch-recovery.md`](docs/decisions/data-fetch-recovery.md) for the incident
   this caused while writing this file.

4. **Build the corpus DB and BM25 index** from `rawdata/`:
   ```
   PYTHONPATH=. .venv/bin/python -m src.corpus.build_db
   PYTHONPATH=. .venv/bin/python -m src.retrieval.export_corpus
   PYTHONPATH=. .venv/bin/python -m src.retrieval.build_index
   ```
   Each step prints where it wrote its output and roughly how long it took. See
   [`specs/01-corpus-construction.md`](specs/01-corpus-construction.md) and
   [`specs/03-lexical-retrieval.md`](specs/03-lexical-retrieval.md) for what these numbers should
   look like and for the tuned index configuration the evaluation results are built on.

5. **Confirm it actually works** — run BM25 with the raw patient narrative as the query and score it:
   ```
   PYTHONPATH=. .venv/bin/python -m src.retrieval.bm25 --index indexes/bm25-base --year 2021 --out runs/bm25.dev.txt
   PYTHONPATH=. .venv/bin/python -m src.eval.score runs/bm25.dev.txt --year 2021
   ```
   or run the Phase 4 extraction smoke path (needs the Gemini keys from step 1):
   ```
   python -m src.extraction.gemini --self-test
   PYTHONPATH=. .venv/bin/python -m src.extraction.extract --year 2021 --limit 5
   ```

## What isn't automatic

- `.venv` and every gitignored data directory must be rebuilt (steps above) or manually copied —
  nothing is fetched automatically except `rawdata/` via `src/fetch_data.py`.
- `.env` is never committed or auto-populated. Secrets and repo IDs are yours to manage.
- Model/backend choices (embeddings, reranking, the extraction/reasoning LLM) are deliberately left
  open — see [`specs/README.md`](specs/README.md) for what's decided so far and why.
