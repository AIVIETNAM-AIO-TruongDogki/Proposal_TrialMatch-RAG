# Rebuilding `rawdata/` on a new machine: what was missing, what was considered, what was chosen

**Date.** 2026-08-30.

## What was found missing

`.gitignore` excludes everything except source code: `rawdata/` (8.9G), `data/` (6.2G), `indexes/`
(3.0G), `runs/`, `results/`, `.venv/` (5.5G), `.env` — roughly 25 GB, none of it git-tracked. That's
intentional for the *derived* directories (`data/`, `indexes/`, `runs/`, `results/` are regenerable
from `rawdata/` + code — `src/corpus/build_db.py`, `src/retrieval/export_corpus.py`,
`src/retrieval/build_index.py`). It was not intentional for `rawdata/` itself: a `src/fetch_data.py`
clearly existed at some point (a stale `src/__pycache__/fetch_data.cpython-312.pyc` remained after
its source was gone), leaving `rawdata/` with no way to reproduce it on a new machine except manual
re-discovery. `src/check_data.py` was in the same state (stale `.pyc`, no source) — **not
recreated**, since its apparent job (confirming NCT IDs resolve, building an `nct_id -> path` index)
is already done properly by `src/corpus/build_db.py` into `data/trials.db`; a standalone duplicate
wasn't worth adding back.

## Data-source investigation (three rounds)

1. **Original TREC sources**, confirmed live this session (`curl`/`WebFetch`, no auth needed):
   - Topics/qrels: `https://trec.nist.gov/data/trials/{topics2021.xml,qrels2021.txt,topics2022.xml,qrels2022.txt}`
     (~1.4 MB total).
   - Corpus: `https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part{1..5}.zip`
     (~1.79 GB compressed). **Behind Cloudflare — rejects requests without a browser-like
     `User-Agent` header.** No signup/data-use agreement encountered.
   - Kept here as a **fallback**, not implemented in code: if the Hugging Face copy described below
     is ever lost, these are the confirmed-working original URLs — no need to re-derive them.

2. **TrialGPT's NCBI-hosted `corpus.jsonl`**, suggested as a simpler alternative. Investigated by
   tracing TrialGPT's actual retrieval code
   (`https://github.com/ncbi-nlp/TrialGPT/blob/main/trialgpt_retrieval/hybrid_fusion_retrieval.py`)
   rather than trusting the README. **Rejected**: `corpus.jsonl` only carries `_id`, `title`, a
   flattened `text` blob, and `metadata.diseases_list` per trial — no separate inclusion/exclusion
   criteria, no structured `minimum_age`/`gender`/`healthy_volunteers`/etc. Adopting it would break
   `src/corpus/parse.py`'s span-offset grounding verification (`store.verify_quote()`, which checks
   an LLM's cited criterion text against the literal ClinicalTrials.gov source) and invalidate
   everything already measured in [`specs/01-corpus-construction.md`](../../specs/01-corpus-construction.md),
   which depends on the original criteria text, not someone else's already-flattened extraction.

3. **Google Drive + `gdown`**, suggested next. Reasonable and simple to upload to, but `gdown` is an
   unofficial scraper of Google Drive's download page — it breaks whenever Google changes that page,
   and Drive throttles/blocks files downloaded too many times. Not a good fit for something
   [`specs/10-end-to-end-system.md`](../../specs/10-end-to-end-system.md) wants as a reliable
   cold-start step.

## Final decision: Hugging Face Hub, one archive

The user uploads their own already-verified `rawdata/` as a single `rawdata.tar.gz` to an HF
**dataset** repo (`truongdogki/Proposal_TrialMatch-RAG`). `src/fetch_data.py` downloads it via the
official `huggingface_hub` client (`hf_hub_download`, resumable, no virus-scan interstitial) and
extracts it with stdlib `tarfile`.

**Upload procedure** (one-time, run from a machine that already has `rawdata/`):
```
tar -czf rawdata.tar.gz rawdata/      # from the PROJECT ROOT — produces an archive whose internal
                                       # paths are rawdata/..., not the contents of rawdata/ directly
hf auth login
hf upload truongdogki/Proposal_TrialMatch-RAG rawdata.tar.gz --repo-type=dataset
```
**Deliberately not** `cd rawdata && hf upload <repo> . --repo-type=dataset` and **deliberately not**
`hf upload <repo> . --repo-type=dataset` from the project root: `hf upload` has no built-in
gitignore-style exclusion (`hf upload --help` confirms only manual `--include`/`--exclude`), so
running it against the project root would try to upload `.env` (real API keys), `.venv/` (5.5G),
`.git/`, `data/`, `indexes/` along with everything else in the tree. Always upload one intentionally
built archive, never a bare `.`.

`src/fetch_data.py` extracts assuming the archive's internal paths start with `rawdata/` (matching
the `tar` command above exactly) — it extracts into the *parent* of `--dest`, not into `--dest`
itself. Idempotent (`--force` to override); prints the same NCT file-count sanity check
`CLAUDE.md` documents (`375,580` expected) as a warning, not a hard failure, since the check is only
as good as what's actually in the archive.

**Env vars** (`.env`, gitignored; `.env.example` tracked with empty placeholders): `HF_DATASET_REPO`
(required), `HF_TOKEN` (only if the dataset repo is private).

**Status: not yet end-to-end tested against the real archive** — the user had not uploaded
`rawdata.tar.gz` as of this writing. `src/fetch_data.py`'s extraction/verification logic was
exercised against a synthetic local archive (no network) to confirm the layout logic is correct
independent of the actual upload; the real Hugging Face round trip needs to be tried once the archive
exists.

## Incident: bare `uv sync` uninstalled the ML stack

While generating `pyproject.toml`/`uv.lock` for this decision, running plain `uv sync` **pruned the
venv to exactly match the new lockfile** and silently uninstalled `pyserini`, `sentence-transformers`,
`torch`, and `transformers` — all installed previously via bare `pip install` (Phase 3/5 work) and
never declared in `requirements.txt`/`pyproject.toml`. Caught immediately via `pip show` after the
sync; recovered with `pip install torch==2.13.0 pyserini==2.3.0 sentence-transformers==6.0.0`
(versions recovered from `specs/04`/`specs/05` text and from `~/.cache/uv/archive-v0/*/*.dist-info`,
which still held metadata for the uninstalled packages even though `pip show` no longer did) — verify
`torch.cuda.is_available()` and `pyserini`/`sentence_transformers` imports still work after any future
`uv sync`.

**Fix going forward:** `uv sync --inexact` (documented in `README.md`) — installs/locks what's in
`pyproject.toml` without removing anything else already present. Bare `uv sync` should not be run in
this repo until `pyserini`/`sentence-transformers`/`torch`/`transformers` are properly declared as
dependencies (a separate, larger task — pinning `torch` in particular needs to account for the CUDA
build and the specific GPU documented in `specs/04-patient-profile-extraction.md`, not just a bare
PyPI version).
