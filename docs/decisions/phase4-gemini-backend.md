# Phase 4 decision: Gemini API (round-robin, 3 keys) replaces Ollama as the primary backend

**Date.** 2026-08-30.

**Decision.** `src/extraction/extract.py` and `src/extraction/query.py` call
`src/extraction/gemini.py` (model `gemini-3.6-flash`) instead of `src/extraction/ollama.py`.
Three Gemini API keys are rotated round-robin — the key advances on **every** call, not only on
error.

**Update, same day.** `src/extraction/ollama.py` was originally left in place, unused, so a revert
would be a one-line import change (see below) — the user then explicitly deleted it outright, on the
grounds that Ollama is no longer a maintained fallback in this repo. Reverting to a local backend now
means restoring it from git history (`git log --all --follow -- src/extraction/ollama.py`) or
rewriting it against `gemini.py`'s `chat_json()` interface.

**Why.** The user has 3 Gemini API keys and wants the throughput/quality available from a cloud
model instead of the 8 GB VRAM ceiling documented in [`specs/04-patient-profile-extraction.md`](../../specs/04-patient-profile-extraction.md#hardware-constraints-measured-not-estimated).
This is a direct reversal of the "fully local, no API" deployment decision recorded there for
Phases 4/8/9 — recorded as a reversal, not folded silently into the original text, because the
trade-off (patient narratives now leave the machine) is real and needs to be re-evaluated,
separately, before Phase 8/9 or any real-patient deployment.

**What was verified empirically before wiring this in** (not assumed):

1. **`gemini-2.5-flash` (the originally planned model) is gone for new accounts.** A live call
   returned `404 ... no longer available to new users`, with Google's error pointing at
   `gemini-3.6-flash` as the replacement. Confirmed knowledge of "the current Gemini model" from an
   earlier point in time does not survive contact with the live API — call it, don't recall it.
2. **Gemini's `response_schema` is an OpenAPI subset, not full JSON Schema.** Tested directly
   against the actual `PROFILE_SCHEMA` (nested objects, arrays, `enum`, `minLength`, `description`,
   nested `required`): everything survived except `additionalProperties`, which Gemini rejects
   outright with `400 INVALID_ARGUMENT — Unknown name "additional_properties"`.
   `gemini._to_gemini_schema()` strips only that key, recursively, and nothing else — confirmed
   sufficient by a full end-to-end call that reproduced correct grounding *and* correct negation
   handling (`"no angiographically apparent flow-limiting coronary artery disease"` →
   `status: "negated"`) on topic `2021_2`, the same narrative `specs/04` names as the case where the
   3B Ollama model previously diagnosed instead of extracted ("hypertrophic cardiomyopathy"). Gemini
   did not make that leap in this test.
3. **`thinking_budget` (the Gemini 2.5-era thinking control) is not accepted by `gemini-3.6-flash`;
   it uses `thinking_config.thinking_level`.** `MINIMAL` was chosen — this is structured extraction,
   not open reasoning, and default thinking spent ~346 tokens "thinking" on a trivial one-field
   probe for no measured benefit.
4. **The 3 keys share one rate-limit pool, not three independent ones.** A live smoke run hit
   `429 RESOURCE_EXHAUSTED` (`generate_content_free_tier_requests`, limit 5/minute for
   `gemini-3.6-flash`) and exhausted all 3 round-robin keys within the same call — strong evidence
   the keys sit under one Google Cloud project's free-tier quota. Round-robin therefore does not
   give ~3× throughput here; its measured benefit is smoothing bursts and providing redundancy, not
   multiplying the rate limit. `gemini.chat_json()` reads the `retryDelay` Gemini's own 429 response
   returns and sleeps that long before trying the next key, rather than failing after 3 immediate
   attempts (confirmed: a 5-topic extraction run that hit this limit completed successfully once
   this was added, ~10s/call average).

**Key storage.** `GEMINI_API_KEY_1/2/3` live in `.env` (gitignored, real values) with `.env.example`
(tracked, empty values) documenting the expected variables. `src/extraction/gemini.py` never logs a
key value. The 3 keys were pasted directly into the chat session that produced this decision —
rotating them in Google AI Studio afterward is recommended, since chat history is not a secret
store.

**Prompt storage.** `SYSTEM_PROMPT`/`USER_TEMPLATE` (`src/extraction/schema.py`) and the HyDE
system/user prompts (`src/extraction/query.py`) moved from Python string constants to
`prompts/*.txt`, loaded via `prompts.load(name)`. Content is unchanged; only the location changed,
so prompt edits are no longer code edits.

**What was not changed.** `src/extraction/verify.py` (grounding/gold-label checks operate on the
returned profile dict, independent of backend). The cache format in
`data/profiles/{year}.{model}.json` (model name substitution into the filename works unchanged for
`gemini-3.6-flash`).

**Open question for Phase 8/9.** If Gemini is adopted there too, the cost model in
[`specs/risk-register.md`](../../specs/risk-register.md) ("local-only, so cost is GPU hours not
dollars") needs re-deriving from the observed 5 req/min/model shared quota and measured
seconds/call, not reused as-is.

## Update: batched extraction calls (same day)

The free tier turned out to also cap at **20 requests/day/project/model**, on top of the 5/min limit
— confirmed by exhausting it during this session's own testing. `src/extraction/extract.py` now
batches `--batch-size` (default 5) patient narratives into a single Gemini call instead of one call
per narrative, cutting request count ~5×.

**Correctness risk and how it's handled.** Batching multiple patients into one prompt risks the model
blending facts across patients — a direct violation of invariant 3 (every claim must be grounded in
*its own* source). Mitigated two ways: (1) the addendum in `prompts/extraction_batch_addendum.txt`
explicitly instructs the model to treat each patient independently and never guess which patient a
detail belongs to; (2) mechanically, `schema.batch_schema()` requires each returned profile to carry
its own `index`, and `extract.py` only accepts an index that is present, in-range, and non-duplicate
— a missing, duplicate, or out-of-range index fails **only that one patient**, not the whole batch.
Verified with a mocked response exercising all three failure modes at once (missing, duplicate,
out-of-range indices in the same batch) plus two correctly-matched entries — each behaved exactly as
designed.

**Failure handling changed too.** A whole-batch transient failure (`GeminiError` — exhausted retries
across all 3 keys) is now **not written to the cache at all**, so the next run (no `--force` needed)
retries it automatically — different from a schema-invalid model response, which is still cached as a
recorded failure since retrying an already-bad output rarely helps. Verified live against the actual
exhausted daily quota: the run completed cleanly (no crash), printed 5 patients as pending, and wrote
nothing to `data/profiles/...json` for them.

**Cost accounting note.** `rec["seconds"]` is the batch's wall time divided evenly across its N
patients, so `verify.py`'s existing `sec_per_call` average still reads as "seconds per patient."
`prompt_tokens`/`output_tokens` are stored as the batch's raw totals, **not divided** — most of a
batch's prompt tokens are the shared system instructions paid once per batch, not once per patient, so
dividing them would understate the real per-call overhead. A new `batch_size` field on each record
makes this explicit for any later analysis.

## Update: quota is per-project, and the daily cap really is 20/day (confirmed twice)

Two more live 429s on separate days, both against the *original* project's 3 keys, both reporting
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaValue: 20` — confirms the 20/day cap is
real and consistently enforced, not a one-off fluke from the first test.

**Rotating keys within the same Google Cloud project does not reset quota.** The user regenerated all
3 keys in AI Studio (new key strings) and the very next call still hit the same 429 — quota is scoped
to the project, not the individual key string.

**Keys from a genuinely different Google Cloud project have independent quota.** The user then created
3 new keys under a separate project. The next call succeeded immediately (topic `2021_1`, single-item
batch, 5.67s, schema-valid, `dropped: []`). This is the fix for "quota exhausted" going forward when
testing needs to continue same-day: create keys under a new project rather than rotating within the
exhausted one — rotating for *secrecy* (chat history isn't a secret store) and rotating for *quota* are
different operations with different effects.

**Extraction quality re-confirmed on a live (non-mocked) call through the actual batched code path**
(topic `2021_1`, `--batch-size 1`): every field's `evidence` was a verbatim substring of the source
narrative (`verify.py` dropped nothing), `age`/`sex` correct, all 7 conditions / 7 prior_treatments / 2
labs / 2 comorbidities correctly grounded, `biomarkers: []` correctly left empty (none mentioned). This
topic had no negated statements, so it re-confirms grounding fidelity but not the `present`/`negated`
distinction — that remains verified only by the earlier `2021_2` test recorded above.

## Update: model selected by measurement, and what per-model quota means

**Quota is scoped per project *per model*.** Confirmed by three separate observations: rotating keys
inside one project does not add quota (new keys, immediate 429); keys from a *different* project do;
and a model that is fully exhausted for the day does not block a *different* model, which still serves
normally. This is the lever that made the rest of Phase 4 finishable after `gemini-3.6-flash` hit its
20/day ceiling.

**Model round-robin: legitimate for production, wrong for a measured arm.** The same rotation pattern
used for API keys would multiply available quota across models. It is fine for Phase 8 (bulk
production — 27,045 calls, no cross-item comparison). It is **not** fine inside a single ablation arm:
if the 75 HyDE descriptions came from a mix of models, the `hyde` arm would measure "HyDE with an
arbitrary model mixture", neither reproducible nor attributable. `query.py` therefore gained
`--hyde-model` — one model for all 75 topics, separable from the model that produced the profiles —
rather than a model rotator.

**Backend changed to `gemini-3.5-flash-lite`, on measured evidence.** Three models were run over all
75 dev topics with identical prompt, schema, and batch size (only the model varying):

| model | ground | age/sex | cover | neg (26 real) | s/call | Phase 8 proj. |
|---|---|---|---|---|---|---|
| `gemini-3.6-flash` | 99.9% | 100/100% | **16.9** | **92%** | 3.87s | 29.1h |
| **`gemini-3.5-flash-lite`** | 99.3% | 100/100% | 13.0 | 58% | **1.36s** | **10.2h** |
| `gemini-3.1-flash-lite` | 98.7% | *93.3/93.2%* | 8.9 | 50% | 1.33s | 10.0h |

`3.1-flash-lite` fails the age/sex threshold and is rejected. `3.5-flash-lite` is selected for its
independent quota and 2.8× speed — **a deliberate trade, not an equivalence claim.** Its negation
recall is 58% against 92%. Retrieval is unaffected (no significant difference on any query variant,
since negated terms are excluded from queries by design), so the cost falls entirely on Phase 8, which
must measure it directly. `data/profiles/2021.gemini-3.6-flash.json` is kept so reverting is free.

**Two measurement lessons.** *Grounding barely separates models* (99.9 / 99.3 / 98.7) because
extracting less makes grounding easier — `cover` and `neg` are the columns that discriminate; a
grounding-only report would have called these three models equivalent. And *quality is not monotonic
in model size*: on the `2021_14` trap, both Lite models cited `"She"` (valid evidence for the patient's
sex) where `3.6-flash` cited `"Daughter"` (invalid — having a daughter does not establish sex).
