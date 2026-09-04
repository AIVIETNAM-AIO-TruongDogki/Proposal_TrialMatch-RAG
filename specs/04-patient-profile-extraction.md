[← specs index](README.md) · Phase 4 of 11 · prev: [3 — Lexical retrieval](03-lexical-retrieval.md) · next: [5 — Dense retrieval](05-dense-retrieval.md)

# Phase 4 — Patient profile extraction

**Goal.** Convert the free-text narrative into a structured clinical profile that both query construction
and eligibility reasoning consume.

**Steps.**
1. Define the schema explicitly — the proposal names the fields:
   ```json
   {
     "age": {"value": 62, "unit": "years", "evidence": "62-year-old"},
     "sex": {"value": "female", "evidence": "woman"},
     "conditions": [{"name": "stage III NSCLC", "evidence": "..."}],
     "stage": {"value": "III", "evidence": "..."},
     "biomarkers": [{"name": "EGFR", "status": "wild-type", "evidence": "..."}],
     "prior_treatments": [{"name": "cisplatin", "evidence": "..."}],
     "labs": [{"name": "creatinine", "value": 1.8, "unit": "mg/dL", "evidence": "..."}],
     "comorbidities": [...],
     "negated": [...]
   }
   ```
2. **Every extracted field carries a verbatim evidence span from the narrative.** A field with no span is
   a hallucination by construction and should be rejected programmatically.
3. **Absent fields are `null`, never guessed.** This is invariant 2 entering the codebase for the first
   time. An LLM asked to fill a schema will happily invent a plausible age. The schema must distinguish
   *stated*, *negated* ("no history of diabetes" — clinically meaningful, not absence), and *absent*.
   Conflating negated with absent is the most common and most damaging bug in this phase.
4. Use constrained/structured decoding (JSON schema or function-calling) rather than parsing free text.
5. Build queries from the profile and evaluate as a **separate BM25 run** against Phase 3. Also try
   HyDE-style expansion — generate a hypothetical trial description from the profile and retrieve with
   that — and measure it. Query expansion via LLM is well-supported on this task [Q1][T2].

**Decide.** Whether extraction feeds retrieval, reasoning, or both (recommended: both, but ablate the
retrieval half — it may not help, and knowing that is a result).

**Deliverable.** `src/extraction/`, extracted profiles for all 125 topics, a BM25+profile run.

**Exit criterion.** Hand-audit 10 topics against the narratives: zero invented values, and
negation handled correctly in every case where the narrative contains a negation.

**Reading.** HyDE [Q1] for the expansion idea; Wornow et al. [E2] for how far careful prompting alone
gets you on clinical eligibility text; IELab [T2] for LLM-generated synthetic patient descriptions used
as training data for this exact collection. Full citations in [reading-list.md](reading-list.md).

## Deployment decision: reversed — Gemini API, round-robin, not local-only

**Superseded.** The original decision below ("fully local, no API") held through the corpus, harness,
and lexical-retrieval phases. For Phase 4 extraction it has been **explicitly reversed**: the primary
backend is now the **Google Gemini API** (`gemini-3.6-flash`), called through `src/extraction/gemini.py`
with **round-robin rotation across 3 API keys** (one key per call, advancing regardless of success —
spreads load across each key's own rate limit rather than only failing over on error). Full record in
`docs/decisions/phase4-gemini-backend.md`.

**The trade-off, stated plainly, not buried.** Patient narratives now leave the local machine and are
sent to Google's API — the opposite of the rationale below. This is judged acceptable *here* because the
TREC 2021/2022 topics are public de-identified benchmark narratives, not real patient data. **This
decision must be revisited, not silently inherited, before it is extended to Phase 8 or Phase 9** (which
process the same kind of text) or before any deployment against real patient input — at that point the
local-only argument regains its original force. `src/extraction/ollama.py` was deleted outright
(2026-08-30, user's explicit call — Ollama is no longer a maintained fallback in this repo). Reverting
to a local backend means restoring it from git history (`git log --all --follow -- src/extraction/ollama.py`)
or rewriting it against `gemini.py`'s `chat_json()` interface, not a one-line import swap anymore.

**Two things learned empirically while wiring this up, worth recording so they aren't re-discovered:**

1. **The model named in an approved plan can stop existing between planning and execution.**
   `gemini-2.5-flash` (the original choice) returned `404 ... no longer available to new users` when
   actually called (August 2026) — Google's own error pointed at `gemini-3.6-flash`, which was then
   verified empirically (schema round-trip, negation handling, grounding) before being adopted. Verify
   model availability by calling it, not by trusting a name written down earlier.
2. **A round-robin of 3 keys does not multiply throughput if the keys share a quota pool.** The observed
   429 (`generate_content_free_tier_requests`, limit 5/minute) was hit and exhausted across all 3 keys
   within the same call — strong evidence the 3 keys sit under one Google Cloud project's free-tier quota,
   not three independent ones. Round-robin still has value (key redundancy, spreading load if the keys
   *are* ever split across projects) but should not be assumed to give 3× throughput. `gemini.py` honors
   the `retryDelay` Gemini's own 429 response returns rather than retrying immediately — a fixed short
   sleep would either under-wait (still 429) or over-wait (real capacity sitting idle).

### Original rationale (kept for context — no longer the operative decision for Phase 4)

Every generative call in this project (Phases 4, 8, 9) was designed to run on local Ollama. For
patient-to-trial matching this was **the medically defensible design, not a budget compromise**: the
input is a clinical narrative, and a system where patient text never leaves the machine is what a
hospital could actually deploy. This reasoning still applies directly to Phase 8/9 until explicitly
revisited there.

The cost was a real ceiling on model capability, and the benchmark below is what measured that ceiling
instead of guessing at it — **historical context now**, since Gemini's selection for Phase 4 was a direct
decision rather than a result of this benchmark methodology (worth naming as a methodology deviation).

### Hardware constraints (measured, not estimated)

| | |
|---|---|
| GPU | RTX 4060 Laptop, **8,188 MiB** — the hard ceiling on every model choice |
| torch | 2.13.0+cu130, CUDA available, sm_89, bf16 supported |
| Narratives | 125 topics, mean 135 words, max 218 — fits every candidate with no truncation |

### Model roster — six candidates, 75 narratives each

**Historical — designed for the local-only decision above, superseded before it was run.** Phase 4's
actual backend (Gemini, see the reversed decision above) was a direct choice, not the output of this
benchmark. Kept here because it's the right methodology *if* Ollama becomes the backend again (e.g. for
Phase 8/9, where local-only still applies).

Model choice is an experiment, not a guess. 6 × 75 = **450 calls**, cheap enough that the table is
simply run rather than argued about.

| Model | Size | Question it answers |
|---|---|---|
| `qwen2.5:3b-instruct` | 1.9 GB | the floor — smallest still usable |
| `qwen3-vl:4b-instruct` | 3.3 GB | does the VL variant cost anything on pure text |
| `qwen3:4b-instruct` | 2.5 GB | text-only 4B — fair control against the VL line |
| **`medgemma:4b`** | 3.3 GB | **does biomedical pretraining beat general at equal size** |
| `qwen2.5:7b-instruct` | 4.7 GB | is the jump in size worth it |
| `qwen3:8b` | 5.2 GB | the ceiling of 8 GB VRAM |

`medgemma:4b` is the interesting one: it asks, for 75 calls, the same question Phase 5 warns about for
embeddings — *the biomedical-beats-general assumption is frequently false and is worth testing rather
than assuming.*

**Seconds per call is a first-class measurement, not a footnote.** Phase 8 needs **27,045 calls** for
the dev set (75 topics × top-20 trials × **18.0 criteria/trial**, measured on `runs/bm25_best.dev.txt` —
not the corpus mean of 13.3). A model 3% better but twice as slow is the wrong choice at that volume,
and this table is where that gets caught.

Determinism: `temperature=0, top_p=1, seed=0` on every call. This reduces sampling noise so the table
measures models rather than variance; it does not make Ollama bit-reproducible (GPU reduction order),
and the write-up should say so rather than imply otherwise.

### Three schema conventions, and where they deviate from the sketch above

1. **`status` is `present` | `negated` only — there is no `absent` value.** Absence is expressed by the
   item not appearing in the list, because you cannot cite evidence for something never mentioned.
2. **Omission, not `null`.** This deviates from step 3's wording above. Allowing `null` yields both
   `{"age": null}` and `{"age": {"value": null}}` and destroys the distinction between "looked and
   found nothing" and "did not look". Absence has exactly one representation.
3. **`negated` is a per-entity status, not a separate top-level list.** A negation belongs next to the
   concept it negates, with the negating phrase as its evidence.

**Known gap:** the schema sketch above names a `stage` field; the implemented schema does not have one.
Staging is load-bearing for oncology trials and should be added as a scalar field before the full run.

## Status: COMPLETE (2026-08-30)

Full write-up in Vietnamese: `docs/phase4-tong-ket.md`.

`src/extraction/` — `schema.py` (JSON Schema + prompt + batch variants), `gemini.py` (round-robin API
client), `verify.py` (mechanical checks + gold labels), `extract.py` (CLI + cache keyed on
`prompt_hash`, batches 5 patients/call), `query.py` (profile → BM25 query, three variants, HyDE
batched + disk-cached, `--hyde-model` separable from `--model`). All 75 dev-set profiles extracted,
**all three query runs recorded** — `results/bm25_{prof,prof_narr,hyde}.dev.json`.

**Headline result.** `prof_narr` is the best lexical query yet (official nDCG@10 0.4528 vs 0.3859,
p=0.0001). But normalizing contamination by judged coverage shows **all three** extraction-based
variants sit *above* the raw-narrative baseline (0.427 / 0.396 / 0.399 vs 0.335) — the Phase 3 finding
reproduced at the query layer, on an independent axis. `hyde` is **undetermined, not refuted**: best
ranking among judged documents (condensed nDCG@10 0.3876, bpref 0.2609 — highest of all four runs) but
judged@10 of only 0.4813, so this collection cannot separate "finds what the pool missed" from
"topic drift into unjudged trials".

**The 6-model *Ollama* roster was superseded, but the comparison it called for was run — in cloud
form.** The original roster chose among local models under an 8 GB VRAM ceiling; that deployment
decision was reversed mid-phase, so the roster no longer applied. It was replaced by a 3-model Gemini
comparison over all 75 dev topics (identical prompt, schema, and `--batch-size 5` — only the model
varies), scored by the same `verify.py` measures. Full table in
`docs/phase4-tong-ket.md` §6.5.

| model | schema | ground | age | sex | cover | neg (26 real) | s/call | Phase 8 |
|---|---|---|---|---|---|---|---|---|
| `gemini-3.6-flash` | 100% | 99.9% | 100% | 100% | **16.9** | **92%** | 3.87s | 29.1h |
| **`gemini-3.5-flash-lite`** ← selected | 100% | 99.3% | 100% | 100% | 13.0 | 58% | **1.36s** | **10.2h** |
| `gemini-3.1-flash-lite` | 100% | 98.7% | *93.3%* ❌ | *93.2%* ❌ | 8.9 | 50% | 1.33s | 10.0h |

**`gemini-3.5-flash-lite` is the selected backend** (`gemini.MODEL`) — chosen for its independent
per-model quota and 2.8× speed, which are what make Phase 8's 27,045 calls feasible at all. This is a
**deliberate quality trade-off, not a claim the models are equivalent**: its negation recall is 58%
against 3.6-flash's 92%, measured over the 26 dev topics carrying a real negation. Retrieval is
unaffected (§7.4 — no significant difference on any of the three query variants, because negated terms
are deliberately excluded from queries anyway), so **the cost lands entirely on Phase 8**, where
`negated` is a required input for concluding `satisfied` on an exclusion criterion. Phase 8 must
measure this directly rather than inherit Phase 4's verdict; `data/profiles/2021.gemini-3.6-flash.json`
is retained so reverting costs no re-extraction.

Two findings worth carrying: grounding barely separates the models (99.9 / 99.3 / 98.7) because
**extracting less makes grounding easier** — `cover` and `neg` are the discriminating columns. And
quality is **not monotonic in model size**: on the `2021_14` fabrication trap both Lite models cited
`"She"` (valid evidence for the patient's sex) where `3.6-flash` cited `"Daughter"` (invalid — having a
daughter does not establish sex).

**Free gold labels, validated before use.** Age and sex sit in the first sentence of a narrative and
regex recovers them precisely: **age 75/75, sex 74/75** on the dev topics. The single miss is
`2021_14` (*"70 y/o with COPD…"*), where the narrative genuinely states no sex — so it is not a regex
gap but a **fabrication trap**: any model that emits a sex there is inventing, caught at zero
annotation cost.

**Three fixes forced by a 2-narrative smoke test**, each of which would have silently corrupted the
benchmark:

1. The 3B model returned `sex: "male"` with `evidence: ""` — schema-valid, and a correct value thrown
   away by the grounding check. Fixed with `minLength: 3` on evidence.
2. **The grounding measure was gameable.** The model quoted an entire paragraph as evidence for each
   field; that is a literal substring, so it scores 100% while localizing nothing. Split into two
   columns: `grounding` (substring) and `localized` (≤ 30 words).
3. Field misassignment is rampant at 3B (`hypertension` filed as a biomarker, `bicuspid aortic valve`
   as a lab) — a coverage number says nothing about whether the fields mean anything.

**What mechanical verification cannot catch — read before trusting the number.** Substring checking
confirms the *evidence* is real. It does not confirm the *name* follows from it. Observed at 3B on
topic 2021_2:

```
name:     "hypertrophic cardiomyopathy"
evidence: "left ventricular hypertrophy with cavity dilation and severe global hypokinesis"
```

The evidence is verbatim; the narrative never says hypertrophic cardiomyopathy. The model **diagnosed
instead of extracting** — precisely what invariant 2 forbids — and the check waved it through.
`grounding` is therefore necessary but not sufficient, which is exactly why the hand-audit exists and
why it cannot be automated away with the data on hand.

**Revised exit criterion** (the original "hand-audit 10 topics" is raised to 25, since it is cheap):

1. Six-model table carrying all six measures: schema validity, grounding, **localized**, age/sex
   accuracy, coverage, **seconds/call** + projected Phase 8 GPU hours.
2. Selected model: schema ≥ 95%, grounding ≥ 90%, age/sex ≥ 95%.
3. Hand-audit 25 narratives: **zero invented values surviving the mechanical check**, negation labelled
   `negated` in every case.
4. All three query runs (`prof`, `prof_narr`, `hyde`) recorded in `results/` — **including if they lose
   to Phase 3**. Extraction is lossy and the narrative is already information-rich; losing here is a
   legitimate result, and Phase 4's real value is its contribution to Phase 8.

Query ablation must run on the Phase 3 winning configuration (`indexes/bm25-critfields`, k1=1.8, b=1.0).
Changing the index or the parameters changes two things at once and the delta becomes unattributable.

**Negation is excluded from queries but kept in the profile.** Feeding a negated term to BM25 retrieves
exactly the trials about the disease the patient does *not* have — the mirror image of the negation trap
found at Phase 3. Phase 8 still needs those terms to conclude `satisfied` on an exclusion criterion.

---
[← specs index](README.md) · prev: [3 — Lexical retrieval](03-lexical-retrieval.md) · next: [5 — Dense retrieval](05-dense-retrieval.md)
