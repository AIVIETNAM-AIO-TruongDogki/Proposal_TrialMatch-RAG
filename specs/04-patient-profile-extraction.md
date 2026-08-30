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

## Deployment decision: fully local, no API

Every generative call in this project (Phases 4, 8, 9) runs on local Ollama. For patient-to-trial
matching this is **the medically defensible design, not a budget compromise**: the input is a clinical
narrative, and a system where patient text never leaves the machine is what a hospital could actually
deploy. Record it in the write-up as a deliberate design choice.

The cost is a real ceiling on model capability, and the benchmark below is what measures that ceiling
instead of guessing at it.

### Hardware constraints (measured, not estimated)

| | |
|---|---|
| GPU | RTX 4060 Laptop, **8,188 MiB** — the hard ceiling on every model choice |
| torch | 2.13.0+cu130, CUDA available, sm_89, bf16 supported |
| Narratives | 125 topics, mean 135 words, max 218 — fits every candidate with no truncation |

### Model roster — six candidates, 75 narratives each

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

## Status: BUILT, NOT YET BENCHMARKED

`src/extraction/` — `schema.py` (JSON Schema + prompt), `ollama.py` (HTTP to localhost, no new
dependency — `requests` ships with pyserini), `verify.py` (mechanical checks + gold labels),
`extract.py` (CLI + cache keyed on `prompt_hash`), `query.py` (profile → BM25 query, three variants).
883 lines, running end to end. The 6-model benchmark has **not** been run yet.

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
