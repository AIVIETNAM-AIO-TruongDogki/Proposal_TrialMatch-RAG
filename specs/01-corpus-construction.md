[← specs index](README.md) · Phase 1 of 11 · prev: [0 — Ground truth](00-ground-truth-environment.md) · next: [2 — Evaluation harness](02-evaluation-harness.md)

# Phase 1 — Corpus construction and criteria segmentation

**Goal.** Turn 375,580 XML files into two things: a retrievable document store, and a **criterion store**
where every criterion is individually addressable with an offset back to its source text.

This is the phase most likely to be underestimated. The proposal already flags it: eligibility criteria
arrive as one unstructured blob and splitting them is a real parsing problem.

**Steps.**
1. **Parse defensively.** Root is `<clinical_study>`; nearly every field is optional and many repeat.
   Extract: `id_info/nct_id`, `brief_title`, `brief_summary/textblock`, `detailed_description/textblock`,
   `condition*`, `intervention/intervention_name*`, `keyword*`, `condition_browse/mesh_term*`,
   `eligibility/{gender,minimum_age,maximum_age,healthy_volunteers}`, `eligibility/criteria/textblock`,
   `overall_status`, `phase`, `study_type`. Never assume presence.
2. **Normalize text blocks.** Strip the literal `&#xD;` carriage returns, then un-wrap hard line breaks
   *without* destroying bullet boundaries — the wrapping and the bullets both use newlines, so the order
   of operations matters. Get this wrong and every criterion downstream is malformed.
3. **Normalize ages** to a numeric field plus a flag: `"14 Years" → 14.0`, `"N/A" → None`. Keep the raw
   string too.
4. **Segment criteria.** Recommended algorithm:
   - locate section headers by case-insensitive regex covering the real variants seen in the wild
     (`Inclusion Criteria:`, `INCLUSION CRITERIA`, `Key Inclusion Criteria`, `Eligibility Criteria`);
   - inside a section, split on bullet markers (`-  `, `*`, `•`, `1.`, `(1)`);
   - treat a line that does not start with a marker as a continuation of the previous criterion;
   - **fallback** when neither headers nor bullets are found: sentence-split and label the section
     `unknown`. Do not silently drop these trials — a meaningful fraction of the corpus has no
     conventional formatting at all.
5. **Store character offsets.** Each criterion record keeps `(start, end)` into the normalized textblock.
   These offsets are what make invariant 3 (every claim is grounded) *verifiable* rather than merely
   asserted — in Phase 9 you can check that a quoted span is a literal substring of the source.
6. **Measure the parser.** Report: % of trials yielding ≥1 inclusion criterion, % yielding ≥1 exclusion
   criterion, % falling back to sentence-splitting, and the criterion-length distribution. Then hand-check
   a random sample of 50 trials and record the error modes.

Suggested record shape:

```json
{
  "nct_id": "NCT00000102",
  "title": "...",
  "summary": "...",
  "conditions": ["..."],
  "interventions": ["..."],
  "mesh_terms": ["..."],
  "gender": "All",
  "min_age_years": 14.0,
  "max_age_years": null,
  "criteria": [
    {"idx": 0, "section": "inclusion", "text": "...", "span": [412, 498]},
    {"idx": 1, "section": "exclusion", "text": "...", "span": [560, 631]}
  ],
  "criteria_parse": {"method": "bulleted", "confidence": "high"}
}
```

**Decide.** The **indexing unit**. Recommended: two stores with different jobs —
a *relevance store* (title + summary + conditions + interventions + MeSH) used for retrieval, and a
*criterion store* (keyed by `nct_id`, one row per criterion) used only by the reasoning stage. Mixing
criteria text into the retrieval document is a defensible alternative but must be ablated, not assumed:
criteria text is long, negation-heavy, and can drown the topical signal.

**Deliverable.** `data/trials.db` (SQLite), a parser-quality report, a named failure-mode taxonomy.

**Exit criterion.** ≥95% of trials that contain an `eligibility/criteria/textblock` produce at least one
segmented criterion, and the audit is written up with named failure modes.

## Status: BUILT AND PASSING

`src/corpus/` implements this phase: `schema.sql`, `parse.py` (normalize + segment + span),
`build_db.py` (parallel walk + quality report), `store.py` (read API + grounding check).
Full corpus build: **375,580 files in 94s, 0 parse failures, 2.92 GB DB.**

| Measure | Result | Target |
|---|---|---|
| Trials with an eligibility blob | 374,648 (99.8%) | — |
| Of those, ≥1 criterion segmented | **374,584 (100.0%)** | ≥95% ✅ |
| ≥1 inclusion / ≥1 exclusion | 96.6% / 94.2% | — |
| Total criteria | **4,985,262** (13.3/trial, p50 10, p95 35, max 148) | — |
| Span verification (`criteria_raw[s:e]` ≡ `text`) | **0 violations / 39,351 sampled** | 0 ✅ |

Segmentation method: bulleted 77.2%, numbered 12.7%, mixed 6.9%, sentence-split 3.1%, none 0.3%.

### Failure-mode taxonomy (the part that matters)

The headline pass rate is not the useful output — this is. Ordered by consequence, not frequency:

1. **Section headers rendered as bullets** (`-  Exclusion Criteria:`) — **found and fixed.** The damage
   was never the header itself; it was that the section never switched, so *every criterion after it*
   inherited the wrong label. 2,962 trials affected, of which **536 had no `exclusion` criterion at all**
   despite plainly having an exclusion section. In Phase 8 this inverts the conclusion — an exclusion
   criterion read as an inclusion criterion turns `violated` into `satisfied`. Fix: test a bullet's body
   against the header pattern before accepting it as a criterion. Residual after fix: 3,726 → **93**.
2. **No section headers anywhere** — 11,286 trials (3.0%), all criteria labelled `unknown`. Mostly
   genuine: the source record really is unsectioned, so `unknown` is the correct label, not a defect.
   Phase 8 must handle `unknown`-section criteria rather than assume every criterion is typed.
3. **Empty criteria blobs** (`-  0`, a bare `-`) — 64 trials. Source-data defect, not a parser defect.
   They correctly yield zero criteria.
4. **Uncuttable long criteria** (>1500 chars) — 2,430 (0.05%). Acceptable.
5. **Lowercase headers with no colon** (`inclusion criteria patients with allergic rhinitis`) —
   **deliberately not fixed.** Catching these means loosening the header regex, and a false section
   switch corrupts every subsequent label while a missed one only leaves them `unknown`. Asymmetric
   damage, so the parser stays conservative. Recorded as a known limitation.

### Two invariants now enforced mechanically, not by convention

- **Spans are verified, not asserted.** `criteria_raw` is stored, every criterion carries offsets into
  it, and `store.verify_quote()` checks an LLM's citation against the stored criterion. Invariant 3 is
  a measurable rate from Phase 8 onward, not a design promise.
- **`lead_in` is preserved.** 343,372 criteria sit under a lead-in line (e.g. `Acute onset of:` before a
  numbered list). Detached, `PaO2/FiO2 ≤ 300` loses its clinical meaning — it is *acute onset*, not a
  standing value. Dropping this field silently strips context from ~7% of all criteria.

**Note on sizing.** Parsed content is ~1.4 GB; the DB is 2.92 GB because indexes roughly double it.
Budget accordingly before adding more indexes.

**Reading.** Chia [D1] and the Leaf corpus [D2] for what a well-annotated criterion looks like and what
entity/relation structure is recoverable; Criteria2Query [D3] for the classical parse-to-query pipeline
and its published error rates — useful as a realistic expectation-setter for step 4. Full citations in
[reading-list.md](reading-list.md).

---
[← specs index](README.md) · prev: [0 — Ground truth](00-ground-truth-environment.md) · next: [2 — Evaluation harness](02-evaluation-harness.md)
