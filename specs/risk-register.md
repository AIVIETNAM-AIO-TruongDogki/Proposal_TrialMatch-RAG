[← specs index](README.md)

# Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Criteria segmentation is worse than expected | High | High | [Phase 1](01-corpus-construction.md) measures it explicitly; the `unknown`-section fallback keeps unparseable trials in the corpus rather than dropping them |
| LLM cost for Phase 8 exceeds budget | High | Medium | **Now measured, not estimated: 27,045 calls for the dev set** (75 × top-20 × 18.0 criteria/trial). Local-only, so the cost is GPU hours, not dollars — [Phase 4](04-patient-profile-extraction.md)'s seconds/call column projects it before [Phase 8](08-eligibility-reasoning.md) starts. Batching all criteria of one trial into a single call cuts this to **1,500 calls (18×)**, at the risk of the model losing track across 18 criteria; measure both. Cache on `(nct_id, criterion_idx, topic_id)`; cap N |
| Official nDCG hides the contribution | **Certain** | High | Dual metric families from [Phase 2](02-evaluation-harness.md); contamination@K is the headline |
| No criterion-level gold labels | High | High | Check the TrialGPT release in [Phase 0](00-ground-truth-environment.md); trial-level aggregation as guaranteed fallback |
| Dense retrieval does not beat BM25 | Medium | Low | It is a valid finding on this task; the ladder is designed to report it |
| Full-corpus encoding burns days of GPU | Medium | Medium | [Phase 5](05-dense-retrieval.md) benchmarks on a judged-plus-distractors subset before committing |
| n2c2 data-use agreement not approved in time | Medium | Low | Started in week 1; TREC alone is sufficient for the core claim |
| Integration week eats the analysis week | Medium | High | [Phase 10](10-end-to-end-system.md) forbids new component choices |

---
[← specs index](README.md)
