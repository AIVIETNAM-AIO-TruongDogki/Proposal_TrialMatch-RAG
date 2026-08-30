[← specs index](README.md) · Phase 8 of 11 · prev: [7 — Reranking](07-reranking.md) · next: [9 — Evidence-grounded generation](09-evidence-grounded-generation.md)

# Phase 8 — Eligibility reasoning (ladder rung 5 — the contribution)

**Goal.** For each of the top-N reranked trials, label **every criterion** `satisfied` / `violated` /
`unverifiable`, with a grounded evidence span.

This is the phase the project exists for. Budget accordingly — it is the reason weeks 6 and 8 are
protected.

**Steps.**

1. **Get criterion-level labels.** TREC qrels are trial-level only; they cannot supervise or evaluate a
   criterion-level classifier. Three options, in increasing order of cost:
   - **(a)** Evaluate only at trial level: aggregate predicted criterion labels into a trial-level
     decision and compare against qrels `2/1/0`. Free, but it measures the aggregation rule as much as
     the reasoning.
   - **(b)** Check what the TrialGPT release [E1] provides — it published criterion-level annotations
     over patient cohorts including TREC-derived ones. If the licence and coverage fit, this removes the
     single largest cost in the project. **Verify this in [Phase 0](00-ground-truth-environment.md), not
     Phase 8.**
   - **(c)** Annotate your own gold set: ~20 topics × 5 trials × ~10 criteria ≈ 1,000 criteria. Expensive,
     but it is the asset that makes the three-state claim defensible.

   Recommended: (a) as the guaranteed path, (b) checked early, (c) as the stretch deliverable.

2. **Per-criterion prompt, structured output.** One criterion at a time, or small batches — not the whole
   criteria blob at once. Enforce a schema:
   ```json
   {
     "criterion_idx": 3,
     "label": "violated",
     "patient_evidence": "received cisplatin in 2021",
     "criterion_quote": "No prior platinum-based chemotherapy",
     "reasoning": "..."
   }
   ```
3. **Enforce grounding mechanically.** Reject any output where `criterion_quote` is not a literal
   substring of the stored criterion text, or `patient_evidence` is not a literal substring of the
   narrative. This converts invariant 3 from a hope into a check. Log the rejection rate — it is a
   faithfulness measurement in its own right and a reportable number.
4. **Make `unverifiable` the default.** Prompt for it explicitly: if the narrative does not state the
   information, the label is `unverifiable` — inference from typical patients is forbidden. Then test the
   invariant adversarially: take narratives with a field removed and confirm the label flips to
   `unverifiable` rather than staying `satisfied`.
5. **Aggregate to a trial score.** Define the rule explicitly and ablate it — for example:
   any `violated` exclusion ⇒ trial excluded; all inclusions `satisfied` ⇒ eligible; otherwise ranked by
   the count of `unverifiable`. The aggregation rule is a free parameter that materially affects the
   headline result, so it must be stated and varied, not buried in code.
6. **Evaluate with paired metrics.** Macro-F1 over the three labels is the proposal's metric, but on its
   own it is gameable: a model that answers `unverifiable` everywhere can score respectably while being
   useless. Always report jointly:
   - macro-F1 over `{satisfied, violated, unverifiable}`;
   - **abstention rate** (share labelled `unverifiable`) and **accuracy on the non-abstained subset**;
   - a **risk–coverage curve** if you have confidence scores.

   And run a **forced-choice ablation** — the same model with `unverifiable` removed from the label set.
   If two-state does as well, the three-state contribution is not yet demonstrated. That ablation is the
   direct empirical test of invariant 1, and reviewers will ask for it.
7. **Cost control.** N trials × M criteria × 125 topics is a large number of LLM calls. Cache aggressively
   keyed on `(nct_id, criterion_idx, topic_id)`, cap N at 20–50 for the full runs, and measure cost per
   topic — it is a reportable systems result.

**Decide.** Reasoning LLM (on reasoning quality, structured-output support, context length, cost,
reproducibility — the proposal's own criteria); N; aggregation rule.

**Deliverable.** `src/reasoning/`, criterion-level predictions, the three-state evaluation table, the
forced-choice ablation, the grounding-violation rate.

**Exit criterion.** Rung 5 is measured against rung 4 on **both** metric families from
[Phase 2](02-evaluation-harness.md) — and the central claim is stated in terms of **exclusion
contamination@10**, where the eligibility stage should show a clear reduction even if official nDCG@10
moves little or drops.

**Reading.** TrialGPT [E1] — closest prior work, criterion-level three-way prediction with published
numbers (87.3% criterion-level accuracy) and released code; Wornow et al. [E2] for zero-shot prompting
on n2c2; Jullien et al. [E3] for controlled/set-guided LLM reasoning evaluated on *this exact collection*
(nDCG@10 0.693, P@10 0.73 on TREC 2022 — a concrete target to compare against); SatIR [E4] for the
constraint-satisfaction framing. Full citations in [reading-list.md](reading-list.md).

---
[← specs index](README.md) · prev: [7 — Reranking](07-reranking.md) · next: [9 — Evidence-grounded generation](09-evidence-grounded-generation.md)
