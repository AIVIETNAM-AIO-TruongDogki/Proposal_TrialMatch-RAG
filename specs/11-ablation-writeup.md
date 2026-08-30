[← specs index](README.md) · Phase 11 of 11 · prev: [10 — End-to-end system](10-end-to-end-system.md)

# Phase 11 — Ablation study, error analysis, write-up

**Goal.** The results that answer the research question.

**Steps.**
1. Run the full ladder on the **held-out 2022 topics** — the first and only time they are touched.
   Report every rung on both metric families.
2. **Error analysis on the discriminating case.** Pull every trial with `qrel == 1` (medically relevant
   but excluded) that the system ranked in the top 10, and categorize why: parsing failure, extraction
   miss, reasoning error, or genuinely ambiguous criterion. This taxonomy is the most valuable single
   table in the report — it is direct evidence about where eligibility-aware matching actually breaks.
3. Report the negative results honestly: components that did not help, and the cost/latency each rung
   added. A clean negative result on one rung is worth more than an unexplained aggregate gain.
4. Write up against the research question as posed: *does explicit eligibility-aware retrieval and
   reasoning improve trial matching over conventional lexical or semantic retrieval?* Answer it in the
   terms the ladder measured.

**Deliverable.** Final results tables, error taxonomy, report.

**Exit criterion.** Every rung has a number on the held-out set, and the headline claim is stated with a
significance test attached.

See also: [what "done" looks like](definition-of-done.md) for the minimum publishable result, and
[research edge](research-edge.md) for where the write-up's contribution actually lives.

---
[← specs index](README.md) · prev: [10 — End-to-end system](10-end-to-end-system.md)
