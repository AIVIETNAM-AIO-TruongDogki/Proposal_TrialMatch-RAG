[← specs index](README.md) · Phase 2 of 11 · prev: [1 — Corpus construction](01-corpus-construction.md) · next: [3 — Lexical retrieval](03-lexical-retrieval.md)

# Phase 2 — Evaluation harness (build this before any retriever)

**Goal.** A single command that takes a run file and prints every metric the project will ever report.

**Steps.**
1. Adopt the TREC run-file format (`topic_id Q0 nct_id rank score run_tag`) and use a standard scorer
   (`pytrec_eval` or `ir_measures`). Do not hand-roll nDCG.
2. Implement **two metric families**, and understand why both are needed:

   *Official measures*, for comparability with published work: nDCG@10, P@10, Recall@1000, MRR — using
   TREC's graded gains (eligible = 2, excluded = 1).

   *Eligibility-aware measures*, for the project's actual claim:
   - **nDCG@10 (eligible-only)** — gain 1 for `eligible`, gain **0** for `excluded` *and* `not relevant`.
   - **Exclusion contamination@K** — the fraction of the top-K that is `qrel == 1`. Lower is better.
   - **Eligible-P@10** — precision counting only `qrel == 2` as correct.

   **This distinction is the single most important thing in the harness.** Under the official measure, a
   medically relevant but exclusion-triggering trial earns *positive* gain. That is exactly the failure
   case the proposal exists to eliminate. If you evaluate the headline claim with official nDCG alone,
   a successful eligibility filter will look like a regression. Report both, always, side by side.
3. Add a significance test — paired bootstrap or a paired t-test over per-topic scores. With 50 topics,
   differences below roughly 0.02 nDCG are noise, and the ablation ladder will produce several
   differences in exactly that range.
4. Log every run as JSON: run tag, git commit, config hash, per-topic scores, aggregate scores.

**Decide.** Primary headline metric (recommended: nDCG@10 official for the literature comparison,
eligible-only nDCG@10 for the contribution claim).

**Deliverable.** `src/eval/` + `results/` with a `python -m src.eval.score run.txt` entry point.

**Exit criterion.** The harness reproduces a known-good number: score a trivial run and confirm the
scorer is wired correctly by checking a hand-computed nDCG@10 for one topic against the tool's output.

## Status: BUILT AND PASSING

`src/eval/` — `data.py` (year-namespaced topics/qrels + label remap), `metrics.py` (both families),
`sig.py` (paired bootstrap), `run_io.py` (TREC run files + result log), `score.py` (CLI),
`verify.py` (the exit-criterion check). Scoring uses `pytrec_eval`; nDCG is not hand-rolled.

Run `python -m src.eval.verify` — all checks pass:

| Check | Result |
|---|---|
| official nDCG@10 vs hand-computed | exact to 1e-9 ✅ |
| eligible-only nDCG@10 vs hand-computed | exact to 1e-9 ✅ |
| contamination@10 vs hand count | exact ✅ |

**Two conventions verified empirically rather than assumed**, because both were guessable wrong:
`trec_eval`/`pytrec_eval` nDCG uses **linear gain** (`gain = rel`), *not* `2^rel - 1`; and
`P`, `recall`, `recip_rank` are **binary at `rel > 0`**, so an `excluded` trial counts as a *hit*.

### The trap, measured on real 2021 qrels

`verify.py` builds an adversarial run that ranks **every excluded trial first** — the precise failure
this project exists to eliminate — and scores it against the real 75-topic qrels:

| Measure | Adversarial | Random | Oracle |
|---|---|---|---|
| **official** nDCG@10 | **0.520** | 0.214 | 0.998 |
| **official** P@10 | **1.000** | — | 0.995 |
| **official** MRR | **1.000** | — | 1.000 |
| eligible-only nDCG@10 | **0.039** | 0.133 | 1.000 |
| contamination@10 | **0.951** | 0.171 | 0.000 |

A system that is 95% wrong scores **P@10 = 1.000 and MRR = 1.000** under the official measures, and its
nDCG@10 of 0.520 is not far off a mediocre real system (published TREC 2022 SOTA is ≈0.693 [E3]). Worse:
**random ranking beats it on every eligibility-aware measure while losing badly on the official ones.**
The official metric prefers a systematically wrong system over chance.

This is the concrete justification for the dual metric families, and it is worth a paragraph in the
final write-up (see [research edge §8.2](research-edge.md)).

**Reference bounds are checked into `results/`.** `_oracle`, `_random`, and `_adversarial` give every
future run a frame: a BM25 nDCG@10 means little alone, but "0.31, against oracle 1.00 and random 0.13"
does.

**Caveat to record now.** TREC qrels are *pooled* — only trials retrieved by 2022 participant systems
were judged. Unjudged trials count as non-relevant. Recall@1000 over a 375k corpus is therefore a lower
bound, and a system that finds genuinely relevant unpooled trials is silently penalized. Note this in the
final write-up rather than discovering it during it.

**Reading.** The TREC overview papers [T1] for the official measure definitions and pool depth. Full
citations in [reading-list.md](reading-list.md).

---
[← specs index](README.md) · prev: [1 — Corpus construction](01-corpus-construction.md) · next: [3 — Lexical retrieval](03-lexical-retrieval.md)
