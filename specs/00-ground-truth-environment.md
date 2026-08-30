[← specs index](README.md) · Phase 0 of 11 · next: [1 — Corpus construction](01-corpus-construction.md)

# Phase 0 — Ground truth and environment

**Goal.** Have the benchmark labels in hand and know exactly what the corpus contains, before writing
any pipeline code.

**Status: the benchmark data is already in place and verified.** Both years of topics and qrels are in
`rawdata/`, and both join cleanly against the local corpus:

| File | Content | Verified |
|---|---|---|
| `rawdata/topics2021.xml` | 75 patient narratives, `number` 1–75 | ✅ |
| `rawdata/qrels2021.txt` | 35,832 judgments, 75 topics | ✅ 25/25 sampled NCT IDs resolve |
| `rawdata/topics2022.xml` | 50 patient narratives, `number` 1–50 | ✅ |
| `rawdata/qrels2022.txt` | 35,394 judgments, 50 topics | ✅ 20/20 sampled NCT IDs resolve |

Both tracks use the **same April 27, 2021 ClinicalTrials.gov snapshot**, which is why the 2021 labels
join against the same `rawdata/` tree. Source: <https://trec.nist.gov/data/trials/qrels2021.txt>.

**Label semantics** — confirmed verbatim from NIST ("Judgment of 0 is non-relevant, 1 is excluded, and
2 is eligible"):

| Label | Meaning | 2021 | 2022 |
|---|---|---|---|
| `2` | eligible | 5,570 (15.5%) | 3,939 (11.1%) |
| `1` | medically relevant **but excluded** | 6,019 (16.8%) | 3,036 (8.6%) |
| `0` | not relevant | 24,243 (67.7%) | 28,419 (80.3%) |

**⚠️ Two traps in this data. Both are silent — nothing crashes, the numbers just come out wrong.**

**Trap 1 — topic IDs collide across years.** Both files number their topics from 1. Topic `1` of 2021 is
a 45-year-old man with anaplastic astrocytoma; topic `1` of 2022 is a 19-year-old man with a sexual-health
concern. Loading both into one structure keyed on the raw number silently merges two unrelated patients.
**Namespace every topic ID at load time** (`2021_1`, `2022_1`) and never let the bare integer out of the
loader.

**Trap 2 — the two years are not distributionally interchangeable.** The 2022 set is markedly sparser in
positives, and critically the `excluded` rate is roughly **half** that of 2021 (8.6% vs 16.8%). Since
`excluded` is the discriminating case this whole project turns on, any absolute score threshold tuned on
2021 will behave differently on 2022. **Tune rank-based parameters (cut-offs, fusion `k`, candidate
depth), not absolute score thresholds.** If a component genuinely needs a score threshold, calibrate it
per-topic rather than globally.

**Steps.**
1. ~~Download topics and qrels.~~ **Done** — see the table above.
2. Confirm the local snapshot matches the benchmark corpus. Count only trial files —
   `find rawdata -name 'NCT*.xml' | wc -l` gives **375,580**. Note that a bare `find rawdata -name '*.xml'`
   returns 375,582, because `topics2021.xml` and `topics2022.xml` also live under `rawdata/`; the two
   extra files are the topics, not stray trials. **Verified.**
3. **Build the `nct_id -> path` index** — a dict mapping each trial ID to its file location on disk, since
   qrels reference trials by ID only and the 375,580 files are spread across 5 directories × ~100 buckets.
   One `os.walk` builds it in ~1s; the 48,714 distinct IDs across both qrels files then resolve in
   ~0.05s. Doing it with one `find` per ID instead costs ~0.067s each, i.e. **~54 minutes** — and Phase 1
   needs far more lookups than that. Use the index to widen the join check to the full qrels set (the
   25-sample check already passed); any ID that fails to resolve shows up as a `None` for free.
   Paths *are* derivable by rule — the bucket is always `NCT` + the first 4 digits + `xxxx`, verified
   across all 375,580 files with 0 exceptions — but the rule needs a hardcoded part-range table and the
   directory names are off by one (`fixed` = part1). The index is built from what is actually on disk, so
   it cannot drift from reality.
4. Record the label semantics as a named constant, not a magic number. `1` meaning *excluded* rather than
   *partially relevant* is the single most consequential fact in the dataset and it reads backwards
   against every other TREC collection.
5. **Lead-time item:** if you intend to use the n2c2 2018 Track 1 cohort-selection data as a second
   evaluation for the reasoning module ([research edge](research-edge.md)), start the data-use agreement
   **this week**. Access approval takes longer than any other item in this plan.
6. Pin the environment: `pyproject.toml` / `uv.lock`, a fixed random seed, and a `results/` directory
   with one JSON per experiment run.

**Decide.** Dev/test split policy — **2021 (75 topics) = dev, 2022 (50 topics) = held-out test**, touched
once at the end. This is now viable because the 2021 labels are present; it was not before. Also decide
whether n2c2 is in scope.

**Deliverable.** `data/topics/`, `data/qrels/`, an `nct_id -> path` index, an environment lockfile, a
one-page corpus audit note.

**Exit criterion.** 100% of distinct NCT IDs across both qrels files resolve to a file in `rawdata/`
(the 45-sample check already passed; confirm it at full scale), and the per-topic label distribution is
tabulated for both years.

**Reading.** TREC CT track overviews [T1]; the survey [S1] for orientation on the field as a whole. Full
citations in [reading-list.md](reading-list.md).

---
[← specs index](README.md) · next: [1 — Corpus construction](01-corpus-construction.md)
