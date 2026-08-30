[← specs index](README.md)

# Target repository layout

```
src/
  corpus/       # XML parsing, normalization, criteria segmentation   (Phase 1)
  eval/         # metrics, run scoring, significance tests            (Phase 2)
  retrieval/    # bm25.py, dense.py, fusion.py, rerank.py             (Phases 3,5,6,7)
  extraction/   # patient narrative -> structured profile             (Phase 4)
  reasoning/    # per-criterion three-state eligibility               (Phase 8)
  generation/   # evidence-grounded explanation                       (Phase 9)
  api/          # FastAPI                                             (Phase 10)
data/
  topics/ qrels/ processed/
runs/           # TREC-format run files
results/        # one JSON per experiment run
docs/decisions/ # one short note per recorded decision
```

Note: `documents/` and `rawdata/` are gitignored (~9 GB) — the plan lives at the repository root
(historically `IMPLEMENTATION_PLAN.md`, now split under `specs/`) so that it is version-controlled
alongside the code.

See [00 — Ground truth and environment](00-ground-truth-environment.md) through
[11 — Ablation study, error analysis, write-up](11-ablation-writeup.md) for the phase that produces
each directory.

---
[← specs index](README.md)
