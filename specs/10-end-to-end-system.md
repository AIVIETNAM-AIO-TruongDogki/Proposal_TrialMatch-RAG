[← specs index](README.md) · Phase 10 of 11 · prev: [9 — Evidence-grounded generation](09-evidence-grounded-generation.md) · next: [11 — Ablation study & write-up](11-ablation-writeup.md)

# Phase 10 — End-to-end system

**Goal.** A running prototype. Deliberately small.

**Steps.** FastAPI endpoint (`POST /match` → narrative in, ranked annotated trials out); a thin UI or
notebook demo; a `docker-compose` bringing up the lexical engine, vector store, and API; caching so a
demo does not re-run the pipeline; latency measured end to end.

**Decide.** Nothing new. Every component choice was made in an earlier phase. **Resist re-opening them
here** — the pull to "just try one more model" during integration week is what causes week 8 to
disappear.

**Deliverable.** Running system, README with setup steps.

**Exit criterion.** A cold start on a fresh machine produces a correct result for one sample narrative.

---
[← specs index](README.md) · prev: [9 — Evidence-grounded generation](09-evidence-grounded-generation.md) · next: [11 — Ablation study & write-up](11-ablation-writeup.md)
