"""Phase 10 — orchestrates `/match`: 7 steps, streaming an event as each one finishes.

Never blocks the whole request on the slowest call. Measured on real data
(`data/reasoning/*.json`, 1,499 real records, Phase 8): median 4.35s, max
72.98s per trial-mode call — blocking would mean a blank screen for up to
73s with no signal. `trial_result` fires as soon as EACH trial finishes
reasoning, not after the final ranking.

Every step reuses functions already built in Phase 3-9 (see specs/10's
"Decide: Nothing new" and the approved plan) — this is only an orchestration
layer, no new algorithm.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncIterator

from src.corpus import store
from src.extraction import extract, gemini, query
from src.generation import render
from src.reasoning import aggregate, reason
from src.retrieval import bm25, fusion

LIVE_TOP_N = 5             # separate from reason.TOP_N=20 (research) — a UX/latency knob
MAX_NARRATIVE_CHARS = 4000  # GET+EventSource limit, capped deliberately
RRF_K = 60                  # winning config from Phase 6
RRF_WEIGHTS = [0.5, 0.5]    # w_lex — winning config from Phase 6
AGG_RULE = "strict"         # winning rule, F1=0.6222 — Phase 8


async def run_match(state, narrative: str) -> AsyncIterator[tuple[str, dict]]:
    """Async generator: yields (event_name, payload) per step.

    A single trial's failure (`trial_error`) doesn't cancel the whole request
    — matches the per-item error handling already in `reason.main()`.
    Extraction failure (step 1) falls back to the raw narrative as the query
    rather than aborting, exactly like `query.main()` does for a topic with a
    failed profile.

    The `done` event's payload carries `gemini_calls` — the caller (app.py)
    uses this for `quota.commit()`, since it's the ACTUAL call count,
    different from the estimate used at `quota.reserve()`.
    """
    request_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    n_gemini_calls = 0

    yield "stage", {"stage": "extracting"}
    clean, dropped, _meta = await asyncio.to_thread(extract.extract_one, narrative)
    n_gemini_calls += 1
    yield "profile", {"profile": clean, "dropped": dropped}

    yield "stage", {"stage": "retrieving"}
    q_text = query.build_query(clean, narrative, "prof_narr") if clean else narrative

    lex_run, dense_scores = await asyncio.gather(
        asyncio.to_thread(bm25.search, query.BEST_INDEX, {request_id: q_text},
                          query.BEST_K1, query.BEST_B, 1000),
        asyncio.to_thread(state.dense.query, narrative, 1000),
    )
    fused = fusion.rrf([lex_run, {request_id: dense_scores}], k=RRF_K,
                       weights=RRF_WEIGHTS)[request_id]
    ranked = sorted(fused, key=lambda d: (-fused[d], d))[:LIVE_TOP_N]
    yield "candidates", {"nct_ids": ranked}

    async def reason_one(nct: str) -> tuple[str, list[dict], int]:
        crit = store.get_criteria(state.conn, nct)
        if not crit:
            return nct, [], 0
        kept, _meta = await asyncio.to_thread(
            reason.run_batch_trial, gemini.MODEL, request_id, nct, crit,
            narrative, state.conn, False, {}, reason.MAX_CRIT_PER_CALL)
        calls = -(-len(crit) // reason.MAX_CRIT_PER_CALL)
        return nct, kept, calls

    tasks = [asyncio.create_task(reason_one(nct)) for nct in ranked]
    decisions_by: dict[str, list[dict]] = {}
    for coro in asyncio.as_completed(tasks):
        try:
            nct, kept, calls = await coro
        except Exception as e:  # noqa: BLE001 — one bad trial must not cancel the whole request
            yield "trial_error", {"message": str(e)}
            continue
        n_gemini_calls += calls
        decisions_by[nct] = kept
        trial = store.get_trial(state.conn, nct)
        if trial is None:
            continue
        score = aggregate.trial_score(kept, rule=AGG_RULE)
        yield "trial_result", render.trial_card(trial, kept, score)

    final_order = sorted(decisions_by,
                         key=lambda n: -aggregate.trial_score(decisions_by[n], rule=AGG_RULE))
    yield "done", {"ranked": final_order, "elapsed_seconds": round(time.time() - t0, 1),
                   "gemini_calls": n_gemini_calls}
