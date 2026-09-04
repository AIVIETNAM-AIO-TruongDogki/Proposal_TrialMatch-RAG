"""Phase 10 — FastAPI backend: /match/stream (SSE), /quota. No frontend here.

    uvicorn src.api.app:app --port 8000

The static UI lives in `frontend/` and is deployed separately (its own static
host, or `python -m http.server` locally) — set `TRIALMATCH_API_BASE` in
`frontend/index.html` to point it at wherever this backend runs. CORS is open
by default (`DEMO_CORS_ORIGINS`, comma-separated) since this is a public demo
with no cookies/credentials to protect; narrow it for anything less throwaway.

See specs/10-end-to-end-system.md and docs/decisions/ for design context.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.api.pipeline import LIVE_TOP_N, MAX_NARRATIVE_CHARS, run_match
from src.api.state import AppState

DAILY_CAP = int(os.environ.get("DEMO_GEMINI_DAILY_CAP", "150"))
MATCH_CONCURRENCY = int(os.environ.get("DEMO_MATCH_CONCURRENCY", "2"))
CORS_ORIGINS = [o.strip() for o in os.environ.get("DEMO_CORS_ORIGINS", "*").split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.trialmatch = AppState(daily_cap=DAILY_CAP, concurrency=MATCH_CONCURRENCY)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=["GET"], allow_headers=["*"])


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@app.get("/")
async def health():
    return {"service": "trialmatch-rag-api", "status": "ok"}


@app.get("/quota")
async def quota_status(request: Request):
    return app.state.trialmatch.quota.status()


@app.get("/match/stream")
async def match_stream(request: Request, narrative: str):
    """ALWAYS returns HTTP 200 + text/event-stream — every error (over budget,
    overloaded, empty narrative...) goes through an `error` EVENT inside the
    stream, never an HTTP status code.

    Why: the browser's `EventSource` can't read the body of a response with
    a non-200 status or wrong content-type — it just sees a bare connection
    failure, `ev.data` is empty, and the user sees a generic "Connection
    lost." even though the server returned a proper reason (e.g. 429 with
    JSON {"error": "over budget..."}). This is a real EventSource
    limitation, not a client decoding bug — the fix is to never put an error
    into the HTTP status for this endpoint.
    """
    state: AppState = app.state.trialmatch
    ip = _client_ip(request)
    narrative = narrative.strip()

    async def event_gen():
        if not narrative:
            yield {"event": "error", "data": json.dumps({"message": "benh an dang trong"})}
            return
        if len(narrative) > MAX_NARRATIVE_CHARS:
            yield {"event": "error", "data": json.dumps(
                {"message": f"benh an vuot {MAX_NARRATIVE_CHARS} ky tu — gioi han "
                            f"co chu dich cua GET+EventSource"})}
            return

        ok, why = state.quota.reserve(ip, estimated_calls=1 + LIVE_TOP_N)
        if not ok:
            yield {"event": "error", "data": json.dumps({"message": why})}
            return

        if not await state.concurrency.try_acquire():
            yield {"event": "error", "data": json.dumps(
                {"message": "demo dang ban, thu lai sau vai giay"})}
            return
        try:
            async for name, payload in run_match(state, narrative):
                if name == "done":
                    state.quota.commit(ip, payload.get("gemini_calls", 0))
                yield {"event": name, "data": json.dumps(payload, ensure_ascii=False)}
        except Exception as e:  # noqa: BLE001 — surface the error over SSE instead of a bare 500
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
        finally:
            await state.concurrency.release()

    return EventSourceResponse(event_gen())
