"""Phase 10 — FastAPI: /match/stream (SSE), /quota, static frontend.

    uvicorn src.api.app:app --port 8000

Xem specs/10-end-to-end-system.md va docs/decisions/ cho boi canh thiet ke.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.api.pipeline import LIVE_TOP_N, MAX_NARRATIVE_CHARS, run_match
from src.api.state import AppState

DAILY_CAP = int(os.environ.get("DEMO_GEMINI_DAILY_CAP", "150"))
MATCH_CONCURRENCY = int(os.environ.get("DEMO_MATCH_CONCURRENCY", "2"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.trialmatch = AppState(daily_cap=DAILY_CAP, concurrency=MATCH_CONCURRENCY)
    yield


app = FastAPI(lifespan=lifespan)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@app.get("/quota")
async def quota_status(request: Request):
    return app.state.trialmatch.quota.status()


@app.get("/match/stream")
async def match_stream(request: Request, narrative: str):
    """LUON tra HTTP 200 + text/event-stream — moi loi (qua ngan sach, qua tai,
    benh an rong...) di qua SU KIEN `error` BEN TRONG luong, khong qua ma trang
    thai HTTP.

    Ly do: trinh duyet `EventSource` KHONG doc duoc body cua mot response ma
    status khac 200/khac content-type — no chi coi do la loi ket noi tran trui,
    `ev.data` rong, va nguoi dung chi thay "Connection lost." chung chung du
    server da tra dung ly do (vd 429 kem JSON {"error": "vuot gioi han..."}).
    Day la gioi han THAT cua chinh EventSource, khong phai loi giai ma phia
    client — cach sua dung la khong bao gio dua loi vao ma trang thai HTTP cho
    endpoint nay.
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
        except Exception as e:  # noqa: BLE001 — bao loi qua SSE thay vi 500 cau cam
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
        finally:
            await state.concurrency.release()

    return EventSourceResponse(event_gen())


# Mount SAU CUNG — dang ky route API truoc de chung khong bi static handler
# nuot mat (StaticFiles(html=True) phuc vu "/" bang index.html).
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
