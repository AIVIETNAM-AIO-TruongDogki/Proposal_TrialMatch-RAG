"""Phase 10 — dieu phoi `/match`: 7 buoc, phat su kien ngay khi tung buoc xong.

Khong chan ca request doi loi cham nhat. Do luong that tren
`data/reasoning/*.json` (1.499 ban ghi that, Phase 8): trung vi 4,35s, toi da
72,98s moi lan goi theo trial — chan cung nghia la nguoi dung nhin man hinh
trang trong toi 73 giay khong mot dau hieu nao. `trial_result` phat NGAY khi
tung trial suy luan xong, khong doi xep hang cuoi cung.

Moi buoc tai dung nguyen ham da co san tu Phase 3-9 (xem specs/10's "Decide:
Nothing new" va ke hoach da duyet) — day chi la lop dieu phoi, khong them
thuat toan moi.
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

LIVE_TOP_N = 5             # tach khoi reason.TOP_N=20 (nghien cuu) — tham so UX/do tre
MAX_NARRATIVE_CHARS = 4000  # gioi han GET+EventSource, ghi ro co chu dich
RRF_K = 60                  # cau hinh da chon o Phase 6
RRF_WEIGHTS = [0.5, 0.5]    # w_lex — cau hinh da chon o Phase 6
AGG_RULE = "strict"         # luat gop thang cuoc, F1=0.6222 — Phase 8


async def run_match(state, narrative: str) -> AsyncIterator[tuple[str, dict]]:
    """Async generator: yield (ten_su_kien, payload) theo tung buoc.

    Loi MOT trial (`trial_error`) khong huy ca request — khop hanh vi bat loi
    tung item da co san o `reason.main()`. Loi trich xuat (buoc 1) LUI VE benh
    an tho lam truy van thay vi dung han, giong het `query.main()` da lam khi
    ho so trich xuat hong cho mot topic.

    `payload` cua su kien `done` mang `gemini_calls` — nguoi goi (app.py) dung
    con so nay de `quota.commit()`, vi day la so LOI GOI THAT SU, khac voi uoc
    tinh dung luc `quota.reserve()`.
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
        except Exception as e:  # noqa: BLE001 — mot trial hong khong duoc huy ca request
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
