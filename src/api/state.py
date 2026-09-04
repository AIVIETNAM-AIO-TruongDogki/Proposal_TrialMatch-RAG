"""Phase 10 — shared state for the whole API process.

Constructed EXACTLY ONCE, in `app.py`'s lifespan — not per request. This is
the real difference from the Phase 3-8 batch scripts, which treat every run
as independent and reload everything from scratch.
"""

from __future__ import annotations

import torch

from src.api.quota import ConcurrencyGuard, DEFAULT_CONCURRENCY, DEFAULT_DAILY_CAP, QuotaGuard
from src.corpus import store
from src.dense.live import LiveDenseIndex
from src.extraction import query
from src.retrieval import bm25

DENSE_VECS = "indexes/dense/qwen3.base.npz"
DENSE_MODEL = "qwen3"
DB_PATH = "data/trials.db"
QUOTA_PATH = "data/demo_quota.json"


class AppState:
    def __init__(self, daily_cap: int = DEFAULT_DAILY_CAP,
                concurrency: int = DEFAULT_CONCURRENCY, device: str | None = None):
        # check_same_thread=False: pipeline steps run via asyncio.to_thread(),
        # i.e. on multiple different threads — see open_db()'s docstring.
        self.conn = store.open_db(DB_PATH, check_same_thread=False)

        # specs/10's "clean machine" exit criterion doesn't say "has a GPU" —
        # auto-detect rather than assume cuda, so cold start doesn't crash on
        # a GPU-less machine. Encoding a short sentence with a 0.6B model on
        # CPU still only takes seconds, negligible next to the 10s+ already
        # spent per Gemini call.
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dense = LiveDenseIndex(DENSE_VECS, DENSE_MODEL, device=dev)

        # Pay pyserini's JVM startup cost (~17s measured on the dev machine)
        # NOW, at server startup, instead of on the first real user's request.
        bm25.search(query.BEST_INDEX, {"__warmup__": "warmup"},
                   query.BEST_K1, query.BEST_B, depth=1)

        self.quota = QuotaGuard(QUOTA_PATH, daily_cap=daily_cap)
        self.concurrency = ConcurrencyGuard(concurrency)
        print(f"AppState san sang: device={dev}, daily_cap={daily_cap}, "
              f"concurrency={concurrency}")
