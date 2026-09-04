"""Phase 10 — trang thai dung chung cho toan bo tien trinh API.

Dung DUNG MOT LAN trong `app.py`'s lifespan, khong phai moi request — day
chinh la diem khac biet voi cac batch script Phase 3-8, von coi moi lan chay
la doc lap va load lai tu dau.
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
        # check_same_thread=False: cac buoc pipeline chay qua asyncio.to_thread(),
        # tuc la tren nhieu thread khac nhau — xem docstring cua open_db().
        self.conn = store.open_db(DB_PATH, check_same_thread=False)

        # "May sach" o tieu chi thoat cua specs/10 khong noi "co GPU" — tu nhan
        # dien thay vi gia dinh cuda luon co, de cold-start khong chet tren mot
        # may khong GPU. Ma hoa mot cau ngan bang model 0.6B tren CPU van chi
        # mat vai giay, khong dang ke so voi 10s+ da ton cho moi loi goi Gemini.
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dense = LiveDenseIndex(DENSE_VECS, DENSE_MODEL, device=dev)

        # Tra truoc chi phi khoi dong JVM cua pyserini (~17s do thuc te tren
        # may dev) NGAY luc khoi dong server, khong phai o request dau tien
        # cua nguoi dung that.
        bm25.search(query.BEST_INDEX, {"__warmup__": "warmup"},
                   query.BEST_K1, query.BEST_B, depth=1)

        self.quota = QuotaGuard(QUOTA_PATH, daily_cap=daily_cap)
        self.concurrency = ConcurrencyGuard(concurrency)
        print(f"AppState san sang: device={dev}, daily_cap={daily_cap}, "
              f"concurrency={concurrency}")
