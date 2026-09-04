"""Phase 10 — bao ve han ngach Gemini VA do dong thoi cho demo web song.

Demo la nguoi tieu thu MOI tren cung pool khoa (`GEMINI_API_KEY_*`) dang phuc
vu Phase 4-8. Khong bao ve thi mot demo mo cho internet co the am tham nuot
het han ngach ma chinh pipeline nghien cuu con dang can — day la rui ro that
duoc neu ro trong ke hoach, khong phai mot tinh nang phu.

V1 co chu dich: tran ngay ghi file (song sot qua restart) + tran/IP/gio trong
bo nho + gioi han dong thoi trong tien trinh. Khong Redis, khong hang doi
ngoai — du cho luu luong demo, khong danh cho traffic that.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque

DEFAULT_DAILY_CAP = 150
DEFAULT_PER_IP_PER_HOUR = 3
DEFAULT_CONCURRENCY = 2


class QuotaGuard:
    """Tran ngay (file) + tran/IP/gio (bo nho).

    `reserve()`/`commit()` KHONG khoa nguyen tu qua hai loi goi — voi gioi han
    dong thoi 2 (xem ConcurrencyGuard) va tran ngay du du so voi chi phi mot
    request (~6 loi goi), cua so dua giua hai request chay sat nhau la khong
    dang ke cho v1. Sua thanh khoa that neu sau nay chay nhieu worker.
    """

    def __init__(self, cap_path: str, daily_cap: int = DEFAULT_DAILY_CAP,
                per_ip_per_hour: int = DEFAULT_PER_IP_PER_HOUR):
        self.cap_path = cap_path
        self.daily_cap = daily_cap
        self.per_ip_per_hour = per_ip_per_hour
        self._ip_hits: dict[str, deque] = defaultdict(deque)
        self._today, self._used = self._load()

    @staticmethod
    def _today_str() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _load(self) -> tuple[str, int]:
        today = self._today_str()
        if not os.path.exists(self.cap_path):
            return today, 0
        try:
            blob = json.load(open(self.cap_path, encoding="utf-8"))
        except json.JSONDecodeError:
            return today, 0
        if blob.get("date") != today:
            return today, 0
        return today, int(blob.get("calls_used", 0))

    def _save(self) -> None:
        # Ghi nguyen tu — dung khuon tmp+fsync+replace da dung o reason.save_cache,
        # de bo dem khong bao gio bi cat cut giua chung neu tien trinh chet.
        os.makedirs(os.path.dirname(self.cap_path) or ".", exist_ok=True)
        tmp = self.cap_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"date": self._today, "calls_used": self._used}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.cap_path)

    def _roll_day(self) -> None:
        today = self._today_str()
        if today != self._today:
            self._today, self._used = today, 0

    def status(self) -> dict:
        self._roll_day()
        return {"calls_used_today": self._used, "daily_cap": self.daily_cap,
                "remaining": max(self.daily_cap - self._used, 0)}

    def _ip_ok(self, ip: str) -> bool:
        now = time.time()
        q = self._ip_hits[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        return len(q) < self.per_ip_per_hour

    def reserve(self, ip: str, estimated_calls: int) -> tuple[bool, str | None]:
        """Goi TRUOC khi chay pipeline. False -> tu choi ngay, khong ton loi goi nao."""
        self._roll_day()
        if not self._ip_ok(ip):
            return False, "vuot gioi han request/gio cho demo nay — thu lai sau"
        if self._used + estimated_calls > self.daily_cap:
            return False, "demo da het ngan sach Gemini hom nay, quay lai ngay mai"
        return True, None

    def commit(self, ip: str, actual_calls: int) -> None:
        """Goi SAU khi pipeline chay xong, voi so loi goi THAT SU da dung."""
        self._roll_day()
        self._used += actual_calls
        self._ip_hits[ip].append(time.time())
        self._save()


class ConcurrencyGuard:
    """Gioi han so request /match dong thoi — tu choi NGAY thay vi xep hang.

    Ly do khong dung asyncio.Semaphore truc tiep: Semaphore.acquire() CHAN cho
    toi khi co cho trong, con o day muon TU CHOI ngay lap tuc (429) khi vuot
    tran — mot hang doi that la qua tay cho v1 (xem "Pham vi" trong ke hoach).
    """

    def __init__(self, limit: int = DEFAULT_CONCURRENCY):
        self.limit = limit
        self._n = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._n >= self.limit:
                return False
            self._n += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._n = max(self._n - 1, 0)
