"""Phase 10 — guard the Gemini quota AND concurrency for the live web demo.

The demo is a NEW consumer on the same key pool (`GEMINI_API_KEY_*`) already
serving Phase 4-8. Without a guard, a demo open to the internet could
silently eat quota the research pipeline still needs — a real risk flagged
in the plan, not a nice-to-have.

v1 is deliberately simple: a daily cap in a file (survives restarts) + an
in-memory per-IP/hour cap + an in-process concurrency limit. No Redis, no
external queue — enough for demo traffic, not real traffic.
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
    """Daily cap (file) + per-IP/hour cap (memory).

    `reserve()`/`commit()` are NOT atomically locked across the two calls —
    with a concurrency limit of 2 (see ConcurrencyGuard) and a daily cap
    generous relative to one request's cost (~6 calls), the race window
    between two near-simultaneous requests is negligible for v1. Add a real
    lock if this ever runs multiple workers.
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
        # Atomic write — same tmp+fsync+replace pattern as reason.save_cache,
        # so the counter is never truncated mid-write if the process dies.
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
        """Call BEFORE running the pipeline. False -> reject immediately, no calls spent."""
        self._roll_day()
        if not self._ip_ok(ip):
            return False, "vuot gioi han request/gio cho demo nay — thu lai sau"
        if self._used + estimated_calls > self.daily_cap:
            return False, "demo da het ngan sach Gemini hom nay, quay lai ngay mai"
        return True, None

    def commit(self, ip: str, actual_calls: int) -> None:
        """Call AFTER the pipeline finishes, with the ACTUAL number of calls used."""
        self._roll_day()
        self._used += actual_calls
        self._ip_hits[ip].append(time.time())
        self._save()


class ConcurrencyGuard:
    """Limits concurrent /match requests — rejects IMMEDIATELY instead of queuing.

    Not a plain asyncio.Semaphore: Semaphore.acquire() BLOCKS until a slot
    frees up, but this needs to REJECT immediately (429) once the limit is
    hit — a real queue would be overkill for v1 (see "Scope" in the plan).
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
