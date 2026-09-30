"""Adaptive concurrency for API calls: go wide, back off when the provider pushes back.

Each model (and the judge) gets one AdaptiveLimiter. It starts at the requested number of
parallel requests. When a call is rate limited or the provider is overloaded (429 / 503 /
529, after the SDK's own retries), the limit halves and the call is retried after a
jittered exponential backoff; every `limit` successes in a row raise it by one again,
up to the starting value. So a high --concurrency is safe: a provider that can't keep up
throttles that one model instead of turning a run into a pile of api_errors.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

RATE_LIMIT_RETRIES = 4  # on top of the SDK's own retries
BACKOFF_BASE_S = 10.0  # 10, 20, 40, 80 s (±50% jitter)
CUT_COOLDOWN_S = 5.0  # one burst of 429s from many in-flight calls halves the limit once
BACKPRESSURE_STATUS = {429, 503, 529}


def is_rate_limited(exc: BaseException) -> bool:
    """True for rate-limit / overload errors from any of the SDKs (or ones they wrap)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        for attr in ("status_code", "code", "status"):
            if getattr(exc, attr, None) in BACKPRESSURE_STATUS:
                return True
        if type(exc).__name__ in ("RateLimitError", "OverloadedError"):
            return True
        exc = exc.__cause__
    return False


class AdaptiveLimiter:
    """An async semaphore whose size shrinks on backpressure and grows back on success."""

    def __init__(self, limit: int):
        self.max_limit = max(1, limit)
        self.limit = self.max_limit
        self.inflight = 0
        self._cond = asyncio.Condition()
        self._streak = 0
        self._last_cut = float("-inf")

    async def __aenter__(self) -> AdaptiveLimiter:
        async with self._cond:
            await self._cond.wait_for(lambda: self.inflight < self.limit)
            self.inflight += 1
        return self

    async def __aexit__(self, *exc) -> None:
        async with self._cond:
            self.inflight -= 1
            self._cond.notify_all()

    def throttle(self) -> None:
        now = time.monotonic()
        if now - self._last_cut >= CUT_COOLDOWN_S:
            self.limit = max(1, self.limit // 2)
            self._last_cut = now
        self._streak = 0

    def succeed(self) -> None:
        if self.limit >= self.max_limit:
            return
        self._streak += 1
        if self._streak >= self.limit:
            self.limit += 1
            self._streak = 0

    async def call(self, send: Callable[[], Awaitable[T]]) -> T:
        """Run `send()` (inside a slot already held) with backoff on rate limiting.
        The slot stays held while backing off, so a throttled model drains rather than
        piling more requests onto the provider."""
        attempt = 0
        while True:
            try:
                result = await send()
            except Exception as e:
                if attempt >= RATE_LIMIT_RETRIES or not is_rate_limited(e):
                    raise
                self.throttle()
                await asyncio.sleep(BACKOFF_BASE_S * 2**attempt * random.uniform(0.5, 1.5))
                attempt += 1
            else:
                self.succeed()
                return result
