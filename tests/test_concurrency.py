import asyncio

import pytest

from scorebench import concurrency
from scorebench.concurrency import AdaptiveLimiter, is_rate_limited


class HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class GoogleAPIError(Exception):
    code = 429


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(concurrency, "BACKOFF_BASE_S", 0)


def test_is_rate_limited():
    assert is_rate_limited(HTTPError(429)) and is_rate_limited(HTTPError(529)) and is_rate_limited(GoogleAPIError())
    wrapped = RuntimeError("retries exhausted")
    wrapped.__cause__ = HTTPError(429)
    assert is_rate_limited(wrapped)
    assert not is_rate_limited(HTTPError(400)) and not is_rate_limited(RuntimeError("503 Service Unavailable"))


def test_limit_halves_once_per_burst_then_recovers():
    lim = AdaptiveLimiter(8)
    lim.throttle()
    lim.throttle()  # same burst of 429s: no second cut
    assert lim.limit == 4
    for _ in range(4):
        lim.succeed()
    assert lim.limit == 5
    for _ in range(100):
        lim.succeed()
    assert lim.limit == 8  # never above the requested concurrency


async def test_never_exceeds_limit():
    lim = AdaptiveLimiter(3)
    peak = 0

    async def work():
        nonlocal peak
        async with lim:
            peak = max(peak, lim.inflight)
            await asyncio.sleep(0.001)

    await asyncio.gather(*(work() for _ in range(20)))
    assert peak == 3 and lim.inflight == 0


async def test_call_retries_rate_limits_only():
    lim = AdaptiveLimiter(4)
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise HTTPError(429)
        return "ok"

    async with lim:
        assert await lim.call(flaky) == "ok"
    assert len(attempts) == 3 and lim.limit == 2

    async def broken():
        raise HTTPError(400)

    async with lim:
        with pytest.raises(HTTPError):
            await lim.call(broken)


async def test_call_gives_up_after_retries():
    lim = AdaptiveLimiter(4)
    attempts = []

    async def always_limited():
        attempts.append(1)
        raise HTTPError(429)

    async with lim:
        with pytest.raises(HTTPError):
            await lim.call(always_limited)
    assert len(attempts) == concurrency.RATE_LIMIT_RETRIES + 1
