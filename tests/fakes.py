"""Shared in-memory fakes. No network, no API keys."""

from __future__ import annotations

import time


class FakeKV:
    """Just enough of redis.asyncio.Redis for RevocationStore and LoginThrottle."""

    def __init__(self) -> None:
        self.data: dict[str, tuple[str, float | None]] = {}
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("redis down")

    def _live(self, name: str) -> str | None:
        item = self.data.get(name)
        if item is None:
            return None
        value, expires = item
        if expires is not None and expires <= time.monotonic():
            del self.data[name]
            return None
        return value

    async def set(self, name, value, ex=None):
        self._check()
        self.data[name] = (str(value), time.monotonic() + ex if ex else None)
        return True

    async def get(self, name):
        self._check()
        return self._live(name)

    async def exists(self, *names):
        self._check()
        return sum(self._live(n) is not None for n in names)

    async def incr(self, name):
        self._check()
        value = int(self._live(name) or 0) + 1
        expires = self.data.get(name, (None, None))[1]
        self.data[name] = (str(value), expires)
        return value

    async def expire(self, name, seconds):
        self._check()
        if name in self.data:
            self.data[name] = (self.data[name][0], time.monotonic() + seconds)
        return True

    async def delete(self, *names):
        self._check()
        for n in names:
            self.data.pop(n, None)
        return len(names)

    async def ping(self):
        self._check()
        return True
