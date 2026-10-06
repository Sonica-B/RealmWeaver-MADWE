"""Scheduler: a priority queue of chunk requests (priority = visit probability / cost), a hard cap on in-flight
generations, and a byte-capped LRU cache whose eviction takes the farthest chunk first."""

from __future__ import annotations

from collections import OrderedDict

Key = tuple[int, int]

HORIZON = 2  # pending requests farther than this many chunks from the player are dropped


def _distance(a: Key, b: Key) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class Scheduler:
    def __init__(self, cost_s: float = 1.0, max_in_flight: int = 2, cache_bytes: int = 512 << 20) -> None:
        self.cost_s, self.max_in_flight, self.cache_bytes = cost_s, max_in_flight, cache_bytes
        self._pending: dict[Key, float] = {}  # key -> score
        self._in_flight: set[Key] = set()
        self._cached: OrderedDict[Key, int] = OrderedDict()  # key -> bytes, least recently used first
        self.bytes_used = 0

    def submit(self, key: Key, priority: float, cost_s: float | None = None) -> None:
        """Queue `key` at score priority / cost, `cost_s` being the measured chunk cost (the constructor's prior when
        None); a resubmission replaces the score, a cached or running key is ignored."""
        if key not in self._cached and key not in self._in_flight:
            self._pending[key] = priority / (self.cost_s if cost_s is None else cost_s)

    def next(self) -> Key | None:
        """Hand out the best pending key and count it in flight; None when nothing is pending or the cap is reached."""
        if len(self._in_flight) >= self.max_in_flight or not self._pending:
            return None
        key = max(self._pending, key=self._pending.__getitem__)
        del self._pending[key]
        self._in_flight.add(key)
        return key

    def done(self, key: Key, nbytes: int) -> None:
        """Record `key` as cached with `nbytes`, most recently used; it leaves the in-flight set and the queue."""
        self._in_flight.discard(key)
        self._pending.pop(key, None)
        self.bytes_used += nbytes - self._cached.pop(key, 0)
        self._cached[key] = nbytes

    def abort(self, key: Key) -> None:
        """Free the in-flight slot of a generation that failed; the key is neither cached nor re-queued."""
        self._in_flight.discard(key)

    def touch(self, key: Key) -> None:
        if key in self._cached:
            self._cached.move_to_end(key)

    def evict_if_needed(self, current: Key) -> list[Key]:
        """Drop pending requests beyond the horizon, then evict until under the byte cap: farthest from `current`
        first, least recently used among equals, never `current` itself. Returns the evicted keys."""
        for key in [k for k in self._pending if _distance(k, current) > HORIZON]:
            del self._pending[key]
        evicted: list[Key] = []
        # ponytail: every eviction rescans the whole cache (O(n) per victim, O(n^2) when the cap forces many out);
        # a heap over (distance, age) maintained by `touch` and `done` is the upgrade path for caches of thousands
        # of chunks.
        while self.bytes_used > self.cache_bytes:
            age = {k: i for i, k in enumerate(self._cached)}
            candidates = [k for k in self._cached if k != current]
            if not candidates:
                break
            victim = max(candidates, key=lambda k: (_distance(k, current), -age[k]))
            self.bytes_used -= self._cached.pop(victim)
            evicted.append(victim)
        return evicted

    @property
    def in_flight(self) -> int:
        return len(self._in_flight)

    @property
    def pending(self) -> int:
        return len(self._pending)

    def __contains__(self, key: Key) -> bool:
        return key in self._cached
