"""Scheduler: priority = visit probability / cost, a hard in-flight cap and a byte-capped, distance-aware LRU."""

from realmweaver.world import Scheduler


def test_never_more_than_max_in_flight_before_done():
    s = Scheduler(max_in_flight=2, cache_bytes=1 << 30)
    for i in range(5):
        s.submit((i, 0), priority=0.5)
    first, second = s.next(), s.next()
    assert first is not None and second is not None and s.in_flight == 2
    assert s.next() is None  # cap reached
    s.done(first, nbytes=10)
    assert s.in_flight == 1 and s.next() is not None and s.next() is None


def test_higher_priority_is_returned_first():
    s = Scheduler(cache_bytes=1 << 30)
    s.submit((0, 1), priority=0.1)
    s.submit((5, 5), priority=0.9)
    s.submit((1, 0), priority=0.4)
    assert s.next() == (5, 5) and s.next() == (1, 0)


def test_evicts_the_farthest_lru_key_once_bytes_exceed_the_cap():
    s = Scheduler(cache_bytes=30)
    for key in [(0, 0), (1, 0), (9, 9), (2, 0)]:
        s.done(key, nbytes=10)  # 40 bytes cached, cap is 30
    assert s.evict_if_needed(current=(0, 0)) == [(9, 9)]
    assert s.bytes_used == 30 and s.evict_if_needed(current=(0, 0)) == []


# --- beyond the plan ---------------------------------------------------------------------------------


def test_priority_is_divided_by_cost():
    s = Scheduler(cost_s=1.0, cache_bytes=1 << 30)
    s.submit((0, 0), priority=0.9, cost_s=3.0)  # score 0.3
    s.submit((1, 1), priority=0.4)  # score 0.4
    assert s.next() == (1, 1)


def test_resubmission_replaces_the_score_and_running_or_cached_keys_are_ignored():
    s = Scheduler(cache_bytes=1 << 30)
    s.submit((0, 0), 0.9)
    s.submit((1, 1), 0.5)
    s.submit((0, 0), 0.1)  # the latest prediction wins
    assert s.next() == (1, 1)
    s.submit((1, 1), 1.0)  # in flight: ignored
    assert s.pending == 1
    s.done((1, 1), 1)
    s.submit((1, 1), 1.0)  # cached: ignored
    assert s.pending == 1 and (1, 1) in s and (0, 0) not in s


def test_touch_makes_a_key_most_recently_used_among_equal_distance():
    s = Scheduler(cache_bytes=15)
    s.done((2, 0), 10)
    s.done((0, 2), 10)  # both two chunks from the origin; (2, 0) is the older one
    s.touch((2, 0))
    assert s.evict_if_needed((0, 0)) == [(0, 2)]


def test_the_current_chunk_is_never_evicted():
    s = Scheduler(cache_bytes=5)
    s.done((0, 0), 10)
    assert s.evict_if_needed((0, 0)) == [] and s.bytes_used == 10
    s.done((1, 0), 10)
    assert s.evict_if_needed((0, 0)) == [(1, 0)] and s.bytes_used == 10


def test_pending_requests_beyond_the_horizon_are_pruned():
    s = Scheduler(cache_bytes=1 << 30)
    s.submit((9, 9), 0.9)
    s.submit((1, 0), 0.1)
    s.evict_if_needed(current=(0, 0))
    assert s.pending == 1 and s.next() == (1, 0)


def test_abort_frees_the_in_flight_slot_without_caching():
    s = Scheduler(max_in_flight=1, cache_bytes=1 << 30)
    s.submit((0, 0), 0.5)
    key = s.next()
    assert s.next() is None
    s.abort(key)
    assert s.in_flight == 0 and key not in s and s.pending == 0


def test_done_on_an_unsubmitted_key_caches_it_and_redoing_updates_bytes():
    s = Scheduler(cache_bytes=1 << 30)
    s.done((3, 3), 100)
    s.done((3, 3), 40)
    assert s.bytes_used == 40 and (3, 3) in s and s.in_flight == 0
