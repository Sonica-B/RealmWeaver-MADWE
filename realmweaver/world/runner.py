"""Runner: the seam that decides on which thread a job runs. `World` submits its generations here and keeps the
in-flight model; two adapters sit at the seam: `InlineRunner` runs each job on the calling thread before `submit`
returns (tests, the CLI), `ThreadRunner` runs one worker thread per slot over a priority queue, player-facing jobs
ahead of prewarm. A generator is called by at most `slots` threads at a time, so the diffusion adapter takes one."""

from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, Literal, Protocol, TypeVar

T = TypeVar("T")
Priority = Literal["player", "prewarm"]
_RANK: dict[Priority, int] = {"player": 0, "prewarm": 1}  # queue order: a player's request pre-empts prewarm
# A queued job: rank, arrival order, the job and its handle; a job of None tells a worker to stop.
_Item = tuple[int, int, Callable[[], Any] | None, Future[Any] | None]


class Runner(Protocol):
    """`slots` is how many jobs run at once; `submit` returns the job's handle (a `concurrent.futures.Future`)."""

    slots: int

    def submit(self, priority: Priority, fn: Callable[[], T]) -> Future[T]: ...


class InlineRunner:
    """One slot on the calling thread: the job has run, and its handle is done, when `submit` returns."""

    slots = 1

    def __init__(self) -> None:
        self._slot = threading.Lock()  # callers on several threads still reach the generator one at a time

    def submit(self, priority: Priority, fn: Callable[[], T]) -> Future[T]:
        handle: Future[T] = Future()
        with self._slot:
            try:
                handle.set_result(fn())
            except Exception as exc:
                handle.set_exception(exc)
        return handle


class ThreadRunner:
    """One daemon worker thread per slot over a priority queue, FIFO within a priority. `slots=1` for a generator
    that must stay on one thread (the diffusion pipeline); the procedural generator takes two."""

    def __init__(self, slots: int = 2) -> None:
        self.slots = slots
        self._queue: queue.PriorityQueue[_Item] = queue.PriorityQueue()
        self._order = itertools.count()
        self._workers = [
            threading.Thread(target=self._work, name=f"realmweaver-slot-{i}", daemon=True)
            for i in range(slots)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, priority: Priority, fn: Callable[[], T]) -> Future[T]:
        handle: Future[T] = Future()
        self._queue.put((_RANK[priority], next(self._order), fn, handle))
        return handle

    def close(self) -> None:
        """Run what is queued, then stop the workers; `submit` after this is never run."""
        for _ in self._workers:
            self._queue.put((max(_RANK.values()) + 1, next(self._order), None, None))
        for worker in self._workers:
            worker.join()

    def _work(self) -> None:
        while True:
            _, _, fn, handle = self._queue.get()
            if fn is None or handle is None:
                return
            if handle.set_running_or_notify_cancel():
                try:
                    handle.set_result(fn())
                except Exception as exc:
                    handle.set_exception(exc)
