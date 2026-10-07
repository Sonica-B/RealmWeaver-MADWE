"""Runner: the seam that decides on which thread a job runs. `World` submits its generations here and keeps the
in-flight model; two adapters sit at the seam: `InlineRunner` runs each job on the calling thread before `submit`
returns (tests, the CLI), `ThreadRunner` runs one worker thread per slot over a priority queue, player-facing jobs
ahead of prewarm. A generator is called by at most `slots` threads at a time, so the diffusion adapter takes one.

A job's handle is a `concurrent.futures.Future`: the caller may make it, so it is registered before the job can
run, or let `submit` make one. It settles with the job's result or exception, or cancelled when `close` found the
job still queued; `close` lets the running jobs finish, stops the workers, and `submit` after it settles its
handle with a RuntimeError instead of running anything."""

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
_STOP = max(_RANK.values()) + 1  # the rank of the job of None that tells a worker to stop
# A queued job: rank, arrival order, the job and its handle; a job of None tells a worker to stop.
_Item = tuple[int, int, Callable[[], Any] | None, Future[Any] | None]


class Runner(Protocol):
    """`slots` is how many jobs run at once; `submit` returns the job's handle (`handle` itself when given) and
    `close` stops the runner, settling every handle still queued."""

    slots: int

    def submit(
        self, priority: Priority, fn: Callable[[], T], handle: Future[T] | None = None
    ) -> Future[T]: ...

    def close(self) -> None: ...


def _settle(handle: Future[T], fn: Callable[[], T]) -> None:
    """Run `fn` for `handle` unless it was cancelled meanwhile; the result or the exception settles the handle."""
    if handle.set_running_or_notify_cancel():
        try:
            handle.set_result(fn())
        except Exception as exc:
            handle.set_exception(exc)


class InlineRunner:
    """One slot on the calling thread: the job has run, and its handle is done, when `submit` returns."""

    slots = 1

    def __init__(self) -> None:
        self._slot = threading.Lock()  # callers on several threads still reach the generator one at a time

    def submit(self, priority: Priority, fn: Callable[[], T], handle: Future[T] | None = None) -> Future[T]:
        handle = Future() if handle is None else handle
        with self._slot:
            _settle(handle, fn)
        return handle

    def close(self) -> None:
        """Nothing to stop: every job ran before its `submit` returned."""


class ThreadRunner:
    """One daemon worker thread per slot over a priority queue, FIFO within a priority. `slots=1` for a generator
    that must stay on one thread (the diffusion pipeline); the procedural generator takes two."""

    def __init__(self, slots: int = 2) -> None:
        self.slots = slots
        self._queue: queue.PriorityQueue[_Item] = queue.PriorityQueue()
        self._order = itertools.count()
        self._lock = threading.Lock()  # `submit` against `close`: a job is queued or refused, never lost
        self._closed = False
        self._workers = [
            threading.Thread(target=self._work, name=f"realmweaver-slot-{i}", daemon=True)
            for i in range(slots)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, priority: Priority, fn: Callable[[], T], handle: Future[T] | None = None) -> Future[T]:
        handle = Future() if handle is None else handle
        with self._lock:
            if not self._closed:
                self._queue.put((_RANK[priority], next(self._order), fn, handle))
                return handle
        if handle.set_running_or_notify_cancel():  # closed: refused, and the handle says so
            handle.set_exception(RuntimeError("ThreadRunner is closed; the job was not run"))
        return handle

    def close(self) -> None:
        """Cancel the handles of the jobs still queued, let each worker finish the job it is on, then stop the
        workers; `submit` after this settles its handle with a RuntimeError. Closing again does nothing."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            queued: list[_Item] = []
            while True:
                try:
                    queued.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            for _ in self._workers:
                self._queue.put((_STOP, next(self._order), None, None))
        for _, _, _, handle in queued:
            if handle is not None:
                handle.cancel()
        for worker in self._workers:
            worker.join()

    def _work(self) -> None:
        while True:
            _, _, fn, handle = self._queue.get()
            if fn is None or handle is None:
                return
            _settle(handle, fn)
