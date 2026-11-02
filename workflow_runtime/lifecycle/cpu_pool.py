"""Invocation-scoped, bounded CPU work with ordered results and spawn isolation."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from itertools import islice
import multiprocessing
import os
from queue import Empty


_WORKER_EVENTS = None


def available_cpu_workers() -> int:
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        count = os.cpu_count() or 1
    return max(1, min(16, count))


def _initialize_cpu_worker(events) -> None:
    global _WORKER_EVENTS
    _WORKER_EVENTS = events
    # Never inherit a GPU backend's thread fan-out in a CPU shard.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _progress_task(payload):
    token, function, item = payload
    events = []
    def progress(event, fields):
        message = (token, len(events), event, fields)
        events.append(message)
        _WORKER_EVENTS.put(message)
    return function(item, progress=progress), events


class CPUWorkPool:
    """Lazily start one persistent pool; retain at most two tasks per worker.

    A single task executes locally. Failures or abandoned iteration cancel the
    remaining tasks and stop children before returning to lifecycle cleanup.
    """

    def __init__(self, workers: int | None = None) -> None:
        self.workers = available_cpu_workers() if workers is None else workers
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("CPU worker count must be a positive integer")
        self._executor = None
        self._closed = False
        self._events = None
        self._generation = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(abort=exc_type is not None)

    @property
    def closed(self) -> bool:
        return self._closed

    def map(self, function, items, *, on_progress=None):
        if self._closed:
            raise RuntimeError("CPU work pool is closed")
        iterator = iter(items)
        first = tuple(islice(iterator, 2))
        if not first:
            return
        if self.workers == 1 or len(first) == 1:
            from threadpoolctl import threadpool_limits
            # Limit only numerical decision work, never a backend invocation.
            for item in first:
                with threadpool_limits(limits=1):
                    result = function(item) if on_progress is None else function(item, progress=on_progress)
                yield result
            for item in iterator:
                with threadpool_limits(limits=1):
                    result = function(item) if on_progress is None else function(item, progress=on_progress)
                yield result
            return
        if self._executor is None:
            context = multiprocessing.get_context("spawn")
            self._events = context.Queue(maxsize=1024)
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=context,
                initializer=_initialize_cpu_worker,
                initargs=(self._events,),
            )
        pending = deque()
        generation = self._generation
        self._generation += 1
        submitted = 0
        seen = set()
        def publish(message):
            token, ordinal, event, fields = message
            key = (token, ordinal)
            if on_progress is not None and token[0] == generation and key not in seen:
                on_progress(event, fields)
                seen.add(key)
        def drain():
            while True:
                try:
                    publish(self._events.get_nowait())
                except Empty:
                    return
        def submit(item):
            nonlocal submitted
            token = (generation, submitted)
            submitted += 1
            if on_progress is None:
                return self._executor.submit(function, item)
            return self._executor.submit(_progress_task, (token, function, item))
        try:
            for item in first:
                pending.append(submit(item))
            for item in islice(iterator, 2 * self.workers - len(first)):
                pending.append(submit(item))
            while pending:
                future = pending.popleft()
                if on_progress is None:
                    result = future.result()
                else:
                    while True:
                        drain()
                        try:
                            result, events = future.result(timeout=0.05)
                            break
                        except FutureTimeout:
                            if future.done():
                                # Propagate a TimeoutError raised by the task.
                                result, events = future.result()
                                break
                    drain()
                    # Queue delivery can lag behind a Future's result. Replay
                    # only missing events, keeping one complete parent-owned log.
                    for event in events:
                        publish(event)
                yield result
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                pending.append(submit(item))
            drain()
        except BaseException:
            for future in pending:
                future.cancel()
            self.close(abort=True)
            raise

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is None:
            return
        executor, self._executor = self._executor, None
        if abort:
            # Python 3.10 has no terminate_workers API. Stop running work as
            # well as queued futures; the executor then drains its own queues.
            processes = tuple((executor._processes or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
        executor.shutdown(wait=True, cancel_futures=abort)
        self._events.close()
        self._events.join_thread()
