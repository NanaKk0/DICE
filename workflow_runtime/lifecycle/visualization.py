from __future__ import annotations

import multiprocessing
from contextlib import ExitStack, nullcontext
import os
import signal
from pathlib import Path
from queue import Empty, Full
from typing import Final


_STOP: Final = None


def _stop_coordinator(_signum, _frame) -> None:
    # Unwind the pool before exiting so its queues and semaphore finalizers
    # run even when the public lifecycle cancels the diagnostic process group.
    raise SystemExit(0)


def _worker_main(jobs, errors) -> None:
    os.setsid()
    signal.signal(signal.SIGTERM, _stop_coordinator)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import numba

    numba.set_num_threads(1)
    from workflow_artifacts.lifecycle.visualization import (
        render_visualization_cache,
    )
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None

    from workflow_runtime.lifecycle.cpu_pool import CPUWorkPool
    with ExitStack() as stack:
        pool = stack.enter_context(CPUWorkPool())
        while True:
            job = jobs.get()
            if job is _STOP:
                return
            try:
                with threadpool_limits(limits=1) if threadpool_limits else nullcontext():
                    if len(job) == 4:
                        _kind, cache, output, payload = job
                        from workflow_core.lifecycle.visualization import visualization_from_draft
                        from workflow_core.structures.formats import read_extxyz_structures
                        from workflow_artifacts.lifecycle.visualization import publish_visualization_request
                        selected = read_extxyz_structures(Path(cache).parent/'screen/train.xyz', allow_empty=True)
                        request = visualization_from_draft(payload, [str(f.info['structure_id']) for f in selected])
                        publish_visualization_request(cache, request)
                    else:
                        cache, output = job
                    render_visualization_cache(cache, output, view_map=pool.map)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                errors.put(f"selection visualization skipped: {type(exc).__name__}: {exc}")
                if pool.closed:
                    pool = stack.enter_context(CPUWorkPool())


class VisualizationWorker:
    """One invocation-owned FIFO renderer process with warning-only failures."""

    def __init__(self) -> None:
        context = multiprocessing.get_context("spawn")
        self._jobs = context.Queue(maxsize=2)
        self._errors = context.Queue()
        self._process = context.Process(
            target=_worker_main,
            args=(self._jobs, self._errors),
            name="dice-visualization",
            daemon=False,
        )
        self._started = False
        self._closed = False
        self._exit_failure_reported = False
        self._scheduled: set[tuple[str, str]] = set()
        self._pending = {}

    def defer(self, owner: Path, draft) -> None:
        """O(1) ownership handoff; do not serialize or traverse descriptors here."""
        self._pending[str(Path(owner).resolve())] = draft

    def release_pending(self, owner: Path) -> bool:
        owner = Path(owner).resolve()
        draft = self._pending.pop(str(owner), None)
        if draft is None:
            return False
        if self._closed:
            self._errors.put('selection visualization skipped: worker is closed')
            return True
        if not self._started:
            self._process.start()
            self._started = True
        try:
            self._jobs.put_nowait(('prepare', str(owner/'.visualize_cache'), str(owner/'visualize'), draft))
        except Full:
            self._errors.put('selection visualization skipped: bounded queue is full')
        return True

    @property
    def alive(self) -> bool:
        return self._started and self._process.is_alive()

    def enqueue(self, cache_dir: str | Path, output_dir: str | Path) -> None:
        if self._closed:
            raise RuntimeError("visualization worker is closed")
        cache = Path(cache_dir).resolve()
        output = Path(output_dir).resolve()
        if not cache.is_dir():
            raise ValueError("visualization cache directory does not exist")
        job = (str(cache), str(output))
        if job in self._scheduled:
            return
        self._scheduled.add(job)
        if not self._started:
            self._process.start()
            self._started = True
        try:
            self._jobs.put_nowait(job)
        except Full:
            self._errors.put('selection visualization skipped: bounded queue is full')

    def join(self) -> tuple[str, ...]:
        if self._closed:
            return self._drain_errors()
        self._closed = True
        if self._started:
            # An explicit diagnostic join may wait, but must not deadlock when
            # the child has already died with a full queue.
            while self._process.is_alive():
                try:
                    self._jobs.put(_STOP, timeout=0.1)
                    break
                except Full:
                    continue
            self._process.join()
            if self._process.exitcode:
                self._signal_group(signal.SIGKILL)
        return self._drain_errors(report_exit_failure=True)

    def cancel(self) -> tuple[str, ...]:
        self._closed = True
        self._pending.clear()
        self._jobs.cancel_join_thread()
        if self._started:
            self._signal_group(signal.SIGTERM)
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=1)
            # The coordinator may already have exited while a view is still
            # unwinding. Its process group also owns every spawned view child.
            self._signal_group(signal.SIGKILL)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1)
        return self._drain_errors()

    def _signal_group(self, sig) -> None:
        pid = getattr(self._process, "pid", None)
        if pid is None:
            return
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass

    def _drain_errors(self, *, report_exit_failure: bool = False) -> tuple[str, ...]:
        errors: list[str] = []
        while True:
            try:
                errors.append(str(self._errors.get_nowait()))
            except Empty:
                break
        if (
            report_exit_failure
            and self._started
            and not self._exit_failure_reported
            and self._process.exitcode not in (None, 0)
        ):
            errors.append(
                "selection visualization skipped: worker exited unexpectedly "
                f"with code {self._process.exitcode}"
            )
            self._exit_failure_reported = True
        return tuple(errors)

    def __enter__(self) -> "VisualizationWorker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.join()
        else:
            self.cancel()
        return False


__all__ = ["VisualizationWorker"]
