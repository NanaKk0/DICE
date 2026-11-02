from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading
import time
from typing import Callable, Iterator, Mapping, Sequence
import warnings

from workflow_runtime.lifecycle.environment import OperationContext
from workflow_runtime.lifecycle.progress.operation import (
    OperationProgressReporter,
    ProgressProbe,
)


@dataclass(frozen=True, slots=True)
class OperationResult:
    role: str
    executable: str
    returncode: int
    stdout_path: Path
    stderr_path: Path
    stdout: str
    stderr: str
    structured_result: str | None = None


class OperationError(RuntimeError):
    def __init__(self, result: OperationResult) -> None:
        detail = result.stderr.strip()
        suffix = f"; stderr: {detail[-1200:]}" if detail else ""
        super().__init__(f"lifecycle operation {result.role} exit {result.returncode}; diagnostics: {result.stderr_path}{suffix}")
        self.result = result


PROCESS_GROUP_GRACE_SECONDS = 10.0


class ProcessGroupRegistry:
    """Track attempt-owned process groups and cancel them as one fail-fast unit."""

    def __init__(self, *, grace_seconds: float = PROCESS_GROUP_GRACE_SECONDS) -> None:
        if grace_seconds <= 0:
            raise ValueError("lifecycle process-group grace interval must be positive")
        self.grace_seconds = float(grace_seconds)
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self._cancelled = False

    def register(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            cancelled = self._cancelled
            if not cancelled:
                self._processes[process.pid] = process
        if cancelled:
            self._cancel_late_process(process)

    def unregister(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._processes.pop(process.pid, None)

    def cancel_all(self) -> None:
        with self._lock:
            self._cancelled = True
            processes = tuple(self._processes.values())
        live = [process for process in processes if process.poll() is None]
        for process in live:
            self._signal(process, signal.SIGTERM)
        deadline = time.monotonic() + self.grace_seconds
        while live and time.monotonic() < deadline:
            live = [process for process in live if process.poll() is None]
            if live:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        # A worker may have exited on TERM while a native child in its owned
        # process group remains alive. Do not use the leader's poll() to skip
        # final group cleanup (GCMC workers launch LAMMPS in this group).
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=self.grace_seconds)
            except OSError:
                pass

    def _cancel_late_process(self, process: subprocess.Popen[bytes]) -> None:
        self._signal(process, signal.SIGTERM)
        try:
            process.wait(timeout=self.grace_seconds)
            return
        except subprocess.TimeoutExpired:
            self._signal(process, signal.SIGKILL)
        try:
            process.wait()
        except OSError:
            pass

    @staticmethod
    def _signal(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except OSError:
            pass


_OPERATION_ENVIRONMENT_OVERRIDES: ContextVar[Mapping[str, str]] = ContextVar(
    "dice_operation_environment_overrides", default={}
)
_OPERATION_PROCESS_REGISTRY: ContextVar[ProcessGroupRegistry | None] = ContextVar(
    "dice_operation_process_registry", default=None
)


def current_operation_environment_overrides() -> dict[str, str]:
    """Return the immutable execution-scope overlay visible to native substeps."""

    return dict(_OPERATION_ENVIRONMENT_OVERRIDES.get())


def current_operation_process_registry() -> ProcessGroupRegistry | None:
    return _OPERATION_PROCESS_REGISTRY.get()


@contextmanager
def operation_execution_scope(
    *,
    environment_overrides: Mapping[str, str],
    process_registry: ProcessGroupRegistry,
) -> Iterator[None]:
    """Keep one member's device and cancellation ownership stable across substeps."""

    environment_token = _OPERATION_ENVIRONMENT_OVERRIDES.set(
        dict(environment_overrides)
    )
    registry_token = _OPERATION_PROCESS_REGISTRY.set(process_registry)
    try:
        yield
    finally:
        _OPERATION_PROCESS_REGISTRY.reset(registry_token)
        _OPERATION_ENVIRONMENT_OVERRIDES.reset(environment_token)


def structured_result_stdout(result: OperationResult, *, prefix: str) -> str:
    """Present one bounded result to existing backend-owned parsers.

    The fallback keeps injected test executors compatible while production
    `OperationResult` values use the explicit bounded result field.
    """
    value = getattr(result, "structured_result", None)
    return result.stdout if value is None else f"{prefix}{value}\n"


def run_operation(
    context: OperationContext,
    logical_executable: str,
    arguments: Sequence[str],
    *,
    cwd: str | Path,
    diagnostic_name: str,
    structured_result_prefix: str | None = None,
    structured_result_limit_bytes: int = 1024 * 1024,
    progress_reporter: OperationProgressReporter | None = None,
    progress_probe: ProgressProbe | None = None,
    probe_interval_seconds: float = 1.0,
    heartbeat_interval_seconds: float = 60.0,
    close_progress_reporter: bool = True,
    environment_overrides: Mapping[str, str] | None = None,
    process_registry: ProcessGroupRegistry | None = None,
    input_producer: Callable[[subprocess.Popen[bytes]], None] | None = None,
    timeout_seconds: float | None = None,
) -> OperationResult:
    """Execute one backend-native command with its isolated activation context."""
    work_dir = Path(cwd)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not diagnostic_name or "/" in diagnostic_name or "\\" in diagnostic_name:
        raise ValueError("lifecycle diagnostic name must be one filename stem")
    if structured_result_prefix is not None and (not structured_result_prefix or "\n" in structured_result_prefix):
        raise ValueError("lifecycle structured result prefix must be one non-empty line prefix")
    if structured_result_limit_bytes <= 0:
        raise ValueError("lifecycle structured result limit must be positive")
    if probe_interval_seconds <= 0.0 or heartbeat_interval_seconds <= 0.0:
        raise ValueError("lifecycle progress probe and heartbeat intervals must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0.0:
        raise ValueError("lifecycle operation timeout must be positive")
    if progress_probe is not None and progress_reporter is None:
        raise ValueError("lifecycle progress probe requires an operation-scoped reporter")
    executable = context.executable(logical_executable)
    sources = " ".join(f"source {shlex.quote(item['path'])};" for item in context.source_scripts)
    # The external profile may tune only an allowlisted allocator.  It is set
    # after activation so overlays cannot accidentally replace it, and we do
    # not project arbitrary profile values into the child environment.
    allocator = context.runtime.get("tf_gpu_allocator")
    allocator_export = "" if allocator is None else f"export TF_GPU_ALLOCATOR={shlex.quote(str(allocator))};"
    scoped_overrides = current_operation_environment_overrides()
    scoped_overrides.update(environment_overrides or {})
    override_exports = ""
    for name, value in scoped_overrides.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"lifecycle operation {context.role!r} has invalid environment override name {name!r}")
        override_exports += f"export {name}={shlex.quote(str(value))};"
    # Native worker modules execute from their lifecycle-local work root. Keep
    # the repository package importable without making a stage path or profile
    # variable a hidden workflow input.
    repo_root = Path(__file__).resolve().parents[1]
    module_path_export = f"export PYTHONPATH={shlex.quote(str(repo_root))}${{PYTHONPATH:+:${{PYTHONPATH}}}};"
    command = f"set -e; {sources} {allocator_export} {override_exports} {module_path_export} exec {shlex.quote(executable)} {' '.join(shlex.quote(str(item)) for item in arguments)}"
    stdout_path = work_dir / f"{diagnostic_name}.stdout.log"
    stderr_path = work_dir / f"{diagnostic_name}.stderr.log"
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=work_dir,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        effective_registry = (
            process_registry
            if process_registry is not None
            else _OPERATION_PROCESS_REGISTRY.get()
        )
        if effective_registry is not None:
            effective_registry.register(process)
        producer_error: list[BaseException] = []
        producer_thread: threading.Thread | None = None
        if input_producer is not None:
            def produce_input() -> None:
                try:
                    input_producer(process)
                except BaseException as exc:
                    producer_error.append(exc)
                    _terminate_process_group(process)

            producer_thread = threading.Thread(
                target=produce_input,
                name=f"dice-input-{diagnostic_name}",
                daemon=True,
            )
            producer_thread.start()
        try:
            returncode = _wait_with_progress(
                process,
                reporter=progress_reporter,
                probe=progress_probe,
                probe_interval_seconds=probe_interval_seconds,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                close_reporter=close_progress_reporter,
                timeout_seconds=timeout_seconds,
            )
        finally:
            if producer_thread is not None:
                producer_thread.join(timeout=5.0)
                if producer_thread.is_alive():
                    _terminate_process_group(process)
                    producer_thread.join(timeout=5.0)
                    if not producer_error:
                        producer_error.append(
                            TimeoutError(
                                "lifecycle operation input producer did not terminate"
                            )
                        )
            if effective_registry is not None:
                effective_registry.unregister(process)
    if producer_error:
        raise producer_error[0]
    structured_result = (
        None
        if structured_result_prefix is None
        else _read_bounded_prefixed_result(
            stdout_path,
            prefix=structured_result_prefix,
            limit_bytes=structured_result_limit_bytes,
        )
    )
    result = OperationResult(
        context.role,
        executable,
        returncode,
        stdout_path,
        stderr_path,
        _read_text_tail(stdout_path),
        _read_text_tail(stderr_path),
        structured_result,
    )
    if returncode:
        raise OperationError(result)
    return result


def _read_text_tail(path: Path, *, limit_bytes: int = 8192) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - limit_bytes))
            return handle.read(limit_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _read_bounded_prefixed_result(path: Path, *, prefix: str, limit_bytes: int) -> str:
    prefix_bytes = prefix.encode("utf-8")
    line_limit = len(prefix_bytes) + limit_bytes
    current = bytearray()
    oversized = False
    oversized_result = False
    matches: list[bytes] = []

    def finish_line() -> None:
        nonlocal current, oversized, oversized_result
        if oversized:
            oversized_result = oversized_result or current.startswith(prefix_bytes)
        elif current.startswith(prefix_bytes):
            matches.append(bytes(current[len(prefix_bytes) :]))
        current = bytearray()
        oversized = False

    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            for value in chunk:
                if value == 10:
                    finish_line()
                    continue
                if oversized:
                    continue
                current.append(value)
                if len(current) > line_limit:
                    oversized_result = current.startswith(prefix_bytes)
                    oversized = True
    if current or oversized:
        finish_line()
    if oversized_result:
        raise ValueError(f"lifecycle structured worker bounded result exceeds {limit_bytes} bytes")
    if len(matches) != 1:
        raise ValueError("lifecycle structured worker must emit exactly one bounded result line")
    if not matches[0]:
        raise ValueError("lifecycle structured worker bounded result is empty")
    return matches[0].decode("utf-8", errors="strict")


def _wait_with_progress(
    process: subprocess.Popen[bytes],
    *,
    reporter: OperationProgressReporter | None,
    probe: ProgressProbe | None,
    probe_interval_seconds: float,
    heartbeat_interval_seconds: float,
    close_reporter: bool,
    timeout_seconds: float | None,
) -> int:
    started = time.monotonic()
    next_probe = started
    next_heartbeat = next_probe + heartbeat_interval_seconds
    last_token: str | None = None
    warned = False

    def warn_once(exc: Exception) -> None:
        nonlocal warned
        if warned:
            return
        warned = True
        warnings.warn(f"lifecycle operation progress degraded: {exc}", RuntimeWarning, stacklevel=3)

    def poll_probe() -> None:
        nonlocal last_token
        if probe is None or reporter is None:
            return
        try:
            update = probe.poll()
            if update is not None:
                if update.warning:
                    warn_once(ValueError(update.warning))
                if update.semantic_token != last_token:
                    last_token = update.semantic_token
                    reporter.report(update.snapshot)
        except Exception as exc:
            warn_once(exc)

    try:
        while True:
            returncode = process.poll()
            now = time.monotonic()
            if now >= next_probe:
                poll_probe()
                next_probe = now + probe_interval_seconds
            if reporter is not None and now >= next_heartbeat:
                try:
                    reporter.heartbeat()
                except Exception as exc:
                    warn_once(exc)
                next_heartbeat = now + heartbeat_interval_seconds
            if returncode is not None:
                poll_probe()
                return returncode
            if timeout_seconds is not None and now - started >= timeout_seconds:
                _terminate_process_group(process)
                raise TimeoutError(
                    f"lifecycle operation process timed out after {timeout_seconds:g} seconds"
                )
            time.sleep(min(probe_interval_seconds, 0.1))
    except BaseException:
        _terminate_process_group(process)
        raise
    finally:
        if reporter is not None and close_reporter:
            reporter.close()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        process.wait()
