from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


@dataclass(frozen=True, slots=True)
class RankProcessReceipt:
    rank: int
    pid: int
    state: str
    return_code: int | None


@dataclass(frozen=True, slots=True)
class CoordinatedProcessResult:
    outcome: str
    receipts: tuple[RankProcessReceipt, ...]


def run_coordinated_processes(
    *,
    commands: tuple[tuple[str, ...], ...],
    cwd: str | Path,
    timeout_seconds: float,
    cancel_event: threading.Event | None = None,
    terminate_grace_seconds: float = 1.0,
) -> CoordinatedProcessResult:
    if not commands or any(not command for command in commands):
        raise ValueError("coordinated launch requires non-empty rank commands")
    if timeout_seconds <= 0 or terminate_grace_seconds <= 0:
        raise ValueError("coordinated launch bounds must be positive")
    root = Path(cwd).resolve()
    if not root.is_dir():
        raise ValueError("coordinated launch cwd must exist")
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for command in commands:
            processes.append(
                subprocess.Popen(
                    list(command),
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            )
    except BaseException:
        _terminate_owned(processes, grace_seconds=terminate_grace_seconds)
        raise

    started = time.monotonic()
    outcome = "succeeded"
    while True:
        return_codes = [process.poll() for process in processes]
        if all(code is not None for code in return_codes):
            outcome = "succeeded" if all(code == 0 for code in return_codes) else "failed"
            break
        if any(code not in (None, 0) for code in return_codes):
            outcome = "failed"
            break
        if cancel_event is not None and cancel_event.is_set():
            outcome = "cancelled"
            break
        if time.monotonic() - started >= timeout_seconds:
            outcome = "timed_out"
            break
        time.sleep(0.02)

    if outcome != "succeeded":
        _terminate_owned(processes, grace_seconds=terminate_grace_seconds)
    for process in processes:
        if process.poll() is None:
            process.wait()
    receipts = tuple(
        RankProcessReceipt(
            rank=rank,
            pid=process.pid,
            state=(
                "succeeded"
                if process.returncode == 0 and outcome == "succeeded"
                else outcome
            ),
            return_code=process.returncode,
        )
        for rank, process in enumerate(processes)
    )
    return CoordinatedProcessResult(outcome=outcome, receipts=receipts)


def _terminate_owned(
    processes: list[subprocess.Popen[bytes]],
    *,
    grace_seconds: float,
) -> None:
    live = [process for process in processes if process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while live and time.monotonic() < deadline:
        live = [process for process in live if process.poll() is None]
        if live:
            time.sleep(0.02)
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
