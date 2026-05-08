from __future__ import annotations

import json
import math
import queue
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


RMSE_TOKEN = re.compile(r"rmse", re.IGNORECASE)
NONFINITE_TOKEN = re.compile(r"\b(?:nan|inf)\b", re.IGNORECASE)


def _ensure_text_log(path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _line_has_nonfinite_rmse(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return bool(RMSE_TOKEN.search(stripped) and NONFINITE_TOKEN.search(stripped))


def _read_recent_lcurve_lines(lcurve_path: Path, max_lines: int = 8) -> list[str]:
    if not lcurve_path.exists():
        return []
    with open(lcurve_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    return lines[-max_lines:]


def _stream_reader(stream, output_queue: queue.Queue[tuple[str, str | None]]) -> None:
    try:
        for raw_line in iter(stream.readline, ""):
            output_queue.put(("line", raw_line))
    finally:
        output_queue.put(("eof", None))


def _terminate_process(proc: subprocess.Popen[str], timeout_seconds: float) -> tuple[bool, bool]:
    terminated = False
    killed = False
    if proc.poll() is not None:
        return terminated, killed

    try:
        proc.terminate()
        terminated = True
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        killed = True
        proc.wait()
    return terminated, killed


def run_dp_train_with_watchdog(
    command: list[str],
    workdir: str | Path,
    log_file: str | Path,
    watch_log_file: str | Path,
    input_json: str | Path,
    attempt: int,
    seed: int,
    *,
    env: dict[str, str] | None = None,
    terminate_timeout_seconds: float = 10.0,
    watch_lcurve_fallback: bool = True,
    lcurve_poll_interval_seconds: float = 1.5,
    quiet: bool = False,
) -> dict[str, Any]:
    workdir = Path(workdir).resolve()
    log_file = _ensure_text_log(log_file)
    watch_log_file = _ensure_text_log(watch_log_file)
    input_json = Path(input_json).resolve()
    command = [str(item) for item in command]
    absolute_workdir = str(workdir)
    absolute_log = str(log_file)
    absolute_watch_log = str(watch_log_file)
    lcurve_path = workdir / "lcurve.out"

    watch_header = {
        "event": "attempt_start",
        "attempt": int(attempt),
        "seed": int(seed),
        "command": command,
        "workdir": absolute_workdir,
        "input_json": str(input_json),
        "log_file": absolute_log,
        "watch_lcurve_fallback": bool(watch_lcurve_fallback),
        "lcurve_poll_interval_seconds": float(lcurve_poll_interval_seconds),
    }

    with open(watch_log_file, "a", encoding="utf-8") as watch_handle:
        watch_handle.write(json.dumps(watch_header, ensure_ascii=False) + "\n")

    process = subprocess.Popen(
        command,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    reader = threading.Thread(target=_stream_reader, args=(process.stdout, output_queue), daemon=True)
    reader.start()

    tail = deque(maxlen=50)
    nan_detected = False
    matched_line: str | None = None
    terminated = False
    killed = False
    eof_seen = False
    last_lcurve_check = 0.0

    with open(log_file, "w", encoding="utf-8") as log_handle, open(
        watch_log_file, "a", encoding="utf-8"
    ) as watch_handle:
        while True:
            try:
                event_type, payload = output_queue.get(timeout=0.2)
            except queue.Empty:
                event_type, payload = None, None

            if event_type == "line" and payload is not None:
                line = payload
                log_handle.write(line)
                log_handle.flush()
                tail.append(line.rstrip("\n"))
                if not quiet:
                    print(line.rstrip())
                if _line_has_nonfinite_rmse(line):
                    nan_detected = True
                    matched_line = line.rstrip("\n")
                    watch_handle.write(
                        json.dumps(
                            {
                                "event": "nan_detected_stdout",
                                "attempt": int(attempt),
                                "seed": int(seed),
                                "matched_line": matched_line,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    terminated, killed = _terminate_process(process, terminate_timeout_seconds)
            elif event_type == "eof":
                eof_seen = True

            if watch_lcurve_fallback and not nan_detected:
                now = time.monotonic()
                if now - last_lcurve_check >= lcurve_poll_interval_seconds:
                    last_lcurve_check = now
                    recent_lines = _read_recent_lcurve_lines(lcurve_path)
                    for recent_line in reversed(recent_lines):
                        if _line_has_nonfinite_rmse(recent_line):
                            nan_detected = True
                            matched_line = recent_line
                            watch_handle.write(
                                json.dumps(
                                    {
                                        "event": "nan_detected_lcurve",
                                        "attempt": int(attempt),
                                        "seed": int(seed),
                                        "matched_line": matched_line,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            terminated, killed = _terminate_process(process, terminate_timeout_seconds)
                            break

            if eof_seen and process.poll() is not None:
                break

        return_code = process.wait()
        reader.join(timeout=1.0)

        result = {
            "status": "nan_detected" if nan_detected else ("success" if return_code == 0 else "failed"),
            "attempt": int(attempt),
            "return_code": int(return_code),
            "nan_detected": bool(nan_detected),
            "matched_line": matched_line,
            "terminated": bool(terminated),
            "killed": bool(killed),
            "workdir": absolute_workdir,
            "log_file": absolute_log,
            "watch_log_file": absolute_watch_log,
            "input_json": str(input_json),
            "command": command,
            "tail": list(tail),
        }
        watch_handle.write(json.dumps({"event": "attempt_end", **result}, ensure_ascii=False) + "\n")
        return result


def summarize_watchdog_attempts(
    attempts: list[dict[str, Any]],
    *,
    workdir: str | Path,
    log_file: str | Path,
) -> dict[str, Any]:
    workdir = Path(workdir).resolve()
    log_file = Path(log_file).resolve()
    final = attempts[-1] if attempts else {}
    nan_attempts = sum(1 for item in attempts if item.get("nan_detected"))
    status = "success" if final.get("status") == "success" else "failed"
    return {
        "status": status,
        "attempts": len(attempts),
        "nan_attempts": nan_attempts,
        "final_return_code": int(final.get("return_code", -1)) if attempts else -1,
        "workdir": str(workdir),
        "log_file": str(log_file),
    }


def metrics_are_nonfinite(metrics: dict[str, float | None]) -> bool:
    for value in metrics.values():
        if value is None:
            continue
        if not math.isfinite(float(value)):
            return True
    return False
