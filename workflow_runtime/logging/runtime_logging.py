import logging
import os
import re
import subprocess
import traceback
import warnings
from collections import deque
from contextlib import contextmanager
from pathlib import Path


KNOWN_WARNING_RULES = [
    {
        "action": "ignore",
        "message": r".*weights_only=False.*",
        "category": FutureWarning
    },
    {
        "action": "ignore",
        "message": r".*weights_only=False.*",
        "category": UserWarning
    }
]

KNOWN_LOG_SUBSTRINGS = [
    "'model_path' argument is deprecated, please use 'model_paths' in the future.",
    "_Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))",
]

QUIET_TERMINAL_SKIP_SUBSTRINGS = [
    "weights_only=False",
    "'model_path' argument is deprecated, please use 'model_paths' in the future.",
    "Cuequivariance acceleration will be disabled.",
    "_Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))",
    "Created device /job:localhost/replica:0/task:0/device:GPU:0",
]

QUIET_TERMINAL_ECHO_PATTERNS = [
    re.compile(r"Initial:"),
    re.compile(r"\bEpoch\b", re.IGNORECASE),
    re.compile(r"\bSWA\b"),
    re.compile(r"\bTRAINING\b"),
    re.compile(r"\bOPTIMIZER INFORMATION\b"),
    re.compile(r"\bStage Two\b"),
    re.compile(r"\bLoaded model\b", re.IGNORECASE),
    re.compile(r"\bcheckpoint\b", re.IGNORECASE),
    re.compile(r"\bvalid\b", re.IGNORECASE),
    re.compile(r"\bloss=", re.IGNORECASE),
    re.compile(r"\bRMSE_", re.IGNORECASE),
    re.compile(r"\bERROR\b"),
    re.compile(r"\bTraceback\b"),
    re.compile(r"\bRuntimeError\b"),
]

QUIET_TERMINAL_ERROR_PATTERNS = [
    re.compile(r"\bTraceback\b"),
    re.compile(r"\bRuntimeError\b"),
    re.compile(r"\bValueError\b"),
    re.compile(r"\bException\b"),
    re.compile(r"\bOutOfMemoryError\b"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"\berror:\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
]


def is_verbose_enabled():
    value = os.environ.get("WORKFLOW_VERBOSE", "")

    return value.lower() in {"1", "true", "yes", "on"}


def ensure_step_log_file(paths, step_name, round_id, filename=None):
    log_dir = paths.root / "logs" / f"r{round_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{step_name}.log"

    return log_dir / filename


def ensure_step_log_dir(paths, round_id):
    log_dir = paths.root / "logs" / f"r{round_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir


class SuppressKnownLogsFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()

        return not any(text in message for text in KNOWN_LOG_SUBSTRINGS)


def _print_failure_summary(log_file, tail_lines):
    print(f"\n[ERROR] Command failed. Full log written to:\n{log_file}")

    if not tail_lines:
        return

    print("\n[ERROR] Last log lines:")

    for line in tail_lines:
        print(line.rstrip())


def _line_matches_error_context(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in QUIET_TERMINAL_ERROR_PATTERNS)


def should_echo_quiet_line(line):
    stripped = line.strip()

    if not stripped:
        return False

    if any(text in stripped for text in QUIET_TERMINAL_SKIP_SUBSTRINGS):
        return False

    return any(pattern.search(stripped) for pattern in QUIET_TERMINAL_ECHO_PATTERNS)


def run_logged_command(cmd, log_file, cwd=None, env=None, quiet=None):
    if quiet is None:
        quiet = not is_verbose_enabled()

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"[INFO] Verbose mode enabled; full command output will stream to the terminal.")
        print(f"[INFO] To switch back to quiet mode, unset WORKFLOW_VERBOSE.")
        print(f"[INFO] Command log file: {log_file}")

        with open(log_file, "w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            assert process.stdout is not None

            for line in process.stdout:
                print(line.rstrip())
                handle.write(line)

            returncode = process.wait()

        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

        return

    tail = deque(maxlen=60)
    error_context = deque(maxlen=120)
    capturing_error_context = False

    with open(log_file, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        assert process.stdout is not None

        for line in process.stdout:
            handle.write(line)
            tail.append(line)

            if should_echo_quiet_line(line):
                print(line.rstrip())

            if _line_matches_error_context(line):
                capturing_error_context = True

            if capturing_error_context:
                error_context.append(line)

        returncode = process.wait()

    if returncode != 0:
        failure_lines = error_context if error_context else tail
        _print_failure_summary(log_file, failure_lines)
        raise subprocess.CalledProcessError(returncode, cmd)

    print(f"[INFO] Command output saved to: {log_file}")


@contextmanager
def quiet_known_third_party_noise():
    if is_verbose_enabled():
        yield
        return

    root_logger = logging.getLogger()
    log_filter = SuppressKnownLogsFilter()
    root_logger.addFilter(log_filter)

    with warnings.catch_warnings():
        for rule in KNOWN_WARNING_RULES:
            warnings.filterwarnings(
                rule["action"],
                message=rule["message"],
                category=rule["category"]
            )

        try:
            yield
        finally:
            root_logger.removeFilter(log_filter)


def print_exception_banner(step_name: str, runtime_root: Path, exc: BaseException) -> None:
    print("\n" + "=" * 80)
    print(f"[ERROR] STEP FAILED: {step_name}")
    print(f"[ERROR] Runtime root: {runtime_root}")
    print(f"[ERROR] Exception: {exc.__class__.__name__}: {exc}")
    print("[ERROR] Full traceback:")
    print("=" * 80)
    traceback.print_exc()
