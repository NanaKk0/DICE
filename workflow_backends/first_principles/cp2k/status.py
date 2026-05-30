from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_backends.first_principles.cp2k.defaults import DEFAULT_DFT_OUTPUT_NAME


CP2K_SUCCESS_MARKERS = (
    "PROGRAM ENDED AT",
    "The number of warnings for this run is",
)

CP2K_FAILURE_MARKERS = (
    "SCF run NOT converged".upper(),
    "SCF NOT CONVERGED",
    "ABORT",
    "SIGSEGV",
    "STACK TRACE",
    "CHOLESKY DECOMPOSE FAILED",
)


def cp2k_output_looks_successful(task_dir: Path, cfg: dict[str, Any]) -> bool:
    output_name = DEFAULT_DFT_OUTPUT_NAME
    output_path = task_dir / output_name
    if not output_path.exists():
        return False

    try:
        output_text = output_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False

    upper_text = output_text.upper()
    has_success_marker = any(marker.upper() in upper_text for marker in CP2K_SUCCESS_MARKERS)
    if not has_success_marker:
        return False

    failure_hits = [marker for marker in CP2K_FAILURE_MARKERS if marker in upper_text]
    if failure_hits:
        return False

    suspicious_error_lines = [
        line for line in upper_text.splitlines() if " ERROR" in f" {line} " or line.startswith("ERROR")
    ]
    if suspicious_error_lines:
        return False

    return True

