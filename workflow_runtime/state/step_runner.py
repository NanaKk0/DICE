from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from workflow_runtime.state.run_state import RunStateManager


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def backup_partial_output(output_dir: str | Path, status: str) -> Path | None:
    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        return None
    suffix = "failed" if status == "FAILED" else "interrupted"
    target = output_dir.with_name(f"{output_dir.name}.{suffix}.{_timestamp()}")
    shutil.move(str(output_dir), str(target))
    return target


def execute_step(step: dict[str, Any], state_mgr: RunStateManager) -> Any:
    step_id = str(step["id"])
    record = state_mgr.get_step_record(step_id)
    previous_status = record.get("status")
    output_dir = step.get("output_dir")

    if (
        state_mgr.backup_partial_outputs
        and step.get("backup_partial_outputs", False)
        and previous_status in {"FAILED", "INTERRUPTED"}
        and output_dir
    ):
        backup_partial_output(output_dir, previous_status)

    state_mgr.mark_step_running(step_id)
    try:
        prepare = step.get("prepare")
        if callable(prepare):
            prepare()

        result = step["run"]()

        success_check = step.get("success_check")
        success_ok = True
        if callable(success_check):
            success_ok = bool(success_check())
        if not success_ok:
            raise RuntimeError(f"success_check failed for step {step_id}")

        state_mgr.mark_step_success(step_id)
        return result
    except KeyboardInterrupt:
        state_mgr.mark_step_interrupted(step_id, error_message="Interrupted by KeyboardInterrupt")
        raise
    except SystemExit as exc:
        state_mgr.mark_step_interrupted(step_id, error_message=f"Interrupted by SystemExit({exc.code})")
        raise
    except Exception as exc:
        state_mgr.mark_step_failed(step_id, error_message=str(exc))
        raise

