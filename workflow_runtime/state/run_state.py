from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


STEP_STATUSES = {"PENDING", "RUNNING", "SUCCESS", "FAILED", "INTERRUPTED", "SKIPPED"}
TERMINAL_STEP_STATUSES = {"SUCCESS", "FAILED", "INTERRUPTED", "SKIPPED"}
MARKER_SUFFIXES = ("running", "success", "failed", "interrupted")


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path


class RunStateManager:
    def __init__(
        self,
        state_path: str | Path,
        marker_dir: str | Path,
        *,
        backup_partial_outputs: bool = True,
        verify_success_markers: bool = True,
    ) -> None:
        self.state_path = Path(state_path).resolve()
        self.marker_dir = Path(marker_dir).resolve()
        self.backup_partial_outputs = bool(backup_partial_outputs)
        self.verify_success_markers = bool(verify_success_markers)
        self.state: dict[str, Any] | None = None

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> dict[str, Any]:
        with open(self.state_path, "r", encoding="utf-8-sig") as handle:
            self.state = json.load(handle)
        return self.state

    def save_atomic(self) -> Path:
        if self.state is None:
            raise RuntimeError("RunStateManager.save_atomic() called before state was initialized")
        return atomic_write_json(self.state_path, self.state)

    def init_state(self, *, run_name: str, round_id: int, steps: list[dict[str, Any]]) -> dict[str, Any]:
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        self.state = {
            "run_name": run_name,
            "round": int(round_id),
            "status": "PENDING",
            "started_at": None,
            "finished_at": None,
            "current_round": int(round_id),
            "current_step_index": 0 if steps else None,
            "current_step_id": steps[0]["id"] if steps else None,
            "steps": [],
        }
        for index, step in enumerate(steps):
            self.state["steps"].append(
                {
                    "id": str(step["id"]),
                    "name": str(step.get("name", step["id"])),
                    "round": int(step.get("round", round_id)),
                    "index": int(index),
                    "status": "PENDING",
                    "started_at": None,
                    "finished_at": None,
                    "attempt": 0,
                    "output_dir": str(step.get("output_dir")) if step.get("output_dir") else None,
                    "error_message": None,
                    "success_marker": str(self.marker_path(step["id"], "success")),
                }
            )
        self.save_atomic()
        return self.state

    def marker_path(self, step_id: str, suffix: str) -> Path:
        return self.marker_dir / f"{step_id}.{suffix}"

    def _clear_markers(self, step_id: str) -> None:
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        for suffix in MARKER_SUFFIXES:
            marker = self.marker_path(step_id, suffix)
            if marker.exists():
                marker.unlink()

    def _set_marker(self, step_id: str, suffix: str) -> None:
        self._clear_markers(step_id)
        marker = self.marker_path(step_id, suffix)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(utc_now(), encoding="utf-8")

    def get_step_record(self, step_id: str) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("RunStateManager.get_step_record() called before state was loaded")
        for record in self.state["steps"]:
            if record["id"] == step_id:
                return record
        raise KeyError(f"Step record not found: {step_id}")

    def _mark_step_state(self, step_id: str, status: str, *, error_message: str | None = None) -> None:
        if status not in STEP_STATUSES:
            raise ValueError(f"Unsupported step status: {status}")
        record = self.get_step_record(step_id)
        record["status"] = status
        if status == "RUNNING":
            record["attempt"] = int(record.get("attempt", 0)) + 1
            record["started_at"] = utc_now()
            record["finished_at"] = None
            record["error_message"] = None
            self._set_marker(step_id, "running")
        else:
            record["finished_at"] = utc_now()
            record["error_message"] = error_message
            marker_suffix = {
                "SUCCESS": "success",
                "FAILED": "failed",
                "INTERRUPTED": "interrupted",
                "SKIPPED": "success",
            }.get(status)
            if marker_suffix:
                self._set_marker(step_id, marker_suffix)
            else:
                self._clear_markers(step_id)
        self.save_atomic()

    def mark_step_running(self, step_id: str) -> None:
        self._mark_step_state(step_id, "RUNNING")

    def mark_step_success(self, step_id: str) -> None:
        self._mark_step_state(step_id, "SUCCESS")

    def mark_step_failed(self, step_id: str, error_message: str | None = None) -> None:
        self._mark_step_state(step_id, "FAILED", error_message=error_message)

    def mark_step_interrupted(self, step_id: str, error_message: str | None = None) -> None:
        self._mark_step_state(step_id, "INTERRUPTED", error_message=error_message)

    def mark_workflow_running(self, *, current_step_index: int | None = None, current_step_id: str | None = None) -> None:
        if self.state is None:
            raise RuntimeError("RunStateManager.mark_workflow_running() called before state was loaded")
        if self.state.get("started_at") is None:
            self.state["started_at"] = utc_now()
        self.state["status"] = "RUNNING"
        self.state["finished_at"] = None
        if current_step_index is not None:
            self.state["current_step_index"] = current_step_index
        if current_step_id is not None:
            self.state["current_step_id"] = current_step_id
        self.save_atomic()

    def mark_workflow_success(self) -> None:
        if self.state is None:
            raise RuntimeError("RunStateManager.mark_workflow_success() called before state was loaded")
        self.state["status"] = "SUCCESS"
        self.state["finished_at"] = utc_now()
        self.state["current_step_id"] = None
        self.state["error_message"] = None
        self.save_atomic()

    def mark_workflow_failed(self, error_message: str | None = None) -> None:
        if self.state is None:
            raise RuntimeError("RunStateManager.mark_workflow_failed() called before state was loaded")
        self.state["status"] = "FAILED"
        self.state["finished_at"] = utc_now()
        if error_message:
            self.state["error_message"] = error_message
        self.save_atomic()

    def mark_workflow_interrupted(self, error_message: str | None = None) -> None:
        if self.state is None:
            raise RuntimeError("RunStateManager.mark_workflow_interrupted() called before state was loaded")
        self.state["status"] = "INTERRUPTED"
        self.state["finished_at"] = utc_now()
        if error_message:
            self.state["error_message"] = error_message
        self.save_atomic()

    def reconcile(self, success_checks: dict[str, Callable[[], bool]] | None = None) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("RunStateManager.reconcile() called before state was loaded")
        success_checks = success_checks or {}
        for record in self.state.get("steps", []):
            step_id = record["id"]
            running_marker = self.marker_path(step_id, "running")
            success_marker = self.marker_path(step_id, "success")
            status = record.get("status", "PENDING")
            if status == "RUNNING" or running_marker.exists():
                record["status"] = "INTERRUPTED"
                record["finished_at"] = utc_now()
                record["error_message"] = "Recovered from unfinished RUNNING state"
                self._set_marker(step_id, "interrupted")
                continue
            if status == "SUCCESS" and self.verify_success_markers:
                success_check = success_checks.get(step_id)
                success_ok = success_marker.exists()
                if success_ok and success_check is not None:
                    try:
                        success_ok = bool(success_check())
                    except Exception:
                        success_ok = False
                if not success_ok:
                    record["status"] = "FAILED"
                    record["finished_at"] = utc_now()
                    record["error_message"] = "Success marker or outputs failed verification during reconcile"
                    self._set_marker(step_id, "failed")
        self.save_atomic()
        return self.state

    def get_resume_index(self) -> int:
        if self.state is None:
            raise RuntimeError("RunStateManager.get_resume_index() called before state was loaded")
        for record in self.state.get("steps", []):
            if record.get("status") != "SUCCESS":
                return int(record["index"])
        return len(self.state.get("steps", []))
