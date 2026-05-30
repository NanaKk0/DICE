from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from workflow_runtime.node_contracts import MARKER_SUFFIXES, runtime_node_marker_path, top_level_runtime_node_contract

STEP_STATUSES = {"PENDING", "RUNNING", "SUCCESS", "FAILED", "INTERRUPTED", "SKIPPED"}


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def read_marker_payload(path: str | Path) -> dict[str, Any]:
    marker = Path(path).resolve()
    if not marker.exists():
        return {}
    text = marker.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"timestamp": text}
    return dict(payload or {}) if isinstance(payload, dict) else {"value": payload}


class MarkerStateManager:
    def __init__(
        self,
        marker_dir: str | Path,
        *,
        run_name: str,
        round_id: int,
        steps: list[dict[str, Any]],
        backup_partial_outputs: bool = True,
        verify_success_markers: bool = True,
    ) -> None:
        self.marker_dir = Path(marker_dir).resolve()
        self.backup_partial_outputs = bool(backup_partial_outputs)
        self.verify_success_markers = bool(verify_success_markers)
        self.state = self._build_state(run_name=run_name, round_id=round_id, steps=steps)

    def exists(self) -> bool:
        return self.marker_dir.exists() and any(self.marker_dir.rglob("*.*"))

    def load(self) -> dict[str, Any]:
        return self.state

    def init_state(self, *, run_name: str, round_id: int, steps: list[dict[str, Any]]) -> dict[str, Any]:
        self.state = self._build_state(run_name=run_name, round_id=round_id, steps=steps)
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        return self.state

    def _build_state(self, *, run_name: str, round_id: int, steps: list[dict[str, Any]]) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            teacher_round = step.get("teacher_round", step.get("round", round_id))
            records.append(
                {
                    "id": str(step["id"]),
                    "name": str(step.get("name", step["id"])),
                    "teacher_round": int(teacher_round),
                    "index": int(step.get("index", index)),
                    "status": "PENDING",
                    "started_at": None,
                    "finished_at": None,
                    "parent_step": step.get("parent_step"),
                    "identity": dict(step.get("identity", {}) or {}),
                    "artifact_contract_id": str(step.get("artifact_contract_id", step.get("name", step["id"]))),
                    "global_invariant_fingerprint": str(step.get("global_invariant_fingerprint", "")),
                    "stage_sequence_fingerprint": str(step.get("stage_sequence_fingerprint", "")),
                }
            )
        return {
            "run_name": run_name,
            "teacher_round": int(round_id),
            "status": "PENDING",
            "started_at": None,
            "finished_at": None,
            "current_step_index": 0 if records else None,
            "current_step_id": records[0]["id"] if records else None,
            "steps": records,
        }

    def marker_path(self, step_id: str, suffix: str) -> Path:
        return runtime_node_marker_path(self.marker_dir, step_id, suffix)

    def _clear_markers(self, step_id: str) -> None:
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        for suffix in MARKER_SUFFIXES:
            marker = self.marker_path(step_id, suffix)
            if marker.exists():
                marker.unlink()

    def _marker_payload(self, record: dict[str, Any], *, suffix: str, status: str) -> dict[str, Any]:
        contract = top_level_runtime_node_contract(record, marker_dir=self.marker_dir)
        payload = contract.marker_payload(suffix=suffix, status=status, timestamp=utc_now())
        payload.update(
            {
                "step_id": str(record["id"]),
                "name": str(record["name"]),
                "index": int(record["index"]),
                "parent_step": record.get("parent_step"),
            }
        )
        return payload

    def _set_marker(self, step_id: str, suffix: str, status: str) -> None:
        self._clear_markers(step_id)
        record = self.get_step_record(step_id)
        marker = self.marker_path(step_id, suffix)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(self._marker_payload(record, suffix=suffix, status=status), indent=2), encoding="utf-8")

    def get_step_record(self, step_id: str) -> dict[str, Any]:
        for record in self.state["steps"]:
            if record["id"] == step_id:
                return record
        raise KeyError(f"Step record not found: {step_id}")

    def _mark_step_state(self, step_id: str, status: str, *, error_message: str | None = None) -> None:
        del error_message
        if status not in STEP_STATUSES:
            raise ValueError(f"Unsupported step status: {status}")
        record = self.get_step_record(step_id)
        record["status"] = status
        if status == "RUNNING":
            record["started_at"] = utc_now()
            record["finished_at"] = None
            self._set_marker(step_id, "running", "RUNNING")
            return
        record["finished_at"] = utc_now()
        marker_suffix = {
            "SUCCESS": "success",
            "FAILED": "failed",
            "INTERRUPTED": "interrupted",
            "SKIPPED": "success",
        }.get(status)
        if marker_suffix:
            self._set_marker(step_id, marker_suffix, status)
        else:
            self._clear_markers(step_id)

    def mark_step_running(self, step_id: str) -> None:
        self._mark_step_state(step_id, "RUNNING")

    def mark_step_success(self, step_id: str) -> None:
        self._mark_step_state(step_id, "SUCCESS")

    def mark_step_failed(self, step_id: str, error_message: str | None = None) -> None:
        self._mark_step_state(step_id, "FAILED", error_message=error_message)

    def mark_step_interrupted(self, step_id: str, error_message: str | None = None) -> None:
        self._mark_step_state(step_id, "INTERRUPTED", error_message=error_message)

    def mark_workflow_running(self, *, current_step_index: int | None = None, current_step_id: str | None = None) -> None:
        if self.state.get("started_at") is None:
            self.state["started_at"] = utc_now()
        self.state["status"] = "RUNNING"
        self.state["finished_at"] = None
        if current_step_index is not None:
            self.state["current_step_index"] = current_step_index
        if current_step_id is not None:
            self.state["current_step_id"] = current_step_id

    def mark_workflow_success(self) -> None:
        self.state["status"] = "SUCCESS"
        self.state["finished_at"] = utc_now()
        self.state["current_step_id"] = None

    def mark_workflow_failed(self, error_message: str | None = None) -> None:
        del error_message
        self.state["status"] = "FAILED"
        self.state["finished_at"] = utc_now()

    def mark_workflow_interrupted(self, error_message: str | None = None) -> None:
        del error_message
        self.state["status"] = "INTERRUPTED"
        self.state["finished_at"] = utc_now()

    def reconcile(self, success_checks: dict[str, Callable[[], bool]] | None = None) -> dict[str, Any]:
        success_checks = success_checks or {}
        for record in self.state.get("steps", []):
            step_id = str(record["id"])
            running_marker = self.marker_path(step_id, "running")
            success_marker = self.marker_path(step_id, "success")
            failed_marker = self.marker_path(step_id, "failed")
            interrupted_marker = self.marker_path(step_id, "interrupted")
            if running_marker.exists():
                record["status"] = "INTERRUPTED"
                record["finished_at"] = utc_now()
                self._set_marker(step_id, "interrupted", "INTERRUPTED")
                continue
            if success_marker.exists():
                success_ok = True
                success_check = success_checks.get(step_id)
                if self.verify_success_markers and success_check is not None:
                    try:
                        success_ok = bool(success_check())
                    except Exception:
                        success_ok = False
                if success_ok:
                    record["status"] = "SUCCESS"
                else:
                    record["status"] = "FAILED"
                    record["finished_at"] = utc_now()
                    self._set_marker(step_id, "failed", "FAILED")
                continue
            if failed_marker.exists():
                record["status"] = "FAILED"
                continue
            if interrupted_marker.exists():
                record["status"] = "INTERRUPTED"
                continue
            record["status"] = "PENDING"
        return self.state

    def get_resume_index(self, success_checks: dict[str, Callable[[], bool]] | None = None) -> int:
        success_checks = success_checks or {}
        for record in self.state.get("steps", []):
            step_id = str(record["id"])
            success_marker = self.marker_path(step_id, "success")
            if not success_marker.exists():
                return int(record["index"])
            success_check = success_checks.get(step_id)
            if success_check is not None:
                try:
                    if not bool(success_check()):
                        return int(record["index"])
                except Exception:
                    return int(record["index"])
        return len(self.state.get("steps", []))
