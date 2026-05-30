from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from workflow_core.student_pipeline.orchestration.runtime_node_contracts import student_stage_runtime_node_contract
from workflow_runtime.state.markers import utc_now
from workflow_core.student_pipeline.runtime.commit_plan import (
    ArtifactPublication,
    StageCommitPlan,
    StagePublishPlan,
)
from workflow_core.student_pipeline.runtime.stage_stdout import write_student_stage_stdout_log
from workflow_runtime.artifacts.checks import xyz_file_complete


def commit_student_pipeline_stage_plan(
    runtime_state: dict[str, Any],
    plan: StageCommitPlan | StagePublishPlan,
) -> dict[str, Any]:
    return _commit_student_pipeline_stage_publish_plan(runtime_state, plan)


def _commit_student_pipeline_stage_publish_plan(
    runtime_state: dict[str, Any],
    plan: StagePublishPlan,
) -> dict[str, Any]:
    student_round = int(plan.student_round)
    stage_run_index = int(plan.stage_run_index)
    invocation = _find_invocation(runtime_state, student_round=student_round)
    stage = _find_stage(invocation, stage_run_index=stage_run_index)
    contract = student_stage_runtime_node_contract(runtime_state, stage)

    _validate_stage_publish_plan(plan)

    stdout_path = _stage_stdout_path(stage)
    touched_paths = _snapshot_paths(
        [
            *[item.target for item in plan.artifact_publications],
            *[item.target for item in plan.backend_log_publications],
            stdout_path,
            contract.marker_path_for("success"),
        ]
    )
    original_state = deepcopy(runtime_state)

    try:
        for publication in (*plan.artifact_publications, *plan.backend_log_publications):
            _publish_file(publication)

        status = deepcopy(dict(runtime_state.get("status", {}) or {}))
        status["current_student_round"] = int(student_round)
        runtime_state["status"] = status
        _mark_stage_complete_in_memory(runtime_state, invocation, stage, student_round, stage_run_index)

        stage_payload = _stage_payload(stage)
        stage_payload["completed"] = True
        if plan.stage_payload:
            stage_payload.update(dict(plan.stage_payload))
        write_student_stage_stdout_log(stage, stage_payload)
        if not contract.artifact_check():
            raise ValueError(
                f"published artifacts do not satisfy contract {contract.artifact_contract_id!r} "
                f"for {contract.node_id!r}"
            )
        _write_stage_marker(runtime_state, stage, "success")
        stage["completed"] = True
        stage["stage_payload"] = stage_payload
        return {
            "status": deepcopy(runtime_state["status"]),
            "stage_payload": stage_payload,
            "published_artifacts": [str(item.target) for item in plan.artifact_publications],
        }
    except Exception:
        _restore_paths(touched_paths)
        runtime_state.clear()
        runtime_state.update(original_state)
        raise


def _find_invocation(runtime_state: dict[str, Any], *, student_round: int) -> dict[str, Any]:
    for invocation in list(runtime_state.get("invocations", []) or []):
        if int(invocation["student_round"]) == int(student_round):
            return invocation
    raise ValueError(f"unknown student_round: {student_round}")


def _find_stage(invocation: dict[str, Any], *, stage_run_index: int) -> dict[str, Any]:
    for stage in list(invocation.get("stages", []) or []):
        if int(stage["stage_run_index"]) == int(stage_run_index):
            return stage
    raise ValueError(f"unknown stage_run_index {stage_run_index} for student_round {invocation['student_round']}")


def _write_stage_marker(runtime_state: dict[str, Any], stage: dict[str, Any], suffix: str) -> Path | None:
    contract = student_stage_runtime_node_contract(runtime_state, stage)
    marker = contract.marker_path_for(suffix)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = contract.marker_payload(suffix=suffix, status=suffix.upper(), timestamp=utc_now())
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def _validate_stage_publish_plan(plan: StagePublishPlan) -> None:
    seen_targets: set[Path] = set()
    for publication in (*plan.artifact_publications, *plan.backend_log_publications):
        if publication.target in seen_targets:
            raise ValueError(f"duplicate publication target: {publication.target}")
        seen_targets.add(publication.target)
        if not publication.source.exists() or not publication.source.is_file():
            if publication.required:
                raise ValueError(f"missing staged publication source: {publication.source}")
            continue
        if publication.validate_extxyz and not xyz_file_complete(publication.source, allow_empty=publication.allow_empty):
            raise ValueError(f"invalid staged extxyz publication source: {publication.source}")
    commit_owned_manifest_fields = {
        "teacher_round",
        "student_round",
        "stage_run_index",
        "stage",
        "config",
        "completed",
    }
    manifest_overrides = sorted(set(plan.stage_payload) & commit_owned_manifest_fields)
    if manifest_overrides:
        raise ValueError(f"stage payload contains commit-owned stage manifest fields: {manifest_overrides}")


def _publish_file(publication: ArtifactPublication) -> None:
    publication.target.parent.mkdir(parents=True, exist_ok=True)
    if publication.source == publication.target:
        return
    shutil.copy2(publication.source, publication.target)


def _mark_stage_complete_in_memory(
    runtime_state: dict[str, Any],
    invocation: dict[str, Any],
    stage: dict[str, Any],
    student_round: int,
    stage_run_index: int,
) -> None:
    status = deepcopy(dict(runtime_state.get("status", {}) or {}))
    completed_indexes = sorted(
        set(int(item) for item in list(invocation.get("completed_stage_run_indexes", []) or []) + [stage_run_index])
    )
    invocation["current_stage_run_index"] = int(stage_run_index)
    invocation["completed_stage_run_indexes"] = completed_indexes
    if len(completed_indexes) >= int(invocation.get("stage_count", len(invocation.get("stages", [])))):
        invocation["completed"] = True
        finished_rounds = sorted(set(int(item) for item in status.get("finished_rounds", []) + [student_round]))
        status["finished_rounds"] = finished_rounds
        status["completed_invocations"] = len(finished_rounds)
        planned = list(status.get("planned_student_rounds", []) or [])
        remaining = [int(item) for item in planned if int(item) not in finished_rounds]
        status["next_student_round"] = remaining[0] if remaining else None
        status["schedule_exhausted"] = not bool(remaining)
    runtime_state["status"] = status
    stage["completed"] = True


def _snapshot_paths(paths: list[Path]) -> dict[Path, bytes | None]:
    snapshots: dict[Path, bytes | None] = {}
    for path in paths:
        snapshots[path] = path.read_bytes() if path.exists() else None
    return snapshots


def _restore_paths(snapshots: dict[Path, bytes | None]) -> None:
    for path, payload in snapshots.items():
        if payload is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _stage_stdout_path(stage: dict[str, Any]) -> Path:
    log_paths = dict(stage.get("log_paths", {}) or {})
    return Path(str(log_paths["stdout_path"]))


def _stage_payload(stage: dict[str, Any]) -> dict[str, Any]:
    payload = dict(stage.get("stage_payload", {}) or {})
    payload.update(
        {
            "teacher_round": int(stage["teacher_round"]),
            "student_round": int(stage["student_round"]),
            "stage_run_index": int(stage["stage_run_index"]),
            "stage": str(stage["stage"]),
        }
    )
    return deepcopy(payload)
