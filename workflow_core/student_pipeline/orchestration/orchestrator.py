from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from workflow_core.student.config import load_student_active_learning_config
from workflow_core.student_pipeline.orchestration.recovery_nodes import build_student_stage_recovery_nodes
from workflow_core.student_pipeline.orchestration.source_snapshots import materialize_initial_seed_pool_snapshot
from workflow_core.student_pipeline.orchestration.state_store import build_student_pipeline_runtime_state
from workflow_runtime.recovery.recovery_plan import RecoveryValidationError, validate_completed_recovery_node

StageRunner = Callable[[dict[str, Any]], dict[str, Any]]


def run_student_pipeline_template_subworkflow(
    cfg_path: str | Path,
    paths,
    outer_round: int,
    metrics_logger=None,
    *,
    stage_runners: dict[str, Callable[..., dict[str, Any]]] | None = None,
    extra_cfg: dict[str, Any] | None = None,
) -> str:
    del metrics_logger
    cfg = load_student_active_learning_config(cfg_path)
    if extra_cfg:
        cfg = {**cfg, **dict(extra_cfg)}
    runtime_state = build_student_pipeline_runtime_state(cfg, teacher_round=int(outer_round), paths=paths)
    prepare_student_pipeline_runtime_state(runtime_state)
    execute_student_pipeline_runtime(runtime_state, stage_runners=stage_runners)
    return str(paths.student_subworkflow_artifacts_dir(int(outer_round)).resolve())


def execute_student_pipeline_runtime(
    runtime_state: dict[str, Any],
    *,
    stage_runners: dict[str, Callable[..., dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    prepare_student_pipeline_runtime_state(runtime_state)
    runners = {**_default_stage_runners(), **dict(stage_runners or {})}
    for invocation in list(runtime_state.get("invocations", []) or []):
        student_round = int(invocation["student_round"])
        for stage in list(invocation.get("stages", []) or []):
            if bool(stage.get("completed")):
                continue
            stage_type = str(stage["stage"])
            runner = runners.get(stage_type)
            if runner is None:
                raise ValueError(f"no runner registered for student pipeline stage: {stage_type}")
            runner(
                runtime_state,
                student_round=student_round,
                stage_run_index=int(stage["stage_run_index"]),
            )
    return _mark_pipeline_schedule_exhausted(runtime_state)


def prepare_student_pipeline_runtime_state(runtime_state: dict[str, Any]) -> dict[str, Any]:
    ensure_initial_seed_pool_snapshot(runtime_state)
    marker_dir = runtime_state["paths"].workflow_round_marker_dir(int(runtime_state["teacher_round"]))
    if not Path(marker_dir).exists():
        return runtime_state
    return reconcile_student_pipeline_runtime_state(runtime_state)


def ensure_initial_seed_pool_snapshot(runtime_state: dict[str, Any]) -> dict[str, Any]:
    paths = runtime_state.get("paths")
    if paths is None:
        raise ValueError("student pipeline runtime_state is missing paths")
    initial_seed_pool = paths.student_subworkflow_initial_seed_pool_xyz(int(runtime_state["teacher_round"]))
    if _runtime_state_requires_seed_pool_snapshot(runtime_state) and not Path(initial_seed_pool).exists():
        materialize_initial_seed_pool_snapshot(
            dict(runtime_state.get("cfg", {}) or {}),
            teacher_round=int(runtime_state["teacher_round"]),
            paths=paths,
        )
    return runtime_state


def reconcile_student_pipeline_runtime_state(runtime_state: dict[str, Any]) -> dict[str, Any]:
    recovery_nodes = {node.node_id: node for node in build_student_stage_recovery_nodes(runtime_state)}
    for invocation in list(runtime_state.get("invocations", []) or []):
        completed_indexes: list[int] = []
        for stage in list(invocation.get("stages", []) or []):
            node_id = _student_stage_node_id(runtime_state, stage)
            node = recovery_nodes.get(node_id)
            is_completed = False
            if node is not None:
                try:
                    validate_completed_recovery_node(node)
                except RecoveryValidationError as exc:
                    raise RuntimeError(str(exc)) from None
                is_completed = True
                completed_indexes.append(int(stage["stage_run_index"]))
            stage["completed"] = is_completed
        invocation["completed_stage_run_indexes"] = sorted(completed_indexes)
        invocation["current_stage_run_index"] = max(completed_indexes) if completed_indexes else 0
        invocation["completed"] = len(completed_indexes) >= len(list(invocation.get("stages", []) or []))
    _rebuild_status_progress_from_markers(runtime_state)
    return runtime_state


def _runtime_state_requires_seed_pool_snapshot(runtime_state: dict[str, Any]) -> bool:
    for invocation in list(runtime_state.get("invocations", []) or []):
        for stage in list(invocation.get("stages", []) or []):
            if str(stage.get("stage", "")).strip() in {"sampling", "seed_pool_refresh"}:
                return True
    return False


def _student_stage_node_id(runtime_state: dict[str, Any], stage: dict[str, Any]) -> str:
    from workflow_core.student_pipeline.orchestration.runtime_node_contracts import student_stage_node_id

    paths = runtime_state.get("paths")
    if paths is None:
        raise ValueError("student pipeline runtime_state is missing paths")
    return student_stage_node_id(
        paths,
        teacher_round=int(stage["teacher_round"]),
        student_round=int(stage["student_round"]),
        stage_run_index=int(stage["stage_run_index"]),
        stage_type=str(stage["stage"]),
    )


def _rebuild_status_progress_from_markers(runtime_state: dict[str, Any]) -> None:
    status = dict(runtime_state.get("status", {}) or {})
    finished_rounds = [
        int(invocation["student_round"])
        for invocation in list(runtime_state.get("invocations", []) or [])
        if bool(invocation.get("completed"))
    ]
    planned = [int(item) for item in status.get("planned_student_rounds", []) or []]
    remaining = [item for item in planned if item not in finished_rounds]
    status["finished_rounds"] = finished_rounds
    status["completed_invocations"] = len(finished_rounds)
    status["next_student_round"] = remaining[0] if remaining else None
    status["schedule_exhausted"] = not bool(remaining)
    status["current_student_round"] = max(finished_rounds) if finished_rounds else int(status.get("current_student_round", 0) or 0)
    runtime_state["status"] = status


def _default_stage_runners() -> dict[str, Callable[..., dict[str, Any]]]:
    from workflow_core.student_pipeline.stages.relabel_stage import run_relabel_stage
    from workflow_core.student_pipeline.stages.sampling_stage import run_sampling_stage
    from workflow_core.student_pipeline.stages.seed_pool_refresh_stage import run_seed_pool_refresh_stage
    from workflow_core.student_pipeline.stages.student_relabel_input_filtering_stage import (
        run_student_relabel_input_filtering_stage,
    )
    from workflow_core.student_pipeline.stages.teacher_dft_candidate_filtering_stage import (
        run_teacher_dft_candidate_filtering_stage,
    )
    from workflow_core.student_pipeline.stages.train_stage import run_train_stage

    return {
        "relabel": run_relabel_stage,
        "train": run_train_stage,
        "sampling": run_sampling_stage,
        "student_relabel_input_filtering": run_student_relabel_input_filtering_stage,
        "teacher_dft_candidate_filtering": run_teacher_dft_candidate_filtering_stage,
        "seed_pool_refresh": run_seed_pool_refresh_stage,
    }


def _mark_pipeline_schedule_exhausted(runtime_state: dict[str, Any]) -> dict[str, Any]:
    status = dict(runtime_state.get("status", {}) or {})
    status["schedule_exhausted"] = True
    status["stopped"] = True
    status["stop_reason"] = "schedule_exhausted"
    latest_committee = _latest_student_committee_for_display(runtime_state)
    if latest_committee is not None:
        status["final_student_model_path"] = str(latest_committee.model_root.resolve())
    else:
        status["final_student_model_path"] = status.get("final_student_model_path")
    runtime_state["status"] = dict(status)
    return status


def _latest_student_committee_for_display(runtime_state: dict[str, Any]):
    from workflow_core.student_pipeline.artifacts.successful_stage_resolver import SuccessfulStageArtifactResolver

    stages = [
        dict(stage)
        for invocation in list(runtime_state.get("invocations", []) or [])
        for stage in list(invocation.get("stages", []) or [])
    ]
    if not stages:
        return None
    last = max(
        stages,
        key=lambda stage: (
            int(stage.get("teacher_round", runtime_state.get("teacher_round", 0))),
            int(stage.get("student_round", 0)),
            int(stage.get("stage_run_index", 0)),
        ),
    )
    sentinel = dict(last)
    sentinel["stage_run_index"] = int(last.get("stage_run_index", 0)) + 1
    return SuccessfulStageArtifactResolver(runtime_state).latest_train_committee_before(sentinel)
