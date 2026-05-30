from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from workflow_core.student_pipeline.services.filtering.student_relabel_input_filtering import filter_student_relabel_inputs
from workflow_core.student_pipeline.stages.base import resolve_stage_config
from workflow_core.student_pipeline.stages.harness import StudentStageRunnerContext, run_student_stage_with_harness
from workflow_core.student_pipeline.stages.publish import (
    build_stage_publish_plan,
    prepare_stage_result_publications,
)


def run_student_relabel_input_filtering_stage(
    runtime_state: dict[str, Any],
    *,
    student_round: int,
    stage_run_index: int,
    stage_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def execute(context: StudentStageRunnerContext) -> dict[str, Any]:
        executor = stage_executor or (
            lambda stage_context: _default_student_relabel_input_filtering_executor(runtime_state, stage_context)
        )
        return dict(executor(context.execution_context) or {})

    def publish_plan(context: StudentStageRunnerContext, stage_result: dict[str, Any]):
        return build_stage_publish_plan(
            student_round=student_round,
            stage_run_index=stage_run_index,
            artifact_publications=prepare_stage_result_publications(
                "student_relabel_input_filtering",
                staging_dir=context.staging_dir,
                target_root=context.stage_context["stage_dir"],
                stage_result=stage_result,
                role_source_fields={
                    "filtered_candidate": "filtered_candidate_path",
                    "filtered_failed": "filtered_failed_path",
                },
            ),
            stage_payload=dict(stage_result.get("stage_payload", {}) or {}),
        )

    return run_student_stage_with_harness(
        runtime_state,
        student_round=student_round,
        stage_run_index=stage_run_index,
        expected_stage_type="student_relabel_input_filtering",
        execute=execute,
        build_publish_plan=publish_plan,
    )


def _default_student_relabel_input_filtering_executor(
    runtime_state: dict[str, Any],
    stage_context: dict[str, Any],
) -> dict[str, Any]:
    stage_cfg = resolve_stage_config(stage_context, str(stage_context["stage"]))
    candidate_atoms, failed_atoms, window_payload = _resolve_filtering_window(runtime_state, stage_context)
    result = filter_student_relabel_inputs(
        candidate_atoms=candidate_atoms,
        failed_atoms=failed_atoms,
        output_dir=Path(stage_context["stage_results_dir"]).resolve(),
        cfg=stage_cfg,
        teacher_round=int(stage_context["teacher_round"]),
        student_round=int(stage_context["student_round"]),
        stage_run_index=int(stage_context["stage_run_index"]),
    )
    stage_payload = result.to_stage_payload()
    stage_payload.update(window_payload)
    return {
        "filtered_candidate_path": str(result.filtered_candidate_path.resolve()),
        "filtered_failed_path": str(result.filtered_failed_path.resolve()),
        "stage_payload": stage_payload,
    }


def _resolve_filtering_window(
    runtime_state: dict[str, Any],
    stage_context: dict[str, Any],
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    del runtime_state
    artifacts = stage_context.get("artifacts")
    inputs = getattr(artifacts, "inputs", None)
    if inputs is None:
        raise ValueError("student_relabel_input_filtering stage artifacts are missing typed inputs")
    candidate_all = _read_atoms(inputs.candidate_xyz)
    failed_all = _read_atoms(inputs.failed_xyz)
    candidate_start, failed_start = 0, 0
    return (
        candidate_all[candidate_start:],
        failed_all[failed_start:],
        {
            "candidate_input_start_index": int(candidate_start),
            "failed_input_start_index": int(failed_start),
            "candidate_input_consumed_count": len(candidate_all),
            "failed_input_consumed_count": len(failed_all),
        },
    )


def _read_atoms(path: Path) -> list[Any]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    from workflow_core.structures.formats import read_extxyz_structures

    return read_extxyz_structures(path)
