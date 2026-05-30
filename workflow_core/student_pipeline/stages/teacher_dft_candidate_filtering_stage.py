from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from workflow_core.student_pipeline.services.filtering.teacher_dft_candidate_filtering import filter_teacher_dft_candidates
from workflow_core.student_pipeline.stages.base import resolve_stage_config
from workflow_core.student_pipeline.stages.harness import StudentStageRunnerContext, run_student_stage_with_harness
from workflow_core.student_pipeline.stages.publish import (
    build_stage_publish_plan,
    prepare_stage_output_publications,
)
from workflow_artifacts.first_principles.handoff import first_principles_handoff_input_path


def run_teacher_dft_candidate_filtering_stage(
    runtime_state: dict[str, Any],
    *,
    student_round: int,
    stage_run_index: int,
    stage_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def execute(context: StudentStageRunnerContext) -> dict[str, Any]:
        executor = stage_executor or (
            lambda stage_context: _default_teacher_dft_candidate_filtering_executor(runtime_state, stage_context)
        )
        return dict(executor(context.execution_context) or {})

    def publish_plan(context: StudentStageRunnerContext, stage_result: dict[str, Any]):
        screened_source = stage_result.get("screened_dft_candidate_path")
        first_principles_source = stage_result.get("first_principles_input_source_path", screened_source)
        return build_stage_publish_plan(
            student_round=student_round,
            stage_run_index=stage_run_index,
            artifact_publications=prepare_stage_output_publications(
                "teacher_dft_candidate_filtering",
                staging_dir=context.staging_dir,
                target_root=context.stage_context["stage_dir"],
                role_sources={
                    "screened_dft_candidates": screened_source,
                    "first_principles_input": first_principles_source,
                },
                extxyz_roles=("screened_dft_candidates",),
            ),
            stage_payload=dict(stage_result.get("stage_payload", {}) or {}),
        )

    return run_student_stage_with_harness(
        runtime_state,
        student_round=student_round,
        stage_run_index=stage_run_index,
        expected_stage_type="teacher_dft_candidate_filtering",
        execute=execute,
        build_publish_plan=publish_plan,
    )


def _default_teacher_dft_candidate_filtering_executor(
    runtime_state: dict[str, Any],
    stage_context: dict[str, Any],
) -> dict[str, Any]:
    stage_cfg = resolve_stage_config(stage_context, "teacher_dft_candidate_filtering")
    artifacts = stage_context.get("artifacts")
    inputs = getattr(artifacts, "inputs", None)
    if inputs is None:
        raise ValueError("teacher_dft_candidate_filtering stage artifacts are missing typed inputs")
    source_path = Path(inputs.teacher_dft_candidates_xyz).resolve()
    atoms_list = _read_atoms(source_path)
    result = filter_teacher_dft_candidates(
        atoms_list=atoms_list,
        output_dir=Path(stage_context["stage_results_dir"]).resolve(),
        cfg=stage_cfg,
        teacher_round=int(stage_context["teacher_round"]),
        student_round=int(stage_context["student_round"]),
        stage_run_index=int(stage_context["stage_run_index"]),
    )
    paths = runtime_state.get("paths")
    if paths is None:
        raise ValueError("runtime_state is missing paths for first-principles handoff publishing")
    first_principles_input = first_principles_handoff_input_path(paths, int(stage_context["teacher_round"]))
    stage_payload = result.to_stage_payload()
    stage_payload.setdefault("selection_mode", str(stage_cfg.get("selection_mode", "top_uncertainty_per_system")))
    stage_payload.setdefault("input_candidate_count", len(atoms_list))
    stage_payload["source_teacher_dft_candidate_pool_path"] = str(source_path.resolve())
    stage_payload["first_principles_input_xyz"] = str(first_principles_input.resolve())
    return {
        "screened_dft_candidate_path": str(result.screened_dft_candidate_path.resolve()),
        "first_principles_input_source_path": str(result.screened_dft_candidate_path.resolve()),
        "first_principles_input_xyz": str(first_principles_input.resolve()),
        "screened_count": int(result.screened_count),
        "selected_count_per_system": dict(result.selected_count_per_system),
        "stage_payload": stage_payload,
    }


def _read_atoms(path: Path) -> list[Any]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    from workflow_core.structures.formats import read_extxyz_structures

    return read_extxyz_structures(path)
