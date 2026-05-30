from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from workflow_core.common.io import read_json
from workflow_backends.sampling import get_sampling_backend
from workflow_backends.sampling.request_adapter import build_md_sampling_request
from workflow_backends.student.request_adapter import build_student_sampling_request
from workflow_core.student_pipeline.request_context import build_backend_identity_context, build_sampling_request_input
from workflow_core.student_pipeline.services.seed_selection import (
    SeedSelectionResult,
    select_sampling_seeds,
    stage_seed_usage_update,
    write_selected_seed_artifact,
)
from workflow_core.student_pipeline.stages.base import resolve_stage_config
from workflow_core.student_pipeline.stages.harness import StudentStageRunnerContext, run_student_stage_with_harness
from workflow_core.student_pipeline.stages.publish import (
    build_backend_log_publications,
    build_stage_publish_plan,
    prepare_stage_output_publications,
    stage_payload_without_commit_fields,
)
from workflow_core.structures.formats import copy_extxyz_file


def run_sampling_stage(
    runtime_state: dict[str, Any],
    *,
    student_round: int,
    stage_run_index: int,
    stage_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def execute(context: StudentStageRunnerContext) -> dict[str, Any]:
        executor = stage_executor or (
            lambda stage_context: _default_sampling_stage_executor(
                runtime_state,
                stage_context,
                context.artifacts,
                staging_dir=context.staging_dir,
            )
        )
        return dict(executor(context.execution_context) or {})

    def publish_plan(context: StudentStageRunnerContext, stage_result: dict[str, Any]):
        staged_candidate = _stage_sampling_output(
            stage_result.get("student_candidate_output_path"),
            context.staging_dir / "candidate.xyz",
        )
        staged_failed = _stage_sampling_output(
            stage_result.get("student_failed_output_path"),
            context.staging_dir / "failed.xyz",
        )
        staged_selected_seed = _stage_sampling_output(
            stage_result.get("selected_seed_xyz_path"),
            context.staging_dir / "seed_pool.xyz",
        )
        staged_seed_successor = _stage_seed_pool_successor(
            context.artifacts,
            stage_result,
            staging_dir=context.staging_dir,
        )
        return build_stage_publish_plan(
            student_round=student_round,
            stage_run_index=stage_run_index,
            stage_payload=stage_payload_without_commit_fields(stage_result.get("stage_payload", {})),
            artifact_publications=prepare_stage_output_publications(
                "sampling",
                staging_dir=context.staging_dir,
                target_root=context.stage_context["stage_dir"],
                role_sources={
                    "selected_seed": staged_selected_seed,
                    "student_candidate": staged_candidate,
                    "student_failed": staged_failed,
                    "seed_pool_successor": staged_seed_successor,
                },
            ),
            backend_log_publications=build_backend_log_publications(
                context.staging_dir,
                Path(context.stage_record["log_paths"]["stdout_path"]).resolve().parent / "backend_logs",
            ),
        )

    return run_student_stage_with_harness(
        runtime_state,
        student_round=student_round,
        stage_run_index=stage_run_index,
        expected_stage_type="sampling",
        execute=execute,
        build_publish_plan=publish_plan,
    )


def _default_sampling_stage_executor(
    runtime_state: dict[str, Any],
    stage_context: dict[str, Any],
    artifacts,
    *,
    staging_dir: Path,
) -> dict[str, Any]:
    cfg = dict(runtime_state.get("cfg", {}) or {})
    paths = runtime_state.get("paths")
    if paths is None:
        raise ValueError("runtime_state is missing paths for default sampling execution")
    stage_cfg = resolve_stage_config(stage_context, "sampling")
    identity = build_backend_identity_context(cfg)
    backend = _load_sampling_backend(identity.sampling_backend)
    selected_seed_path, selection_result = _select_sampling_seeds(
        runtime_state,
        stage_context,
        stage_cfg,
        artifacts,
        selected_seed_target=staging_dir / "seed_pool.xyz",
    )
    if artifacts.adapter_context.committee_summary_path is None:
        raise ValueError("sampling requires a latest successful train committee before the sampling stage")
    student_request = build_student_sampling_request(
        build_sampling_request_input(
            cfg=cfg,
            stage_context=stage_context,
            stage_cfg=stage_cfg,
            artifacts=artifacts,
            work_root=staging_dir,
            extra_context={"seed_xyz_path": str(Path(selected_seed_path).resolve())},
        )
    )
    request = build_md_sampling_request(student_request)
    summary_path = backend.sample_with_committee(paths, request)
    summary_payload = read_json(summary_path)
    sampling_summary = backend.normalize_sampling_summary(
        paths,
        request,
        Path(summary_path).resolve(),
        dict(summary_payload or {}),
    )
    payload = dict(getattr(sampling_summary, "payload", {}) or {})
    candidate_count = int(getattr(sampling_summary, "candidate_count", payload.get("candidate_count", 0)) or 0)
    failed_count = int(getattr(sampling_summary, "failed_count", payload.get("failed_count", 0)) or 0)
    effective_count = int(getattr(sampling_summary, "effective_sample_count", payload.get("effective_sample_count", 0)) or 0)
    payload.setdefault("candidate_count", candidate_count)
    payload.setdefault("failed_count", failed_count)
    payload.setdefault("accurate_count", max(0, effective_count - candidate_count - failed_count))
    payload.setdefault("task_count", int(payload.get("task_total", effective_count) or 0))
    payload.setdefault("failed_task_count", int(payload.get("task_failed_count", 0) or 0))
    payload.setdefault(
        "succeeded_task_count",
        max(0, int(payload.get("task_count", 0) or 0) - int(payload.get("failed_task_count", 0) or 0)),
    )
    payload.setdefault("candidate_count_per_system", {})
    payload.setdefault("failed_count_per_system", {})
    payload.setdefault("accurate_count_per_system", {})
    payload.update(
        {
            "selected_seed_xyz_path": str(Path(selected_seed_path).resolve()),
            "selection_mode": str(selection_result.selection_mode),
            "selected_seed_count": len(selection_result.selected_structure_ids),
            "selected_seed_count_per_system": dict(selection_result.selected_count_per_system),
            "selected_structure_ids": list(selection_result.selected_structure_ids),
            "seed_usage_updated_count": len(selection_result.selected_structure_ids),
        }
    )
    return {
        "student_candidate_output_path": payload.get("candidate_xyz"),
        "student_failed_output_path": payload.get("failed_xyz"),
        "selected_seed_xyz_path": str(Path(selected_seed_path).resolve()),
        "stage_payload": payload,
    }


def _load_sampling_backend(name: str):
    return get_sampling_backend(name)


def _select_sampling_seeds(
    runtime_state: dict[str, Any],
    stage_context: dict[str, Any],
    stage_cfg: dict[str, Any],
    artifacts,
    *,
    selected_seed_target: Path,
) -> tuple[Path, SeedSelectionResult]:
    del runtime_state
    selection_result = select_sampling_seeds(
        artifacts.inputs.seed_pool_path,
        dict(stage_cfg or {}),
        teacher_round=int(stage_context["teacher_round"]),
        student_round=int(stage_context["student_round"]),
    )
    selected_seed_path = write_selected_seed_artifact(selection_result, selected_seed_target)
    return selected_seed_path, selection_result


def _stage_seed_pool_successor(artifacts, stage_result: dict[str, Any], *, staging_dir: Path) -> Path:
    selected_seed_xyz_path = stage_result.get("selected_seed_xyz_path")
    target = staging_dir / "seed_pool_after_sampling.xyz"
    if selected_seed_xyz_path in (None, ""):
        return _copy_or_empty(
            artifacts.inputs.seed_pool_path,
            target,
        )
    return stage_seed_usage_update(
        artifacts.inputs.seed_pool_path,
        selected_seed_xyz_path,
        target,
    )


def _stage_sampling_output(source: Any, target: Path) -> Path:
    if source in (None, ""):
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("", encoding="utf-8")
        return target.resolve()
    return _copy_or_empty(Path(str(source)).resolve(), target)


def _copy_or_empty(source: Path, target: Path) -> Path:
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists() and source_path != target_path:
        copy_extxyz_file(source_path, target_path, allow_empty=True)
    elif source_path.exists() and source_path == target_path:
        pass
    elif not target_path.exists():
        target_path.write_text("", encoding="utf-8")
    return target_path
