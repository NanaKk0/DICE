from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from workflow_core.student_pipeline.artifacts.stage_artifacts import (
    build_artifact_resolution_context,
    resolve_stage_artifacts,
)
from workflow_core.student_pipeline.config.stage_config import canonical_stage_type
from workflow_core.student_pipeline.orchestration.derived_inputs import materialize_stage_inputs
from workflow_core.student_pipeline.runtime.commit import commit_student_pipeline_stage_plan
from workflow_core.student_pipeline.runtime.commit_plan import StagePublishPlan
from workflow_core.student_pipeline.stages.base import resolve_stage_config, resolve_stage_context
from workflow_core.student_pipeline.stages.publish import stage_publish_staging_dir


@dataclass(frozen=True, slots=True)
class StudentStageRunnerContext:
    runtime_state: dict[str, Any]
    invocation: dict[str, Any]
    stage_record: dict[str, Any]
    stage_context: dict[str, Any]
    execution_context: dict[str, Any]
    stage_cfg: dict[str, Any]
    artifacts: Any
    staging_dir: Path


StageExecutor = Callable[[StudentStageRunnerContext], dict[str, Any]]
StagePublishPlanBuilder = Callable[[StudentStageRunnerContext, dict[str, Any]], StagePublishPlan]


def run_student_stage_with_harness(
    runtime_state: dict[str, Any],
    *,
    student_round: int,
    stage_run_index: int,
    expected_stage_type: str,
    execute: StageExecutor,
    build_publish_plan: StagePublishPlanBuilder,
) -> dict[str, Any]:
    invocation, stage_context = resolve_stage_context(
        runtime_state,
        student_round=student_round,
        stage_run_index=stage_run_index,
    )
    if canonical_stage_type(stage_context["stage"]) != canonical_stage_type(expected_stage_type):
        raise ValueError(
            f"run_{canonical_stage_type(expected_stage_type)}_stage requires a "
            f"{canonical_stage_type(expected_stage_type)} stage, got {stage_context['stage']!r}"
        )
    stage_record = _stage_record(invocation, stage_run_index)
    materialize_stage_inputs(runtime_state, stage_record)
    stage_cfg = resolve_stage_config(stage_context, expected_stage_type)
    artifacts = resolve_stage_artifacts(build_artifact_resolution_context(runtime_state, stage_context, stage_cfg))
    staging_dir = stage_publish_staging_dir(stage_context["stage_dir"])
    execution_context = dict(stage_context)
    execution_context["stage_results_dir"] = str(staging_dir.resolve())
    execution_context["artifacts"] = artifacts
    runner_context = StudentStageRunnerContext(
        runtime_state=runtime_state,
        invocation=invocation,
        stage_record=stage_record,
        stage_context=stage_context,
        execution_context=execution_context,
        stage_cfg=stage_cfg,
        artifacts=artifacts,
        staging_dir=staging_dir,
    )
    stage_result = dict(execute(runner_context) or {})
    commit_result = commit_student_pipeline_stage_plan(
        runtime_state,
        build_publish_plan(runner_context, stage_result),
    )
    return {
        "stage_context": stage_context,
        "stage_result": stage_result,
        "commit_result": commit_result,
    }


def _stage_record(invocation: dict[str, Any], stage_run_index: int) -> dict[str, Any]:
    for stage in list(invocation.get("stages", []) or []):
        if int(stage.get("stage_run_index", 0)) == int(stage_run_index):
            return stage
    raise ValueError(f"unknown stage_run_index: {stage_run_index}")
