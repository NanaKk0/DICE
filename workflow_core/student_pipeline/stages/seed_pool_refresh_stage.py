from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from workflow_core.common.io import write_candidate_xyz
from workflow_core.round_context import StudentRoundContext
from workflow_core.seed_pool.metadata import combine_grouped_atoms, group_atoms_by_system
from workflow_core.student_pipeline.services.seed_pool_rescore import (
    rescore_seed_pool_refresh_inputs as _rescore_refresh_inputs,
)
from workflow_core.student_pipeline.stages.base import resolve_stage_config
from workflow_core.student_pipeline.stages.harness import StudentStageRunnerContext, run_student_stage_with_harness
from workflow_core.student_pipeline.stages.publish import (
    build_stage_publish_plan,
    prepare_stage_result_publications,
)
from workflow_policies.seed_pool_refresh import (
    SeedPoolRefreshPolicy,
    apply_seed_pool_refresh_policy,
    merge_refresh_sources,
)


def run_seed_pool_refresh_stage(
    runtime_state: dict[str, Any],
    *,
    student_round: int,
    stage_run_index: int,
    stage_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def execute(context: StudentStageRunnerContext) -> dict[str, Any]:
        executor = stage_executor or (lambda stage_context: _default_seed_pool_refresh_executor(runtime_state, stage_context))
        return dict(executor(context.execution_context) or {})

    def publish_plan(context: StudentStageRunnerContext, stage_result: dict[str, Any]):
        seed_pool_path = stage_result.get("seed_pool_path")
        if not seed_pool_path:
            raise ValueError("seed_pool_refresh stage executor must return seed_pool_path")
        return build_stage_publish_plan(
            student_round=student_round,
            stage_run_index=stage_run_index,
            artifact_publications=prepare_stage_result_publications(
                "seed_pool_refresh",
                staging_dir=context.staging_dir,
                target_root=context.stage_context["stage_dir"],
                stage_result=stage_result,
                role_source_fields={"seed_pool": "seed_pool_path"},
            ),
            stage_payload=dict(stage_result.get("stage_payload", {}) or {}),
        )

    return run_student_stage_with_harness(
        runtime_state,
        student_round=student_round,
        stage_run_index=stage_run_index,
        expected_stage_type="seed_pool_refresh",
        execute=execute,
        build_publish_plan=publish_plan,
    )


def _default_seed_pool_refresh_executor(runtime_state: dict[str, Any], stage_context: dict[str, Any]) -> dict[str, Any]:
    stage_cfg = resolve_stage_config(stage_context, "seed_pool_refresh")
    artifacts = stage_context.get("artifacts")
    inputs = getattr(artifacts, "inputs", None)
    if inputs is None:
        raise ValueError("seed_pool_refresh stage artifacts are missing typed inputs")
    current_seed_atoms = _read_atoms(inputs.seed_pool_xyz)
    candidate_increment = _read_atoms(inputs.candidate_increment_xyz)
    failed_increment = _read_atoms(inputs.failed_increment_xyz)
    rescored_seed_atoms, rescored_candidate_increment, rescored_failed_increment = _rescore_refresh_inputs(
        runtime_state,
        stage_context,
        current_seed_atoms,
        candidate_increment,
        failed_increment,
    )

    source_grouped = merge_refresh_sources(
        group_atoms_by_system(rescored_seed_atoms),
        group_atoms_by_system(rescored_candidate_increment),
        group_atoms_by_system(rescored_failed_increment),
    )
    policy = _resolve_stage_policy(stage_cfg)
    refresh_result = apply_seed_pool_refresh_policy(
        source_grouped,
        policy=policy,
        context=StudentRoundContext(
            teacher_round=int(stage_context["teacher_round"]),
            student_round=int(stage_context["student_round"]),
        ),
    )
    refreshed_atoms = combine_grouped_atoms(refresh_result.refreshed_grouped)
    seed_pool_path = write_candidate_xyz(
        Path(stage_context["stage_results_dir"]).resolve() / "seed_pool.xyz",
        refreshed_atoms,
    ).resolve()
    return {
        "seed_pool_path": str(seed_pool_path),
        "stage_payload": {
            "seed_pool_path": str(seed_pool_path),
            "candidate_increment_start_index": 0,
            "failed_increment_start_index": 0,
            "candidate_increment_consumed_count": len(candidate_increment),
            "failed_increment_consumed_count": len(failed_increment),
            "current_seed_pool_count": len(current_seed_atoms),
            "candidate_increment_count": len(candidate_increment),
            "failed_increment_count": len(failed_increment),
            "rescored_candidate_increment_count": len(rescored_candidate_increment),
            "rescored_failed_increment_count": len(rescored_failed_increment),
            "rescored_seed_pool_count": len(rescored_seed_atoms),
            "seed_pool_size_after": len(refreshed_atoms),
            "eligible_count_per_system": dict(refresh_result.eligible_count_per_system),
            "selected_count_per_system": dict(refresh_result.selected_count_per_system),
        },
    }


def _resolve_stage_policy(stage_cfg: dict[str, Any]) -> SeedPoolRefreshPolicy:
    keep_if_ge_low = stage_cfg.get("keep_if_ge_low")
    return SeedPoolRefreshPolicy(
        keep_if_ge_low=None if keep_if_ge_low is None else float(keep_if_ge_low),
        keep_if_lt_high=float(stage_cfg.get("keep_if_lt_high", 0.40)),
        max_pool_size_per_system=int(stage_cfg.get("max_pool_size_per_system", 2000)),
        selection_mode=str(stage_cfg.get("selection_mode", "top_uncertainty_per_system")).strip().lower(),
        random_seed=int(stage_cfg.get("random_seed", 20260502)),
        dedup_enabled=bool(stage_cfg.get("dedup_enabled", True)),
        reset_times_used=bool(stage_cfg.get("reset_times_used", False)),
    )


def _read_atoms(path: Path) -> list[Any]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    from workflow_core.structures.formats import read_extxyz_structures

    return read_extxyz_structures(path)
