from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from workflow_core.student_pipeline.orchestration.runtime_node_contracts import student_stage_runtime_node_contract
from workflow_runtime.plan import StepPlan

StageRunner = Callable[[dict[str, Any], int, int], dict[str, Any]]


def build_concrete_student_stage_step_plan(
    runtime_state: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    stage_runners: Mapping[str, StageRunner] | None = None,
    index: int | None = None,
) -> StepPlan:
    contract = student_stage_runtime_node_contract(runtime_state, stage)
    stage_type = str(stage["stage"])
    runner = _resolve_stage_runner(stage_type, stage_runners)
    teacher_round = int(stage["teacher_round"])
    student_round = int(stage["student_round"])
    stage_run_index = int(stage["stage_run_index"])
    stage_dir = Path(str(stage["stage_dir"])).resolve()

    def run() -> Any:
        prepared_runtime_state = dict(runtime_state)
        if stage_type in {"sampling", "seed_pool_refresh"}:
            from workflow_core.student_pipeline.orchestration.orchestrator import ensure_initial_seed_pool_snapshot

            prepared_runtime_state = ensure_initial_seed_pool_snapshot(prepared_runtime_state)
        return runner(prepared_runtime_state, student_round, stage_run_index)

    return StepPlan(
        id=contract.node_id,
        name=stage_type,
        round_id=teacher_round,
        index=int(index if index is not None else max(0, stage_run_index - 1)),
        run=run,
        success_check=lambda: contract.artifact_check(),
        output_dir=stage_dir,
        backup_partial_outputs=False,
        parent_step="committee_cycle",
        visibility="user",
        identity=dict(contract.identity),
        artifact_contract_id=contract.artifact_contract_id,
        global_invariant_fingerprint=contract.global_invariant_fingerprint,
        stage_sequence_fingerprint=contract.stage_sequence_fingerprint,
        stage_local_config_fingerprint=contract.stage_local_config_fingerprint,
        artifact_roots=contract.artifact_roots,
        semantic_log_roots=contract.semantic_log_roots,
        backend_log_roots=contract.backend_log_roots,
        recovery_nodes=(contract.to_recovery_node(),),
    )


def build_concrete_student_stage_step_plans(
    runtime_state: Mapping[str, Any],
    *,
    stage_runners: Mapping[str, StageRunner] | None = None,
) -> tuple[StepPlan, ...]:
    steps: list[StepPlan] = []
    index = 0
    for invocation in list(runtime_state.get("invocations", []) or []):
        for stage in list(invocation.get("stages", []) or []):
            steps.append(
                build_concrete_student_stage_step_plan(
                    runtime_state,
                    stage,
                    stage_runners=stage_runners,
                    index=index,
                )
            )
            index += 1
    return tuple(steps)


def _resolve_stage_runner(
    stage_type: str,
    stage_runners: Mapping[str, StageRunner] | None,
) -> StageRunner:
    if stage_runners and stage_type in stage_runners:
        return stage_runners[stage_type]
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

    default_runners: dict[str, StageRunner] = {
        "relabel": lambda runtime_state, student_round, stage_run_index: run_relabel_stage(
            runtime_state,
            student_round=student_round,
            stage_run_index=stage_run_index,
        ),
        "train": lambda runtime_state, student_round, stage_run_index: run_train_stage(
            runtime_state,
            student_round=student_round,
            stage_run_index=stage_run_index,
        ),
        "sampling": lambda runtime_state, student_round, stage_run_index: run_sampling_stage(
            runtime_state,
            student_round=student_round,
            stage_run_index=stage_run_index,
        ),
        "student_relabel_input_filtering": lambda runtime_state, student_round, stage_run_index: run_student_relabel_input_filtering_stage(
            runtime_state,
            student_round=student_round,
            stage_run_index=stage_run_index,
        ),
        "seed_pool_refresh": lambda runtime_state, student_round, stage_run_index: run_seed_pool_refresh_stage(
            runtime_state,
            student_round=student_round,
            stage_run_index=stage_run_index,
        ),
        "teacher_dft_candidate_filtering": lambda runtime_state, student_round, stage_run_index: run_teacher_dft_candidate_filtering_stage(
            runtime_state,
            student_round=student_round,
            stage_run_index=stage_run_index,
        ),
    }
    try:
        return default_runners[stage_type]
    except KeyError as exc:
        raise ValueError(f"unsupported student pipeline stage type for concrete plan: {stage_type!r}") from exc
