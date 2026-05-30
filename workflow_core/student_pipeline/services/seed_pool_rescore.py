from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.seed_pool.rescore import rescore_atoms_with_student_backend
from workflow_core.student_pipeline.artifacts.successful_stage_resolver import SuccessfulStageArtifactResolver


def rescore_seed_pool_refresh_inputs(
    runtime_state: dict[str, Any],
    stage_context: dict[str, Any],
    current_seed_atoms: list[Any],
    candidate_increment: list[Any],
    failed_increment: list[Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    summary_path = _latest_student_committee_path(runtime_state, stage_context)
    cfg = dict(runtime_state.get("cfg", {}) or {})
    temp_root = Path(stage_context["stage_results_dir"]).resolve() / "_rescore"
    rescored_seed = rescore_atoms_with_student_backend(
        current_seed_atoms,
        cfg=cfg,
        committee_summary_path=summary_path,
        temp_root=temp_root,
        prefix="seed_pool",
    )
    rescored_candidate = rescore_atoms_with_student_backend(
        candidate_increment,
        cfg=cfg,
        committee_summary_path=summary_path,
        temp_root=temp_root,
        prefix="candidate_increment",
    )
    rescored_failed = rescore_atoms_with_student_backend(
        failed_increment,
        cfg=cfg,
        committee_summary_path=summary_path,
        temp_root=temp_root,
        prefix="failed_increment",
    )
    return rescored_seed, rescored_candidate, rescored_failed


def _latest_student_committee_path(runtime_state: dict[str, Any], stage_context: dict[str, Any]) -> str:
    current_stage = _stage_record_for_context(runtime_state, stage_context)
    if current_stage is None:
        raise RuntimeError("seed_pool_refresh requires a stage context for latest successful train resolution")
    latest_committee = SuccessfulStageArtifactResolver(runtime_state).latest_train_committee_before(current_stage)
    if latest_committee is None:
        raise RuntimeError("seed_pool_refresh requires latest_student_committee_path for rescoring")
    path = latest_committee.committee_summary_path.resolve()
    if not path.exists():
        raise RuntimeError(f"seed_pool_refresh committee summary is missing: {path}")
    return str(path)


def _stage_record_for_context(runtime_state: dict[str, Any], stage_context: dict[str, Any]) -> dict[str, Any] | None:
    if not stage_context:
        return None
    student_round = int(stage_context["student_round"])
    stage_run_index = int(stage_context["stage_run_index"])
    for invocation in list(runtime_state.get("invocations", []) or []):
        if int(invocation.get("student_round", 0)) != student_round:
            continue
        for stage in list(invocation.get("stages", []) or []):
            if int(stage.get("stage_run_index", 0)) == stage_run_index:
                return dict(stage)
    return dict(stage_context)
