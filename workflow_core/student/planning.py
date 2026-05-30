from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.student.config import load_student_active_learning_config
from workflow_core.student_pipeline.orchestration.concrete_stage_plan import build_concrete_student_stage_step_plans
from workflow_core.student_pipeline.orchestration.recovery_nodes import build_student_stage_recovery_nodes
from workflow_core.student_pipeline.orchestration.state_store import build_student_pipeline_runtime_state


def build_student_subworkflow_step_plans(
    cfg: dict[str, Any],
    paths,
    teacher_round: int,
) -> tuple[StepPlan, ...]:
    round_id = int(teacher_round)
    runtime_state = _student_runtime_state(cfg, paths, round_id)
    if not runtime_state.get("invocations"):
        raise ValueError("student_subworkflow config produced no concrete student_pipeline_stage nodes")

    return tuple(build_concrete_student_stage_step_plans(runtime_state))


def _student_stage_recovery_nodes(cfg: dict[str, Any], paths, teacher_round: int) -> tuple:
    if paths is None:
        return ()
    runtime_state = _student_runtime_state(cfg, paths, int(teacher_round))
    if not runtime_state.get("invocations"):
        return ()
    return tuple(build_student_stage_recovery_nodes(runtime_state))


def _student_runtime_state(cfg: dict[str, Any], paths, teacher_round: int) -> dict[str, Any]:
    if paths is None:
        raise ValueError("student_subworkflow concrete planning requires runtime paths")
    student_subworkflow_cfg = dict(cfg.get("student_subworkflow", {}) or {})
    config_path = Path(str(student_subworkflow_cfg.get("config", ""))).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"student_subworkflow config not found: {config_path}")
    student_cfg = load_student_active_learning_config(config_path)
    if isinstance(cfg.get("teacher_committee_finetune"), dict):
        student_cfg["teacher_committee_finetune"] = dict(cfg["teacher_committee_finetune"])
    if isinstance(cfg.get("teacher_committee_finetune_train_config_snapshot"), dict):
        student_cfg["teacher_committee_finetune_train_config_snapshot"] = dict(
            cfg["teacher_committee_finetune_train_config_snapshot"]
        )
    if isinstance(cfg.get("environment_profile"), dict):
        student_cfg["environment_profile"] = dict(cfg["environment_profile"])
    return build_student_pipeline_runtime_state(student_cfg, teacher_round=int(teacher_round), paths=paths)
