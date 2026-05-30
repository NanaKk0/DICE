from __future__ import annotations

from typing import Any

from workflow_core.student.subworkflow_service import run_student_active_learning_subworkflow
from workflow_runtime.artifacts.checks import student_subworkflow_complete


def student_subworkflow_success_check(paths, teacher_round: int) -> bool:
    return student_subworkflow_complete(paths.student_subworkflow_artifacts_dir(teacher_round))


def run_student_subworkflow_substep(cfg: dict[str, Any], paths, teacher_round: int, metrics_logger=None) -> str:
    return run_student_active_learning_subworkflow(
        cfg["student_subworkflow"]["config"],
        paths,
        teacher_round,
        metrics_logger=metrics_logger,
        extra_cfg={"teacher_committee_finetune": dict(cfg.get("teacher_committee_finetune", {}) or {})},
    )
