from __future__ import annotations

from typing import Any

from workflow_core.committee_cycle.substeps.teacher_committee_finetune import (
    run_teacher_committee_finetune_substep,
    teacher_committee_finetune_success_check,
)
from workflow_core.student import planning as student_planning
from workflow_runtime.plan import StepPlan


def build_committee_cycle_plan(
    cfg: dict[str, Any],
    paths,
    teacher_round: int,
) -> list[StepPlan]:
    round_id = int(teacher_round)
    teacher_step = StepPlan(
        id=f"r{round_id}_committee_cycle.teacher_committee_finetune",
        name="teacher_committee_finetune",
        round_id=round_id,
        index=0,
        run=lambda: run_teacher_committee_finetune_substep(cfg, paths, round_id),
        success_check=lambda: teacher_committee_finetune_success_check(paths, round_id, cfg),
        output_dir=paths.teacher_committee_finetune_artifacts_dir(round_id) if paths is not None else None,
        backend_log_roots=(
            paths.teacher_committee_finetune_backend_logs_dir(round_id),
        )
        if paths is not None
        else (),
        backup_partial_outputs=False,
        parent_step="committee_cycle",
        visibility="user",
    )

    student_steps = list(student_planning.build_student_subworkflow_step_plans(cfg, paths, round_id))
    for offset, step in enumerate(student_steps, start=1):
        step.index = offset
    return [teacher_step, *student_steps]
