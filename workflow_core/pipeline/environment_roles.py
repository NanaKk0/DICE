from __future__ import annotations

from typing import Any

from workflow_core.student_pipeline.request_context import canonical_sampling_backend_name, canonical_student_backend_name
from workflow_runtime.environment.roles import sampling_backend_role, student_backend_role, teacher_backend_role


def selected_environment_roles_for_workflow(workflow: dict[str, Any]) -> set[str]:
    selected: set[str] = set()
    pipeline = [str(item) for item in list(workflow.get("pipeline", []) or [])]
    if "teacher_foundation_train" in pipeline:
        foundation_cfg = dict(workflow.get("teacher_foundation_train", {}) or {})
        selected.add(teacher_backend_role(foundation_cfg.get("teacher_backend", "mace"), "foundation_train"))
    if "committee_cycle" in pipeline:
        selected.update(_committee_cycle_environment_roles(workflow.get("committee_cycle")))
    return selected


def _committee_cycle_environment_roles(raw_committee_ref: Any) -> set[str]:
    committee_ref = dict(raw_committee_ref or {})
    config_path = committee_ref.get("config")
    if config_path in (None, ""):
        return set()
    from workflow_core.committee_cycle.config import load_committee_cycle_config

    committee_cfg = load_committee_cycle_config(config_path)
    teacher_backend = str(
        dict(committee_cfg.get("teacher_committee_finetune", {}) or {}).get("teacher_backend", "mace")
    )
    selected = {teacher_backend_role(teacher_backend, "finetune")}
    selected.update(_student_subworkflow_environment_roles(committee_cfg.get("student_subworkflow"), teacher_backend))
    return selected


def _student_subworkflow_environment_roles(raw_student_ref: Any, teacher_backend: str) -> set[str]:
    student_ref = dict(raw_student_ref or {})
    config_path = student_ref.get("config")
    if config_path in (None, ""):
        return set()
    from workflow_core.student.config import load_student_active_learning_config

    student_cfg = load_student_active_learning_config(config_path)
    student_backend = canonical_student_backend_name(str(student_cfg.get("student_backend", "deepmd") or "deepmd"))
    sampling_backend = canonical_sampling_backend_name(
        str(student_cfg.get("sampling_backend", "") or ""),
        student_backend=student_backend,
    )
    selected: set[str] = set()
    for stage_type in _scheduled_student_stage_types(student_cfg):
        if stage_type == "relabel":
            selected.add(teacher_backend_role(teacher_backend, "relabel"))
        elif stage_type == "train":
            selected.add(student_backend_role(student_backend, "train"))
        elif stage_type == "sampling":
            selected.add(sampling_backend_role(sampling_backend, "sample"))
    return selected


def _scheduled_student_stage_types(student_cfg: dict[str, Any]) -> tuple[str, ...]:
    resolved_templates = dict(student_cfg.get("resolved_student_pipeline_templates", {}) or {})
    stage_types: list[str] = []
    for schedule_entry in list(student_cfg.get("student_pipeline_schedule", []) or []):
        template_name = str(schedule_entry.get("template", ""))
        template_payload = dict(resolved_templates.get(template_name, {}) or {})
        for stage in list(template_payload.get("stages", []) or []):
            stage_types.append(str(stage.get("stage", "")).strip())
    return tuple(stage_types)
