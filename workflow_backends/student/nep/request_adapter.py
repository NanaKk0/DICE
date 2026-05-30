from __future__ import annotations

from pathlib import Path

from workflow_backends.student.base import StudentSamplingRequest, StudentTrainRequest
from workflow_backends.student.config import load_nep_training_config
from workflow_backends.student.request_adapter import (
    StudentSamplingRequestInput,
    StudentTrainRequestInput,
    resolved_train_decision_context,
    validate_canonical_sampling_stage_config,
)
from workflow_core.student_pipeline.config.train_regime import INITIALIZATION_MODE_PREVIOUS_COMMITTEE


def build_train_request(request_input: StudentTrainRequestInput) -> StudentTrainRequest:
    if request_input.dataset_path is None:
        raise ValueError("NEP training adapter requires dataset_path")
    selector = str(Path(request_input.train_recipe_ref).resolve())
    decision_context = resolved_train_decision_context(request_input)
    recipe_branch = str(decision_context["effective_train_recipe_branch"])
    training_stage = (
        "finetune"
        if str(decision_context["initialization_mode"]) == INITIALIZATION_MODE_PREVIOUS_COMMITTEE
        else "bootstrap"
    )
    if selector:
        nep_cfg = load_nep_training_config(
            selector,
            training_stage=training_stage,
            recipe_branch=recipe_branch,
        )
    else:
        nep_cfg = {}
    workflow_nep_cfg = dict(
        dict(request_input.extra_context or {})
        .get("backend_workflow_config", {})
        .get("nep", {})
        or {}
    )
    for key, value in workflow_nep_cfg.items():
        if value is not None:
            nep_cfg[key] = value
    identity = dict(request_input.student_model_identity or {})
    for key, value in identity.items():
        if value is not None:
            nep_cfg[key] = value

    work_root = Path(request_input.work_root).resolve()
    summary_path = Path(request_input.summary_path).resolve()
    dataset_path = Path(request_input.dataset_path).resolve()
    context = {
        "dataset_path": str(dataset_path),
        "work_root": str(work_root),
        "summary_path": str(summary_path),
        "previous_model_root": (
            str(Path(request_input.previous_model_root).resolve())
            if request_input.previous_model_root is not None
            else None
        ),
        "continue_training_override": bool(decision_context["requires_previous_committee"]),
        "training_stage": training_stage,
        "backend_log_dir": str((work_root / "backend_logs").resolve()),
        "log_prefix": f"student_subworkflow_r{request_input.teacher_round}_student{request_input.student_round}_train",
        "stage_run_index": int(request_input.stage_run_index),
        "selected_train_recipe_selector": selector,
        "resolved_train_decision": decision_context,
        "foundation_bootstrap": request_input.foundation_context,
    }
    context.update(dict(request_input.extra_context or {}))
    return StudentTrainRequest(
        backend_name="nep",
        outer_round=int(request_input.teacher_round),
        student_round=int(request_input.student_round),
        work_root=str(work_root),
        summary_path=str(summary_path),
        backend_cfg=nep_cfg,
        dataset_path=str(dataset_path),
        dataset_roots=list(request_input.dataset_roots or []),
        previous_model_root=context["previous_model_root"],
        continue_training=bool(decision_context["requires_previous_committee"]),
        training_stage=training_stage,
        execution_context=request_input.execution_context,
        context=context,
    )


def build_sampling_request(request_input: StudentSamplingRequestInput) -> StudentSamplingRequest:
    sampling_backend = str(request_input.sampling_backend or "").strip().lower()
    if sampling_backend != "gpumd":
        raise ValueError(
            f"Unsupported sampling_backend for NEP sampling request adapter: {request_input.sampling_backend!r}; "
            "supported sampling_backend values: gpumd"
        )
    stage_cfg = validate_canonical_sampling_stage_config(request_input.stage_config, sampling_backend="gpumd")
    md_cfg = dict(stage_cfg.get("md", {}) or {})
    execution_cfg = dict(stage_cfg.get("execution", {}) or {})
    backend_identity = dict(request_input.sampling_backend_identity or {})
    native_recipe = {}
    gpumd_cfg = dict(backend_identity)
    if stage_cfg.get("native_recipe_ref"):
        native_recipe["run_template_ref"] = str(Path(str(stage_cfg["native_recipe_ref"])).resolve())
        gpumd_cfg.update(native_recipe)

    if md_cfg.get("temperature_schedule") is not None:
        gpumd_cfg["temperatures"] = list(md_cfg.get("temperature_schedule") or [])
    if "temperature_schedule" in md_cfg:
        gpumd_cfg["temperature_override_declared"] = True
    gpumd_cfg["steps"] = int(md_cfg.get("steps", gpumd_cfg.get("steps_short", gpumd_cfg.get("steps", 500))))
    if "steps" in md_cfg:
        gpumd_cfg["steps_override_declared"] = True
    for key in ("active_interval", "dump_interval"):
        if key in md_cfg:
            gpumd_cfg[key] = int(md_cfg[key])
    if "active_outputs" in md_cfg:
        gpumd_cfg["active_outputs"] = dict(md_cfg["active_outputs"])
    for key, value in execution_cfg.items():
        gpumd_cfg[key] = value
    if gpumd_cfg.get("max_parallel_tasks") is not None:
        gpumd_cfg["max_parallel_tasks"] = int(gpumd_cfg["max_parallel_tasks"])

    uncertainty = dict(request_input.student_uncertainty or {})
    context = {
        "committee_summary_path": str(Path(request_input.committee_summary_path).resolve()),
        "seed_xyz_path": str(Path(request_input.seed_xyz_path).resolve()),
        "work_root": str(Path(request_input.work_root).resolve()),
        "gpumd_cfg": gpumd_cfg,
        "uncertainty_cfg": uncertainty,
        "stage_run_index": int(request_input.stage_run_index),
        "student_identity": dict(request_input.student_model_identity or {}),
    }
    context.update(dict(request_input.extra_context or {}))
    seed_xyz_path = Path(str(context.get("seed_xyz_path") or request_input.seed_xyz_path)).resolve()
    return StudentSamplingRequest(
        backend_name="nep",
        outer_round=int(request_input.teacher_round),
        student_round=int(request_input.student_round),
        committee_summary_path=str(Path(request_input.committee_summary_path).resolve()),
        seed_xyz_path=str(seed_xyz_path),
        work_root=str(Path(request_input.work_root).resolve()),
        backend_cfg={
            "committee_summary_path": str(Path(request_input.committee_summary_path).resolve()),
            "seed_xyz_path": str(seed_xyz_path),
            "gpumd": gpumd_cfg,
            "uncertainty": uncertainty,
        },
        reference_xyz_files=list(request_input.reference_xyz_files or []),
        uncertainty_cfg=uncertainty,
        summary_extra=dict(request_input.summary_extra or {}),
        execution_context=request_input.execution_context,
        context=context,
        backend_identity=backend_identity,
        stage_execution=execution_cfg,
        md_expansion=md_cfg,
        native_recipe=native_recipe,
        scalar_overrides={
            "temperature_schedule": "temperature_schedule" in md_cfg,
            "steps": "steps" in md_cfg,
        },
        output_requirements=_gpumd_output_requirements(md_cfg),
    )


def _gpumd_output_requirements(md_cfg: dict) -> dict:
    requirements = {}
    for key in ("active_interval", "dump_interval"):
        if key in md_cfg:
            requirements[key] = int(md_cfg[key])
    if "active_outputs" in md_cfg:
        requirements["active_outputs"] = dict(md_cfg["active_outputs"])
    return requirements
