from __future__ import annotations

from pathlib import Path

from workflow_backends.student.base import StudentSamplingRequest, StudentTrainRequest
from workflow_backends.student.request_adapter import (
    StudentSamplingRequestInput,
    StudentTrainRequestInput,
    resolved_train_decision_context,
    validate_canonical_sampling_stage_config,
)
from workflow_core.student_pipeline.config.train_regime import INITIALIZATION_MODE_PREVIOUS_COMMITTEE


def build_train_request(request_input: StudentTrainRequestInput) -> StudentTrainRequest:
    selector = str(Path(request_input.train_recipe_ref).resolve())
    decision_context = resolved_train_decision_context(request_input)
    continue_training = bool(decision_context["requires_previous_committee"])
    training_stage = (
        "finetune"
        if str(decision_context["initialization_mode"]) == INITIALIZATION_MODE_PREVIOUS_COMMITTEE
        else "bootstrap"
    )

    work_root = Path(request_input.work_root).resolve()
    context = {
        "selected_train_recipe_selector": selector,
        "resolved_train_decision": decision_context,
        "student_identity": dict(request_input.student_model_identity or {}),
        "stage_run_index": int(request_input.stage_run_index),
        "backend_log_dir": str((work_root / "backend_logs").resolve()),
        "foundation_bootstrap": request_input.foundation_context,
    }
    context.update(dict(request_input.extra_context or {}))
    return StudentTrainRequest(
        backend_name="deepmd",
        outer_round=int(request_input.teacher_round),
        student_round=int(request_input.student_round),
        work_root=str(work_root),
        summary_path=str(Path(request_input.summary_path).resolve()),
        backend_cfg=str(selector),
        dataset_roots=list(request_input.dataset_roots or []),
        dataset_path=str(Path(request_input.dataset_path).resolve()) if request_input.dataset_path is not None else None,
        previous_model_root=(
            str(Path(request_input.previous_model_root).resolve())
            if request_input.previous_model_root is not None
            else None
        ),
        continue_training=continue_training,
        training_stage=training_stage,
        execution_context=request_input.execution_context,
        context=context,
    )


def build_sampling_request(request_input: StudentSamplingRequestInput) -> StudentSamplingRequest:
    sampling_backend = str(request_input.sampling_backend or "").strip().lower()
    if sampling_backend != "lammps":
        raise ValueError(
            f"Unsupported sampling_backend for DeepMD sampling request adapter: {request_input.sampling_backend!r}; "
            "supported sampling_backend values: lammps"
        )
    stage_cfg = validate_canonical_sampling_stage_config(request_input.stage_config, sampling_backend="lammps")
    md_cfg = dict(stage_cfg.get("md", {}) or {})
    execution_cfg = dict(stage_cfg.get("execution", {}) or {})
    backend_identity = dict(request_input.sampling_backend_identity or {})
    native_recipe = {}
    lammps_cfg = dict(backend_identity)
    if stage_cfg.get("native_recipe_ref"):
        native_recipe["input_template_ref"] = str(Path(str(stage_cfg["native_recipe_ref"])).resolve())
        lammps_cfg.update(native_recipe)
    if "temperature_schedule" in md_cfg:
        lammps_cfg["temperature_override_declared"] = True
    if "steps" in md_cfg:
        lammps_cfg["steps_override_declared"] = True
    if "type_map" not in lammps_cfg:
        type_map = list(dict(request_input.student_model_identity or {}).get("type_map", []) or [])
        if type_map:
            lammps_cfg["type_map"] = [str(item) for item in type_map]
            backend_identity["type_map"] = [str(item) for item in type_map]
    _apply_md_lammps_defaults(lammps_cfg, md_cfg)
    uncertainty = dict(request_input.student_uncertainty or {})
    if "trust_lo" in uncertainty and "f_trust_lo" not in lammps_cfg:
        lammps_cfg["f_trust_lo"] = float(uncertainty["trust_lo"])
    if "trust_hi" in uncertainty and "f_trust_hi" not in lammps_cfg:
        lammps_cfg["f_trust_hi"] = float(uncertainty["trust_hi"])
    profile_lammps_cfg = {
        **lammps_cfg,
        **execution_cfg,
    }
    context = {
        "sampling_progress_poll_interval_sec": 2.0,
        "stage_run_index": int(request_input.stage_run_index),
        "student_identity": dict(request_input.student_model_identity or {}),
    }
    context.update(dict(request_input.extra_context or {}))
    seed_xyz_path = Path(str(context.get("seed_xyz_path") or request_input.seed_xyz_path)).resolve()
    return StudentSamplingRequest(
        backend_name="deepmd",
        outer_round=int(request_input.teacher_round),
        student_round=int(request_input.student_round),
        committee_summary_path=str(Path(request_input.committee_summary_path).resolve()),
        seed_xyz_path=str(seed_xyz_path),
        work_root=str(Path(request_input.work_root).resolve()),
        backend_cfg={
            "backend": sampling_backend,
            "driver_policy": "round_robin",
            "md": md_cfg,
            "lammps": profile_lammps_cfg,
            "candidate_train_selection": {},
        },
        reference_xyz_files=list(request_input.reference_xyz_files or []),
        uncertainty_cfg={},
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
        output_requirements=_lammps_output_requirements(md_cfg),
    )


def _apply_md_lammps_defaults(lammps_cfg: dict, md_cfg: dict) -> None:
    for key in ("model_devi_interval", "dump_interval"):
        if key in md_cfg:
            lammps_cfg[key] = int(md_cfg[key])
    if "atomic_model_devi" in md_cfg:
        lammps_cfg["atomic_model_devi"] = bool(md_cfg["atomic_model_devi"])


def _lammps_output_requirements(md_cfg: dict) -> dict:
    requirements = {}
    for key in ("model_devi_interval", "dump_interval"):
        if key in md_cfg:
            requirements[key] = int(md_cfg[key])
    if "atomic_model_devi" in md_cfg:
        requirements["atomic_model_devi"] = bool(md_cfg["atomic_model_devi"])
    return requirements
