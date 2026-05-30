from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workflow_backends.student.base import StudentSamplingRequest, StudentTrainRequest
from workflow_backends.student.registry import get_student_model_catalog_backend
from workflow_core.student_pipeline.config.stage_config import normalize_pipeline_stage_config
from workflow_core.student_pipeline.config.train_regime import ResolvedTrainDecision


@dataclass(slots=True)
class StudentTrainRequestInput:
    student_backend: str
    teacher_round: int
    student_round: int
    stage_run_index: int
    stage_config: dict[str, Any]
    student_model_identity: dict[str, Any]
    work_root: str | Path
    summary_path: str | Path
    train_recipe_ref: str | Path
    resolved_train_decision: ResolvedTrainDecision
    dataset_roots: list[str] = field(default_factory=list)
    dataset_path: str | Path | None = None
    previous_model_root: str | Path | None = None
    foundation_context: dict[str, Any] | None = None
    execution_context: dict[str, Any] | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StudentSamplingRequestInput:
    student_backend: str
    sampling_backend: str
    teacher_round: int
    student_round: int
    stage_run_index: int
    stage_config: dict[str, Any]
    student_model_identity: dict[str, Any]
    sampling_backend_identity: dict[str, Any]
    student_uncertainty: dict[str, Any]
    committee_summary_path: str | Path
    seed_xyz_path: str | Path
    work_root: str | Path
    reference_xyz_files: list[str] = field(default_factory=list)
    summary_extra: dict[str, Any] = field(default_factory=dict)
    execution_context: dict[str, Any] | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)


def build_student_train_request(request_input: StudentTrainRequestInput) -> StudentTrainRequest:
    backend_name = _canonical_backend_name(request_input.student_backend)
    if backend_name == "deepmd":
        from workflow_backends.student.deepmd.request_adapter import build_train_request

        return build_train_request(request_input)
    if backend_name == "nep":
        from workflow_backends.student.nep.request_adapter import build_train_request

        return build_train_request(request_input)
    raise ValueError(f"Unsupported student backend for train request adapter: {request_input.student_backend!r}")


def build_student_sampling_request(request_input: StudentSamplingRequestInput) -> StudentSamplingRequest:
    backend_name = _canonical_backend_name(request_input.student_backend)
    if backend_name == "deepmd":
        from workflow_backends.student.deepmd.request_adapter import build_sampling_request

        request = build_sampling_request(request_input)
    elif backend_name == "nep":
        from workflow_backends.student.nep.request_adapter import build_sampling_request

        request = build_sampling_request(request_input)
    else:
        raise ValueError(f"Unsupported student_backend for sampling request adapter: {request_input.student_backend!r}")
    summary_backend = get_student_model_catalog_backend(backend_name)
    request.committee_model_paths = [
        artifact.model_path
        for artifact in summary_backend.load_committee_summary(request_input.committee_summary_path).members
        if artifact.model_path
    ]
    return request


def validate_canonical_sampling_stage_config(
    stage_config: dict[str, Any],
    *,
    sampling_backend: str | None = None,
) -> dict[str, Any]:
    payload = dict(stage_config or {})
    legacy_keys = sorted(set(payload) & {"profile_cfg", "profile_name", "sampling_seed_policy", "sampling"})
    if legacy_keys:
        raise ValueError(f"student sampling request adapter rejects legacy public sampling fields: {legacy_keys}")
    allowed = {"seed_selection", "md", "execution", "native_recipe_ref"}
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ValueError(f"student sampling request adapter only accepts canonical sampling fields {sorted(allowed)}; got {extra}")
    return normalize_pipeline_stage_config(
        "sampling",
        payload,
        sampling_backend=sampling_backend,
    )


def _canonical_backend_name(value: str) -> str:
    normalized = str(value or "deepmd").strip().lower()
    if normalized == "dp":
        return "deepmd"
    return normalized


def resolved_train_decision_context(request_input: StudentTrainRequestInput) -> dict[str, Any]:
    decision = request_input.resolved_train_decision
    return {
        "train_recipe_ref": str(decision.train_recipe_ref),
        "effective_train_recipe_branch": str(decision.effective_train_recipe_branch),
        "dataset_mode": str(decision.dataset_mode),
        "initialization_mode": str(decision.initialization_mode),
        "is_entry_train": bool(decision.is_entry_train),
        "uses_foundation_model": bool(decision.uses_foundation_model),
        "requires_previous_committee": bool(decision.requires_previous_committee),
    }
