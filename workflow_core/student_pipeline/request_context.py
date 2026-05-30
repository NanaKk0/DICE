from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow_backends.student.request_adapter import StudentSamplingRequestInput, StudentTrainRequestInput
from workflow_core.student_pipeline.config.train_regime import resolve_train_decision
from workflow_runtime.environment import context_with_execution_context, sampling_backend_role, student_backend_role


@dataclass(frozen=True, slots=True)
class BackendIdentityContext:
    student_backend: str
    sampling_backend: str
    student_model_identity: dict[str, Any]
    sampling_backend_identity: dict[str, Any]
    train_regime: dict[str, bool]

    def global_invariant_payload(self) -> dict[str, Any]:
        return {
            "student_backend": self.student_backend,
            "sampling_backend": self.sampling_backend,
            "student_model_identity": dict(self.student_model_identity),
            "sampling_backend_identity": dict(self.sampling_backend_identity),
            "train_regime": dict(self.train_regime),
        }


def build_backend_identity_context(cfg: dict[str, Any]) -> BackendIdentityContext:
    payload = dict(cfg or {})
    student_backend = canonical_student_backend_name(str(payload.get("student_backend", "deepmd") or "deepmd"))
    sampling_backend = canonical_sampling_backend_name(
        str(payload.get("sampling_backend", "") or ""),
        student_backend=student_backend,
    )
    train_regime = dict(payload.get("train_regime", {}) or {})
    sampling_backend_config = dict(payload.get("sampling_backend_config", {}) or {})
    return BackendIdentityContext(
        student_backend=student_backend,
        sampling_backend=sampling_backend,
        student_model_identity=_identity_payload(payload, "student_model"),
        sampling_backend_identity=_identity_payload(sampling_backend_config, sampling_backend, sampling_backend),
        train_regime={
            "use_foundation_model": bool(train_regime.get("use_foundation_model", False)),
            "continue_training": bool(train_regime.get("continue_training", False)),
        },
    )


def canonical_student_backend_name(value: str) -> str:
    normalized = str(value or "deepmd").strip().lower()
    if normalized == "dp":
        return "deepmd"
    return normalized


def canonical_sampling_backend_name(value: str, *, student_backend: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized:
        return normalized
    return "gpumd" if canonical_student_backend_name(student_backend) == "nep" else "lammps"


def build_sampling_request_input(
    *,
    cfg: dict[str, Any],
    stage_context: dict[str, Any],
    stage_cfg: dict[str, Any],
    artifacts,
    work_root: str | Path,
    reference_xyz_files: list[str] | None = None,
    summary_extra: dict[str, Any] | None = None,
    extra_context: dict[str, Any] | None = None,
) -> StudentSamplingRequestInput:
    identity = build_backend_identity_context(cfg)
    request_extra_context = _extra_context_with_execution_context(
        cfg,
        role=sampling_backend_role(identity.sampling_backend, "sample"),
        extra_context=extra_context,
    )
    execution_context = request_extra_context.get("execution_context")
    return StudentSamplingRequestInput(
        student_backend=identity.student_backend,
        sampling_backend=identity.sampling_backend,
        teacher_round=int(stage_context["teacher_round"]),
        student_round=int(stage_context["student_round"]),
        stage_run_index=int(stage_context["stage_run_index"]),
        stage_config=dict(stage_cfg or {}),
        student_model_identity=dict(identity.student_model_identity),
        sampling_backend_identity=dict(identity.sampling_backend_identity),
        student_uncertainty=dict(dict(cfg or {}).get("uncertainty", {}) or {}),
        committee_summary_path=str(artifacts.adapter_context.committee_summary_path.resolve()),
        seed_xyz_path=str(artifacts.adapter_context.selected_seed_xyz.resolve()),
        work_root=str(Path(work_root).resolve()),
        reference_xyz_files=list(reference_xyz_files or []),
        summary_extra=dict(summary_extra or {}),
        execution_context=execution_context if isinstance(execution_context, dict) else None,
        extra_context=request_extra_context,
    )


def build_train_request_input(
    *,
    cfg: dict[str, Any],
    stage_context: dict[str, Any],
    stage_cfg: dict[str, Any],
    artifacts,
    foundation_context: dict[str, Any] | None,
    previous_committee: str | Path | None = None,
    extra_context: dict[str, Any] | None = None,
) -> StudentTrainRequestInput:
    payload = dict(cfg or {})
    identity = build_backend_identity_context(payload)
    train_recipe_ref = str(stage_cfg["train_recipe_ref"])
    decision = resolve_train_decision(
        train_regime=dict(payload.get("train_regime", {}) or {}),
        is_entry_train=_is_entry_train(stage_context),
        train_recipe_ref=train_recipe_ref,
    )
    stage_results_dir = Path(stage_context["stage_results_dir"]).resolve()
    request_extra_context = _extra_context_with_execution_context(
        payload,
        role=student_backend_role(identity.student_backend, "train"),
        extra_context=extra_context,
    )
    execution_context = request_extra_context.get("execution_context")
    return StudentTrainRequestInput(
        student_backend=identity.student_backend,
        teacher_round=int(stage_context["teacher_round"]),
        student_round=int(stage_context["student_round"]),
        stage_run_index=int(stage_context["stage_run_index"]),
        stage_config=dict(stage_cfg or {}),
        student_model_identity=dict(identity.student_model_identity),
        work_root=str(stage_results_dir),
        summary_path=str((stage_results_dir / "student_committee_summary.json").resolve()),
        train_recipe_ref=train_recipe_ref,
        resolved_train_decision=decision,
        dataset_roots=[str(path.resolve()) for path in artifacts.adapter_context.dataset_roots],
        dataset_path=str(artifacts.adapter_context.dataset_path.resolve()),
        previous_model_root=None if previous_committee is None else str(Path(previous_committee).resolve()),
        foundation_context=foundation_context,
        execution_context=execution_context if isinstance(execution_context, dict) else None,
        extra_context={
            "backend_workflow_config": {
                identity.student_backend: dict(payload.get(identity.student_backend, {}) or {}),
            },
            **request_extra_context,
        },
    )


def _identity_payload(cfg: dict[str, Any], explicit_key: str, backend_key: str | None = None) -> dict[str, Any]:
    explicit = cfg.get(explicit_key)
    if isinstance(explicit, dict):
        return dict(explicit)
    if backend_key is None:
        return {}
    payload = cfg.get(backend_key)
    return dict(payload or {}) if isinstance(payload, dict) else {}


def _is_entry_train(stage_context: dict[str, Any]) -> bool:
    return int(stage_context["teacher_round"]) == 0 and int(stage_context["student_round"]) == 1


def _extra_context_with_execution_context(
    cfg: dict[str, Any],
    *,
    role: str,
    extra_context: dict[str, Any] | None,
) -> dict[str, Any]:
    return context_with_execution_context(cfg, role=role, context=extra_context)
