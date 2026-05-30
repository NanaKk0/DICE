from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from workflow_backends.student import get_student_train_backend
from workflow_backends.student.foundation import resolve_train_regime_foundation_asset
from workflow_backends.student.request_adapter import build_student_train_request
from workflow_core.common.io import read_json
from workflow_core.student_pipeline.request_context import build_train_request_input
from workflow_core.student_pipeline.stages.publish import (
    build_stage_file_publications,
    build_stage_publish_plan,
)
from workflow_core.student_pipeline.stages.base import resolve_stage_config
from workflow_core.student_pipeline.stages.harness import StudentStageRunnerContext, run_student_stage_with_harness
from workflow_core.student_pipeline.config.train_regime import (
    INITIALIZATION_MODE_FOUNDATION,
    INITIALIZATION_MODE_PREVIOUS_COMMITTEE,
)


def run_train_stage(
    runtime_state: dict[str, Any],
    *,
    student_round: int,
    stage_run_index: int,
    stage_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def execute(context: StudentStageRunnerContext) -> dict[str, Any]:
        executor = stage_executor or (
            lambda stage_context: _default_train_stage_executor(runtime_state, stage_context, context.artifacts)
        )
        return dict(executor(context.execution_context) or {})

    def publish_plan(context: StudentStageRunnerContext, stage_result: dict[str, Any]):
        committee_summary_path = stage_result.get("committee_summary_path")
        if not committee_summary_path:
            raise ValueError("train stage executor must return committee_summary_path")
        _stage_train_outputs(Path(committee_summary_path), context.staging_dir)
        return build_stage_publish_plan(
            student_round=student_round,
            stage_run_index=stage_run_index,
            artifact_publications=build_stage_file_publications("train", context.staging_dir, context.stage_context["stage_dir"]),
            stage_payload=dict(stage_result.get("stage_payload", {}) or {}),
        )

    return run_student_stage_with_harness(
        runtime_state,
        student_round=student_round,
        stage_run_index=stage_run_index,
        expected_stage_type="train",
        execute=execute,
        build_publish_plan=publish_plan,
    )


def _default_train_stage_executor(runtime_state: dict[str, Any], stage_context: dict[str, Any], artifacts) -> dict[str, Any]:
    cfg = dict(runtime_state.get("cfg", {}) or {})
    paths = runtime_state.get("paths")
    if paths is None:
        raise ValueError("runtime_state is missing paths for default train execution")
    stage_cfg = resolve_stage_config(stage_context, "train")
    train_recipe_ref = str(stage_cfg["train_recipe_ref"])
    backend = _load_student_backend(str(cfg.get("student_backend", "deepmd")))
    provisional_request = build_train_request_input(
        cfg=cfg,
        stage_context=stage_context,
        stage_cfg=stage_cfg,
        artifacts=artifacts,
        foundation_context=None,
    )
    decision = provisional_request.resolved_train_decision
    stage_results_dir = Path(stage_context["stage_results_dir"]).resolve()
    previous_committee = (
        str(artifacts.inputs.previous_committee_path.resolve()) if artifacts.inputs.previous_committee_path is not None else None
    )
    foundation_context = _resolve_train_foundation_bootstrap_context(
        cfg,
        backend,
        decision=decision,
        train_recipe_ref=train_recipe_ref,
    )
    backend_log_dir = paths.student_stage_backend_logs_dir(
        int(stage_context["teacher_round"]),
        int(stage_context["student_round"]),
        int(stage_context["stage_run_index"]),
        str(stage_context["stage"]),
    )
    if decision.initialization_mode != INITIALIZATION_MODE_PREVIOUS_COMMITTEE:
        previous_committee = None
    training_stage = _decision_training_stage(decision)
    request = build_student_train_request(
        build_train_request_input(
            cfg=cfg,
            stage_context=stage_context,
            stage_cfg=stage_cfg,
            artifacts=artifacts,
            foundation_context=foundation_context,
            previous_committee=previous_committee if decision.requires_previous_committee else None,
            extra_context={
                "backend_log_dir": str(backend_log_dir.resolve()),
                "resolved_train_decision": _resolved_train_decision_payload(decision),
                "foundation_bootstrap": foundation_context,
            },
        )
    )
    summary_path = backend.train_committee(paths, request)
    member_counts = _committee_member_counts(Path(summary_path).resolve())
    return {
        "committee_summary_path": Path(summary_path).resolve(),
        "stage_payload": {
            "effective_train_recipe_branch": decision.effective_train_recipe_branch,
            "dataset_mode": decision.dataset_mode,
            "initialization_mode": decision.initialization_mode,
            **member_counts,
        },
    }


def _load_student_backend(name: str):
    return get_student_train_backend(name)


def _committee_member_counts(summary_path: Path) -> dict[str, int]:
    payload = _read_committee_summary_payload(summary_path)
    members = [item for item in list(payload.get("members", []) or []) if isinstance(item, dict)]
    total = _int_or_none(payload.get("committee_member_count"))
    if total is None:
        total = _int_or_none(payload.get("member_count"))
    if total is None:
        total = _int_or_none(payload.get("committee_size"))
    if total is None:
        total = len(members) if members else len(list(payload.get("models", []) or []))

    explicit_success = _int_or_none(payload.get("successful_member_count"))
    if explicit_success is None:
        explicit_success = _int_or_none(payload.get("success_member_count"))
    if explicit_success is None:
        explicit_success = _int_or_none(payload.get("usable_member_count"))
    if explicit_success is None:
        explicit_success = sum(1 for member in members if _member_succeeded(member))

    explicit_failed = _int_or_none(payload.get("failed_member_count"))
    if explicit_failed is None:
        explicit_failed = max(0, int(total) - int(explicit_success))

    return {
        "committee_member_count": int(total),
        "successful_member_count": int(explicit_success),
        "failed_member_count": int(explicit_failed),
    }


def _read_committee_summary_payload(summary_path: Path) -> dict[str, Any]:
    path = Path(summary_path).resolve()
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return dict(payload or {}) if isinstance(payload, dict) else {}


def _member_succeeded(member: dict[str, Any]) -> bool:
    status = str(member.get("status", "")).strip().lower()
    if status in {"success", "successful", "ok", "complete", "completed"}:
        return True
    if status in {"failed", "failure", "error"}:
        return False
    return any(member.get(key) for key in ("model_path", "graph_pb", "nep_txt", "restart_path"))


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_train_foundation_bootstrap_context(
    cfg: dict[str, Any],
    backend,
    *,
    decision,
    train_recipe_ref: str,
) -> dict[str, Any] | None:
    if decision.initialization_mode != INITIALIZATION_MODE_FOUNDATION:
        return None
    backend_name = str(getattr(backend, "backend_name", cfg.get("student_backend", "")) or "").strip().lower()
    foundation_spec = resolve_train_regime_foundation_asset(cfg, backend_name, decision)
    if foundation_spec is None:
        return None
    element_order = backend.resolve_element_order(
        cfg,
        config_path=train_recipe_ref,
        training_stage=decision.effective_train_recipe_branch,
    )
    backend.validate_foundation_bootstrap(cfg, foundation_spec, element_order=element_order)
    return backend.prepare_foundation_bootstrap_context(cfg, foundation_spec)


def _decision_training_stage(decision) -> str:
    if decision.initialization_mode == INITIALIZATION_MODE_PREVIOUS_COMMITTEE:
        return "finetune"
    return "bootstrap"


def _is_entry_train(stage_context: dict[str, Any]) -> bool:
    return int(stage_context["teacher_round"]) == 0 and int(stage_context["student_round"]) == 1


def _resolved_train_decision_payload(decision) -> dict[str, Any]:
    return {
        "train_recipe_ref": decision.train_recipe_ref,
        "effective_train_recipe_branch": decision.effective_train_recipe_branch,
        "dataset_mode": decision.dataset_mode,
        "initialization_mode": decision.initialization_mode,
        "is_entry_train": decision.is_entry_train,
        "uses_foundation_model": decision.uses_foundation_model,
        "requires_previous_committee": decision.requires_previous_committee,
    }


def _stage_train_outputs(summary_path: Path, staging_dir: Path) -> None:
    source_summary = Path(summary_path).resolve()
    target_summary = staging_dir / "student_committee_summary.json"
    target_summary.parent.mkdir(parents=True, exist_ok=True)
    if source_summary.exists() and source_summary != target_summary:
        shutil.copy2(source_summary, target_summary)
    committee_target = staging_dir / "student_committee"
    committee_target.mkdir(parents=True, exist_ok=True)
    payload = read_json(source_summary) if source_summary.exists() else {}
    model_paths = []
    for entry in list(dict(payload or {}).get("members", []) or []):
        if isinstance(entry, dict):
            for key in ("model_path", "snapshot_model_path", "restart_path"):
                value = entry.get(key)
                if value:
                    model_paths.append(Path(str(value)).resolve())
    for value in list(dict(payload or {}).get("models", []) or []):
        model_paths.append(Path(str(value)).resolve())
    seen_sources: set[Path] = set()
    for model_path in model_paths:
        if model_path in seen_sources:
            continue
        seen_sources.add(model_path)
        member_dir = model_path.parent
        if not member_dir.exists() or not member_dir.is_dir():
            continue
        target = committee_target / member_dir.name
        shutil.copytree(member_dir, target, dirs_exist_ok=True)
    arch_meta = source_summary.parent / "arch_meta.json"
    if arch_meta.exists() and arch_meta.is_file():
        shutil.copy2(arch_meta, committee_target / "arch_meta.json")
