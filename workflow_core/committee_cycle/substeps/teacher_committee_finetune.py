from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_runtime.environment import context_with_execution_context, teacher_backend_role
from workflow_runtime.artifacts.checks import teacher_committee_finetune_complete


def teacher_committee_finetune_success_check(paths, teacher_round: int, cfg: dict[str, Any]) -> bool:
    from workflow_backends.teacher.registry import get_teacher_backend

    finetune_cfg = dict(cfg.get("teacher_committee_finetune", {}) or {})
    train_cfg = dict(cfg.get("teacher_committee_finetune_train_config_snapshot", {}) or {})
    teacher_count = int(
        train_cfg.get(
            "teacher_num",
            finetune_cfg.get("teacher_num", finetune_cfg.get("committee_size", 1)),
        )
        or 1
    )
    artifact_root = paths.teacher_committee_finetune_artifacts_dir(teacher_round)
    backend = get_teacher_backend(finetune_cfg.get("teacher_backend", "mace"))
    discovered = [
        artifact
        for artifact in backend.collect_models(str(artifact_root))
        if Path(artifact.model_path).is_file() and Path(artifact.model_path).stat().st_size > 0
    ]
    if len(discovered) >= teacher_count:
        return True
    if bool(getattr(backend, "strict_model_discovery_completion", False)):
        return False
    return teacher_committee_finetune_complete(
        artifact_root,
        teacher_count=teacher_count,
    )


def run_teacher_committee_finetune_substep(
    cfg: dict[str, Any],
    paths,
    teacher_round: int,
    *,
    input_resolver=None,
) -> str:
    from workflow_core.teacher.committee_finetune import run_finetune
    from workflow_core.teacher.training_data_policy import resolve_teacher_finetune_inputs

    resolver = input_resolver or resolve_teacher_finetune_inputs
    training_input = resolver(paths, teacher_round, cfg)
    finetune_cfg = dict(cfg["teacher_committee_finetune"])
    artifact_root = paths.teacher_committee_finetune_artifacts_dir(teacher_round)
    context = {
        "training_input": training_input,
        "train_config_snapshot": dict(cfg.get("teacher_committee_finetune_train_config_snapshot", {}) or {}),
        "teacher_backend": finetune_cfg.get("teacher_backend", "mace"),
        "merged_xyz_override": str((artifact_root / "input" / "merged.xyz").resolve()),
        "runtime_json_override": str((artifact_root / "teacher_committee_finetune_runtime.json").resolve()),
        "work_root_override": str(artifact_root.resolve()),
        "backend_log_dir_override": str(
            paths.teacher_committee_finetune_backend_logs_dir(teacher_round).resolve()
        ),
        "reuse_existing_committee": bool(finetune_cfg.get("reuse_existing_committee", True)),
    }
    context = context_with_execution_context(
        cfg,
        role=teacher_backend_role(finetune_cfg.get("teacher_backend", "mace"), "finetune"),
        context=context,
    )
    run_finetune(
        {
            "teacher_backend": finetune_cfg.get("teacher_backend", "mace"),
            "config": finetune_cfg["config"],
        },
        paths,
        teacher_round,
        context=context,
    )
    return str(artifact_root.resolve())
