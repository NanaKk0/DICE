from __future__ import annotations

from pathlib import Path

from workflow_backends.teacher import get_teacher_backend
from workflow_backends.teacher.base import TeacherFoundationTrainRequest
from workflow_runtime.environment import context_with_execution_context, teacher_backend_role


def run_foundation_train(cfg, paths, r, context=None):
    context = dict(context or {})
    teacher_backend_name = str(
        cfg.get("teacher_backend")
        or context.get("teacher_backend")
        or cfg.get("mace_backend")
        or context.get("mace_backend")
        or "mace"
    )
    backend = get_teacher_backend(teacher_backend_name)
    artifact_root = paths.teacher_foundation_train_artifacts_dir(r).resolve()
    request_context = context_with_execution_context(
        cfg,
        role=teacher_backend_role(backend.backend_name, "foundation_train"),
        context=context,
    )
    request_context.setdefault("teacher_backend", backend.backend_name)
    if cfg.get("_foundation_runner") is not None:
        request_context["foundation_runner"] = cfg["_foundation_runner"]
    request = TeacherFoundationTrainRequest(
        cfg=dict(cfg),
        paths=paths,
        teacher_round=int(r),
        artifact_root=artifact_root,
        foundation_model_root=(artifact_root / "teacher_foundation").resolve(),
        runtime_json=(artifact_root / "teacher_foundation_train_runtime.json").resolve(),
        backend_log_dir=(artifact_root / "backend_logs").resolve(),
        context=request_context,
    )
    return backend.train_foundation(request)
