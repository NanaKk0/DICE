from __future__ import annotations

from workflow_runtime.environment.requirements import validate_environment_bindings

TEACHER_BACKEND_ACTIONS = frozenset({"foundation_train", "finetune", "relabel", "evaluate"})
STUDENT_BACKEND_ACTIONS = frozenset({"train", "evaluate"})
SAMPLING_BACKEND_ACTIONS = frozenset({"sample"})


def teacher_backend_role(backend: str, action: str) -> str:
    return _backend_role("teacher_backend", backend, action, TEACHER_BACKEND_ACTIONS)


def student_backend_role(backend: str, action: str) -> str:
    return _backend_role("student_backend", _canonical_student_backend_name(backend), action, STUDENT_BACKEND_ACTIONS)


def sampling_backend_role(backend: str, action: str) -> str:
    return _backend_role("sampling_backend", backend, action, SAMPLING_BACKEND_ACTIONS)


def _backend_role(namespace: str, backend: str, action: str, supported_actions: frozenset[str]) -> str:
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in supported_actions:
        supported = ", ".join(sorted(supported_actions))
        raise ValueError(f"unsupported {namespace} action {normalized_action!r}; supported actions: {supported}")
    normalized_backend = str(backend or "").strip().lower()
    if not normalized_backend:
        raise ValueError(f"{namespace} role requires a non-empty backend name")
    return f"{namespace}.{normalized_backend}.{normalized_action}"


def _canonical_student_backend_name(value: str) -> str:
    normalized = str(value or "deepmd").strip().lower()
    if normalized == "dp":
        return "deepmd"
    return normalized
