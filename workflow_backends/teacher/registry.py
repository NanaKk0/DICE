from __future__ import annotations

from workflow_backends.teacher.base import TeacherBackend
from workflow_backends.teacher.deepmd import DeepMDTeacherBackend
from workflow_backends.teacher.mace import MaceTeacherBackend


_BACKENDS: list[TeacherBackend] = [MaceTeacherBackend(), DeepMDTeacherBackend()]


def register_teacher_backend(name: str, backend: TeacherBackend) -> None:
    normalized = str(name or "").strip().lower()
    if not normalized:
        raise ValueError("teacher backend name must be non-empty")
    for index, existing in enumerate(_BACKENDS):
        if existing.matches_name(normalized):
            _BACKENDS[index] = backend
            return
    _BACKENDS.append(backend)


def get_teacher_backend(name: str | None) -> TeacherBackend:
    normalized = str(name or "mace").strip().lower()
    for backend in _BACKENDS:
        if backend.matches_name(normalized):
            return backend
    supported = ", ".join(sorted({backend.backend_name for backend in _BACKENDS}))
    raise ValueError(f"Unknown teacher backend: {name!r}. Supported backends: {supported}")
