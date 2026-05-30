from __future__ import annotations

from workflow_backends.student.base import (
    StudentBackend,
    StudentConfigBackend,
    StudentEvaluationBackend,
    StudentModelCatalogBackend,
    StudentSamplingExecutionBackend,
    StudentTrainExecutionBackend,
)


_BACKEND_CACHE: dict[str, StudentBackend] = {}


def _backend_factories():
    from workflow_backends.student.deepmd import DeepMDStudentBackend
    from workflow_backends.student.nep import NEPStudentBackend

    return {
        "deepmd": DeepMDStudentBackend,
        "nep": NEPStudentBackend,
    }


def get_student_backend(name: str | None) -> StudentBackend:
    normalized = str(name or "deepmd").strip().lower()
    if normalized == "dp":
        normalized = "deepmd"
    if normalized in _BACKEND_CACHE:
        return _BACKEND_CACHE[normalized]
    factories = _backend_factories()
    backend_cls = factories.get(normalized)
    if backend_cls is None:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unknown student backend: {name!r}. Supported backends: {supported}")
    backend = backend_cls()
    _BACKEND_CACHE[normalized] = backend
    for alias in getattr(backend, "aliases", ()):
        _BACKEND_CACHE[str(alias).strip().lower()] = backend
    _BACKEND_CACHE[backend.backend_name] = backend
    return backend


def get_student_train_backend(name: str | None) -> StudentTrainExecutionBackend:
    backend = get_student_backend(name)
    if not isinstance(backend, StudentTrainExecutionBackend):
        raise TypeError(f"Student backend {name!r} does not implement training execution")
    return backend


def get_student_sampling_backend(name: str | None) -> StudentSamplingExecutionBackend:
    """Return the legacy fixed-loop student sampling execution facade.

    Target-mode `student_pipeline_template` callers must use
    `workflow_backends.sampling.get_sampling_backend` for MD execution and
    `workflow_backends.student.request_adapter.build_student_sampling_request`
    only for request translation.
    """
    backend = get_student_backend(name)
    if not isinstance(backend, StudentSamplingExecutionBackend):
        raise TypeError(f"Student backend {name!r} does not implement sampling execution")
    return backend


def get_student_model_catalog_backend(name: str | None) -> StudentModelCatalogBackend:
    backend = get_student_backend(name)
    if not isinstance(backend, StudentModelCatalogBackend):
        raise TypeError(f"Student backend {name!r} does not implement model catalog access")
    return backend


def get_student_config_backend(name: str | None) -> StudentConfigBackend:
    backend = get_student_backend(name)
    if not isinstance(backend, StudentConfigBackend):
        raise TypeError(f"Student backend {name!r} does not implement config helpers")
    return backend


def get_student_evaluation_backend(name: str | None) -> StudentEvaluationBackend:
    backend = get_student_backend(name)
    if not isinstance(backend, StudentEvaluationBackend):
        raise TypeError(f"Student backend {name!r} does not implement evaluation")
    return backend
