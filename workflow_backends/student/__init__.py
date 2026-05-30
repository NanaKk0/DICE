from workflow_backends.student.base import (
    CommitteeModelArtifact,
    StudentBackend,
    StudentConfigBackend,
    StudentEvaluationBackend,
    StudentModelCatalogBackend,
    StudentSamplingExecutionBackend,
    StudentTrainExecutionBackend,
)
from workflow_backends.student.request_adapter import (
    StudentSamplingRequestInput,
    StudentTrainRequestInput,
    build_student_sampling_request,
    build_student_train_request,
)


def get_student_backend(name: str | None):
    from workflow_backends.student.registry import get_student_backend as _get_student_backend

    return _get_student_backend(name)

def get_student_train_backend(name: str | None):
    from workflow_backends.student.registry import get_student_train_backend as _get_student_train_backend

    return _get_student_train_backend(name)


def get_student_sampling_backend(name: str | None):
    from workflow_backends.student.registry import get_student_sampling_backend as _get_student_sampling_backend

    return _get_student_sampling_backend(name)


def get_student_config_backend(name: str | None):
    from workflow_backends.student.registry import get_student_config_backend as _get_student_config_backend

    return _get_student_config_backend(name)


__all__ = [
    "CommitteeModelArtifact",
    "StudentBackend",
    "StudentConfigBackend",
    "StudentEvaluationBackend",
    "StudentModelCatalogBackend",
    "StudentSamplingRequestInput",
    "StudentSamplingExecutionBackend",
    "StudentTrainRequestInput",
    "StudentTrainExecutionBackend",
    "build_student_sampling_request",
    "build_student_train_request",
    "get_student_backend",
    "get_student_config_backend",
    "get_student_sampling_backend",
    "get_student_train_backend",
]
