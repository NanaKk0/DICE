from __future__ import annotations

__all__ = [
    "load_student_active_learning_config",
    "run_student_active_learning_subworkflow",
]


def load_student_active_learning_config(*args, **kwargs):
    from workflow_core.student.config import load_student_active_learning_config as _impl

    return _impl(*args, **kwargs)


def run_student_active_learning_subworkflow(*args, **kwargs):
    from workflow_core.student.subworkflow_service import run_student_active_learning_subworkflow as _impl

    return _impl(*args, **kwargs)
