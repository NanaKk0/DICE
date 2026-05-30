"""Compatibility shim for MACE teacher foundation training.

New target-mode code should enter through workflow_core.teacher.foundation_train
and MaceTeacherBackend.train_foundation.
"""

from workflow_backends.teacher.base import TeacherFoundationTrainRequest
from workflow_backends.teacher.mace.foundation import load_mace_foundation_train_config, run_mace_foundation_train


def run_train_mace(cfg, paths, r):
    artifact_root = paths.teacher_foundation_train_artifacts_dir(r).resolve()
    request = TeacherFoundationTrainRequest(
        cfg=dict(cfg),
        paths=paths,
        teacher_round=int(r),
        artifact_root=artifact_root,
        foundation_model_root=(artifact_root / "teacher_foundation").resolve(),
        runtime_json=(artifact_root / "teacher_foundation_train_runtime.json").resolve(),
        backend_log_dir=(artifact_root / "backend_logs").resolve(),
        context={"teacher_backend": "mace"},
    )
    return run_mace_foundation_train(request)
