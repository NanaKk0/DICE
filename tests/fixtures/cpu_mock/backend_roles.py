from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.lifecycle.composition import (
    BackendComposition,
    sampling_backend_binding,
    student_backend_binding,
    teacher_backend_binding,
)
from workflow_core.lifecycle.data import resolve_label_contract
from workflow_core.lifecycle.config import RunConfig, load_run_config
from workflow_core.lifecycle.student_facade import LifecycleStudentFacade
from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator
from workflow_core.lifecycle.teacher_facade import LifecycleTeacherFacade
from workflow_core.lifecycle.screen_module import LifecycleScreenModule
from workflow_backends.teacher.deepmd.adapter import DeepMDTeacherExecutor
from workflow_backends.teacher.mace.adapter import MaceExecutor
from workflow_app.composition import compose_screen_io_adapter


class _UnusedRole:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"test route unexpectedly used unbound role capability {name!r}")


def compose_test_backend_roles(
    run_json_or_config: str | Path | RunConfig,
    *,
    teacher: MaceExecutor | DeepMDTeacherExecutor | None = None,
    student: Any = None,
    sampling: Any = None,
    first_principles: Any = None,
    observation_chunk_size: int = 64,
    gmm_score_chunk_size: int = 500,
) -> BackendComposition:
    """Bind native test doubles before they cross the lifecycle role boundary."""

    source = getattr(run_json_or_config, "config", run_json_or_config)
    config = source if isinstance(source, RunConfig) else load_run_config(source)
    teacher_role: LifecycleTeacherFacade | _UnusedRole = _UnusedRole()
    if teacher is not None:
        if config.teacher["backend"] == "MACE":
            from workflow_backends.teacher.mace.adapter import (
                MaceTeacherBackendFactory,
            )

            teacher_factory = MaceTeacherBackendFactory(teacher)
        else:
            from workflow_backends.teacher.deepmd.adapter import (
                DeepMDTeacherBackendFactory,
            )

            teacher_factory = DeepMDTeacherBackendFactory(
                teacher, cli=str(config.teacher["cli"])
            )
        selected_teacher = teacher_factory.bind(teacher_backend_binding(config))
        teacher_contract, _student_contract = resolve_label_contract(
            config.root / config.teacher["recipe"],
            config.root / config.student["recipe"],
        )
        teacher_role = LifecycleTeacherFacade(
            delegate=selected_teacher,
            label_contract=teacher_contract,
        )
    student_role: Any = _UnusedRole()
    if student is not None:
        if config.student["backend"] == "DeepMD":
            from workflow_backends.student.deepmd.adapter import (
                DeepMDStudentBackendFactory,
            )

            student_factory = DeepMDStudentBackendFactory(
                student, cli=str(config.student["cli"])
            )
        else:
            from workflow_backends.student.nep.adapter import NepStudentBackendFactory

            student_factory = NepStudentBackendFactory(student)
        selected_student = student_factory.bind(student_backend_binding(config))
        student_role = LifecycleStudentFacade(delegate=selected_student)

    sampling_role: Any = _UnusedRole()
    if sampling is not None:
        if config.sampler["backend"] == "LAMMPS":
            from workflow_backends.lammps.adapter import LammpsSamplingBackendFactory

            sampling_factory = LammpsSamplingBackendFactory(sampling)
        else:
            from workflow_backends.gpumd.adapter import GpumdSamplingBackendFactory

            sampling_factory = GpumdSamplingBackendFactory(sampling)
        sampling_role = LifecycleSamplingCoordinator(
            sampling_factory.bind(sampling_backend_binding(config)),
            student_role.sampling_models,
        )

    screen_role = LifecycleScreenModule.bind(
        config=config,
        io=compose_screen_io_adapter(
            config=config,
            teacher=teacher_role,
            student=student_role,
            gmm_score_chunk_size=gmm_score_chunk_size,
        ),
        observation_chunk_size=observation_chunk_size,
        gmm_score_chunk_size=gmm_score_chunk_size,
    )

    return BackendComposition(
        teacher=teacher_role,
        student=student_role,
        sampling=sampling_role,
        screen=screen_role,
        first_principles=first_principles,
        observation_chunk_size=observation_chunk_size,
        gmm_score_chunk_size=gmm_score_chunk_size,
    )


def compose_teacher_test_roles(
    run_json_or_config: Any, *, teacher: MaceExecutor | DeepMDTeacherExecutor
) -> BackendComposition:
    """Build the narrow composition used by teach/train-only unit plans."""
    return compose_test_backend_roles(
        run_json_or_config,
        teacher=teacher,
    )


__all__ = ["compose_teacher_test_roles", "compose_test_backend_roles"]
