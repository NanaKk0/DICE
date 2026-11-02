from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from workflow_backends.lammps.native import NativeLammpsExecutor
from workflow_backends.gpumd.native import NativeGpumdExecutor
from workflow_backends.student.nep.native import NativeNepExecutor
from workflow_backends.teacher.deepmd.native import NativeDeepMDTeacherExecutor
from workflow_backends.student.deepmd.native import NativeDeepMDStudentExecutor
from workflow_backends.teacher.mace.native import NativeMaceExecutor
from workflow_backends.sampling import SamplingBackendFactory
from workflow_backends.student.role import StudentBackendFactory
from workflow_backends.teacher.role import TeacherBackendFactory
from workflow_runtime.lifecycle.environment import OperationContext


@dataclass(frozen=True, slots=True)
class SelectedBackendAdapters:
    """Backend-owned capabilities selected by one resolved lifecycle route."""

    teacher: TeacherBackendFactory
    student: StudentBackendFactory
    sampling: SamplingBackendFactory
    first_principles: Any | None = None


@dataclass(frozen=True, slots=True)
class _ObservationOnlyExecutor:
    """Carries immutable operation evidence without exposing an execution method."""

    compress_context: OperationContext | None = None
    gmm_screen_context: OperationContext | None = None
    gmm_reference_context: OperationContext | None = None
    gmm_score_context: OperationContext | None = None


def build_observation_backend_adapters(
    config: Any,
    contexts: Mapping[str, OperationContext],
) -> SelectedBackendAdapters:
    """Compose validation/recovery adapters for a read-only lifecycle probe."""

    teacher_spec = config.teacher
    if teacher_spec["backend"] == "MACE":
        from workflow_backends.teacher.mace.adapter import MaceTeacherBackendFactory

        teacher: TeacherBackendFactory = MaceTeacherBackendFactory(
            _ObservationOnlyExecutor(
                gmm_reference_context=contexts.get("teacher.mace.gmm_reference"),
                gmm_score_context=contexts.get("teacher.mace.gmm_score"),
            )
        )
    else:
        from workflow_backends.teacher.deepmd.adapter import (
            DeepMDTeacherBackendFactory,
        )

        cli = str(teacher_spec["cli"])
        teacher = DeepMDTeacherBackendFactory(
            _ObservationOnlyExecutor(
                gmm_reference_context=contexts.get(
                    f"teacher.deepmd.{cli}.gmm_reference"
                ),
                gmm_score_context=contexts.get(
                    f"teacher.deepmd.{cli}.gmm_score"
                ),
            ),
            cli=cli,
        )

    student_spec = config.student
    if student_spec["backend"] == "DeepMD":
        from workflow_backends.student.deepmd.adapter import (
            DeepMDStudentBackendFactory,
        )

        cli = str(student_spec["cli"])
        prefix = f"student.deepmd.{cli}"
        student: StudentBackendFactory = DeepMDStudentBackendFactory(
            _ObservationOnlyExecutor(
                compress_context=contexts.get(f"{prefix}.compress"),
                gmm_screen_context=contexts.get(f"{prefix}.gmm_screen"),
                gmm_reference_context=contexts.get(f"{prefix}.gmm_reference"),
                gmm_score_context=contexts.get(f"{prefix}.gmm_score"),
            ),
            cli=cli,
        )
    else:
        from workflow_backends.student.nep.adapter import NepStudentBackendFactory

        student = NepStudentBackendFactory(
            _ObservationOnlyExecutor(
                gmm_reference_context=contexts.get("student.nep.gmm_reference"),
                gmm_score_context=contexts.get("student.nep.gmm_score"),
            )
        )

    if config.sampler["backend"] == "LAMMPS":
        from workflow_backends.lammps.adapter import LammpsSamplingBackendFactory

        sampling: SamplingBackendFactory = LammpsSamplingBackendFactory(
            _ObservationOnlyExecutor()
        )
    else:
        from workflow_backends.gpumd.adapter import GpumdSamplingBackendFactory

        sampling = GpumdSamplingBackendFactory(_ObservationOnlyExecutor())
    return SelectedBackendAdapters(
        teacher=teacher,
        student=student,
        sampling=sampling,
    )


def build_selected_backend_adapters(
    contexts: Mapping[str, OperationContext],
    *,
    dormant_roles: tuple[str, ...] = (),
) -> SelectedBackendAdapters:
    """Build exactly the native capabilities selected by one lifecycle route."""
    if dormant_roles:
        contexts = _with_dormant_prefix_contexts(
            contexts,
            dormant_roles=dormant_roles,
        )
    selected = set(contexts)
    if not selected:
        raise ValueError("lifecycle executor factory is missing registered teacher operation contexts")

    mace_teacher, mace_roles = _mace_teacher(contexts)
    deepmd_teacher, deepmd_teacher_roles = _deepmd_teacher(contexts)
    if mace_teacher is not None and deepmd_teacher is not None:
        raise ValueError("lifecycle executor factory cannot select both MACE and DeepMD teacher services")
    if mace_teacher is None and deepmd_teacher is None:
        raise ValueError("lifecycle executor factory is missing registered teacher operation contexts")

    deepmd_student, deepmd_student_roles = _deepmd_student(contexts)
    lammps, lammps_roles = _lammps_sampler(contexts)
    nep_student, nep_student_roles = _nep_student(contexts)
    gpumd, gpumd_roles = _gpumd_sampler(contexts)
    neb_context, neb_roles = _neb_sampler(contexts)
    relax_context, relax_roles = _relax_sampler(contexts)
    if deepmd_student is not None and nep_student is not None:
        raise ValueError(
            "lifecycle executor factory cannot select both DeepMD and NEP student services"
        )
    if deepmd_student is None and nep_student is None:
        raise ValueError(
            "lifecycle executor factory is missing registered student operation contexts"
        )
    if lammps is not None and gpumd is not None:
        raise ValueError(
            "lifecycle executor factory cannot select both LAMMPS and GPUMD sampling services"
        )
    if lammps is None and gpumd is None and neb_context is None and relax_context is None:
        raise ValueError(
            "lifecycle executor factory is missing registered sampling operation contexts"
        )
    if neb_context is None and relax_context is None and (deepmd_student is None) != (lammps is None):
        raise ValueError(
            "lifecycle executor factory requires DeepMD/LAMMPS or NEP/GPUMD selected capabilities"
        )
    if neb_context is not None:
        if deepmd_student is not None and neb_context.role not in {
            "student.deepmd.tf.neb",
            "student.deepmd.pt.neb",
        }:
            raise ValueError("DeepMD student requires its matching ASE NEB role")
        if nep_student is not None and neb_context.role != "student.nep.neb":
            raise ValueError("NEP student requires the NEP ASE NEB role")
    if relax_context is not None:
        if deepmd_student is not None and relax_context.role not in {
            "student.deepmd.tf.relax",
            "student.deepmd.pt.relax",
        }:
            raise ValueError("DeepMD student requires its matching ASE relax role")
        if nep_student is not None and relax_context.role != "student.nep.relax":
            raise ValueError("NEP student requires the NEP ASE relax role")
    if deepmd_student is not None:
        student_cli = next(iter(deepmd_student_roles)).split(".")[2]
        if (
            neb_context is not None
            and neb_context.role != f"student.deepmd.{student_cli}.neb"
        ):
            raise ValueError(
                "DeepMD student and ASE NEB roles require matching CLI families"
            )
        if (
            relax_context is not None
            and relax_context.role != f"student.deepmd.{student_cli}.relax"
        ):
            raise ValueError(
                "DeepMD student and ASE relax roles require matching CLI families"
            )
        sampler_cli = (
            None
            if not lammps_roles
            else next(iter(lammps_roles)).split(".")[2]
        )
        if sampler_cli is not None and student_cli != sampler_cli:
            raise ValueError(
                "lifecycle executor factory requires matching DeepMD and LAMMPS CLI families"
            )
    first_principles, first_principles_roles = _first_principles(contexts)
    expected = mace_roles | deepmd_teacher_roles | deepmd_student_roles | lammps_roles | nep_student_roles | gpumd_roles | neb_roles | relax_roles | first_principles_roles
    unregistered = sorted(selected - expected)
    if unregistered:
        raise ValueError(f"lifecycle executor factory has unregistered selected operation context(s): {', '.join(unregistered)}")
    model_family = (
        f"deepmd-{student_cli}" if deepmd_student is not None else "nep"
    )
    md_factory = None
    if lammps is not None:
        from workflow_backends.lammps.adapter import LammpsSamplingBackendFactory

        md_factory = LammpsSamplingBackendFactory(lammps)
    elif gpumd is not None:
        from workflow_backends.gpumd.adapter import GpumdSamplingBackendFactory

        md_factory = GpumdSamplingBackendFactory(gpumd)
    neb_factory = None
    if neb_context is not None:
        from workflow_backends.ase_neb.adapter import AseNebSamplingBackendFactory

        neb_factory = AseNebSamplingBackendFactory(
            neb_context, model_family=model_family
        )
    relax_factory = None
    if relax_context is not None:
        from workflow_backends.ase_relax.adapter import AseRelaxSamplingBackendFactory

        relax_factory = AseRelaxSamplingBackendFactory(
            relax_context, model_family=model_family
        )
    branches = tuple(
        branch for branch in (md_factory, neb_factory, relax_factory) if branch is not None
    )
    if len(branches) == 1:
        sampling: SamplingBackendFactory = branches[0]
    else:
        from workflow_backends.hybrid_sampling import HybridSamplingBackendFactory

        sampling = HybridSamplingBackendFactory(
            md=md_factory,
            neb=neb_factory,
            relax=relax_factory,
        )
    if deepmd_student is not None:
        from workflow_backends.student.deepmd.adapter import (
            DeepMDStudentBackendFactory,
        )

        student: StudentBackendFactory = DeepMDStudentBackendFactory(
            deepmd_student,
            cli=student_cli,
        )
    else:
        from workflow_backends.student.nep.adapter import NepStudentBackendFactory

        student = NepStudentBackendFactory(nep_student)
    if mace_teacher is not None:
        from workflow_backends.teacher.mace.adapter import MaceTeacherBackendFactory

        teacher_factory: TeacherBackendFactory = MaceTeacherBackendFactory(
            mace_teacher
        )
    else:
        from workflow_backends.teacher.deepmd.adapter import (
            DeepMDTeacherBackendFactory,
        )

        teacher_cli = next(iter(deepmd_teacher_roles)).split(".")[2]
        teacher_factory = DeepMDTeacherBackendFactory(
            deepmd_teacher, cli=teacher_cli
        )
    return SelectedBackendAdapters(
        teacher=teacher_factory,
        student=student,
        sampling=sampling,
        first_principles=first_principles,
    )


def _neb_sampler(
    contexts: Mapping[str, OperationContext],
) -> tuple[OperationContext | None, set[str]]:
    roles = (
        "student.deepmd.tf.neb",
        "student.deepmd.pt.neb",
        "student.nep.neb",
    )
    selected = tuple(role for role in roles if role in contexts)
    if not selected:
        return None, set()
    if len(selected) != 1:
        raise ValueError(
            "lifecycle executor factory requires exactly one ASE NEB operation context"
        )
    return contexts[selected[0]], {selected[0]}


def _relax_sampler(
    contexts: Mapping[str, OperationContext],
) -> tuple[OperationContext | None, set[str]]:
    roles = (
        "student.deepmd.tf.relax",
        "student.deepmd.pt.relax",
        "student.nep.relax",
    )
    selected = tuple(role for role in roles if role in contexts)
    if not selected:
        return None, set()
    if len(selected) != 1:
        raise ValueError(
            "lifecycle executor factory requires exactly one ASE relax operation context"
        )
    return contexts[selected[0]], {selected[0]}


def _with_dormant_prefix_contexts(
    contexts: Mapping[str, OperationContext], *, dormant_roles: tuple[str, ...]
) -> dict[str, OperationContext]:
    """Complete provider constructors without preflighting skipped-prefix roles."""

    result = dict(contexts)
    for role in dormant_roles:
        if role in result or role.startswith("first_principles."):
            continue
        result[role] = OperationContext(
            role=role,
            profile_id="dormant-completed-prefix",
            source_scripts=(),
            executables={},
            runtime={"execution_forbidden": True},
        )
    return result


def _mace_teacher(contexts: Mapping[str, OperationContext]) -> tuple[NativeMaceExecutor | None, set[str]]:
    base = ("teacher.mace.train", "teacher.mace.label")
    gmm = ("teacher.mace.gmm_reference", "teacher.mace.gmm_score")
    selected = {role for role in (*base, *gmm) if role in contexts}
    if not selected:
        return None, set()
    missing = [role for role in base if role not in contexts]
    if missing:
        raise ValueError(f"lifecycle executor factory missing required MACE operation context(s): {', '.join(missing)}")
    if any(role in contexts for role in gmm) and not all(role in contexts for role in gmm):
        raise ValueError("lifecycle executor factory requires both MACE gmm_reference and gmm_score contexts")
    return NativeMaceExecutor(
        train_context=contexts[base[0]],
        label_context=contexts[base[1]],
        gmm_reference_context=contexts.get(gmm[0]),
        gmm_score_context=contexts.get(gmm[1]),
    ), selected


def _deepmd_teacher(contexts: Mapping[str, OperationContext]) -> tuple[NativeDeepMDTeacherExecutor | None, set[str]]:
    matching: list[tuple[tuple[str, ...], set[str]]] = []
    for cli in ("tf", "pt"):
        base = (f"teacher.deepmd.{cli}.train", f"teacher.deepmd.{cli}.label")
        gmm = (f"teacher.deepmd.{cli}.gmm_reference", f"teacher.deepmd.{cli}.gmm_score")
        if any(role in contexts for role in gmm):
            if cli != "pt":
                raise ValueError("lifecycle DeepMD teacher GMM contexts are supported only for PT/DPA4")
            roles = (*base, *gmm)
        else:
            roles = base
        selected = {role for role in roles if role in contexts}
        if selected:
            missing = [role for role in roles if role not in contexts]
            if missing:
                raise ValueError(f"lifecycle executor factory missing required DeepMD operation context(s): {', '.join(missing)}")
            matching.append((roles, selected))
    if not matching:
        return None, set()
    if len(matching) != 1:
        raise ValueError("lifecycle executor factory requires exactly one DeepMD teacher CLI family")
    roles, selected = matching[0]
    return NativeDeepMDTeacherExecutor(
        train_context=contexts[roles[0]],
        label_context=contexts[roles[1]],
        gmm_reference_context=contexts.get("teacher.deepmd.pt.gmm_reference"),
        gmm_score_context=contexts.get("teacher.deepmd.pt.gmm_score"),
    ), selected


def _deepmd_student(contexts: Mapping[str, OperationContext]) -> tuple[NativeDeepMDStudentExecutor | None, set[str]]:
    matching: list[tuple[tuple[str, ...], set[str]]] = []
    for cli in ("tf", "pt"):
        base = (f"student.deepmd.{cli}.train",)
        compress = (f"student.deepmd.{cli}.compress",)
        gmm = (f"student.deepmd.{cli}.gmm_reference", f"student.deepmd.{cli}.gmm_score")
        fused = f"student.deepmd.{cli}.gmm_screen"
        if fused in contexts and any(role in contexts for role in gmm):
            raise ValueError(
                "lifecycle executor factory forbids mixing fused and phase-specific "
                "DeepMD student GMM contexts"
            )
        roles = (
            *base,
            *(compress if compress[0] in contexts else ()),
            *((fused,) if fused in contexts else ()),
            *(gmm if any(role in contexts for role in gmm) else ()),
        )
        selected = {role for role in roles if role in contexts}
        if selected:
            missing = [role for role in roles if role not in contexts]
            if missing:
                raise ValueError(f"lifecycle executor factory missing required DeepMD student operation context(s): {', '.join(missing)}")
            matching.append((roles, selected))
    if not matching:
        return None, set()
    if len(matching) != 1:
        raise ValueError("lifecycle executor factory requires exactly one DeepMD student CLI family")
    roles, selected = matching[0]
    operation_prefix = roles[0].rsplit(".", 1)[0]
    fused_context = contexts.get(f"{operation_prefix}.gmm_screen")
    return NativeDeepMDStudentExecutor(
        train_context=contexts[roles[0]],
        compress_context=contexts.get(f"{operation_prefix}.compress"),
        gmm_screen_context=fused_context,
        gmm_reference_context=(
            fused_context or contexts.get(f"{operation_prefix}.gmm_reference")
        ),
        gmm_score_context=(
            fused_context or contexts.get(f"{operation_prefix}.gmm_score")
        ),
    ), selected


def _lammps_sampler(contexts: Mapping[str, OperationContext]) -> tuple[NativeLammpsExecutor | None, set[str]]:
    matching = [role for role in ("sampler.lammps.tf", "sampler.lammps.pt") if role in contexts]
    if not matching:
        return None, set()
    if len(matching) != 1:
        raise ValueError("lifecycle executor factory requires exactly one LAMMPS sampler CLI family")
    role = matching[0]
    return NativeLammpsExecutor(contexts[role]), {role}


def _nep_student(contexts: Mapping[str, OperationContext]) -> tuple[NativeNepExecutor | None, set[str]]:
    base = ("student.nep.train",)
    gmm = ("student.nep.gmm_reference", "student.nep.gmm_score")
    roles = (*base, *gmm) if any(role in contexts for role in gmm) else base
    selected = {role for role in (*base, *gmm) if role in contexts}
    if not selected:
        return None, set()
    missing = [role for role in roles if role not in contexts]
    if missing:
        raise ValueError(f"lifecycle executor factory missing required NEP student operation context(s): {', '.join(missing)}")
    return NativeNepExecutor(
        train_context=contexts[base[0]],
        gmm_reference_context=contexts.get(gmm[0]),
        gmm_score_context=contexts.get(gmm[1]),
    ), selected


def _gpumd_sampler(contexts: Mapping[str, OperationContext]) -> tuple[NativeGpumdExecutor | None, set[str]]:
    role = "sampler.gpumd"
    if role not in contexts:
        return None, set()
    return NativeGpumdExecutor(contexts[role]), {role}


def _first_principles(
    contexts: Mapping[str, OperationContext],
) -> tuple[object | None, set[str]]:
    matching = [
        role
        for role in (
            "first_principles.cp2k.label",
            "first_principles.abacus.label",
        )
        if role in contexts
    ]
    if not matching:
        return None, set()
    if len(matching) != 1:
        raise ValueError(
            "lifecycle executor factory requires exactly one first-principles calculator"
        )
    role = matching[0]
    if role == "first_principles.cp2k.label":
        from workflow_backends.first_principles.cp2k.adapter import (
            CP2KCalculatorAdapter,
        )

        adapter = CP2KCalculatorAdapter()
    else:
        from workflow_backends.first_principles.abacus.adapter import (
            ABACUSCalculatorAdapter,
        )

        adapter = ABACUSCalculatorAdapter()
    from workflow_runtime.lifecycle.fp_service import (
        LocalFirstPrinciplesLabelService,
    )

    return (
        LocalFirstPrinciplesLabelService(
            adapter=adapter, operation_context=contexts[role]
        ),
        {role},
    )
