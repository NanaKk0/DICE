from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from workflow_core.identities import RevisionedIdentity
from workflow_policies.identities import (
    GMM_REFERENCE_TAIL,
    PAIR_GUARD,
    STUDENT_REFERENCE_TEACHER_REFERENCE,
    STUDENT_DEVIATION_TEACHER_REFERENCE,
    TEACHER_DEVIATION_STUDENT_REFERENCE,
    selector_identity,
)

class RouteKind(str, Enum):
    SINGLE_TEACHER_SINGLE_STUDENT = "single-teacher/single-student"
    SINGLE_TEACHER_COMMITTEE_STUDENT = "single-teacher/committee-student"
    COMMITTEE_TEACHER_SINGLE_STUDENT = "committee-teacher/single-student"
    COMMITTEE_TEACHER_COMMITTEE_STUDENT = "committee-teacher/committee-student"


class EvidenceKind(str, Enum):
    REFERENCE_TAIL = "reference-tail"
    FORCE_DEVIATION = "force-deviation"


class ScreenArtifactContract(str, Enum):
    GMM_REFERENCE_TAIL = "gmm-reference-tail"
    SINGLE_TEACHER_COMMITTEE_STUDENT = "single-teacher/committee-student"
    COMMITTEE_TEACHER_SINGLE_STUDENT = "committee-teacher/single-student"
    COMMITTEE_FORCE_DEVIATION = "committee-force-deviation"


@dataclass(frozen=True, slots=True)
class RouteSpec:
    kind: RouteKind
    teacher_evidence_kind: EvidenceKind
    student_evidence_kind: EvidenceKind
    has_teach_scan: bool
    teacher_requires_reference_tail: bool
    student_requires_reference_tail: bool
    screen_routing_policy_id: RevisionedIdentity | None
    reference_tail_policy_id: RevisionedIdentity | None
    screen_artifact_contract: ScreenArtifactContract
    teacher_operation_prefix: str
    student_operation_prefix: str
    sampler_operation_role: str
    sample_compression_role: str | None

    def report_identity(self) -> dict[str, object]:
        identity: dict[str, object] = {"name": self.kind.value, "revision": 1}
        if self.kind in {
            RouteKind.SINGLE_TEACHER_SINGLE_STUDENT,
            RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT,
        }:
            if self.screen_routing_policy_id is None:
                raise ValueError("single-teacher route requires a directional policy identity")
            identity.update(
                {
                    "directional_policy": self.screen_routing_policy_id.as_dict(),
                }
            )
        return identity

    def policy_evidence(
        self, run_json: Mapping[str, Any], block: str
    ) -> dict[str, Any]:
        if block == "teach/scan":
            return (
                {"gmm_router": self.reference_tail_policy_id.as_dict()}
                if self.has_teach_scan and self.reference_tail_policy_id is not None
                else {}
            )
        if block.startswith("r") and block.endswith("/screen"):
            screen = run_json["screen"]
            joint_prescreen = screen.get("prescreen") in {
                "JOINT_DIRECT",
                "ELEMENT_JOINT_DIRECT",
            }
            evidence: dict[str, Any] = {
                "selector": selector_identity(str(screen["policy"])).as_dict(),
                "fill_underfilled_from_safe_region": bool(
                    screen["fill_underfilled_from_safe_region"]
                ),
            }
            if "pair_guard" in screen:
                evidence["pair_guard"] = PAIR_GUARD.as_dict()
            if joint_prescreen:
                evidence["prescreen"] = selector_identity(
                    str(screen["prescreen"])
                ).as_dict()
            if self.kind is RouteKind.SINGLE_TEACHER_SINGLE_STUDENT:
                evidence.update({
                    "gmm_router": self.reference_tail_policy_id.as_dict(),
                    "teacher_p": dict(screen["teacher_p"]),
                    "routing": self.screen_routing_policy_id.as_dict(),
                })
                if not joint_prescreen:
                    evidence["student_p"] = dict(screen["student_p"])
            elif self.kind is RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT:
                evidence.update(
                    {
                        "routing": self.screen_routing_policy_id.as_dict(),
                        "teacher_p": dict(screen["teacher_p"]),
                        "student_force_deviation": {
                            "lo": screen["lo"],
                            "hi": screen["hi"],
                        },
                        "fp_pool": screen["fp_pool"],
                        "pool": screen["pool"],
                        "keep": screen["keep"],
                    }
                )
            elif self.kind is RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT:
                teacher = run_json["teacher"]
                evidence.update(
                    {
                        "routing": self.screen_routing_policy_id.as_dict(),
                        "teacher_force_deviation": {
                            "lo": teacher["lo"],
                            "hi": teacher["hi"],
                        },
                        "pool": screen["pool"],
                        "keep": screen["keep"],
                    }
                )
                if not joint_prescreen:
                    evidence["student_p"] = dict(screen["student_p"])
            return evidence
        if block == "fp":
            return {"selector": selector_identity(str(run_json["fp"]["policy"])).as_dict()}
        return {}

    def operation_roles(self, block: str) -> tuple[str, ...]:
        if block == "teach/train":
            return (f"{self.teacher_operation_prefix}.train",)
        if block == "teach/screen":
            return (f"{self.teacher_operation_prefix}.label",)
        if block == "teach/scan":
            if not self.has_teach_scan:
                return ()
            return (f"{self.teacher_operation_prefix}.gmm_reference",)
        if block.startswith("r") and block.endswith("/train"):
            return (f"{self.student_operation_prefix}.train",)
        if block.startswith("r") and block.endswith("/sample"):
            return (
                (self.sample_compression_role, self.sampler_operation_role)
                if self.sample_compression_role is not None
                else (self.sampler_operation_role,)
            )
        if block.startswith("r") and block.endswith("/screen"):
            student_gmm_roles = (
                (f"{self.student_operation_prefix}.gmm_screen",)
                if self.student_operation_prefix == "student.deepmd.pt"
                else (
                    f"{self.student_operation_prefix}.gmm_reference",
                    f"{self.student_operation_prefix}.gmm_score",
                )
            )
            if self.kind is RouteKind.SINGLE_TEACHER_SINGLE_STUDENT:
                return (
                    f"{self.teacher_operation_prefix}.gmm_score",
                    f"{self.teacher_operation_prefix}.label",
                    *student_gmm_roles,
                )
            if self.kind is RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT:
                return (
                    f"{self.teacher_operation_prefix}.gmm_score",
                    f"{self.teacher_operation_prefix}.label",
                )
            if self.kind is RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT:
                return (
                    f"{self.teacher_operation_prefix}.label",
                    *student_gmm_roles,
                )
            return (f"{self.teacher_operation_prefix}.label",)
        if block == "fp":
            return ()
        raise ValueError(f"unknown lifecycle lifecycle block {block!r}")


def resolve_route(
    *,
    teacher: Mapping[str, Any],
    student: Mapping[str, Any],
    sampler: Mapping[str, Any],
) -> RouteSpec:
    _validate_route_identities(teacher=teacher, student=student, sampler=sampler)
    teacher_shape = teacher.get("model")
    student_shape = student.get("model")
    if teacher_shape not in {"single", "committee"} or student_shape not in {
        "single",
        "committee",
    }:
        raise ValueError("lifecycle route resolution requires single or committee model shapes")
    expected_sampler = "LAMMPS" if student.get("backend") == "DeepMD" else "GPUMD"
    if sampler.get("backend") != expected_sampler:
        raise ValueError(
            f"student backend {student.get('backend')} requires sampler backend {expected_sampler}"
        )

    teacher_requires_reference_tail = teacher_shape == "single"
    student_requires_reference_tail = student_shape == "single"
    teacher_reference_capable = teacher.get("backend") == "MACE" or (
        teacher.get("backend") == "DeepMD" and teacher.get("cli") == "pt"
    )
    student_reference_capable = (
        student.get("backend") == "DeepMD" and student.get("cli") in {"tf", "pt"}
    ) or student.get("backend") == "NEP"
    if (
        teacher_requires_reference_tail
        and not teacher_reference_capable
        or student_requires_reference_tail
        and not student_reference_capable
    ):
        raise ValueError(
            "supported lifecycle GMM and mixed routes require a single MACE or DeepMD-PT DPA4 "
            "teacher and/or a single DeepMD-TF se_e2_a, DeepMD-PT DPA4C, or NEP4 student; DeepMD committee "
            "roles may use TF or PT, NEP committee students use GPUMD, and committee "
            "teachers may use MACE or DeepMD TF/PT"
        )

    teacher_prefix = _operation_prefix("teacher", teacher)
    student_prefix = _operation_prefix("student", student)
    sampler_role = _sampler_operation_role(student, sampler)
    compression_role = (
        f"{student_prefix}.compress"
        if student.get("backend") == "DeepMD" and student.get("compress") is True
        else None
    )
    route_key = (teacher_shape, student_shape)
    if route_key == ("single", "single"):
        return RouteSpec(
            kind=RouteKind.SINGLE_TEACHER_SINGLE_STUDENT,
            teacher_evidence_kind=EvidenceKind.REFERENCE_TAIL,
            student_evidence_kind=EvidenceKind.REFERENCE_TAIL,
            has_teach_scan=True,
            teacher_requires_reference_tail=True,
            student_requires_reference_tail=True,
            screen_routing_policy_id=STUDENT_REFERENCE_TEACHER_REFERENCE,
            reference_tail_policy_id=GMM_REFERENCE_TAIL,
            screen_artifact_contract=ScreenArtifactContract.GMM_REFERENCE_TAIL,
            teacher_operation_prefix=teacher_prefix,
            student_operation_prefix=student_prefix,
            sampler_operation_role=sampler_role,
            sample_compression_role=compression_role,
        )
    if route_key == ("single", "committee"):
        return RouteSpec(
            kind=RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT,
            teacher_evidence_kind=EvidenceKind.REFERENCE_TAIL,
            student_evidence_kind=EvidenceKind.FORCE_DEVIATION,
            has_teach_scan=True,
            teacher_requires_reference_tail=True,
            student_requires_reference_tail=False,
            screen_routing_policy_id=STUDENT_DEVIATION_TEACHER_REFERENCE,
            reference_tail_policy_id=GMM_REFERENCE_TAIL,
            screen_artifact_contract=ScreenArtifactContract.SINGLE_TEACHER_COMMITTEE_STUDENT,
            teacher_operation_prefix=teacher_prefix,
            student_operation_prefix=student_prefix,
            sampler_operation_role=sampler_role,
            sample_compression_role=compression_role,
        )
    if route_key == ("committee", "single"):
        return RouteSpec(
            kind=RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT,
            teacher_evidence_kind=EvidenceKind.FORCE_DEVIATION,
            student_evidence_kind=EvidenceKind.REFERENCE_TAIL,
            has_teach_scan=False,
            teacher_requires_reference_tail=False,
            student_requires_reference_tail=True,
            screen_routing_policy_id=TEACHER_DEVIATION_STUDENT_REFERENCE,
            reference_tail_policy_id=GMM_REFERENCE_TAIL,
            screen_artifact_contract=ScreenArtifactContract.COMMITTEE_TEACHER_SINGLE_STUDENT,
            teacher_operation_prefix=teacher_prefix,
            student_operation_prefix=student_prefix,
            sampler_operation_role=sampler_role,
            sample_compression_role=compression_role,
        )
    return RouteSpec(
        kind=RouteKind.COMMITTEE_TEACHER_COMMITTEE_STUDENT,
        teacher_evidence_kind=EvidenceKind.FORCE_DEVIATION,
        student_evidence_kind=EvidenceKind.FORCE_DEVIATION,
        has_teach_scan=False,
        teacher_requires_reference_tail=False,
        student_requires_reference_tail=False,
        screen_routing_policy_id=None,
        reference_tail_policy_id=None,
        screen_artifact_contract=ScreenArtifactContract.COMMITTEE_FORCE_DEVIATION,
        teacher_operation_prefix=teacher_prefix,
        student_operation_prefix=student_prefix,
        sampler_operation_role=sampler_role,
        sample_compression_role=compression_role,
    )


def _operation_prefix(role: str, spec: Mapping[str, Any]) -> str:
    backend = str(spec["backend"]).lower()
    cli = f".{spec['cli']}" if backend == "deepmd" else ""
    return f"{role}.{backend}{cli}"


def _validate_route_identities(
    *,
    teacher: Mapping[str, Any],
    student: Mapping[str, Any],
    sampler: Mapping[str, Any],
) -> None:
    teacher_backend = teacher.get("backend")
    if teacher_backend not in {"MACE", "DeepMD"}:
        raise ValueError("lifecycle route resolution received an unsupported teacher backend")
    if teacher_backend == "DeepMD" and teacher.get("cli") not in {"tf", "pt"}:
        raise ValueError("lifecycle route resolution requires teacher.cli to be tf or pt")
    if teacher_backend == "MACE" and "cli" in teacher:
        raise ValueError("lifecycle route resolution forbids teacher.cli for MACE")

    student_backend = student.get("backend")
    if student_backend not in {"DeepMD", "NEP"}:
        raise ValueError("lifecycle route resolution received an unsupported student backend")
    if student_backend == "DeepMD" and student.get("cli") not in {"tf", "pt"}:
        raise ValueError("lifecycle route resolution requires student.cli to be tf or pt")
    if student_backend == "NEP" and "cli" in student:
        raise ValueError("lifecycle route resolution forbids student.cli for NEP")

    if sampler.get("backend") not in {"LAMMPS", "GPUMD"}:
        raise ValueError("lifecycle route resolution received an unsupported sampler backend")


def _sampler_operation_role(
    student: Mapping[str, Any], sampler: Mapping[str, Any]
) -> str:
    backend = str(sampler["backend"]).lower()
    cli = f".{student['cli']}" if backend == "lammps" else ""
    return f"sampler.{backend}{cli}"
