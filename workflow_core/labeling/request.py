from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PathLike = str | Path


@dataclass(slots=True)
class TeacherLabelingInput:
    structure_file: PathLike
    include_user_student_train: bool = True
    merged_xyz: PathLike | None = None


@dataclass(slots=True)
class TeacherCommitteeRequest:
    teacher_round: int
    committee_dir: PathLike
    backend: str | None = None
    teacher_num: int | None = None


@dataclass(slots=True)
class TeacherLabelingOutputTargets:
    labeled_output_dir: PathLike
    active_learning_output_dir: PathLike
    append_active_learning_xyz: PathLike | None = None


@dataclass(slots=True)
class StudentLabelExportRequest:
    train_committee_config: PathLike | None = None
    training_stage: str | None = None
    valid_fraction: float | None = None
    element_order: list[str] | None = None


@dataclass(slots=True)
class TeacherLabelingRequest:
    labeling_input: TeacherLabelingInput
    teacher_committee: TeacherCommitteeRequest
    output_targets: TeacherLabelingOutputTargets
    student_export: StudentLabelExportRequest


def build_teacher_labeling_request(
    *,
    structure_file: PathLike,
    teacher_round: int,
    teacher_dir: PathLike,
    labeled_output_dir: PathLike,
    active_learning_output_dir: PathLike,
    include_user_student_train: bool = False,
    merged_xyz: PathLike | None = None,
    append_active_learning_xyz: PathLike | None = None,
    train_committee_config: PathLike | None = None,
    training_stage: str | None = None,
    teacher_backend: str | None = None,
    teacher_num: int | None = None,
    valid_fraction: float | None = None,
    element_order: list[str] | None = None,
) -> TeacherLabelingRequest:
    return TeacherLabelingRequest(
        labeling_input=TeacherLabelingInput(
            structure_file=structure_file,
            include_user_student_train=bool(include_user_student_train),
            merged_xyz=merged_xyz,
        ),
        teacher_committee=TeacherCommitteeRequest(
            teacher_round=int(teacher_round),
            committee_dir=teacher_dir,
            backend=teacher_backend,
            teacher_num=int(teacher_num) if teacher_num is not None else None,
        ),
        output_targets=TeacherLabelingOutputTargets(
            labeled_output_dir=labeled_output_dir,
            active_learning_output_dir=active_learning_output_dir,
            append_active_learning_xyz=append_active_learning_xyz,
        ),
        student_export=StudentLabelExportRequest(
            train_committee_config=train_committee_config,
            training_stage=training_stage,
            valid_fraction=valid_fraction,
            element_order=element_order,
        ),
    )
