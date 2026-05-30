from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TeacherRoundContext:
    teacher_round: int

    def as_dict(self) -> dict[str, int]:
        return {
            "teacher_round": int(self.teacher_round),
        }


@dataclass(frozen=True, slots=True)
class StudentRoundContext:
    teacher_round: int
    student_round: int

    def as_dict(self) -> dict[str, int]:
        return {
            "teacher_round": int(self.teacher_round),
            "student_round": int(self.student_round),
        }


def coerce_teacher_round_context(teacher_round: int | TeacherRoundContext) -> TeacherRoundContext:
    if isinstance(teacher_round, TeacherRoundContext):
        return teacher_round
    return TeacherRoundContext(teacher_round=int(teacher_round))


def coerce_student_round_context(
    teacher_round: int | TeacherRoundContext | StudentRoundContext,
    student_round: int | None = None,
) -> StudentRoundContext:
    if isinstance(teacher_round, StudentRoundContext):
        if student_round is not None:
            raise ValueError("student_round must be omitted when passing StudentRoundContext")
        return teacher_round
    if isinstance(teacher_round, TeacherRoundContext):
        if student_round is None:
            raise ValueError("student_round is required when passing TeacherRoundContext")
        return StudentRoundContext(teacher_round=teacher_round.teacher_round, student_round=int(student_round))
    if student_round is None:
        raise ValueError("student_round is required when teacher_round is not a StudentRoundContext")
    return StudentRoundContext(teacher_round=int(teacher_round), student_round=int(student_round))
