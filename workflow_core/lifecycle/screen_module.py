from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from workflow_core.lifecycle.routes import RouteKind, RouteSpec


@dataclass(frozen=True, slots=True)
class _SingleSingleCapabilities:
    score_teacher: Callable[..., Any]
    score_student: Callable[..., Any]
    label_teacher: Callable[..., list[Any]]


@dataclass(frozen=True, slots=True)
class _SingleTeacherCommitteeStudentCapabilities:
    score_teacher: Callable[..., Any]
    label_teacher: Callable[..., list[Any]]


@dataclass(frozen=True, slots=True)
class _CommitteeTeacherSingleStudentCapabilities:
    score_student: Callable[..., Any]
    label_teacher: Callable[..., list[Any]]


@dataclass(frozen=True, slots=True)
class _CommitteeCommitteeCapabilities:
    label_teacher: Callable[..., list[Any]]


class ScreenIOCapability(Protocol):
    """Imperative screening shell supplied by the application composition root."""

    score_teacher: Callable[..., Any]
    score_student: Callable[..., Any]
    label_teacher: Callable[..., list[Any]]
    read_student_vectors: Callable[..., Any]
    teacher_gmm_profile: Any
    teacher_label_profile: Any
    student_profile_values: tuple[Any, Any]


@dataclass(frozen=True, slots=True)
class LifecycleScreenModule:
    """Route-bound lifecycle projection into the closed screen executor."""

    config: Any
    route: RouteSpec
    io: ScreenIOCapability
    _capabilities: (
        _SingleSingleCapabilities | _SingleTeacherCommitteeStudentCapabilities
        | _CommitteeTeacherSingleStudentCapabilities
        | _CommitteeCommitteeCapabilities
    )
    observation_chunk_size: int
    gmm_score_chunk_size: int

    @classmethod
    def bind(
        cls,
        *,
        config: Any,
        io: ScreenIOCapability,
        observation_chunk_size: int = 64,
        gmm_score_chunk_size: int = 500,
    ) -> LifecycleScreenModule:
        route = config.route
        if not isinstance(route, RouteSpec):
            raise TypeError("lifecycle screen Module requires a resolved RouteSpec")
        if route.kind not in {
            RouteKind.SINGLE_TEACHER_SINGLE_STUDENT,
            RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT,
            RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT,
            RouteKind.COMMITTEE_TEACHER_COMMITTEE_STUDENT,
        }:
            raise NotImplementedError(
                "lifecycle screen Module currently binds only single/single"
            )

        if route.kind is RouteKind.SINGLE_TEACHER_SINGLE_STUDENT:
            capabilities = _SingleSingleCapabilities(
                score_teacher=io.score_teacher,
                score_student=io.score_student,
                label_teacher=io.label_teacher,
            )
        elif route.kind is RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT:
            capabilities = _SingleTeacherCommitteeStudentCapabilities(
                score_teacher=io.score_teacher,
                label_teacher=io.label_teacher,
            )
        elif route.kind is RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT:
            capabilities = _CommitteeTeacherSingleStudentCapabilities(
                score_student=io.score_student,
                label_teacher=io.label_teacher,
            )
        else:
            capabilities = _CommitteeCommitteeCapabilities(
                label_teacher=io.label_teacher,
            )
        return cls(
            config=config,
            route=route,
            io=io,
            _capabilities=capabilities,
            observation_chunk_size=_positive_int(
                observation_chunk_size, "observation chunk size"
            ),
            gmm_score_chunk_size=_positive_int(
                gmm_score_chunk_size, "GMM score chunk size"
            ),
        )

    def execute(
        self,
        *,
        round_index: int,
        stage: Path,
        source: Any,
        report: Any,
        selection_plan: Any | None = None,
    ):
        from workflow_core.lifecycle.screen_executor import (
            ScreenExecutionRequest,
            ScreenOperationProfileEvidence,
            ScreenRoundInputs,
            ScreenServices,
            execute_screen,
        )

        raw = self.config.raw

        def score_teacher(**kwargs):
            return self._capabilities.score_teacher(
                **kwargs,
                source=source,
                round_index=round_index,
            )

        def label_teacher(**kwargs):
            return self._capabilities.label_teacher(
                **kwargs,
                round_index=round_index,
            )

        teacher_score = None
        student_score = None
        student_vectors = None
        teacher_gmm_profile = None
        student_profile_values = (None, None)
        if isinstance(self._capabilities, _SingleSingleCapabilities):
            def score_student(**kwargs):
                return self._capabilities.score_student(
                    **kwargs,
                    source=source,
                    round_index=round_index,
                )

            student_score = score_student
            def read_student_vectors(**kwargs):
                return self.io.read_student_vectors(
                    **kwargs,
                    source=source,
                    round_index=round_index,
                )

            student_vectors = read_student_vectors
            student_profile_values = self.io.student_profile_values
            teacher_score = score_teacher
            teacher_gmm_profile = self.io.teacher_gmm_profile
        elif isinstance(
            self._capabilities, _SingleTeacherCommitteeStudentCapabilities
        ):
            teacher_score = score_teacher
            teacher_gmm_profile = self.io.teacher_gmm_profile
        elif isinstance(
            self._capabilities, _CommitteeTeacherSingleStudentCapabilities
        ):
            def score_student(**kwargs):
                return self._capabilities.score_student(
                    **kwargs,
                    source=source,
                    round_index=round_index,
                )

            student_score = score_student
            def read_student_vectors(**kwargs):
                return self.io.read_student_vectors(
                    **kwargs,
                    source=source,
                    round_index=round_index,
                )

            student_vectors = read_student_vectors
            student_profile_values = self.io.student_profile_values

        return execute_screen(
            ScreenExecutionRequest(
                route=self.route,
                round_index=round_index,
                iter_index=int(self.config.iter_index),
                stage=Path(stage),
                source=source,
                inputs=_round_inputs(
                    self.config,
                    self.route,
                    round_index,
                    selection_plan=selection_plan,
                ),
                services=ScreenServices(
                    score_teacher=teacher_score,
                    score_student=student_score,
                    label_teacher=label_teacher,
                    read_student_vectors=student_vectors,
                    cpu_pool_factory=getattr(self.io, "cpu_pool_factory", None),
                    operation_profiles=ScreenOperationProfileEvidence(
                        teacher_gmm_score=teacher_gmm_profile,
                        teacher_label=self.io.teacher_label_profile,
                        student_gmm_reference=student_profile_values[0],
                        student_gmm_score=student_profile_values[1],
                    ),
                ),
                report=report,
                artifacts=self.io.artifacts,
                observation_chunk_size=self.observation_chunk_size,
                teacher_router=(
                    self.config.root
                    / f"iter{self.config.iter_index}"
                    / "teach"
                    / "scan"
                    / "gmm_router"
                    if self.route.teacher_requires_reference_tail
                    else None
                ),
                student_train_path=(
                    self.config.root
                    / f"iter{self.config.iter_index}"
                    / f"r{round_index}"
                    / "train"
                    / "dataset"
                    / "train.xyz"
                ),
                student_type_map=tuple(self.config.student["type_map"]),
            )
        )


def _round_inputs(
    config: Any,
    route: RouteSpec,
    round_index: int,
    *,
    selection_plan: Any | None = None,
):
    from workflow_core.lifecycle.screen_executor import ScreenRoundInputs

    raw = config.raw
    screen = raw["screen"]
    sampling_plan = getattr(config, "sampling_plan", None)
    if selection_plan is None:
        if hasattr(sampling_plan, "selection_plan_for_round"):
            selection_plan = sampling_plan.selection_plan_for_round(round_index)
        else:
            selection_plan = getattr(config, "selection_plan", None)
    if selection_plan is None:
        from workflow_core.lifecycle.sampling_plan import SeedRole, SelectionPlan

        declarations = raw["sample"].get(
            "seeds", raw["sample"].get("systems", {})
        )
        selection_plan = SelectionPlan.from_seeds(
            tuple(SeedRole(str(role_id), str(role_id)) for role_id in declarations)
        )
    return ScreenRoundInputs(
        system_ids=selection_plan.group_ids,
        selection_plan=selection_plan,
        require_explicit_provenance=bool(
            getattr(sampling_plan, "explicit", "seeds" in raw["sample"])
        ),
        pair_guard=screen.get("pair_guard"),
        screen_policy=screen["policy"],
        fp_policy=raw["fp"]["policy"],
        pool=_round_value(screen["pool"], round_index),
        keep=_round_value(screen["keep"], round_index),
        fp_pool=int(screen["fp_pool"]) if "fp_pool" in screen else None,
        teacher_p=dict(screen["teacher_p"]) if "teacher_p" in screen else None,
        student_p=dict(screen["student_p"]) if "student_p" in screen else None,
        entropy=f"{config.path}/iter{config.iter_index}/r{round_index}",
        fill_underfilled_from_safe_region=bool(
            screen["fill_underfilled_from_safe_region"]
        ),
        prescreen=screen.get("prescreen", "GMM"),
        visualize=bool(
            getattr(config, "visualize_enabled", raw.get("visualize") is not False)
        ),
        student_force_lo=(
            float(screen["lo"])
            if route.kind
            in {
                RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT,
                RouteKind.COMMITTEE_TEACHER_COMMITTEE_STUDENT,
            }
            else None
        ),
        student_force_hi=(
            float(screen["hi"])
            if route.kind
            in {
                RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT,
                RouteKind.COMMITTEE_TEACHER_COMMITTEE_STUDENT,
            }
            else None
        ),
        teacher_force_lo=(
            float(config.teacher["lo"])
            if route.kind
            in {
                RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT,
                RouteKind.COMMITTEE_TEACHER_COMMITTEE_STUDENT,
            }
            else None
        ),
        teacher_force_hi=(
            float(config.teacher["hi"])
            if route.kind
            in {
                RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT,
                RouteKind.COMMITTEE_TEACHER_COMMITTEE_STUDENT,
            }
            else None
        ),
    )


def _round_value(value: object, index: int) -> int:
    return int(value[index] if isinstance(value, list) else value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"lifecycle screen {label} must be a positive integer")
    return value


__all__ = ["LifecycleScreenModule", "ScreenIOCapability"]
