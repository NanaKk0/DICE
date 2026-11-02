from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from ase import Atoms

from workflow_artifacts.lifecycle.screen_report import (
    COMMITTEE_EVIDENCE,
    GMM_EVIDENCE,
    JOINT_DIRECT_EVIDENCE,
    ScreenReport,
    ScreenReportRole,
    ScreenReportRow,
)
from workflow_core.lifecycle.routes import EvidenceKind, RouteSpec


def build_committee_teach_screen_report(
    *,
    iter_index: int,
    route: RouteSpec,
    input_frames: Sequence[Atoms],
    labeled_frames: Sequence[Atoms],
    lo: float,
    hi: float,
    published_train: int,
    published_fp_candidates: int,
) -> ScreenReport:
    """Build the total-only committee teacher bootstrap report.

    ``input_frames`` defines N.  Labels supply only the already-computed force
    deviations; this helper never filters or changes the canonical output.
    """
    if route.teacher_evidence_kind is not EvidenceKind.FORCE_DEVIATION:
        raise ValueError("committee teach screen report requires committee teacher evidence")
    if len(input_frames) != len(labeled_frames):
        raise ValueError("committee teach screen report labels do not cover input frames")
    counts = _committee_counts(labeled_frames, lo=lo, hi=hi)
    role = ScreenReportRole(
        role="teacher",
        evidence=COMMITTEE_EVIDENCE,
        thresholds={"lo": float(lo), "hi": float(hi)},
        systems=(),
        total=ScreenReportRow(
            system_id="TOTAL",
            denominator=len(input_frames),
            counts=counts,
        ),
    )
    return ScreenReport(
        iter_index=iter_index,
        block="teach/screen",
        route=_route_identity(route),
        roles=(role,),
        published_train=published_train,
        published_fp_candidates=published_fp_candidates,
    )


def build_committee_role(
    *,
    role: str,
    frames: Sequence[Atoms],
    system_ids: tuple[str, ...],
    lo: float,
    hi: float,
    value: Callable[[Atoms], float] | None = None,
) -> ScreenReportRole:
    values = tuple(float(item.info["force_uncertainty"]) if value is None else float(value(item)) for item in frames)
    by_system = _group_values(frames, values, system_ids)
    system_rows = tuple(
        ScreenReportRow(
            system_id=system_id,
            denominator=len(items),
            counts=_committee_value_counts(item_values, lo=lo, hi=hi),
        )
        for system_id, items, item_values in by_system
    )
    return ScreenReportRole(
        role=role,
        evidence=COMMITTEE_EVIDENCE,
        thresholds={"lo": float(lo), "hi": float(hi)},
        systems=system_rows,
        total=ScreenReportRow(
            system_id="TOTAL",
            denominator=len(frames),
            counts=_committee_value_counts(values, lo=lo, hi=hi),
        ),
    )


def build_gmm_role(
    *,
    role: str,
    frames: Sequence[Atoms],
    flags: Mapping[str, Mapping[str, bool]],
    system_ids: tuple[str, ...],
    thresholds: Mapping[str, float | None],
) -> ScreenReportRole:
    if role not in {"teacher", "student"}:
        raise ValueError("GMM screen report role is invalid")
    prefix = "teacher" if role == "teacher" else "student"
    keys = {
        "covered": f"{prefix}_covered",
        "candidate": f"{prefix}_candidate",
        "discard": f"{prefix}_floor_discard",
    }

    def counts(items: Sequence[Atoms]) -> dict[str, int]:
        identities = [str(frame.info["structure_id"]) for frame in items]
        return {
            column: sum(bool(flags.get(identity, {}).get(flag)) for identity in identities)
            for column, flag in keys.items()
        }

    grouped = _group_frames(frames, system_ids)
    rows = tuple(
        ScreenReportRow(system_id=system_id, denominator=len(items), counts=counts(items))
        for system_id, items in grouped
    )
    normalized = {
        "floor": float(thresholds["floor"]),
        "trust": float(thresholds["trust"]) if role == "teacher" else None,
        "ceiling": (
            None
            if thresholds.get("ceiling") is None
            else float(thresholds["ceiling"])
        ),
    }
    if role == "student":
        normalized.pop("trust")
    return ScreenReportRole(
        role=role,
        evidence=GMM_EVIDENCE,
        thresholds=normalized,
        systems=rows,
        total=ScreenReportRow(
            system_id="TOTAL",
            denominator=len(frames),
            counts=counts(frames),
        ),
    )


def build_joint_direct_role(
    *,
    frames: Sequence[Atoms],
    outcomes: Mapping[str, Mapping[str, int | bool]],
    system_ids: tuple[str, ...],
    pool: int,
) -> ScreenReportRole:
    rows = tuple(
        ScreenReportRow(
            system_id=system_id,
            denominator=int(outcomes.get(system_id, {}).get("candidate_count", 0)),
            counts={
                "candidate": int(outcomes.get(system_id, {}).get("candidate_count", 0)),
                "proposal": int(outcomes.get(system_id, {}).get("selected_candidate_count", 0)),
                "underfill": int(outcomes.get(system_id, {}).get("underfill", 0)),
            },
        )
        for system_id in system_ids
    )
    totals = {
        key: sum(row.counts[key] for row in rows)
        for key in ("candidate", "proposal", "underfill")
    }
    return ScreenReportRole(
        role="student",
        evidence=JOINT_DIRECT_EVIDENCE,
        thresholds={"pool": int(pool)},
        systems=rows,
        total=ScreenReportRow(
            system_id="TOTAL", denominator=len(frames), counts=totals
        ),
    )


def build_joint_single_teacher_round_screen_report(
    *,
    iter_index: int,
    round_index: int,
    route: RouteSpec,
    system_ids: tuple[str, ...],
    teacher_frames: Sequence[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    teacher_thresholds: Mapping[str, float | None],
    observations: Sequence[Atoms],
    outcomes: Mapping[str, Mapping[str, int | bool]],
    pool: int,
    published_train: int,
    published_fp_candidates: int,
) -> ScreenReport:
    return ScreenReport(
        iter_index=iter_index,
        block=f"r{round_index}/screen",
        route=_route_identity(route),
        roles=(
            build_gmm_role(
                role="teacher",
                frames=teacher_frames,
                flags=teacher_flags,
                system_ids=system_ids,
                thresholds=teacher_thresholds,
            ),
            build_joint_direct_role(
                frames=observations,
                outcomes=outcomes,
                system_ids=system_ids,
                pool=pool,
            ),
        ),
        published_train=published_train,
        published_fp_candidates=published_fp_candidates,
    )


def build_joint_committee_teacher_round_screen_report(
    *,
    iter_index: int,
    round_index: int,
    route: RouteSpec,
    system_ids: tuple[str, ...],
    teacher_frames: Sequence[Atoms],
    teacher_lo: float,
    teacher_hi: float,
    observations: Sequence[Atoms],
    outcomes: Mapping[str, Mapping[str, int | bool]],
    pool: int,
    published_train: int,
    published_fp_candidates: int,
) -> ScreenReport:
    return ScreenReport(
        iter_index=iter_index,
        block=f"r{round_index}/screen",
        route=_route_identity(route),
        roles=(
            build_committee_role(
                role="teacher",
                frames=teacher_frames,
                system_ids=system_ids,
                lo=teacher_lo,
                hi=teacher_hi,
            ),
            build_joint_direct_role(
                frames=observations,
                outcomes=outcomes,
                system_ids=system_ids,
                pool=pool,
            ),
        ),
        published_train=published_train,
        published_fp_candidates=published_fp_candidates,
    )


def build_gmm_round_screen_report(
    *,
    iter_index: int,
    round_index: int,
    route: RouteSpec,
    system_ids: tuple[str, ...],
    teacher_frames: Sequence[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    teacher_thresholds: Mapping[str, float | None],
    student_frames: Sequence[Atoms],
    student_flags: Mapping[str, Mapping[str, bool]],
    student_thresholds: Mapping[str, float | None],
    published_train: int,
    published_fp_candidates: int,
) -> ScreenReport:
    """Build the two-role GMM report without changing route decisions."""
    return ScreenReport(
        iter_index=iter_index,
        block=f"r{round_index}/screen",
        route=_route_identity(route),
        roles=(
            build_gmm_role(
                role="teacher",
                frames=teacher_frames,
                flags=teacher_flags,
                system_ids=system_ids,
                thresholds=teacher_thresholds,
            ),
            build_gmm_role(
                role="student",
                frames=student_frames,
                flags=student_flags,
                system_ids=system_ids,
                thresholds=student_thresholds,
            ),
        ),
        published_train=published_train,
        published_fp_candidates=published_fp_candidates,
    )


def build_gmm_committee_round_screen_report(
    *,
    iter_index: int,
    round_index: int,
    route: RouteSpec,
    system_ids: tuple[str, ...],
    teacher_frames: Sequence[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    teacher_thresholds: Mapping[str, float | None],
    student_frames: Sequence[Atoms],
    student_lo: float,
    student_hi: float,
    published_train: int,
    published_fp_candidates: int,
) -> ScreenReport:
    """Build the single-teacher/committee-student round report."""
    return ScreenReport(
        iter_index=iter_index,
        block=f"r{round_index}/screen",
        route=_route_identity(route),
        roles=(
            build_gmm_role(
                role="teacher",
                frames=teacher_frames,
                flags=teacher_flags,
                system_ids=system_ids,
                thresholds=teacher_thresholds,
            ),
            build_committee_role(
                role="student",
                frames=student_frames,
                system_ids=system_ids,
                lo=student_lo,
                hi=student_hi,
            ),
        ),
        published_train=published_train,
        published_fp_candidates=published_fp_candidates,
    )


def build_committee_gmm_round_screen_report(
    *,
    iter_index: int,
    round_index: int,
    route: RouteSpec,
    system_ids: tuple[str, ...],
    teacher_frames: Sequence[Atoms],
    teacher_lo: float,
    teacher_hi: float,
    student_frames: Sequence[Atoms],
    student_flags: Mapping[str, Mapping[str, bool]],
    student_thresholds: Mapping[str, float | None],
    published_train: int,
    published_fp_candidates: int,
) -> ScreenReport:
    """Build the committee-teacher/single-student round report."""
    return ScreenReport(
        iter_index=iter_index,
        block=f"r{round_index}/screen",
        route=_route_identity(route),
        roles=(
            build_committee_role(
                role="teacher",
                frames=teacher_frames,
                system_ids=system_ids,
                lo=teacher_lo,
                hi=teacher_hi,
            ),
            build_gmm_role(
                role="student",
                frames=student_frames,
                flags=student_flags,
                system_ids=system_ids,
                thresholds=student_thresholds,
            ),
        ),
        published_train=published_train,
        published_fp_candidates=published_fp_candidates,
    )


def build_committee_committee_round_screen_report(
    *,
    iter_index: int,
    round_index: int,
    route: RouteSpec,
    system_ids: tuple[str, ...],
    teacher_frames: Sequence[Atoms],
    teacher_lo: float,
    teacher_hi: float,
    student_frames: Sequence[Atoms],
    student_lo: float,
    student_hi: float,
    published_train: int,
    published_fp_candidates: int,
) -> ScreenReport:
    """Build the committee-teacher/committee-student round report."""
    return ScreenReport(
        iter_index=iter_index,
        block=f"r{round_index}/screen",
        route=_route_identity(route),
        roles=(
            build_committee_role(
                role="teacher",
                frames=teacher_frames,
                system_ids=system_ids,
                lo=teacher_lo,
                hi=teacher_hi,
            ),
            build_committee_role(
                role="student",
                frames=student_frames,
                system_ids=system_ids,
                lo=student_lo,
                hi=student_hi,
            ),
        ),
        published_train=published_train,
        published_fp_candidates=published_fp_candidates,
    )


def _committee_counts(frames: Sequence[Atoms], *, lo: float, hi: float) -> dict[str, int]:
    return _committee_value_counts(
        tuple(float(frame.info["force_uncertainty"]) for frame in frames),
        lo=lo,
        hi=hi,
    )


def _committee_value_counts(values: Sequence[float], *, lo: float, hi: float) -> dict[str, int]:
    if not 0 <= float(lo) < float(hi):
        raise ValueError("committee report thresholds must satisfy 0 <= lo < hi")
    if any(value < 0 for value in values):
        raise ValueError("committee report deviations must be non-negative")
    return {
        "accurate": sum(value < lo for value in values),
        "candidate": sum(lo <= value < hi for value in values),
        "failed": sum(value >= hi for value in values),
    }


def _group_frames(frames: Sequence[Atoms], system_ids: tuple[str, ...]):
    return tuple(
        (system_id, tuple(frame for frame in frames if str(frame.info.get("system_id", "")) == system_id))
        for system_id in system_ids
    )


def _group_values(frames: Sequence[Atoms], values: Sequence[float], system_ids: tuple[str, ...]):
    by_identity = {id(frame): value for frame, value in zip(frames, values, strict=True)}
    return tuple(
        (
            system_id,
            tuple(frame for frame in frames if str(frame.info.get("system_id", "")) == system_id),
            tuple(by_identity[id(frame)] for frame in frames if str(frame.info.get("system_id", "")) == system_id),
        )
        for system_id in system_ids
    )


def _route_identity(route: RouteSpec) -> dict[str, object]:
    return route.report_identity()


__all__ = [
    "build_committee_role",
    "build_committee_teach_screen_report",
    "build_gmm_role",
    "build_gmm_committee_round_screen_report",
    "build_gmm_round_screen_report",
    "build_committee_gmm_round_screen_report",
    "build_committee_committee_round_screen_report",
]
