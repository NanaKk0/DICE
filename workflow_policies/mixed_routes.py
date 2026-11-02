from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms

from workflow_policies.selection import p_space_sample, select_representatives
from workflow_policies.identities import (
    STUDENT_DEVIATION_TEACHER_REFERENCE,
    TEACHER_DEVIATION_STUDENT_REFERENCE,
)

SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_ID = (
    STUDENT_DEVIATION_TEACHER_REFERENCE.name
)
SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_REVISION = (
    STUDENT_DEVIATION_TEACHER_REFERENCE.revision
)
COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID = (
    TEACHER_DEVIATION_STUDENT_REFERENCE.name
)
COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_REVISION = (
    TEACHER_DEVIATION_STUDENT_REFERENCE.revision
)


def select_single_teacher_committee_student_train(
    frames: Sequence[Atoms],
    *,
    pool: int,
    keep: int,
    lo: float,
    hi: float,
    policy: str,
    entropy: str,
) -> tuple[list[Atoms], dict[str, dict[str, bool]]]:
    """Apply the committee-student band, pool, and selector after teacher trust."""
    lower = float(lo)
    upper = float(hi)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper:
        raise ValueError("committee-student force-deviation band must satisfy 0 <= lo < hi")
    if pool == 0 or pool < -1:
        raise ValueError("committee-student screen.pool must be positive or -1")
    if keep <= 0 or (pool > 0 and keep > pool):
        raise ValueError("committee-student screen.keep must be positive and not exceed pool")

    values: dict[str, float] = {}
    flags: dict[str, dict[str, bool]] = {}
    for frame in frames:
        identity = str(frame.info["structure_id"])
        if identity in values:
            raise ValueError("committee-student selection identities must be unique")
        value = float(frame.info["force_uncertainty"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("student force deviation must be finite and non-negative")
        values[identity] = value
        flags[identity] = {
            "student_in_band": lower <= value < upper,
            "student_pool": False,
            "train_selected": False,
        }

    selected_ids: set[str] = set()
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [
            frame
            for frame in frames
            if str(frame.info["system_id"]) == system_id
            and flags[str(frame.info["structure_id"])]["student_in_band"]
        ]
        candidates.sort(
            key=lambda frame: (
                -values[str(frame.info["structure_id"])],
                str(frame.info["structure_id"]),
            )
        )
        pooled = candidates if pool == -1 else candidates[:pool]
        identities = [str(frame.info["structure_id"]) for frame in pooled]
        for identity in identities:
            flags[identity]["student_pool"] = True
        if not identities:
            continue
        vectors = np.asarray([[values[identity]] for identity in identities], dtype=float)
        chosen = select_representatives(
            identities,
            vectors,
            keep=min(keep, len(identities)),
            policy=policy,
            entropy=f"{entropy}/{system_id}",
        )
        selected_ids.update(chosen)
        for identity in chosen:
            flags[identity]["train_selected"] = True

    return (
        [frame for frame in frames if str(frame.info["structure_id"]) in selected_ids],
        flags,
    )


def propose_single_teacher_committee_student_pool(
    frames: Sequence[Atoms],
    *,
    pool: int,
    lo: float,
    hi: float,
) -> tuple[list[Atoms], dict[str, dict[str, bool]]]:
    """Form the native descending-deviation proposal before teacher scoring."""
    lower = float(lo)
    upper = float(hi)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper:
        raise ValueError("committee-student force-deviation band must satisfy 0 <= lo < hi")
    if pool == 0 or pool < -1:
        raise ValueError("committee-student screen.pool must be positive or -1")
    values: dict[str, float] = {}
    flags: dict[str, dict[str, bool]] = {}
    for frame in frames:
        identity = str(frame.info["structure_id"])
        if identity in values:
            raise ValueError("committee-student proposal identities must be unique")
        value = float(frame.info["force_uncertainty"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("student force deviation must be finite and non-negative")
        values[identity] = value
        flags[identity] = {
            "student_in_band": lower <= value < upper,
            "student_pool": False,
            "train_selected": False,
        }
    pooled_ids: set[str] = set()
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [
            frame
            for frame in frames
            if str(frame.info["system_id"]) == system_id
            and flags[str(frame.info["structure_id"])]["student_in_band"]
        ]
        candidates.sort(
            key=lambda frame: (
                -values[str(frame.info["structure_id"])],
                str(frame.info["structure_id"]),
            )
        )
        pooled = candidates if pool == -1 else candidates[:pool]
        for frame in pooled:
            identity = str(frame.info["structure_id"])
            pooled_ids.add(identity)
            flags[identity]["student_pool"] = True
    return (
        [frame for frame in frames if str(frame.info["structure_id"]) in pooled_ids],
        flags,
    )


def select_committee_student_safe_fill(
    frames: Sequence[Atoms],
    *,
    normal_selected: Sequence[Atoms],
    flags: dict[str, dict[str, bool]],
    keep: int,
    lo: float,
    enabled: bool,
    entropy: str,
) -> list[Atoms]:
    """Select a committee-student shortfall from the strict lower region."""
    lower = float(lo)
    if not math.isfinite(lower) or lower < 0 or keep <= 0:
        raise ValueError("committee-student safe fill requires finite lo and positive keep")
    normal_ids = {str(frame.info["structure_id"]) for frame in normal_selected}
    for frame in frames:
        identity = str(frame.info["structure_id"])
        value = float(frame.info["force_uncertainty"])
        flags[identity]["safe_fill_eligible"] = bool(enabled and value < lower)
        flags[identity]["safe_fill_attempted"] = False
    selected: set[str] = set()
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        owned = [frame for frame in frames if str(frame.info["system_id"]) == system_id]
        shortfall = max(
            0,
            keep - sum(str(frame.info["structure_id"]) in normal_ids for frame in owned),
        )
        candidates = [
            frame for frame in owned
            if flags[str(frame.info["structure_id"])]["safe_fill_eligible"]
            and str(frame.info["structure_id"]) not in normal_ids
        ]
        identities = sorted(str(frame.info["structure_id"]) for frame in candidates)
        chosen = select_representatives(
            identities,
            np.zeros((len(identities), 1), dtype=float),
            keep=min(shortfall, len(identities)),
            policy="random",
            entropy=f"{entropy}/safe-fill/{system_id}",
        )
        selected.update(chosen)
        for identity in chosen:
            flags[identity]["safe_fill_attempted"] = True
    return [frame for frame in frames if str(frame.info["structure_id"]) in selected]


def propose_committee_teacher_single_student_pool(
    frames: Sequence[Atoms],
    *,
    pool: int,
    floor: float,
    ceiling: float | None,
    entropy: str,
) -> tuple[list[Atoms], dict[str, dict[str, bool]]]:
    """Form the bounded single-student p-space proposal before teacher work."""
    lower = float(floor)
    upper = None if ceiling is None else float(ceiling)
    if not math.isfinite(lower) or not 0 <= lower < 1:
        raise ValueError("single-student p floor must be finite in [0, 1)")
    if upper is not None and (not math.isfinite(upper) or not lower < upper <= 1):
        raise ValueError("single-student p ceiling must be finite in (floor, 1]")
    if pool == -1 and upper is None:
        raise ValueError("committee-teacher screen.pool=-1 requires screen.student_p.ceiling")
    if pool == 0 or pool < -1:
        raise ValueError("committee-teacher screen.pool must be positive or conditional -1")

    values: dict[str, float] = {}
    flags: dict[str, dict[str, bool]] = {}
    for frame in frames:
        identity = str(frame.info["structure_id"])
        if identity in values:
            raise ValueError("single-student proposal identities must be unique")
        value = float(frame.info["student_p"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("student p values must be finite in [0, 1]")
        values[identity] = value
        candidate = value >= lower and (upper is None or value < upper)
        flags[identity] = {
            "student_floor_discard": value < lower,
            "student_candidate": candidate,
            "student_pool": False,
        }

    pool_ids: set[str] = set()
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [
            frame
            for frame in frames
            if str(frame.info["system_id"]) == system_id
            and flags[str(frame.info["structure_id"])]["student_candidate"]
        ]
        records = [
            (str(frame.info["structure_id"]), values[str(frame.info["structure_id"])])
            for frame in candidates
        ]
        chosen = (
            [identity for identity, _value in records]
            if pool == -1
            else p_space_sample(
                records,
                keep=min(pool, len(records)),
                entropy=f"{entropy}/{system_id}",
                lower_bound=lower,
                upper_bound=upper,
            )
        )
        pool_ids.update(chosen)
        for identity in chosen:
            flags[identity]["student_pool"] = True
    return (
        [frame for frame in frames if str(frame.info["structure_id"]) in pool_ids],
        flags,
    )


def select_committee_teacher_single_student_safe_fill(
    frames: Sequence[Atoms],
    *,
    normal_selected: Sequence[Atoms],
    flags: dict[str, dict[str, bool]],
    keep: int,
    ceiling: float | None,
    enabled: bool,
    entropy: str,
) -> list[Atoms]:
    """Choose GMM-covered frames for a committee-teacher post-gate shortfall."""
    normal_ids = {str(frame.info["structure_id"]) for frame in normal_selected}
    for frame in frames:
        identity = str(frame.info["structure_id"])
        probability = float(frame.info["student_p"])
        flags[identity]["safe_fill_eligible"] = bool(
            enabled and ceiling is not None and probability >= float(ceiling)
        )
        flags[identity]["safe_fill_attempted"] = False
    selected: set[str] = set()
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        owned = [frame for frame in frames if str(frame.info["system_id"]) == system_id]
        shortfall = max(
            0,
            keep - sum(str(frame.info["structure_id"]) in normal_ids for frame in owned),
        )
        candidates = [
            frame for frame in owned
            if flags[str(frame.info["structure_id"])]["safe_fill_eligible"]
            and str(frame.info["structure_id"]) not in normal_ids
        ]
        identities = sorted(str(frame.info["structure_id"]) for frame in candidates)
        chosen = select_representatives(
            identities,
            np.zeros((len(identities), 1), dtype=float),
            keep=min(shortfall, len(identities)),
            policy="random",
            entropy=f"{entropy}/safe-fill/{system_id}",
        )
        selected.update(chosen)
        for identity in chosen:
            flags[identity]["safe_fill_attempted"] = True
    return [frame for frame in frames if str(frame.info["structure_id"]) in selected]


__all__ = [
    "COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID",
    "SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_ID",
    "propose_single_teacher_committee_student_pool",
    "propose_committee_teacher_single_student_pool",
    "select_committee_teacher_single_student_safe_fill",
    "select_committee_student_safe_fill",
    "select_single_teacher_committee_student_train",
]
