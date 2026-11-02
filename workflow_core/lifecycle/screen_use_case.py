from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

from ase import Atoms
import numpy as np

from workflow_core.lifecycle.routes import RouteKind, RouteSpec
from workflow_core.structures.exact_identity import exact_structure_hash
from workflow_policies.mixed_routes import (
    propose_committee_teacher_single_student_pool,
    propose_single_teacher_committee_student_pool,
    select_committee_student_safe_fill,
    select_committee_teacher_single_student_safe_fill,
    select_single_teacher_committee_student_train,
)
from workflow_policies.pair_guard import (
    PairGuardDecision,
    PairGuardThresholds,
    evaluate_pair_guard,
)
from workflow_policies.selection import (
    mean_descriptor_frames,
    p_space_sample,
    select_representatives,
)


_ROUTES = frozenset(RouteKind)


@dataclass(frozen=True, slots=True)
class ScreenRoundUseCase:
    """One pure decision owner shared by every closed screening route.

    The imperative executor supplies ordered in-memory structures and provider
    results.  This use case owns identity, guard, routing, quota, selector and
    safe-fill decisions, but no path, codec, provider, progress or publication.
    """

    route: RouteSpec
    ordered_map: Callable | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route, RouteSpec) or self.route.kind not in _ROUTES:
            raise TypeError("screen use case requires one closed RouteSpec")

    def deduplicate(self, frames: Sequence[Atoms]) -> list[Atoms]:
        return _deduplicate(frames)

    def pair_guard(
        self,
        frames: Sequence[Atoms],
        thresholds: PairGuardThresholds,
    ) -> tuple[list[Atoms], tuple[PairGuardDecision, ...]]:
        ordered = _deduplicate(frames)
        decisions = evaluate_pair_guard(ordered, thresholds)
        return (
            [
                frame
                for frame, decision in zip(ordered, decisions, strict=True)
                if decision.passed
            ],
            decisions,
        )

    def reserve_teacher_reference(self, *args, **kwargs):
        return reserve_teacher_gmm_fp(*args, **kwargs)

    def select_student_reference(self, *args, **kwargs):
        return select_student_gmm_train(*args, **kwargs)

    def select_student_joint_proposal(
        self, frames: Sequence[Atoms], *, pool: int,
        candidate_vectors: Mapping[str, np.ndarray],
        background_vectors: Mapping[str, np.ndarray],
    ) -> tuple[list[Atoms], dict[str, dict[str, int | bool]]]:
        from workflow_policies.screen_tasks import joint_selection_task
        if isinstance(pool, bool) or not isinstance(pool, int) or pool <= 0:
            raise ValueError("joint DIRECT screen.pool must be a positive integer")
        ordered = _deduplicate(frames)
        grouped: dict[str, list[Atoms]] = {}
        for frame in ordered:
            grouped.setdefault(str(frame.info["system_id"]), []).append(frame)
        background_ids = tuple(str(identity) for identity in background_vectors)
        jobs = {}
        background = None
        for group_id, candidates in grouped.items():
            if pool >= len(candidates):
                continue
            candidate_ids = tuple(str(frame.info["structure_id"]) for frame in candidates)
            try:
                vectors = np.stack([np.asarray(candidate_vectors[identity], dtype=float)
                                    for identity in candidate_ids])
                if background is None:
                    background = np.stack([np.asarray(background_vectors[identity], dtype=float)
                                           for identity in background_ids])
            except KeyError as exc:
                raise ValueError(f"joint DIRECT screen evidence is missing for {exc.args[0]!r}") from exc
            jobs[group_id] = (candidate_ids, vectors, background_ids, background, pool)
        results = dict(zip(jobs, (self.ordered_map or map)(joint_selection_task, jobs.values()), strict=True))
        selected_ids: set[str] = set()
        outcomes = {}
        for group_id, candidates in grouped.items():
            bypass = group_id not in results
            if bypass:
                chosen = tuple(str(frame.info["structure_id"]) for frame in candidates)
                audit = dict(realized_clusters=len(chosen), selected_background_count=0,
                             selected_candidate_count=len(chosen), quota_completion_count=0, underfill=0)
            else:
                chosen, audit = results[group_id]
            selected_ids.update(chosen)
            outcomes[group_id] = {
                "candidate_count": len(candidates),
                "background_count": 0 if bypass else len(background_ids),
                "bypass": bypass,
                "selected_candidate_count": len(chosen),
                "selected_background_count": audit["selected_background_count"],
                "realized_clusters": audit["realized_clusters"],
                "quota_completion_count": audit["quota_completion_count"],
                "underfill": audit["underfill"],
            }
        return ([frame for frame in ordered if str(frame.info["structure_id"]) in selected_ids], outcomes)

    def select_student_element_joint_proposal(
        self, frames: Sequence[Atoms], *, background_frames: Sequence[Atoms], pool: int,
        candidate_element_vectors: Mapping[str, Mapping[str, np.ndarray]],
        background_element_vectors: Mapping[str, Mapping[str, np.ndarray]],
        type_map: Sequence[str],
    ) -> tuple[list[Atoms], dict[str, dict[str, object]]]:
        from workflow_policies.screen_tasks import element_joint_selection_task
        if isinstance(pool, bool) or not isinstance(pool, int) or pool <= 0:
            raise ValueError("element joint DIRECT screen.pool must be a positive integer")
        ordered = _deduplicate(frames)
        background = _deduplicate(background_frames)
        if not background:
            raise ValueError("element joint DIRECT student training background is empty")
        background_ids, background_rows, background_presence = _element_policy_rows(
            background, element_vectors=background_element_vectors, type_map=type_map,
        )
        grouped: dict[str, list[Atoms]] = {}
        for frame in ordered:
            grouped.setdefault(str(frame.info["system_id"]), []).append(frame)
        jobs = {}
        for group_id, candidates in grouped.items():
            if pool >= len(candidates):
                continue
            candidate_ids, rows, presence = _element_policy_rows(
                candidates, element_vectors=candidate_element_vectors, type_map=type_map,
            )
            if candidate_ids != tuple(str(frame.info["structure_id"]) for frame in candidates):
                raise ValueError("element joint DIRECT changed canonical candidate order")
            jobs[group_id] = (candidate_ids, rows, presence, background_ids, background_rows,
                              background_presence, tuple(type_map), pool)
        results = dict(zip(jobs, (self.ordered_map or map)(element_joint_selection_task, jobs.values()), strict=True))
        selected_ids: set[str] = set()
        outcomes = {}
        for group_id, candidates in grouped.items():
            if group_id not in results:
                chosen = tuple(str(frame.info["structure_id"]) for frame in candidates)
                audit = dict(candidate_count=len(candidates), background_count=0, bypass=True,
                             selected_candidate_count=len(chosen), selected_background_count=0,
                             realized_clusters=len(chosen), quota_completion_count=0, underfill=0,
                             active_elements=(), inactive_elements=tuple(str(item) for item in type_map),
                             element_dimensions={}, presence_counts={})
            else:
                chosen, audit = results[group_id]
                audit["bypass"] = False
            selected_ids.update(chosen)
            outcomes[group_id] = audit
        return ([frame for frame in ordered if str(frame.info["structure_id"]) in selected_ids], outcomes)

    def fill_student_reference(self, *args, **kwargs):
        return fill_student_gmm_underfill(*args, **kwargs)

    def select_per_system(self, *args, **kwargs):
        return per_system_select(*args, **kwargs)

    def select_element_per_system(
        self,
        frames: Sequence[Atoms],
        *,
        keep: int,
        element_vectors: Mapping[str, Mapping[str, np.ndarray]],
        type_map: Sequence[str],
    ) -> tuple[list[Atoms], dict[str, dict[str, object]]]:
        if isinstance(keep, bool) or not isinstance(keep, int) or keep <= 0:
            raise ValueError("element DIRECT screen.keep must be a positive integer")
        ordered = _deduplicate(frames)
        grouped: dict[str, list[Atoms]] = {}
        for frame in ordered:
            grouped.setdefault(str(frame.info["system_id"]), []).append(frame)
        selected_ids: set[str] = set()
        outcomes: dict[str, dict[str, object]] = {}
        from workflow_policies.screen_tasks import element_selection_task
        jobs = {}
        for group_id, candidates in grouped.items():
            identities, rows, presence = _element_policy_rows(
                candidates, element_vectors=element_vectors, type_map=type_map,
            )
            jobs[group_id] = (identities, rows, presence, tuple(type_map), keep)
        for group_id, (chosen, audit) in zip(
            jobs, (self.ordered_map or map)(element_selection_task, jobs.values()), strict=True,
        ):
            selected_ids.update(chosen)
            outcomes[group_id] = audit
        return (
            [
                frame
                for frame in ordered
                if str(frame.info["structure_id"]) in selected_ids
            ],
            outcomes,
        )

    def propose_committee_teacher_single_student(self, *args, **kwargs):
        return propose_committee_teacher_single_student_pool(*args, **kwargs)

    def propose_single_teacher_committee_student(self, *args, **kwargs):
        return propose_single_teacher_committee_student_pool(*args, **kwargs)

    def select_single_teacher_committee_student(self, *args, **kwargs):
        return select_single_teacher_committee_student_train(*args, **kwargs)

    def fill_committee_student(self, *args, **kwargs):
        return select_committee_student_safe_fill(*args, **kwargs)

    def fill_committee_teacher_single_student(self, *args, **kwargs):
        return select_committee_teacher_single_student_safe_fill(*args, **kwargs)

    def route_teacher_force(
        self,
        frames: Sequence[Atoms],
        *,
        lower: float,
        upper: float,
        missing_default: float | None = None,
    ) -> tuple[list[Atoms], list[Atoms]]:
        """Apply the accepted strict-low/lower-inclusive-middle routing."""

        lo = float(lower)
        hi = float(upper)
        if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
            raise ValueError("teacher force-deviation band must be finite and ordered")
        train: list[Atoms] = []
        fp: list[Atoms] = []
        for frame in _deduplicate(frames):
            value = float(
                frame.info["force_uncertainty"]
                if missing_default is None
                else frame.info.get("force_uncertainty", missing_default)
            )
            if not math.isfinite(value):
                raise ValueError("teacher force uncertainty must be finite")
            if value < lo:
                train.append(frame)
            elif value < hi:
                fp.append(frame)
        return train, fp

    def select_student_force_band(
        self,
        frames: Sequence[Atoms],
        *,
        lower: float,
        upper: float,
        pool: int,
        keep: int,
        policy: str,
        entropy: str,
    ) -> tuple[list[Atoms], list[Atoms]]:
        lo = float(lower)
        hi = float(upper)
        if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
            raise ValueError("student force-deviation band must be finite and ordered")
        candidates = [
            frame
            for frame in _deduplicate(frames)
            if lo <= float(frame.info["force_uncertainty"]) < hi
        ]
        return candidates, per_system_select(
            candidates,
            pool=pool,
            keep=keep,
            policy=policy,
            entropy=entropy,
            feature_vectors=None,
        )

    def restore_order(
        self,
        reference: Sequence[Atoms],
        selected: Sequence[Atoms],
    ) -> list[Atoms]:
        ordered = _deduplicate(reference)
        by_id = {
            str(frame.info["structure_id"]): frame
            for frame in _deduplicate(selected)
        }
        return [
            by_id[str(frame.info["structure_id"])]
            for frame in ordered
            if str(frame.info["structure_id"]) in by_id
        ]


def reserve_teacher_gmm_fp(
    observations: Sequence[Atoms],
    *,
    fp_pool: int,
    teacher_floor: float,
    teacher_trust: float,
    teacher_ceiling: float | None,
    entropy: str,
) -> tuple[list[Atoms], list[Atoms], dict[str, dict[str, bool]]]:
    if fp_pool == -1 and teacher_ceiling is None:
        raise ValueError("single-model GMM screen.fp_pool=-1 requires a teacher ceiling")
    if fp_pool == 0 or fp_pool < -1:
        raise ValueError("single-model GMM screen.fp_pool must be positive or conditional -1")
    floor, ceiling = _validated_p_candidate_band(
        floor=teacher_floor, ceiling=teacher_ceiling, role="teacher"
    )
    trust = float(teacher_trust)
    if not math.isfinite(trust) or not 0 <= trust <= 1:
        raise ValueError("single-model GMM teacher trust must be finite in [0, 1]")
    frames = _deduplicate(observations)
    flags = {
        str(frame.info["structure_id"]): {
            "teacher_floor_discard": float(frame.info["teacher_p"]) < floor,
            "teacher_candidate": (
                float(frame.info["teacher_p"]) >= floor
                and (ceiling is None or float(frame.info["teacher_p"]) < ceiling)
            ),
            "teacher_fp_candidate": False,
            "teacher_covered": (
                float(frame.info["teacher_p"]) >= floor
                and float(frame.info["teacher_p"]) > trust
            ),
            "student_eligible": False,
            "teacher_trust_reject": False,
        }
        for frame in frames
    }
    candidate_ids: set[str] = set()
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [
            frame
            for frame in frames
            if str(frame.info["system_id"]) == system_id
            and float(frame.info["teacher_p"]) >= floor
            and (ceiling is None or float(frame.info["teacher_p"]) < ceiling)
        ]
        chosen = (
            [str(frame.info["structure_id"]) for frame in candidates]
            if fp_pool == -1
            else p_space_sample(
                [
                    (str(frame.info["structure_id"]), float(frame.info["teacher_p"]))
                    for frame in candidates
                ],
                keep=min(fp_pool, len(candidates)),
                entropy=f"{entropy}/teacher-p/{system_id}",
                lower_bound=floor,
                upper_bound=ceiling,
            )
        )
        candidate_ids.update(chosen)
    fp_candidates: list[Atoms] = []
    student_eligible: list[Atoms] = []
    for frame in frames:
        identity = str(frame.info["structure_id"])
        probability = float(frame.info["teacher_p"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("teacher p values must be finite in [0, 1]")
        if flags[identity]["teacher_floor_discard"]:
            continue
        if identity in candidate_ids:
            flags[identity]["teacher_fp_candidate"] = True
            fp_candidates.append(frame)
        if probability > trust:
            flags[identity]["student_eligible"] = True
            student_eligible.append(frame)
        else:
            flags[identity]["teacher_trust_reject"] = True
    return fp_candidates, student_eligible, flags


def select_student_gmm_train(
    frames: Sequence[Atoms],
    *,
    pool: int,
    keep: int,
    policy: str,
    entropy: str,
    feature_vectors: Mapping[str, np.ndarray] | None,
    student_floor: float,
    student_ceiling: float | None,
) -> tuple[list[Atoms], dict[str, dict[str, bool]]]:
    if pool == -1 and student_ceiling is None:
        raise ValueError("single-model GMM screen.pool=-1 requires a student ceiling")
    if pool == 0 or pool < -1:
        raise ValueError("single-model GMM screen.pool must be positive or conditional -1")
    floor, ceiling = _validated_p_candidate_band(
        floor=student_floor, ceiling=student_ceiling, role="student"
    )
    probabilities = {
        str(frame.info["structure_id"]): float(frame.info["student_p"])
        for frame in frames
    }
    if len(probabilities) != len(frames):
        raise ValueError("student p-band identities must be unique")
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in probabilities.values()
    ):
        raise ValueError("student p values must be finite in [0, 1]")
    selected: list[Atoms] = []
    flags = {
        str(frame.info["structure_id"]): {
            "student_floor_discard": probabilities[str(frame.info["structure_id"])] < floor,
            "student_candidate": (
                probabilities[str(frame.info["structure_id"])] >= floor
                and (
                    ceiling is None
                    or probabilities[str(frame.info["structure_id"])] < ceiling
                )
            ),
            "student_covered": (
                ceiling is not None
                and probabilities[str(frame.info["structure_id"])] >= ceiling
            ),
            "student_pool": False,
            "train_selected": False,
        }
        for frame in frames
    }
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [
            frame
            for frame in frames
            if str(frame.info["system_id"]) == system_id
            and float(frame.info["student_p"]) >= floor
            and (ceiling is None or float(frame.info["student_p"]) < ceiling)
        ]
        pool_ids = (
            [str(frame.info["structure_id"]) for frame in candidates]
            if pool == -1
            else p_space_sample(
                [
                    (str(frame.info["structure_id"]), float(frame.info["student_p"]))
                    for frame in candidates
                ],
                keep=min(pool, len(candidates)),
                entropy=f"{entropy}/student-p/{system_id}",
                lower_bound=floor,
                upper_bound=ceiling,
            )
        )
        by_id = {str(frame.info["structure_id"]): frame for frame in candidates}
        pooled = [by_id[identity] for identity in sorted(pool_ids)]
        for identity in pool_ids:
            flags[identity]["student_pool"] = True
        identities = [str(frame.info["structure_id"]) for frame in pooled]
        if policy == "random":
            vectors = np.zeros((len(pooled), 1), dtype=float)
        else:
            if feature_vectors is None:
                raise ValueError(
                    f"{policy} student GMM selection requires readout-input frame means"
                )
            try:
                vectors = np.stack(
                    [np.asarray(feature_vectors[identity], dtype=float) for identity in identities]
                )
            except KeyError as exc:
                raise ValueError(
                    "student GMM representative evidence is missing for "
                    f"{exc.args[0]!r}"
                ) from exc
            if vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise ValueError(
                    "student GMM representative frame means must be finite frame x feature rows"
                )
        chosen = set(
            select_representatives(
                identities,
                vectors,
                keep=min(keep, len(pooled)),
                policy=policy,
                entropy=f"{entropy}/selector/{system_id}",
            )
        )
        for frame in pooled:
            identity = str(frame.info["structure_id"])
            if identity in chosen:
                flags[identity]["train_selected"] = True
                selected.append(frame)
    return selected, flags


def fill_student_gmm_underfill(
    frames: Sequence[Atoms],
    *,
    normal_selected: Sequence[Atoms],
    flags: dict[str, dict[str, bool]],
    keep: int,
    enabled: bool,
    entropy: str,
) -> list[Atoms]:
    selected_ids = {str(frame.info["structure_id"]) for frame in normal_selected}
    for identity, values in flags.items():
        values["safe_fill_eligible"] = bool(enabled and values["student_covered"])
        values["safe_fill_selected"] = False
    fill: list[Atoms] = []
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        system_frames = [
            frame for frame in frames if str(frame.info["system_id"]) == system_id
        ]
        shortfall = max(
            0,
            keep
            - sum(
                str(frame.info["structure_id"]) in selected_ids
                for frame in system_frames
            ),
        )
        candidates = [
            frame
            for frame in system_frames
            if flags[str(frame.info["structure_id"])]["safe_fill_eligible"]
            and str(frame.info["structure_id"]) not in selected_ids
        ]
        identities = sorted(str(frame.info["structure_id"]) for frame in candidates)
        chosen = set(
            select_representatives(
                identities,
                np.zeros((len(identities), 1), dtype=float),
                keep=min(shortfall, len(identities)),
                policy="random",
                entropy=f"{entropy}/safe-fill/{system_id}",
            )
        )
        for frame in candidates:
            identity = str(frame.info["structure_id"])
            if identity in chosen:
                flags[identity]["safe_fill_selected"] = True
                fill.append(frame)
    return fill


def per_system_select(
    frames: Sequence[Atoms],
    *,
    pool: int,
    keep: int,
    policy: str,
    entropy: str,
    feature_vectors: Mapping[str, np.ndarray] | None = None,
) -> list[Atoms]:
    result: list[Atoms] = []
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [
            frame for frame in frames if str(frame.info["system_id"]) == system_id
        ]
        candidates.sort(key=lambda frame: str(frame.info["structure_id"]))
        candidates.sort(key=lambda frame: (-_score(frame), str(frame.info["structure_id"])))
        pooled = candidates if pool < 0 else candidates[:pool]
        identities = [str(frame.info["structure_id"]) for frame in pooled]
        if feature_vectors is None:
            vectors = np.asarray([[_score(frame)] for frame in pooled], dtype=float)
        else:
            try:
                raw_vectors = [
                    np.asarray(feature_vectors[str(frame.info["structure_id"])])
                    for frame in pooled
                ]
            except KeyError as exc:
                raise ValueError(
                    "representative descriptor evidence is missing for structure "
                    f"{exc.args[0]!r}"
                ) from exc
            vectors = (
                np.stack(raw_vectors).astype(float, copy=False)
                if raw_vectors and all(vector.ndim == 1 for vector in raw_vectors)
                else mean_descriptor_frames(raw_vectors)
            )
        chosen = set(
            select_representatives(
                identities,
                vectors,
                keep=min(keep, len(pooled)),
                policy=policy,
                entropy=f"{entropy}/{system_id}",
            )
        )
        result.extend(
            frame
            for frame in pooled
            if str(frame.info["structure_id"]) in chosen
        )
    return result


def _element_policy_rows(
    frames: Sequence[Atoms],
    *,
    element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    type_map: Sequence[str],
) -> tuple[
    tuple[str, ...],
    tuple[dict[str, np.ndarray], ...],
    tuple[tuple[str, ...], ...],
]:
    elements = tuple(str(item) for item in type_map)
    if not elements or len(elements) != len(set(elements)):
        raise ValueError("element DIRECT type_map must be ordered and unique")
    if not set(element_vectors).issubset(elements):
        raise ValueError("element DIRECT evidence contains an unknown element")
    identities: list[str] = []
    seen_identities: set[str] = set()
    rows: list[dict[str, np.ndarray]] = []
    presence: list[tuple[str, ...]] = []
    for frame in frames:
        identity = str(frame.info.get("structure_id", ""))
        if not identity or identity in seen_identities:
            raise ValueError(
                "element DIRECT frames require ordered unique canonical identities"
            )
        seen_identities.add(identity)
        chemical = set(frame.get_chemical_symbols())
        if not chemical or not chemical.issubset(elements):
            raise ValueError(
                "element DIRECT frame composition must use the configured type_map: "
                f"structure {identity!r} has {sorted(chemical)!r}, configured "
                f"{list(elements)!r}"
            )
        frame_presence = tuple(element for element in elements if element in chemical)
        frame_rows: dict[str, np.ndarray] = {}
        for element in frame_presence:
            try:
                vector = np.asarray(element_vectors[element][identity], dtype=float)
            except KeyError as exc:
                raise ValueError(
                    "element DIRECT evidence is missing for chemically present "
                    f"{element!r} in {identity!r}"
                ) from exc
            if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError(
                    "element DIRECT adapter evidence must be finite frame means"
                )
            frame_rows[element] = vector[None, :]
        for element, values in element_vectors.items():
            if identity in values and element not in chemical:
                raise ValueError(
                    "element DIRECT evidence is present for a chemically absent element"
                )
        identities.append(identity)
        rows.append(frame_rows)
        presence.append(frame_presence)
    return tuple(identities), tuple(rows), tuple(presence)


def _validated_p_candidate_band(
    *, floor: float, ceiling: float | None, role: str
) -> tuple[float, float | None]:
    lower = float(floor)
    if not math.isfinite(lower) or not 0 <= lower < 1:
        raise ValueError(f"single-model GMM {role} floor must be finite in [0, 1)")
    if ceiling is None:
        return lower, None
    upper = float(ceiling)
    if not math.isfinite(upper) or not lower < upper <= 1:
        raise ValueError(
            f"single-model GMM {role} ceiling must be finite in (floor, 1]"
        )
    return lower, upper


def _score(frame: Atoms) -> float:
    return float(
        frame.info.get(
            "acquisition",
            frame.info.get(
                "force_uncertainty",
                frame.info.get("teacher_p", frame.info.get("p", 0.0)),
            ),
        )
    )


def _deduplicate(frames: Sequence[Atoms]) -> list[Atoms]:
    selected: dict[str, Atoms] = {}
    for frame in frames:
        identity = str(frame.info["structure_id"])
        previous = selected.get(identity)
        if previous is None:
            selected[identity] = frame
        elif exact_structure_hash(previous) != exact_structure_hash(frame):
            raise ValueError(
                "conflicting geometry for duplicate structure_id before screening"
            )
    return list(selected.values())


__all__ = ["ScreenRoundUseCase"]
