"""Pure deterministic membership selection for the terminal FP publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from ase import Atoms

from workflow_policies.selection import (
    mean_descriptor_frames,
    select_element_joint_representatives,
    select_element_representatives,
    select_joint_representatives_with_audit,
    select_representatives,
)


def select_final_fp_frames(
    frames: Sequence[Atoms],
    *,
    keep: int,
    policy: str,
    entropy: str,
    feature_vectors: Mapping[str, np.ndarray] | None = None,
    group_ids: Sequence[str] | None = None,
) -> list[Atoms]:
    """Select exactly ``min(keep, N)`` frames per resolved population."""

    result: list[Atoms] = []
    observed = {str(frame.info["system_id"]) for frame in frames}
    groups = (
        tuple(sorted(observed))
        if group_ids is None
        else tuple(str(group_id) for group_id in group_ids)
    )
    if (
        any(not group_id for group_id in groups)
        or len(set(groups)) != len(groups)
        or not observed.issubset(set(groups))
    ):
        raise ValueError("terminal FP group order is invalid")
    for system_id in groups:
        candidates = [
            frame
            for frame in frames
            if str(frame.info["system_id"]) == system_id
        ]
        candidates.sort(key=lambda frame: str(frame.info["structure_id"]))
        candidates.sort(
            key=lambda frame: (
                -_score(frame),
                str(frame.info["structure_id"]),
            )
        )
        identities = [str(frame.info["structure_id"]) for frame in candidates]
        if not identities:
            continue
        if feature_vectors is None:
            vectors = np.asarray([[_score(frame)] for frame in candidates], dtype=float)
        else:
            try:
                raw = [
                    np.asarray(feature_vectors[identity]) for identity in identities
                ]
            except KeyError as exc:
                raise ValueError(
                    "terminal FP representative evidence is missing for "
                    f"{exc.args[0]!r}"
                ) from exc
            vectors = (
                np.stack(raw).astype(float, copy=False)
                if raw and all(value.ndim == 1 for value in raw)
                else mean_descriptor_frames(raw)
            )
        chosen = set(
            select_representatives(
                identities,
                vectors,
                keep=min(keep, len(candidates)),
                policy=policy,
                entropy=f"{entropy}/{system_id}",
            )
        )
        result.extend(
            frame
            for frame in candidates
            if str(frame.info["structure_id"]) in chosen
        )
    return result


def select_final_fp_joint_frames(
    frames: Sequence[Atoms],
    *,
    keep: int,
    candidate_vectors: Mapping[str, np.ndarray],
    background_vectors: Mapping[str, np.ndarray],
    group_ids: Sequence[str] | None = None,
) -> list[Atoms]:
    """Apply joint DIRECT once to every non-empty ordered FP group.

    ``keep >= N`` still evaluates the background-aware policy, then returns all
    candidates. Background representatives influence membership but never
    consume candidate capacity.
    """
    selected, _outcomes = select_final_fp_joint_frames_with_outcomes(
        frames,
        keep=keep,
        candidate_vectors=candidate_vectors,
        background_vectors=background_vectors,
        group_ids=group_ids,
    )
    return selected


def select_final_fp_joint_frames_with_outcomes(
    frames: Sequence[Atoms],
    *,
    keep: int,
    candidate_vectors: Mapping[str, np.ndarray],
    background_vectors: Mapping[str, np.ndarray],
    group_ids: Sequence[str] | None = None,
) -> tuple[list[Atoms], dict[str, dict[str, int | bool | str]]]:
    """Select terminal joint-DIRECT candidates and return per-group audit facts."""
    if isinstance(keep, bool) or not isinstance(keep, int) or keep <= 0:
        raise ValueError("terminal joint DIRECT keep must be positive")
    observed = {str(frame.info["system_id"]) for frame in frames}
    groups = (
        tuple(sorted(observed))
        if group_ids is None
        else tuple(str(group_id) for group_id in group_ids)
    )
    if (
        any(not group_id for group_id in groups)
        or len(set(groups)) != len(groups)
        or not observed.issubset(set(groups))
    ):
        raise ValueError("terminal FP group order is invalid")
    background_ids = tuple(str(identity) for identity in background_vectors)
    result: list[Atoms] = []
    outcomes: dict[str, dict[str, int | bool | str]] = {}
    for group_id in groups:
        candidates = [
            frame for frame in frames if str(frame.info["system_id"]) == group_id
        ]
        candidates.sort(key=lambda frame: str(frame.info["structure_id"]))
        candidates.sort(key=lambda frame: (-_score(frame), str(frame.info["structure_id"])))
        identities = tuple(str(frame.info["structure_id"]) for frame in candidates)
        if not identities:
            outcomes[group_id] = {
                "candidate_count": 0,
                "background_count": 0,
                "bypass": False,
                "realized_clusters": 0,
                "selected_background_count": 0,
                "selected_candidate_count": 0,
                "quota_completion_count": 0,
                "underfill": 0,
                "underfill_reason": "empty-candidates",
            }
            continue
        try:
            candidate_rows = np.stack(
                [np.asarray(candidate_vectors[identity], dtype=float) for identity in identities]
            )
            background_rows = np.stack(
                [np.asarray(background_vectors[identity], dtype=float) for identity in background_ids]
            )
        except KeyError as exc:
            raise ValueError(
                f"terminal joint DIRECT evidence is missing for {exc.args[0]!r}"
            ) from exc
        selected_identities, audit = select_joint_representatives_with_audit(
            identities,
            candidate_rows,
            background_ids,
            background_rows,
            keep=keep,
        )
        chosen = set(selected_identities)
        result.extend(
            frame
            for frame in candidates
            if str(frame.info["structure_id"]) in chosen
        )
        outcomes[group_id] = {
            "candidate_count": len(candidates),
            "background_count": len(background_ids),
            "bypass": False,
            **audit,
            "underfill_reason": (
                "insufficient-candidates"
                if audit["underfill"]
                else "none"
            ),
        }
    return result, outcomes


def select_final_fp_element_frames_with_outcomes(
    frames: Sequence[Atoms],
    *,
    keep: int,
    candidate_element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    type_map: Sequence[str],
    group_ids: Sequence[str] | None = None,
) -> tuple[list[Atoms], dict[str, dict[str, object]]]:
    """Apply candidate-only element DIRECT once per ordered terminal group."""

    groups = _terminal_groups(frames, group_ids)
    result: list[Atoms] = []
    outcomes: dict[str, dict[str, object]] = {}
    for group_id in groups:
        candidates = _ordered_group_frames(frames, group_id)
        if not candidates:
            outcomes[group_id] = _empty_element_outcome(keep)
            continue
        identities, rows, presence = _element_rows(
            candidates,
            element_vectors=candidate_element_vectors,
            type_map=type_map,
        )
        selection = select_element_representatives(
            identities,
            rows,
            presence=presence,
            type_map=type_map,
            keep=keep,
        )
        chosen = set(selection.selected_identities)
        result.extend(
            frame
            for frame in candidates
            if str(frame.info["structure_id"]) in chosen
        )
        outcomes[group_id] = {
            **dict(selection.audit),
            "bypass": False,
            "underfill_reason": (
                "insufficient-candidates"
                if int(selection.audit["underfill"])
                else "none"
            ),
        }
    return result, outcomes


def select_final_fp_element_joint_frames_with_outcomes(
    frames: Sequence[Atoms],
    *,
    keep: int,
    candidate_element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    background_frames: Sequence[Atoms],
    background_element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    type_map: Sequence[str],
    group_ids: Sequence[str] | None = None,
) -> tuple[list[Atoms], dict[str, dict[str, object]]]:
    """Apply element joint DIRECT with exact non-selectable teacher training."""

    groups = _terminal_groups(frames, group_ids)
    background = tuple(background_frames)
    if not background:
        raise ValueError("terminal element joint DIRECT background must be non-empty")
    background_ids, background_rows, background_presence = _element_rows(
        background,
        element_vectors=background_element_vectors,
        type_map=type_map,
    )
    result: list[Atoms] = []
    outcomes: dict[str, dict[str, object]] = {}
    for group_id in groups:
        candidates = _ordered_group_frames(frames, group_id)
        if not candidates:
            outcomes[group_id] = _empty_element_outcome(keep)
            continue
        identities, rows, presence = _element_rows(
            candidates,
            element_vectors=candidate_element_vectors,
            type_map=type_map,
        )
        selection = select_element_joint_representatives(
            identities,
            rows,
            candidate_presence=presence,
            background_identities=background_ids,
            background_element_rows=background_rows,
            background_presence=background_presence,
            type_map=type_map,
            keep=keep,
        )
        chosen = set(selection.selected_identities)
        result.extend(
            frame
            for frame in candidates
            if str(frame.info["structure_id"]) in chosen
        )
        outcomes[group_id] = {
            **dict(selection.audit),
            "bypass": False,
            "underfill_reason": (
                "insufficient-candidates"
                if int(selection.audit["underfill"])
                else "none"
            ),
        }
    return result, outcomes


def _terminal_groups(
    frames: Sequence[Atoms], group_ids: Sequence[str] | None
) -> tuple[str, ...]:
    observed = {str(frame.info["system_id"]) for frame in frames}
    groups = (
        tuple(sorted(observed))
        if group_ids is None
        else tuple(str(group_id) for group_id in group_ids)
    )
    if (
        any(not group_id for group_id in groups)
        or len(set(groups)) != len(groups)
        or not observed.issubset(set(groups))
    ):
        raise ValueError("terminal FP group order is invalid")
    return groups


def _ordered_group_frames(
    frames: Sequence[Atoms], group_id: str
) -> list[Atoms]:
    candidates = [
        frame for frame in frames if str(frame.info["system_id"]) == group_id
    ]
    candidates.sort(key=lambda frame: str(frame.info["structure_id"]))
    candidates.sort(
        key=lambda frame: (-_score(frame), str(frame.info["structure_id"]))
    )
    return candidates


def _element_rows(
    frames: Sequence[Atoms],
    *,
    element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    type_map: Sequence[str],
) -> tuple[
    tuple[str, ...],
    tuple[dict[str, np.ndarray], ...],
    tuple[tuple[str, ...], ...],
]:
    configured = tuple(str(item) for item in type_map)
    allowed = set(configured)
    if not set(element_vectors).issubset(allowed):
        raise ValueError("terminal element DIRECT evidence contains unknown element")
    identities: list[str] = []
    rows: list[dict[str, np.ndarray]] = []
    presence: list[tuple[str, ...]] = []
    for frame in frames:
        identity = str(frame.info["structure_id"])
        chemical = set(frame.get_chemical_symbols())
        if not chemical or not chemical.issubset(allowed):
            raise ValueError("terminal element DIRECT chemistry is outside type_map")
        identities.append(identity)
        presence.append(tuple(item for item in configured if item in chemical))
        frame_rows: dict[str, np.ndarray] = {}
        for element in configured:
            has_evidence = identity in element_vectors.get(element, {})
            if (element in chemical) != has_evidence:
                raise ValueError(
                    "terminal element DIRECT evidence must match chemical presence"
                )
            if has_evidence:
                vector = np.asarray(element_vectors[element][identity], dtype=float)
                if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                    raise ValueError("terminal element DIRECT vector is malformed")
                frame_rows[element] = vector[None, :]
        rows.append(frame_rows)
    return tuple(identities), tuple(rows), tuple(presence)


def _empty_element_outcome(keep: int) -> dict[str, object]:
    return {
        "candidate_count": 0,
        "background_count": 0,
        "bypass": False,
        "realized_clusters": 0,
        "selected_background_count": 0,
        "selected_candidate_count": 0,
        "quota_completion_count": 0,
        "underfill": int(keep),
        "underfill_reason": "empty-candidates",
        "active_elements": (),
        "inactive_elements": (),
        "element_dimensions": {},
        "presence_counts": {},
    }


def _score(frame: Atoms) -> float:
    value = float(
        frame.info.get(
            "acquisition",
            frame.info.get(
                "force_uncertainty",
                frame.info.get("teacher_p", frame.info.get("p", 0.0)),
            ),
        )
    )
    if not np.isfinite(value):
        raise ValueError("terminal FP acquisition score must be finite")
    return value


__all__ = [
    "select_final_fp_element_frames_with_outcomes",
    "select_final_fp_element_joint_frames_with_outcomes",
    "select_final_fp_frames",
    "select_final_fp_joint_frames",
    "select_final_fp_joint_frames_with_outcomes",
]
