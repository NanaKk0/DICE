from __future__ import annotations

import math
from itertools import product
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree

from workflow_policies.identities import PAIR_GUARD


POLICY_ID = PAIR_GUARD.name
POLICY_REVISION = PAIR_GUARD.revision
_MAX_PERIODIC_IMAGE_COPIES = 125
_MAX_PERIODIC_IMAGE_ATOMS = 200_000


def canonical_pair_key(raw_key: str) -> str:
    parts = [part.strip() for part in str(raw_key).replace("_", "-").split("-") if part.strip()]
    if len(parts) != 2 or any(part not in atomic_numbers for part in parts):
        raise ValueError(f"invalid pair-guard pair key: {raw_key!r}")
    return _pair_key(parts[0], parts[1])


def _pair_key(left: str, right: str) -> str:
    return "-".join(sorted((str(left), str(right))))


@dataclass(frozen=True)
class PairGuardThresholds:
    minimum: float
    pairs: Mapping[str, float]

    def __post_init__(self) -> None:
        minimum = float(self.minimum)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("pair-guard minimum must be finite and positive")
        normalized: dict[str, float] = {}
        for raw_key, raw_threshold in self.pairs.items():
            key = canonical_pair_key(raw_key)
            if key in normalized:
                raise ValueError(f"duplicate canonical pair-guard rule: {key}")
            threshold = float(raw_threshold)
            if not math.isfinite(threshold) or threshold < minimum:
                raise ValueError(f"pair-guard threshold for {key} must be finite and at least the minimum")
            normalized[key] = threshold
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "pairs", MappingProxyType(normalized))


@dataclass(frozen=True)
class PairGuardDecision:
    structure_id: str
    system_id: str
    passed: bool
    reject_rule: str
    reject_distance: float
    reject_pair: str
    reject_atom_indices: tuple[int, int] | None
    evaluation_path: str


@dataclass(frozen=True)
class _Violation:
    margin: float
    pair_key: str
    atom_i: int
    atom_j: int
    rule: str
    distance: float


def evaluate_pair_guard(
    observations: Sequence[Atoms] | Iterable[Atoms],
    thresholds: PairGuardThresholds,
    *,
    algorithm: str = "auto",
) -> tuple[PairGuardDecision, ...]:
    """Evaluate ordered canonical structures with the reject-only pair guard."""
    if algorithm not in {"auto", "ckdtree", "ase"}:
        raise ValueError(f"unknown pair-guard algorithm: {algorithm!r}")
    decisions: list[PairGuardDecision] = []
    identities: set[str] = set()
    for frame in observations:
        structure_id, system_id = _validated_identity(frame)
        if structure_id in identities:
            raise ValueError(f"duplicate pair-guard structure identity: {structure_id}")
        identities.add(structure_id)
        decisions.append(
            _evaluate_validated_frame(
                frame, thresholds, algorithm, structure_id, system_id
            )
        )
    return tuple(decisions)


def _evaluate_validated_frame(
    frame: Atoms,
    thresholds: PairGuardThresholds,
    algorithm: str,
    structure_id: str,
    system_id: str,
) -> PairGuardDecision:
    use_fast_path = _supports_ckdtree(frame)
    if algorithm == "ckdtree" and not use_fast_path:
        raise ValueError(
            "pair-guard cKDTree path requires a fully periodic orthorhombic cell"
        )
    if algorithm == "ckdtree" or (algorithm == "auto" and use_fast_path):
        path = "ckdtree"
        violation = _worst_ckdtree_violation(frame, thresholds)
    elif algorithm == "auto":
        cutoff = max([thresholds.minimum, *thresholds.pairs.values()])
        periodic_repeats = _periodic_image_repeats(frame, cutoff)
        if periodic_repeats is None:
            path = "ase"
            violation = _worst_fallback_violation(frame, thresholds)
        else:
            path = "periodic"
            violation = _worst_periodic_violation(
                frame, thresholds, periodic_repeats
            )
    else:
        path = "ase"
        violation = _worst_fallback_violation(frame, thresholds)
    return _decision(structure_id, system_id, violation, path)


def _validated_identity(frame: Atoms) -> tuple[str, str]:
    if not isinstance(frame, Atoms) or len(frame) == 0:
        raise ValueError("pair guard requires non-empty ASE structures")
    structure_id = frame.info.get("structure_id")
    system_id = frame.info.get("system_id")
    if not isinstance(structure_id, str) or not structure_id or structure_id != structure_id.strip():
        raise ValueError("pair-guard structure_id must be a non-empty string")
    if not isinstance(system_id, str) or not system_id or system_id != system_id.strip():
        raise ValueError("pair-guard system_id must be a non-empty string")
    positions = np.asarray(frame.positions)
    if positions.shape != (len(frame), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("pair-guard positions must be finite atom-by-three coordinates")
    return structure_id, system_id


def _worst_fallback_violation(frame: Atoms, thresholds: PairGuardThresholds) -> _Violation | None:
    cutoff = max([thresholds.minimum, *thresholds.pairs.values()])
    pair_i, pair_j, distances = neighbor_list("ijd", frame, cutoff=cutoff, self_interaction=False)
    violations: list[_Violation] = []
    symbols = frame.get_chemical_symbols()
    for raw_i, raw_j, raw_distance in zip(pair_i, pair_j, distances, strict=True):
        i, j = sorted((int(raw_i), int(raw_j)))
        if i == j or int(raw_i) > int(raw_j):
            continue
        pair_key = _pair_key(symbols[i], symbols[j])
        threshold = thresholds.pairs.get(pair_key, thresholds.minimum)
        distance = float(raw_distance)
        violation = _violation(
            i,
            j,
            pair_key,
            distance,
            threshold,
            pair_key if pair_key in thresholds.pairs else "min",
        )
        if violation is not None:
            violations.append(violation)
    return min(violations, key=lambda item: (item.margin, item.pair_key, item.atom_i, item.atom_j), default=None)


def _supports_ckdtree(frame: Atoms) -> bool:
    pbc = np.asarray(frame.pbc)
    cell = np.asarray(frame.cell.array, dtype=float)
    diagonal = np.diag(cell)
    off_diagonal = cell - np.diag(diagonal)
    return bool(
        pbc.shape == (3,)
        and np.all(pbc)
        and cell.shape == (3, 3)
        and np.all(np.isfinite(cell))
        and np.all(diagonal > 0.0)
        and np.max(np.abs(off_diagonal)) <= 1.0e-8
    )


def _periodic_image_repeats(frame: Atoms, cutoff: float) -> np.ndarray | None:
    pbc = np.asarray(frame.pbc)
    cell = np.asarray(frame.cell.array, dtype=float)
    if not (
        pbc.shape == (3,)
        and np.all(pbc)
        and cell.shape == (3, 3)
        and np.all(np.isfinite(cell))
    ):
        return None
    volume = abs(float(np.linalg.det(cell)))
    face_areas = np.asarray(
        (
            np.linalg.norm(np.cross(cell[1], cell[2])),
            np.linalg.norm(np.cross(cell[2], cell[0])),
            np.linalg.norm(np.cross(cell[0], cell[1])),
        ),
        dtype=float,
    )
    if (
        not math.isfinite(volume)
        or volume <= 1.0e-12
        or not np.all(np.isfinite(face_areas))
        or np.any(face_areas <= 1.0e-12)
    ):
        return None
    heights = volume / face_areas
    repeats = np.maximum(1, np.ceil(cutoff / heights).astype(int))
    image_copies = int(np.prod(2 * repeats + 1))
    if (
        image_copies > _MAX_PERIODIC_IMAGE_COPIES
        or image_copies * len(frame) > _MAX_PERIODIC_IMAGE_ATOMS
    ):
        return None
    return repeats


def _worst_periodic_violation(
    frame: Atoms,
    thresholds: PairGuardThresholds,
    repeats: np.ndarray,
) -> _Violation | None:
    """Evaluate a fully periodic general cell through bounded image search."""

    cell = np.asarray(frame.cell.array, dtype=float)
    positions = np.asarray(frame.get_scaled_positions(wrap=True), dtype=float) @ cell
    symbols = np.asarray(frame.get_chemical_symbols(), dtype=np.str_)
    cutoff = max([thresholds.minimum, *thresholds.pairs.values()])
    shifts = np.asarray(
        tuple(
            product(
                range(-int(repeats[0]), int(repeats[0]) + 1),
                range(-int(repeats[1]), int(repeats[1]) + 1),
                range(-int(repeats[2]), int(repeats[2]) + 1),
            )
        ),
        dtype=float,
    )
    images = (positions[None, :, :] + (shifts @ cell)[:, None, :]).reshape(-1, 3)
    tree = cKDTree(images)
    search_cutoff = float(np.nextafter(cutoff, math.inf))
    atom_count = len(frame)
    violations: list[_Violation] = []
    for raw_i, neighbors in enumerate(
        tree.query_ball_point(positions, search_cutoff)
    ):
        for image_index in neighbors:
            raw_j = int(image_index) % atom_count
            if raw_i >= raw_j:
                continue
            pair_key = _pair_key(symbols[raw_i], symbols[raw_j])
            threshold = thresholds.pairs.get(pair_key, thresholds.minimum)
            distance = float(
                np.linalg.norm(positions[raw_i] - images[int(image_index)])
            )
            violation = _violation(
                raw_i,
                raw_j,
                pair_key,
                distance,
                threshold,
                (
                    pair_key
                    if pair_key in thresholds.pairs
                    else "min"
                ),
            )
            if violation is not None:
                violations.append(violation)
    return min(
        violations,
        key=lambda item: (item.margin, item.pair_key, item.atom_i, item.atom_j),
        default=None,
    )


def _worst_ckdtree_violation(frame: Atoms, thresholds: PairGuardThresholds) -> _Violation | None:
    boxsize = np.diag(np.asarray(frame.cell.array, dtype=float))
    positions = np.mod(np.asarray(frame.positions, dtype=float), boxsize)
    symbols = np.asarray(frame.get_chemical_symbols(), dtype=np.str_)
    violations: list[_Violation] = []

    all_tree = cKDTree(positions, boxsize=boxsize)
    for i, j in all_tree.query_pairs(thresholds.minimum, output_type="set"):
        distance = _periodic_distance(positions[int(i)], positions[int(j)], boxsize)
        pair_key = _pair_key(symbols[int(i)], symbols[int(j)])
        if pair_key in thresholds.pairs:
            continue
        violation = _violation(int(i), int(j), pair_key, distance, thresholds.minimum, "min")
        if violation is not None:
            violations.append(violation)

    indices_by_symbol = {
        symbol: np.flatnonzero(symbols == symbol)
        for symbol in sorted(set(symbols.tolist()))
    }
    for pair_key, threshold in thresholds.pairs.items():
        left, right = pair_key.split("-")
        if left not in indices_by_symbol or right not in indices_by_symbol:
            continue
        left_indices = indices_by_symbol[left]
        right_indices = indices_by_symbol[right]
        left_tree = cKDTree(positions[left_indices], boxsize=boxsize)
        if left == right:
            for local_i, local_j in left_tree.query_pairs(threshold, output_type="set"):
                i = int(left_indices[int(local_i)])
                j = int(left_indices[int(local_j)])
                distance = _periodic_distance(positions[i], positions[j], boxsize)
                violation = _violation(i, j, pair_key, distance, threshold, pair_key)
                if violation is not None:
                    violations.append(violation)
            continue
        right_tree = cKDTree(positions[right_indices], boxsize=boxsize)
        nearby = left_tree.sparse_distance_matrix(right_tree, threshold, output_type="dict")
        for (local_i, local_j), raw_distance in nearby.items():
            i = int(left_indices[int(local_i)])
            j = int(right_indices[int(local_j)])
            violation = _violation(i, j, pair_key, float(raw_distance), threshold, pair_key)
            if violation is not None:
                violations.append(violation)

    return min(violations, key=lambda item: (item.margin, item.pair_key, item.atom_i, item.atom_j), default=None)


def _periodic_distance(left: np.ndarray, right: np.ndarray, boxsize: np.ndarray) -> float:
    delta = np.abs(left - right)
    return float(np.linalg.norm(np.minimum(delta, boxsize - delta)))


def _violation(
    raw_i: int,
    raw_j: int,
    pair_key: str,
    distance: float,
    threshold: float,
    rule: str,
) -> _Violation | None:
    if distance >= threshold:
        return None
    i, j = sorted((int(raw_i), int(raw_j)))
    return _Violation(distance - threshold, pair_key, i, j, rule, distance)


def _decision(
    structure_id: str,
    system_id: str,
    violation: _Violation | None,
    evaluation_path: str,
) -> PairGuardDecision:
    if violation is None:
        return PairGuardDecision(structure_id, system_id, True, "", math.nan, "", None, evaluation_path)
    return PairGuardDecision(
        structure_id,
        system_id,
        False,
        violation.rule,
        violation.distance,
        violation.pair_key,
        (violation.atom_i, violation.atom_j),
        evaluation_path,
    )


__all__ = [
    "POLICY_ID",
    "PairGuardDecision",
    "PairGuardThresholds",
    "canonical_pair_key",
    "evaluate_pair_guard",
]
