from __future__ import annotations

import hashlib
import math
import random
import warnings
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from workflow_policies.identities import (
    ELEMENT_DIRECT,
    ELEMENT_JOINT_DIRECT,
    GMM_REFERENCE_TAIL,
    JOINT_DIRECT,
    PAIR_GUARD,
    TEACHER_DEVIATION_STUDENT_REFERENCE,
)


POLICY_EVIDENCE = {
    GMM_REFERENCE_TAIL.name: {
        "revision": GMM_REFERENCE_TAIL.revision,
        "constants": {
            "frame_top_fraction": 0.05,
            "candidate_interval": "[floor, upper)",
            "p_space_density": "-log(p)",
        },
    },
    TEACHER_DEVIATION_STUDENT_REFERENCE.name: {
        "revision": TEACHER_DEVIATION_STUDENT_REFERENCE.revision,
        "constants": {
            "proposal": "student-reference-tail-pool",
            "teacher_evaluation_bound": "student-pool",
            "train_fp_overlap": False,
        },
    },
    PAIR_GUARD.name: {
        "revision": PAIR_GUARD.revision,
        "constants": {
            "comparison": "distance < threshold",
            "equality": "pass",
            "units": "angstrom",
        },
    },
    "DIRECT": {
        "revision": 3,
        "constants": {
            "frame_pooling": "mean-if-atom-resolved",
            "normalization": "per-frame-l2",
            "standardization": "population-standard-deviation",
            "zero_variance_divisor": 1.0,
            "pca_retention": "eigenvalue > 1.0",
            "pca_weight": "explained-variance-ratio",
            "clustering": "BIRCH",
            "threshold": 0.5,
            "branching_factor": 50,
            "target_clusters": "min(keep, population)",
            "per_cluster_keep": 1,
            "representative": "centroid-nearest-canonical-first-tie",
            "quota": "exact-min-keep-candidate-population",
            "completion": "greedy-farthest-from-selected",
            "underfill": "zero-against-achievable-quota",
            "fallback": None,
        },
    },
    "JOINT_DIRECT": {
        "revision": JOINT_DIRECT.revision,
        "constants": {
            "population": "ordered-candidates-then-training-background",
            "selectable": "candidates-only",
            "preprocessing": "DIRECT-revision-3-once",
            "target_clusters": "min(keep, candidate-population)",
            "representative": "centroid-nearest-canonical-first-tie",
            "quota": "exact-min-keep-candidate-population",
            "completion": "greedy-farthest-from-background-and-selected",
            "underfill": "zero-against-achievable-quota",
            "fallback": None,
        },
    },
    "ELEMENT_DIRECT": {
        "revision": ELEMENT_DIRECT.revision,
        "constants": {
            "population": "ordered-candidates-only",
            "selectable": "candidates-only",
            "frame_pooling": "per-element-mean",
            "element_preprocessing": "l2-standardize-strict-kaiser-variance-rms",
            "presence_coordinate": 1.0,
            "frame_composition": "type-map-order-divide-sqrt-present-elements",
            "whole_frame_preprocessing": None,
            "clustering": "DIRECT-revision-3-BIRCH-and-completion",
            "fallback": "committee-descriptor-owner-to-random",
        },
    },
    "ELEMENT_JOINT_DIRECT": {
        "revision": ELEMENT_JOINT_DIRECT.revision,
        "constants": {
            "population": "ordered-candidates-then-training-background",
            "selectable": "candidates-only",
            "frame_pooling": "per-element-mean",
            "element_preprocessing": "l2-standardize-strict-kaiser-variance-rms",
            "presence_coordinate": 1.0,
            "frame_composition": "type-map-order-divide-sqrt-present-elements",
            "whole_frame_preprocessing": None,
            "clustering": "joint-direct-revision-2-BIRCH-and-completion",
            "fallback": None,
        },
    },
    "FPS": {
        "revision": 1,
        "constants": {"start": "pool_medoid", "strategy": "greedy_farthest"},
    },
    "random": {
        "revision": 1,
        "constants": {"strategy": "stable_uniform_without_replacement"},
    },
}


@dataclass(frozen=True, slots=True)
class ElementFrameEmbedding:
    """Prepared fixed-width element-explicit frame evidence."""

    vectors: np.ndarray
    element_means: Mapping[str, np.ndarray]
    presence: np.ndarray
    active_elements: tuple[str, ...]
    inactive_elements: tuple[str, ...]
    element_dimensions: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ElementSelectionResult:
    """Element-explicit candidate membership plus replayable preparation."""

    selected_identities: tuple[str, ...]
    embedding: ElementFrameEmbedding
    audit: Mapping[str, object]


def p_space_sample(
    records: Sequence[tuple[str, float]],
    *,
    keep: int,
    entropy: str,
    lower_bound: float = 0.0,
    upper_bound: float | None = None,
) -> list[str]:
    """Sample records by g(p)=-log(p) on one validated conditional interval.

    ``upper_bound=None`` is the legacy inclusive ``[lower_bound, 1]`` form.
    An explicit upper bound describes ``[lower_bound, upper_bound)``.
    """
    if keep < 0:
        raise ValueError("p-space keep must be non-negative")
    normalized_records = [(str(identity), float(value)) for identity, value in records]
    stable_records = sorted(normalized_records, key=lambda item: item[0])
    if len({identity for identity, _ in stable_records}) != len(stable_records):
        raise ValueError("p-space identities must be unique")
    masses = p_space_information_masses(
        [value for _, value in stable_records],
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if keep >= len(records):
        return [identity for identity, _ in normalized_records]
    if not records or keep == 0:
        return []
    weights = {identity: float(mass) for (identity, _), mass in zip(stable_records, masses)}
    seed = int.from_bytes(hashlib.sha256(entropy.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    keys = [(rng.random() ** (1.0 / weights[identity]), index, identity) for index, (identity, _) in enumerate(stable_records)]
    return [identity for _, _, identity in sorted(keys, key=lambda item: (-item[0], item[1]))[:keep]]


def p_space_information_masses(
    p_values: Sequence[float],
    *,
    lower_bound: float = 0.0,
    upper_bound: float | None = None,
) -> np.ndarray:
    """Return midpoint-bounded conditional p-space mass, sharing ties equally.

    This is the policy-owned finite-pool implementation of the validated
    analysis method, not raw per-frame ``-log(p_i)`` weighting.  Omitting the
    upper bound preserves the inclusive ``[lower_bound, 1]`` interval; an
    explicit upper bound uses ``[lower_bound, upper_bound)``.
    """
    floor = float(lower_bound)
    if not math.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("p-space bounds must satisfy 0 <= lower bound < upper bound <= 1")
    upper = 1.0 if upper_bound is None else float(upper_bound)
    if not math.isfinite(upper) or not floor < upper <= 1:
        raise ValueError("p-space bounds must satisfy 0 <= lower bound < upper bound <= 1")
    grouped: dict[float, list[int]] = {}
    for index, raw_p in enumerate(p_values):
        p = float(raw_p)
        within_upper = p <= 1 if upper_bound is None else p < upper
        if not math.isfinite(p) or not floor <= p or not within_upper:
            raise ValueError("p-space probabilities must be finite values within the conditional bounds")
        grouped.setdefault(p, []).append(index)
    masses = np.empty(len(p_values), dtype=float)
    if not grouped:
        return masses
    unique = sorted(grouped)
    for index, p in enumerate(unique):
        left = floor if index == 0 else (unique[index - 1] + p) / 2.0
        right = upper if index == len(unique) - 1 else (p + unique[index + 1]) / 2.0
        group_mass = (_g_cdf(right) - _g_cdf(left)) / (_g_cdf(upper) - _g_cdf(floor))
        for record_index in grouped[p]:
            masses[record_index] = max(group_mass / len(grouped[p]), np.finfo(float).tiny)
    return masses


def reduce_descriptors(descriptors: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """Mean-pool, normalize, standardize, and Kaiser-reduce lifecycle descriptors."""
    if isinstance(descriptors, np.ndarray):
        values = np.asarray(descriptors, dtype=float)
    else:
        committee = [np.asarray(item, dtype=float) for item in descriptors]
        if not committee:
            raise ValueError("descriptor committee cannot be empty")
        values = np.mean(np.stack(committee, axis=0), axis=0)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("descriptors must have frame x atom x feature shape")
    return _preprocess_frame_vectors(np.mean(values, axis=1))


def reduce_descriptor_frames(descriptors: Sequence[np.ndarray]) -> np.ndarray:
    """Reduce potentially ragged per-frame atom features for representative selection."""
    return _preprocess_frame_vectors(mean_descriptor_frames(descriptors))


def mean_descriptor_frames(descriptors: Sequence[np.ndarray]) -> np.ndarray:
    """Mean-pool potentially ragged atom features into finite frame vectors."""
    if not descriptors:
        raise ValueError("descriptor frame list cannot be empty")
    frames = [np.asarray(frame, dtype=float) for frame in descriptors]
    if any(frame.ndim != 2 or frame.shape[0] == 0 or frame.shape[1] == 0 or not np.isfinite(frame).all() for frame in frames):
        raise ValueError("descriptor frames must be non-empty finite atom x feature arrays")
    widths = {frame.shape[1] for frame in frames}
    if len(widths) != 1:
        raise ValueError("descriptor frames must have one shared feature width")
    return np.stack([np.mean(frame, axis=0) for frame in frames], axis=0)


def prepare_element_frame_vectors(
    element_rows: Sequence[Mapping[str, np.ndarray]],
    *,
    presence: Sequence[Sequence[str]],
    type_map: Sequence[str],
) -> ElementFrameEmbedding:
    """Prepare one equal-element, presence-aware frame embedding.

    Atom rows are mean-pooled within each chemically present element.  Every
    active element is normalized, standardized, Kaiser-reduced and RMS-scaled
    independently before fixed-order frame composition.  No whole-frame
    preprocessing is applied after composition.
    """

    elements = tuple(str(item) for item in type_map)
    if not elements or any(not item for item in elements) or len(set(elements)) != len(elements):
        raise ValueError("element DIRECT type_map must be non-empty and ordered unique")
    frames = tuple(element_rows)
    declared_presence = tuple(tuple(str(item) for item in row) for row in presence)
    if not frames or len(frames) != len(declared_presence):
        raise ValueError("element DIRECT frames and presence must have matching non-empty order")

    element_set = set(elements)
    presence_mask = np.zeros((len(frames), len(elements)), dtype=bool)
    atom_rows: dict[str, list[tuple[int, np.ndarray]]] = {element: [] for element in elements}
    descriptor_width: int | None = None
    for frame_index, (frame_rows, frame_presence) in enumerate(
        zip(frames, declared_presence, strict=True)
    ):
        if not isinstance(frame_rows, Mapping):
            raise TypeError("element DIRECT frame evidence must be an element mapping")
        if len(frame_presence) != len(set(frame_presence)):
            raise ValueError("element DIRECT frame presence must be unique")
        present_set = set(frame_presence)
        if not present_set or not present_set.issubset(element_set):
            raise ValueError("element DIRECT frame presence must use the configured type_map")
        evidence_elements = {str(item) for item in frame_rows}
        if evidence_elements != present_set:
            raise ValueError(
                "element DIRECT evidence must match chemically present elements exactly"
            )
        for element_index, element in enumerate(elements):
            if element not in present_set:
                continue
            values = np.asarray(frame_rows[element], dtype=float)
            if (
                values.ndim != 2
                or values.shape[0] == 0
                or values.shape[1] == 0
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "element DIRECT descriptors must be finite non-empty atom x feature rows"
                )
            if descriptor_width is None:
                descriptor_width = int(values.shape[1])
            elif values.shape[1] != descriptor_width:
                raise ValueError("element DIRECT descriptors must share one feature width")
            presence_mask[frame_index, element_index] = True
            atom_rows[element].append((frame_index, values))

    active_elements = tuple(
        element for element_index, element in enumerate(elements)
        if bool(np.any(presence_mask[:, element_index]))
    )
    inactive_elements = tuple(element for element in elements if element not in active_elements)
    if descriptor_width is None:
        raise ValueError("element DIRECT fitting population has no descriptor evidence")

    means_by_element: dict[str, np.ndarray] = {}
    coordinates_by_element: dict[str, np.ndarray] = {}
    dimensions: dict[str, int] = {}
    for element in active_elements:
        means = np.zeros((len(frames), descriptor_width), dtype=float)
        present_indices: list[int] = []
        for frame_index, rows in atom_rows[element]:
            means[frame_index] = np.mean(rows, axis=0)
            present_indices.append(frame_index)
        present = np.asarray(present_indices, dtype=int)
        reduced = _preprocess_present_element_means(means[present])
        rms = float(np.sqrt(np.mean(np.sum(reduced * reduced, axis=1))))
        if rms > 0:
            reduced = reduced / rms
        coordinates = np.zeros((len(frames), reduced.shape[1]), dtype=float)
        coordinates[present] = reduced
        means_by_element[element] = means
        coordinates_by_element[element] = coordinates
        dimensions[element] = int(reduced.shape[1] + 1)

    present_counts = np.sum(presence_mask, axis=1)
    blocks: list[np.ndarray] = []
    for element in active_elements:
        element_index = elements.index(element)
        block = np.concatenate(
            (
                coordinates_by_element[element],
                presence_mask[:, element_index : element_index + 1].astype(float),
            ),
            axis=1,
        )
        blocks.append(block / np.sqrt(present_counts)[:, None])
    vectors = np.concatenate(blocks, axis=1)
    if not np.isfinite(vectors).all():
        raise ValueError("element DIRECT frame composition produced non-finite values")
    return ElementFrameEmbedding(
        vectors=vectors,
        element_means=means_by_element,
        presence=presence_mask,
        active_elements=active_elements,
        inactive_elements=inactive_elements,
        element_dimensions=dimensions,
    )


def _preprocess_present_element_means(means: np.ndarray) -> np.ndarray:
    values = np.asarray(means, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("element DIRECT preprocessing produced non-finite L2 norms")
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    centered = normalized - np.mean(normalized, axis=0, keepdims=True)
    deviations = np.std(normalized, axis=0, keepdims=True)
    standardized = centered / np.where(deviations > 0, deviations, 1.0)
    if len(standardized) < 2:
        return standardized
    covariance = np.cov(standardized, rowvar=False)
    if covariance.ndim == 0:
        return standardized
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    retained = eigenvalues > 1.0
    if not np.any(retained):
        return standardized
    positive_total = float(np.sum(eigenvalues[eigenvalues > 0]))
    reduced = (standardized @ eigenvectors[:, retained]) * (
        eigenvalues[retained] / positive_total
    )
    if not np.isfinite(reduced).all():
        raise ValueError("element DIRECT preprocessing produced non-finite values")
    return reduced


def select_element_representatives(
    candidate_identities: Sequence[str],
    candidate_element_rows: Sequence[Mapping[str, np.ndarray]],
    *,
    presence: Sequence[Sequence[str]],
    type_map: Sequence[str],
    keep: int,
) -> ElementSelectionResult:
    """Select an exact candidate quota in the prepared element space."""

    if not isinstance(keep, int) or isinstance(keep, bool) or keep <= 0:
        raise ValueError("element DIRECT keep must be a positive integer")
    identities = tuple(str(identity) for identity in candidate_identities)
    if len(identities) != len(set(identities)):
        raise ValueError("element DIRECT candidate identities must be unique")
    if len(identities) != len(candidate_element_rows):
        raise ValueError("element DIRECT candidate identities and evidence must align")
    embedding = prepare_element_frame_vectors(
        candidate_element_rows,
        presence=presence,
        type_map=type_map,
    )
    target = min(keep, len(identities))
    representatives = _birch_representative_indices(embedding.vectors, target)
    selected_indices = _complete_quota_indices(
        embedding.vectors,
        selectable_indices=range(len(identities)),
        selected_indices=representatives,
        anchor_indices=(),
        target=target,
    )
    selected_set = set(selected_indices)
    selected = tuple(
        identity for index, identity in enumerate(identities) if index in selected_set
    )
    return ElementSelectionResult(
        selected_identities=selected,
        embedding=embedding,
        audit={
            "candidate_count": len(identities),
            "background_count": 0,
            "realized_clusters": len(representatives),
            "selected_background_count": 0,
            "selected_candidate_count": len(selected),
            "quota_completion_count": len(selected) - len(representatives),
            "underfill": max(0, target - len(selected)),
            "active_elements": embedding.active_elements,
            "inactive_elements": embedding.inactive_elements,
            "element_dimensions": dict(embedding.element_dimensions),
            "presence_counts": _element_presence_counts(embedding, type_map),
        },
    )


def select_element_joint_representatives(
    candidate_identities: Sequence[str],
    candidate_element_rows: Sequence[Mapping[str, np.ndarray]],
    *,
    candidate_presence: Sequence[Sequence[str]],
    background_identities: Sequence[str],
    background_element_rows: Sequence[Mapping[str, np.ndarray]],
    background_presence: Sequence[Sequence[str]],
    type_map: Sequence[str],
    keep: int,
) -> ElementSelectionResult:
    """Select candidates with exact training background as distance anchors."""

    if not isinstance(keep, int) or isinstance(keep, bool) or keep <= 0:
        raise ValueError("element joint DIRECT keep must be a positive integer")
    candidate_names = tuple(str(identity) for identity in candidate_identities)
    background_names = tuple(str(identity) for identity in background_identities)
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("element joint DIRECT candidate identities must be unique")
    if not background_names:
        raise ValueError("element joint DIRECT background must be non-empty")
    if len(background_names) != len(set(background_names)):
        raise ValueError("element joint DIRECT background identities must be unique")
    if len(candidate_names) != len(candidate_element_rows):
        raise ValueError("element joint DIRECT candidate identities and evidence must align")
    if len(background_names) != len(background_element_rows):
        raise ValueError("element joint DIRECT background identities and evidence must align")
    combined_rows = tuple(candidate_element_rows) + tuple(background_element_rows)
    combined_presence = tuple(candidate_presence) + tuple(background_presence)
    embedding = prepare_element_frame_vectors(
        combined_rows,
        presence=combined_presence,
        type_map=type_map,
    )
    candidate_count = len(candidate_names)
    target = min(keep, candidate_count)
    representatives = _birch_representative_indices(embedding.vectors, target)
    candidate_seeds = [index for index in representatives if index < candidate_count]
    selected_indices = _complete_quota_indices(
        embedding.vectors,
        selectable_indices=range(candidate_count),
        selected_indices=candidate_seeds,
        anchor_indices=range(candidate_count, len(combined_rows)),
        target=target,
    )
    selected_set = set(selected_indices)
    selected = tuple(
        identity
        for index, identity in enumerate(candidate_names)
        if index in selected_set
    )
    return ElementSelectionResult(
        selected_identities=selected,
        embedding=embedding,
        audit={
            "candidate_count": candidate_count,
            "background_count": len(background_names),
            "realized_clusters": len(representatives),
            "selected_background_count": sum(
                index >= candidate_count for index in representatives
            ),
            "selected_candidate_count": len(selected),
            "quota_completion_count": len(selected) - len(candidate_seeds),
            "underfill": max(0, target - len(selected)),
            "active_elements": embedding.active_elements,
            "inactive_elements": embedding.inactive_elements,
            "element_dimensions": dict(embedding.element_dimensions),
            "presence_counts": _element_presence_counts(embedding, type_map),
        },
    )


def _element_presence_counts(
    embedding: ElementFrameEmbedding, type_map: Sequence[str]
) -> dict[str, int]:
    return {
        str(element): int(np.sum(embedding.presence[:, index]))
        for index, element in enumerate(type_map)
    }


def _preprocess_frame_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("DIRECT frame vectors must be non-empty finite frame x feature rows")
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("DIRECT frame-vector preprocessing produced non-finite L2 norms")
    normalized = np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > 0,
    )
    means = np.mean(normalized, axis=0, keepdims=True)
    standard_deviations = np.std(normalized, axis=0, keepdims=True)
    standardized = (normalized - means) / np.where(
        standard_deviations > 0,
        standard_deviations,
        1.0,
    )
    if len(standardized) < 2:
        return standardized
    covariance = np.cov(standardized, rowvar=False)
    if covariance.ndim == 0:
        return standardized
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    retained = eigenvalues > 1.0
    if not np.any(retained):
        return standardized
    positive_total = float(np.sum(eigenvalues[eigenvalues > 0]))
    reduced = (standardized @ eigenvectors[:, retained]) * (
        eigenvalues[retained] / positive_total
    )
    if not np.isfinite(reduced).all():
        raise ValueError("DIRECT frame-vector preprocessing produced non-finite values")
    return reduced


def select_representatives(
    identities: Sequence[str],
    vectors: np.ndarray,
    *,
    keep: int,
    policy: str,
    entropy: str,
    runtime_tuning: Mapping[str, object] | None = None,
) -> list[str]:
    """Apply one fixed named selector. Runtime tuning is intentionally ignored."""
    del runtime_tuning
    values = np.asarray(vectors, dtype=float)
    if len(identities) != len(values) or values.ndim != 2:
        raise ValueError("selector identities and vectors must have matching frame x feature shape")
    if keep < 0:
        raise ValueError("representative keep must be non-negative")
    if keep == 0 or not identities:
        return []
    if keep >= len(identities):
        return list(identities)
    if policy == "FPS":
        selected = _fps_indices(values, keep)
    elif policy == "random":
        seed = int.from_bytes(hashlib.sha256(entropy.encode("utf-8")).digest()[:8], "big")
        selected = sorted(random.Random(seed).sample(range(len(identities)), keep))
    elif policy == "DIRECT":
        selected = _direct_indices(values, keep)
    else:
        raise ValueError("lifecycle representative policy must be DIRECT, FPS, or random")
    return [identities[index] for index in selected]


def select_joint_representatives(
    candidate_identities: Sequence[str],
    candidate_vectors: np.ndarray,
    background_identities: Sequence[str],
    background_vectors: np.ndarray,
    *,
    keep: int,
) -> list[str]:
    """Apply DIRECT once to candidates plus a non-selectable background."""
    selected, _audit = select_joint_representatives_with_audit(
        candidate_identities,
        candidate_vectors,
        background_identities,
        background_vectors,
        keep=keep,
    )
    return selected


def select_joint_representatives_with_audit(
    candidate_identities: Sequence[str],
    candidate_vectors: np.ndarray,
    background_identities: Sequence[str],
    background_vectors: np.ndarray,
    *,
    keep: int,
) -> tuple[list[str], dict[str, int]]:
    """Apply joint DIRECT once and expose its realized-cluster accounting."""
    if not isinstance(keep, int) or isinstance(keep, bool) or keep <= 0:
        raise ValueError("joint DIRECT keep must be a positive integer")
    candidate_names = [str(identity) for identity in candidate_identities]
    background_names = [str(identity) for identity in background_identities]
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("joint DIRECT candidate identities must be unique")
    if not background_names:
        raise ValueError("joint DIRECT background must be non-empty")
    if len(set(background_names)) != len(background_names):
        raise ValueError("joint DIRECT background identities must be unique")
    candidate_values = np.asarray(candidate_vectors, dtype=float)
    background_values = np.asarray(background_vectors, dtype=float)
    if (
        candidate_values.ndim != 2
        or background_values.ndim != 2
        or len(candidate_values) != len(candidate_names)
        or len(background_values) != len(background_names)
    ):
        raise ValueError(
            "joint DIRECT identities and vectors must have matching frame x feature shapes"
        )
    if (
        candidate_values.shape[1] == 0
        or candidate_values.shape[1] != background_values.shape[1]
    ):
        raise ValueError("joint DIRECT partitions must share one non-empty feature width")
    if not np.isfinite(candidate_values).all() or not np.isfinite(background_values).all():
        raise ValueError("joint DIRECT vectors must be finite")
    candidate_count = len(candidate_names)
    if candidate_count == 0:
        return [], {
            "realized_clusters": 0,
            "selected_background_count": 0,
            "selected_candidate_count": 0,
            "quota_completion_count": 0,
            "underfill": 0,
        }
    joint_values = np.concatenate((candidate_values, background_values), axis=0)
    reduced = _preprocess_frame_vectors(joint_values)
    target = min(keep, candidate_count)
    representatives = _birch_representative_indices(reduced, target)
    candidate_seeds = [index for index in representatives if index < candidate_count]
    selected = _complete_quota_indices(
        reduced,
        selectable_indices=range(candidate_count),
        selected_indices=candidate_seeds,
        anchor_indices=range(candidate_count, len(joint_values)),
        target=target,
    )
    selected_indices = set(selected)
    selected = [
        identity
        for index, identity in enumerate(candidate_names)
        if index in selected_indices
    ]
    return selected, {
        "realized_clusters": len(representatives),
        "selected_background_count": sum(
            index >= candidate_count for index in representatives
        ),
        "selected_candidate_count": len(selected),
        "quota_completion_count": len(selected) - len(candidate_seeds),
        "underfill": max(0, target - len(selected)),
    }


def _g_cdf(p: float) -> float:
    clipped = min(max(float(p), 1e-12), 1.0)
    return clipped * (1.0 - math.log(clipped))


def _fps_indices(vectors: np.ndarray, keep: int) -> list[int]:
    mean = np.mean(vectors, axis=0)
    distances = np.linalg.norm(vectors - mean, axis=1)
    selected = [int(np.argmin(distances))]
    nearest = np.linalg.norm(vectors - vectors[selected[0]], axis=1)
    nearest[selected[0]] = -np.inf
    while len(selected) < keep:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(vectors - vectors[index], axis=1))
        nearest[index] = -np.inf
    return selected


def _direct_indices(vectors: np.ndarray, keep: int) -> list[int]:
    reduced = _preprocess_frame_vectors(vectors)
    target = min(keep, len(vectors))
    seeds = _birch_representative_indices(reduced, target)
    return _complete_quota_indices(
        reduced,
        selectable_indices=range(len(vectors)),
        selected_indices=seeds,
        anchor_indices=(),
        target=target,
    )


def _birch_representative_indices(
    reduced: np.ndarray, target_clusters: int
) -> list[int]:
    from sklearn.cluster import Birch
    from sklearn.exceptions import ConvergenceWarning

    if target_clusters <= 0:
        return []
    if target_clusters == 1:
        centroid = np.mean(reduced, axis=0)
        return [int(np.argmin(np.linalg.norm(reduced - centroid, axis=1)))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = Birch(
            threshold=0.5,
            branching_factor=50,
            n_clusters=target_clusters,
        ).fit_predict(reduced)
    medoids: list[int] = []
    for label in sorted(set(int(item) for item in labels)):
        indices = np.flatnonzero(labels == label)
        centroid = np.mean(reduced[indices], axis=0)
        medoids.append(
            int(
                indices[
                    np.argmin(np.linalg.norm(reduced[indices] - centroid, axis=1))
                ]
            )
        )
    return medoids


def _complete_quota_indices(
    reduced: np.ndarray,
    *,
    selectable_indices: Iterable[int],
    selected_indices: Iterable[int],
    anchor_indices: Iterable[int],
    target: int,
) -> list[int]:
    """Complete one candidate quota by deterministic max-min distance.

    BIRCH representatives seed the selection.  Non-selectable background rows
    remain distance anchors, so they influence membership without consuming
    candidate capacity.  Equal distances retain selectable input order.
    """

    selectable = np.asarray(tuple(selectable_indices), dtype=int)
    if target <= 0 or len(selectable) == 0:
        return []
    target = min(int(target), len(selectable))
    selectable_set = set(int(index) for index in selectable)
    selected = []
    selected_set: set[int] = set()
    for raw_index in selected_indices:
        index = int(raw_index)
        if index in selectable_set and index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    if len(selected) >= target:
        return selected[:target]

    anchors = [int(index) for index in anchor_indices]
    anchors.extend(selected)
    selectable_rows = reduced[selectable]
    nearest_squared = np.full(len(selectable), np.inf, dtype=float)
    if anchors:
        row_norms = np.sum(selectable_rows * selectable_rows, axis=1)
        for start in range(0, len(anchors), 256):
            anchor_rows = reduced[np.asarray(anchors[start : start + 256], dtype=int)]
            squared = (
                row_norms[:, None]
                + np.sum(anchor_rows * anchor_rows, axis=1)[None, :]
                - 2.0 * (selectable_rows @ anchor_rows.T)
            )
            nearest_squared = np.minimum(
                nearest_squared,
                np.min(np.maximum(squared, 0.0), axis=1),
            )
    else:
        centroid = np.mean(selectable_rows, axis=0)
        nearest_squared = np.sum((selectable_rows - centroid) ** 2, axis=1)

    selectable_positions = {
        int(index): position for position, index in enumerate(selectable)
    }
    for index in selected:
        nearest_squared[selectable_positions[index]] = -np.inf
    while len(selected) < target:
        position = int(np.argmax(nearest_squared))
        index = int(selectable[position])
        selected.append(index)
        selected_set.add(index)
        squared = np.sum((selectable_rows - reduced[index]) ** 2, axis=1)
        nearest_squared = np.minimum(nearest_squared, squared)
        nearest_squared[position] = -np.inf
    return selected


__all__ = [
    "ElementFrameEmbedding",
    "ElementSelectionResult",
    "POLICY_EVIDENCE",
    "p_space_sample",
    "p_space_information_masses",
    "mean_descriptor_frames",
    "prepare_element_frame_vectors",
    "reduce_descriptors",
    "reduce_descriptor_frames",
    "select_joint_representatives",
    "select_joint_representatives_with_audit",
    "select_element_joint_representatives",
    "select_element_representatives",
    "select_representatives",
]
