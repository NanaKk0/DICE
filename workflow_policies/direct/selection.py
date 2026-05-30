from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .clustering import compute_cluster_centers, group_indices_by_label
from .utils import evenly_spaced_indices, stable_random_seed


def select_from_clusters(
    X_pca: Any,
    labels: Any,
    *,
    samples_per_cluster: int,
    selection_criteria: str,
    n_sites: Sequence[int] | None = None,
    random_state: int = 42,
) -> tuple[list[int], dict[int, list[int]]]:
    if samples_per_cluster <= 0:
        return [], {}

    features = np.asarray(X_pca, dtype=float)
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if features.shape[0] != label_array.shape[0]:
        raise ValueError("DIRECT labels must have the same length as X_pca")
    grouped = group_indices_by_label(label_array)
    centers = compute_cluster_centers(features, label_array)
    sites_array = None if n_sites is None else np.asarray(n_sites, dtype=int).reshape(-1)
    if sites_array is not None and sites_array.shape[0] != features.shape[0]:
        raise ValueError("DIRECT n_sites must have the same length as X_pca")

    selected_indices: list[int] = []
    selected_by_label: dict[int, list[int]] = {}

    for label in sorted(grouped):
        members = grouped[label]
        chosen = _select_within_cluster(
            features,
            members,
            center=centers[label],
            samples_per_cluster=samples_per_cluster,
            selection_criteria=selection_criteria,
            n_sites=sites_array,
            random_state=random_state,
            cluster_label=label,
        )
        selected_indices.extend(chosen)
        selected_by_label[int(label)] = chosen

    return selected_indices, selected_by_label


def _select_within_cluster(
    features: np.ndarray,
    members: list[int],
    *,
    center: np.ndarray,
    samples_per_cluster: int,
    selection_criteria: str,
    n_sites: np.ndarray | None,
    random_state: int,
    cluster_label: int,
) -> list[int]:
    if not members:
        return []
    if samples_per_cluster >= len(members):
        return list(members)

    ranked = _rank_members(
        features,
        members,
        center=center,
        selection_criteria=selection_criteria,
        n_sites=n_sites,
        random_state=random_state,
        cluster_label=cluster_label,
    )
    if samples_per_cluster == 1:
        return [ranked[0]]

    positions = evenly_spaced_indices(len(ranked), samples_per_cluster)
    return [ranked[position] for position in positions]


def _rank_members(
    features: np.ndarray,
    members: list[int],
    *,
    center: np.ndarray,
    selection_criteria: str,
    n_sites: np.ndarray | None,
    random_state: int,
    cluster_label: int,
) -> list[int]:
    mode = str(selection_criteria).strip().lower()
    if mode == "random":
        seed = stable_random_seed("direct", "random", cluster_label, len(members), base_seed=random_state)
        rng = np.random.default_rng(seed)
        shuffled = list(members)
        if len(shuffled) > 1:
            rng.shuffle(shuffled)
        return shuffled
    if mode == "smallest":
        if n_sites is None:
            raise ValueError("DIRECT selection_criteria='smallest' requires n_sites")
        return sorted(
            members,
            key=lambda index: (
                int(n_sites[index]),
                float(np.linalg.norm(features[index] - center)),
                int(index),
            ),
        )

    return sorted(
        members,
        key=lambda index: (
            float(np.linalg.norm(features[index] - center)),
            int(index),
        ),
    )
