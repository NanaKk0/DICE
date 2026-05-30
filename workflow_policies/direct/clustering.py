from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import Birch

from .utils import ensure_2d_float_array


def fit_birch_clusterer(
    X_pca: Any,
    *,
    threshold: float = 0.5,
    branching_factor: int = 50,
    n_clusters: int | None = None,
) -> Birch:
    features = ensure_2d_float_array(X_pca)
    clusterer = Birch(
        threshold=float(threshold),
        branching_factor=int(branching_factor),
        n_clusters=n_clusters,
    )
    clusterer.fit(features)
    return clusterer


def predict_cluster_labels(clusterer: Birch, X_pca: Any) -> np.ndarray:
    features = ensure_2d_float_array(X_pca)
    return np.asarray(clusterer.predict(features), dtype=int)


def group_indices_by_label(labels: Any) -> dict[int, list[int]]:
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(label_array):
        grouped[int(label)].append(int(index))
    return {label: members for label, members in sorted(grouped.items())}


def compute_cluster_centers(X_pca: Any, labels: Any) -> dict[int, np.ndarray]:
    features = ensure_2d_float_array(X_pca)
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    centers: dict[int, np.ndarray] = {}
    for label in sorted(set(label_array.tolist())):
        members = features[label_array == label]
        if members.size == 0:
            centers[int(label)] = np.zeros(features.shape[1], dtype=float)
        else:
            centers[int(label)] = members.mean(axis=0)
    return centers
