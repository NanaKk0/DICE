from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .clustering import compute_cluster_centers, fit_birch_clusterer
from .schema import normalize_direct_config, validate_direct_inputs, validate_smallest_mode_requirements
from .selection import select_from_clusters
from .types import DIRECTConfig, DIRECTProjection, DIRECTResult
from .utils import apply_explained_variance_weighting, ensure_2d_float_array, safe_kaiser_component_count


def fit_direct_projection(
    X: Any,
    *,
    cfg: DIRECTConfig | dict[str, Any] | None = None,
) -> DIRECTProjection:
    validate_direct_inputs(X)
    features = ensure_2d_float_array(X)
    normalized_cfg = normalize_direct_config(cfg)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    n_components = max(1, min(X_scaled.shape[0], X_scaled.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="full")
    X_pca_full = pca.fit_transform(X_scaled)
    retained_component_count = safe_kaiser_component_count(pca.explained_variance_)
    retained_component_indices = tuple(range(min(retained_component_count, X_pca_full.shape[1])))
    X_pca = X_pca_full[:, retained_component_indices]
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_[list(retained_component_indices)], dtype=float)

    if normalized_cfg.use_explained_variance_weighting:
        X_pca = apply_explained_variance_weighting(X_pca, explained_variance_ratio)

    return DIRECTProjection(
        X_scaled=np.asarray(X_scaled, dtype=float),
        X_pca=np.asarray(X_pca, dtype=float),
        explained_variance_ratio_=explained_variance_ratio,
        scaler=scaler,
        pca=pca,
        retained_component_indices=retained_component_indices,
        use_explained_variance_weighting=bool(normalized_cfg.use_explained_variance_weighting),
    )


def transform_direct_projection(projection: DIRECTProjection, X: Any) -> np.ndarray:
    features = ensure_2d_float_array(X)
    X_scaled = projection.scaler.transform(features)
    X_pca_full = projection.pca.transform(X_scaled)
    X_pca = X_pca_full[:, list(projection.retained_component_indices)]
    if projection.use_explained_variance_weighting:
        X_pca = apply_explained_variance_weighting(X_pca, projection.explained_variance_ratio_)
    return np.asarray(X_pca, dtype=float)


def direct_select(
    X: Any,
    *,
    samples_per_cluster: int = 1,
    selection_criteria: str = "center",
    n_sites: Any | None = None,
    use_explained_variance_weighting: bool = False,
    birch_threshold: float = 0.5,
    birch_branching_factor: int = 50,
    birch_n_clusters: int | None = None,
    random_state: int = 42,
) -> DIRECTResult:
    cfg = DIRECTConfig(
        samples_per_cluster=int(samples_per_cluster),
        selection_criteria=selection_criteria,  # normalized below
        use_explained_variance_weighting=bool(use_explained_variance_weighting),
        birch_threshold=float(birch_threshold),
        birch_branching_factor=int(birch_branching_factor),
        birch_n_clusters=birch_n_clusters,
        random_state=int(random_state),
    )
    normalized_cfg = normalize_direct_config(cfg)
    if normalized_cfg.selection_criteria == "smallest":
        validate_smallest_mode_requirements(n_sites, ensure_2d_float_array(X).shape[0])

    projection = fit_direct_projection(X, cfg=normalized_cfg)
    clusterer = fit_birch_clusterer(
        projection.X_pca,
        threshold=normalized_cfg.birch_threshold,
        branching_factor=normalized_cfg.birch_branching_factor,
        n_clusters=normalized_cfg.birch_n_clusters,
    )
    labels = np.asarray(clusterer.labels_, dtype=int)
    label_centers = compute_cluster_centers(projection.X_pca, labels)
    selected_indices, selected_by_label = select_from_clusters(
        projection.X_pca,
        labels,
        samples_per_cluster=normalized_cfg.samples_per_cluster,
        selection_criteria=normalized_cfg.selection_criteria,
        n_sites=n_sites,
        random_state=normalized_cfg.random_state,
    )
    cluster_members = _group_members(labels)

    return DIRECTResult(
        selected_indices=selected_indices,
        labels=labels,
        label_centers=label_centers,
        cluster_members=cluster_members,
        clusterer=clusterer,
        projection=projection,
        selected_by_label=selected_by_label,
    )


def _group_members(labels: np.ndarray) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(np.asarray(labels, dtype=int).reshape(-1)):
        grouped.setdefault(int(label), []).append(int(index))
    return grouped
