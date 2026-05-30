from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SelectionCriteria = Literal["center", "random", "smallest"]


@dataclass(frozen=True, slots=True)
class DIRECTConfig:
    samples_per_cluster: int = 1
    selection_criteria: SelectionCriteria = "center"
    use_explained_variance_weighting: bool = False
    birch_threshold: float = 0.5
    birch_branching_factor: int = 50
    birch_n_clusters: int | None = None
    random_state: int = 42


@dataclass(slots=True)
class DIRECTProjection:
    X_scaled: np.ndarray
    X_pca: np.ndarray
    explained_variance_ratio_: np.ndarray
    scaler: StandardScaler
    pca: PCA
    retained_component_indices: tuple[int, ...]
    use_explained_variance_weighting: bool = False


@dataclass(slots=True)
class DIRECTResult:
    selected_indices: list[int]
    labels: np.ndarray
    label_centers: dict[int, np.ndarray]
    cluster_members: dict[int, list[int]]
    clusterer: Birch
    projection: DIRECTProjection
    selected_by_label: dict[int, list[int]] = field(default_factory=dict)

    @property
    def X_scaled(self) -> np.ndarray:
        return self.projection.X_scaled

    @property
    def X_pca(self) -> np.ndarray:
        return self.projection.X_pca

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.projection.explained_variance_ratio_

    @property
    def scaler(self) -> StandardScaler:
        return self.projection.scaler

    @property
    def pca(self) -> PCA:
        return self.projection.pca
