from __future__ import annotations

from typing import Any

from .types import DIRECTConfig

_SELECTION_CRITERIA = {"center", "random", "smallest"}


def normalize_direct_config(payload: dict[str, Any] | DIRECTConfig | None) -> DIRECTConfig:
    if payload is None:
        return DIRECTConfig()
    if isinstance(payload, DIRECTConfig):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"DIRECT config must be a dict or DIRECTConfig, got {type(payload)!r}")

    selection_criteria = resolve_selection_criteria(payload.get("selection_criteria", "center"))
    samples_per_cluster = int(payload.get("samples_per_cluster", payload.get("n_per_cluster", 1)))
    if samples_per_cluster <= 0:
        raise ValueError("DIRECT samples_per_cluster must be > 0")

    birch_n_clusters = payload.get("birch_n_clusters", None)
    if birch_n_clusters in ("", None):
        birch_n_clusters = None
    else:
        birch_n_clusters = int(birch_n_clusters)
        if birch_n_clusters <= 0:
            raise ValueError("DIRECT birch_n_clusters must be > 0 when provided")

    return DIRECTConfig(
        samples_per_cluster=samples_per_cluster,
        selection_criteria=selection_criteria,
        use_explained_variance_weighting=bool(payload.get("use_explained_variance_weighting", False)),
        birch_threshold=float(payload.get("birch_threshold", 0.5)),
        birch_branching_factor=int(payload.get("birch_branching_factor", 50)),
        birch_n_clusters=birch_n_clusters,
        random_state=int(payload.get("random_state", 42)),
    )


def resolve_selection_criteria(value: Any) -> str:
    selection_criteria = str(value or "").strip().lower()
    if selection_criteria not in _SELECTION_CRITERIA:
        raise ValueError(
            f"unsupported DIRECT selection_criteria {selection_criteria!r}; "
            f"supported values: {sorted(_SELECTION_CRITERIA)}"
        )
    return selection_criteria


def validate_smallest_mode_requirements(n_sites: Any | None, n_samples: int) -> None:
    if n_sites is None:
        raise ValueError("DIRECT selection_criteria='smallest' requires n_sites")
    if len(n_sites) != n_samples:
        raise ValueError("DIRECT n_sites must have the same length as the feature matrix")


def validate_direct_inputs(X: Any) -> None:
    from .utils import ensure_2d_float_array

    ensure_2d_float_array(X)
