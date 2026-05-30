from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np


def ensure_2d_float_array(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError("DIRECT requires a non-empty feature matrix")
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"DIRECT requires a 2D feature matrix, got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"DIRECT requires a non-empty feature matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("DIRECT feature matrix contains non-finite values")
    return np.asarray(arr, dtype=float)


def safe_kaiser_component_count(explained_variance: Sequence[float]) -> int:
    values = np.asarray(explained_variance, dtype=float).reshape(-1)
    if values.size == 0:
        return 1
    count = int(np.sum(values > 1.0))
    return max(1, count)


def apply_explained_variance_weighting(X_pca: np.ndarray, explained_variance_ratio: Sequence[float]) -> np.ndarray:
    projected = ensure_2d_float_array(X_pca)
    weights = np.asarray(explained_variance_ratio, dtype=float).reshape(-1)
    if weights.size != projected.shape[1]:
        raise ValueError(
            "explained_variance_ratio length must match the retained PCA feature dimension"
        )
    return projected * weights.reshape(1, -1)


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    raw = np.linspace(0, length - 1, num=count)
    selected = [int(round(value)) for value in raw]
    selected = [min(length - 1, max(0, index)) for index in selected]

    deduped: list[int] = []
    seen: set[int] = set()
    for index in selected:
        if index in seen:
            continue
        seen.add(index)
        deduped.append(index)

    if len(deduped) < count:
        for index in range(length):
            if index in seen:
                continue
            deduped.append(index)
            seen.add(index)
            if len(deduped) == count:
                break
    return deduped[:count]


def stable_random_seed(*parts: object, base_seed: int) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return (int(digest, 16) ^ int(base_seed)) % (2**32)
