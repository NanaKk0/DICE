from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .core import direct_select
from .types import DIRECTResult
from .utils import ensure_2d_float_array


def encode_structures_to_matrix(
    structures: Sequence[Any],
    *,
    structure_encoder: Callable[[Sequence[Any]], Any],
) -> np.ndarray:
    if structure_encoder is None:
        raise ValueError("structure_encoder is required")
    matrix = structure_encoder(structures)
    return ensure_2d_float_array(matrix)


def direct_select_structures(
    structures: Sequence[Any],
    *,
    structure_encoder: Callable[[Sequence[Any]], Any],
    **kwargs: Any,
) -> DIRECTResult:
    X = encode_structures_to_matrix(structures, structure_encoder=structure_encoder)
    return direct_select(X, **kwargs)
