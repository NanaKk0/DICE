"""Stable semantic model-family identities shared across workflow boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DPA4_DESCRIPTOR_TYPES = frozenset({"dpa4", "DPA4", "SeZM", "sezm"})


def canonical_deepmd_descriptor_family(value: object) -> str:
    if value in DPA4_DESCRIPTOR_TYPES:
        return "dpa4"
    if value == "dpa4c":
        return "dpa4c"
    return "standard"


def deepmd_recipe_model_family(rendered: Mapping[str, Any]) -> str:
    model = rendered.get("model", {})
    if not isinstance(model, Mapping):
        return "standard"
    descriptor = model.get("descriptor", rendered.get("descriptor", {}))
    if isinstance(descriptor, Mapping):
        family = canonical_deepmd_descriptor_family(descriptor.get("type"))
        if family != "standard":
            return family
    return canonical_deepmd_descriptor_family(model.get("type"))
