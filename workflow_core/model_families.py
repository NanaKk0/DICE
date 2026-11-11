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


def deepmd_learned_model_config(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Locate the unique learned config in native analytical bridging.

    This is structural routing only: DeePMD owns expansion, the potential
    parameters and their validation. General linear ensembles are unchanged.
    """
    if model.get("type") != "linear_ener":
        return model
    children = model.get("models", ())
    if not isinstance(children, (list, tuple)):
        return model
    inner = [child for child in children
             if isinstance(child, Mapping) and child.get("type") == "inner_potential"]
    if not inner:
        return model
    if (len(children) != 2 or len(inner) != 1
            or not all(isinstance(child, Mapping) for child in children)
            or model.get("weights") != "sum"):
        raise ValueError(
            "DeepMD analytical bridging requires one learned model and one "
            "inner_potential with weights='sum'"
        )
    return next(child for child in children if child.get("type") != "inner_potential")


def deepmd_recipe_model_family(rendered: Mapping[str, Any]) -> str:
    model = rendered.get("model", {})
    if not isinstance(model, Mapping):
        return "standard"
    model = deepmd_learned_model_config(model)
    descriptor = model.get("descriptor", rendered.get("descriptor", {}))
    if isinstance(descriptor, Mapping):
        family = canonical_deepmd_descriptor_family(descriptor.get("type"))
        if family != "standard":
            return family
    family = canonical_deepmd_descriptor_family(model.get("type"))
    if family != "standard":
        return family
    # Native DeePMD owns preset validation and architecture expansion. Only
    # the declared family is needed here, before selecting its runtime.
    preset = model.get("preset")
    if isinstance(preset, str):
        return canonical_deepmd_descriptor_family(preset.lower().split("-", 1)[0])
    return "standard"
