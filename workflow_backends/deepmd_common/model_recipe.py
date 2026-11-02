"""Per-member DeepMD random-seed handling."""

from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence
from typing import Any


def synchronize_model_random_seeds(rendered: MutableMapping[str, Any], seed: int) -> None:
    """Use one deterministic seed for every random component of one member."""

    training = rendered.setdefault("training", {})
    if not isinstance(training, MutableMapping):
        raise ValueError("DeepMD native training configuration must be an object")
    training["seed"] = int(seed)
    model = rendered.setdefault("model", {})
    if not isinstance(model, MutableMapping):
        raise ValueError("DeepMD native model configuration must be an object")
    for container_name, container in (("model", model), ("top-level", rendered)):
        for key in ("descriptor", "fitting_net"):
            component = container.get(key)
            if component is not None:
                if not isinstance(component, MutableMapping):
                    raise ValueError(
                        f"DeepMD native {container_name} {key} must be an object"
                    )
                component["seed"] = int(seed)
    _replace_existing_nested_seeds(rendered, int(seed))


def _replace_existing_nested_seeds(value: object, seed: int) -> None:
    if isinstance(value, MutableMapping):
        for key, nested in value.items():
            if key == "seed":
                value[key] = seed
            else:
                _replace_existing_nested_seeds(nested, seed)
    elif isinstance(value, MutableSequence):
        for nested in value:
            _replace_existing_nested_seeds(nested, seed)
