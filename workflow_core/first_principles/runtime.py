from __future__ import annotations

from typing import Any

from workflow_backends.first_principles.registry import (
    get_first_principles_backend,
    resolve_first_principles_backend_name,
)


def resolve_dft_runtime(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compatibility name for resolving first-principles backend runtime config."""

    backend_name = resolve_first_principles_backend_name(cfg)
    backend = get_first_principles_backend(backend_name)
    runtime = backend.resolve_runtime(cfg)
    runtime["first_principles_backend"] = backend_name
    runtime.setdefault("first_principles_runtime_resolution", {"backend": backend_name})
    runtime.setdefault("dft_runtime_resolution", runtime["first_principles_runtime_resolution"])
    return runtime


resolve_first_principles_runtime = resolve_dft_runtime
