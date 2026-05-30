from __future__ import annotations

from typing import Any

from workflow_backends.first_principles.base import ExternalHandoffFirstPrinciplesBackend, FirstPrinciplesBackend


_REGISTRY: dict[str, FirstPrinciplesBackend] = {}


def register_first_principles_backend(name: str, backend: FirstPrinciplesBackend) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("first-principles backend name must be non-empty")
    _REGISTRY[key] = backend


def _ensure_defaults_registered() -> None:
    if "external_handoff" not in _REGISTRY:
        register_first_principles_backend("external_handoff", ExternalHandoffFirstPrinciplesBackend())
    if "cp2k" not in _REGISTRY:
        from workflow_backends.first_principles.cp2k.backend import CP2KFirstPrinciplesBackend

        register_first_principles_backend("cp2k", CP2KFirstPrinciplesBackend())


def resolve_first_principles_backend_name(cfg: dict[str, Any]) -> str:
    explicit = cfg.get("first_principles_backend")
    if explicit:
        return str(explicit).strip().lower()
    labelling_style = str(cfg.get("labelling_style", "cp2k")).strip().lower()
    if labelling_style in {"external_handoff", "cp2k"}:
        return labelling_style
    return "cp2k"


def get_first_principles_backend(name: str | None = None) -> FirstPrinciplesBackend:
    _ensure_defaults_registered()
    key = str(name or "cp2k").strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported first_principles_backend={key!r}") from exc
