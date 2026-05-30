from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from workflow_backends.first_principles.registry import (
    get_first_principles_backend,
    resolve_first_principles_backend_name,
)
from workflow_core.first_principles.runtime import resolve_dft_runtime


@dataclass(frozen=True, slots=True)
class FirstPrinciplesSubstepContext:
    substep: str
    cfg: dict[str, Any]
    backend_name: str
    backend: Any
    task_root: Path | None = None
    output_root: Path | None = None


FirstPrinciplesSubstepExecutor = Callable[[FirstPrinciplesSubstepContext], dict[str, Any]]


def run_first_principles_substep(
    cfg: dict[str, Any],
    *,
    substep: str,
    execute: FirstPrinciplesSubstepExecutor,
) -> dict[str, Any]:
    resolved_cfg = resolve_dft_runtime(dict(cfg or {}))
    backend_name = resolve_first_principles_backend_name(resolved_cfg)
    backend = get_first_principles_backend(backend_name)
    context = FirstPrinciplesSubstepContext(
        substep=str(substep),
        cfg=resolved_cfg,
        backend_name=backend_name,
        backend=backend,
        task_root=_optional_path(resolved_cfg.get("task_root")),
        output_root=_optional_path(resolved_cfg.get("output_root")),
    )
    payload = dict(execute(context) or {})
    payload.setdefault("first_principles_substep", str(substep))
    payload.setdefault("first_principles_backend", backend_name)
    payload.setdefault(
        "first_principles_runtime_resolution",
        resolved_cfg.get("first_principles_runtime_resolution"),
    )
    payload.setdefault("dft_runtime_resolution", resolved_cfg.get("dft_runtime_resolution"))
    return payload


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).resolve()
