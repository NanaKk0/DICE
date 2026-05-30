from __future__ import annotations

from typing import Any

__all__ = ["TASK_REGISTRY"]


def __getattr__(name: str) -> Any:
    if name == "TASK_REGISTRY":
        from workflow_core.entrypoints.registry import TASK_REGISTRY

        return TASK_REGISTRY
    raise AttributeError(name)
