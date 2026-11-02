"""The independent DICE lifecycle iter/round workflow route."""

from typing import Any

__all__ = ["prepare_lifecycle_launch", "DataService", "LabelContract", "execute_lifecycle"]


def __getattr__(name: str) -> Any:
    if name == "prepare_lifecycle_launch":
        from workflow_core.lifecycle.launch import prepare_lifecycle_launch

        return prepare_lifecycle_launch
    if name in {"DataService", "LabelContract"}:
        from workflow_core.lifecycle.data import DataService, LabelContract

        return {"DataService": DataService, "LabelContract": LabelContract}[name]
    if name == "execute_lifecycle":
        from workflow_core.lifecycle.runner import execute_lifecycle

        return execute_lifecycle
    raise AttributeError(name)
