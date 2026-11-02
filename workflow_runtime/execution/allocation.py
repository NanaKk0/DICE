from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from workflow_runtime.execution.profile import (
    ResourceClass,
    validate_operation_resource_shape,
)


@dataclass(frozen=True, slots=True)
class Allocation:
    """Backend-safe allocation facts with no scheduler authority."""

    nodes: int
    gpus_per_node: int
    cpus_per_task: int
    memory_mb: int
    launcher: str
    mode: str
    per_gpu: int | None

    def __post_init__(self) -> None:
        for name in ("nodes", "gpus_per_node", "cpus_per_task", "memory_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"allocation {name} must be an integer")
        if self.nodes != 1:
            raise ValueError("allocation requires exactly one node")
        if self.gpus_per_node < 0:
            raise ValueError("allocation gpus_per_node must be non-negative")
        if self.cpus_per_task <= 0 or self.memory_mb <= 0:
            raise ValueError("allocation CPU and memory values must be positive")
        if self.mode == "pool":
            if self.launcher != "direct":
                raise ValueError("pool allocation requires direct launcher")
            if (
                isinstance(self.per_gpu, bool)
                or not isinstance(self.per_gpu, int)
                or self.per_gpu <= 0
            ):
                raise ValueError("pool allocation per_gpu must be positive")
            return
        if self.mode == "distributed":
            if self.per_gpu is not None:
                raise ValueError("distributed allocation forbids per_gpu")
            if self.gpus_per_node < 2 or self.launcher != "torchrun":
                raise ValueError(
                    "distributed allocation requires at least two GPUs and torchrun"
                )
            return
        raise ValueError(f"allocation mode is unsupported: {self.mode!r}")

    @property
    def capacity(self) -> int:
        if self.mode == "distributed":
            return 1
        if self.per_gpu is None:
            raise AssertionError("pool allocation has no per_gpu")
        return self.gpus_per_node * self.per_gpu

    @classmethod
    def local_default(cls) -> Allocation:
        return cls(
            nodes=1,
            gpus_per_node=1,
            cpus_per_task=1,
            memory_mb=1,
            launcher="direct",
            mode="pool",
            per_gpu=1,
        )

    @classmethod
    def from_resource_class(cls, resource: ResourceClass) -> Allocation:
        return cls(
            nodes=resource.nodes,
            gpus_per_node=resource.gpus_per_node,
            cpus_per_task=resource.cpus_per_process,
            memory_mb=resource.memory_mb,
            launcher=resource.launcher.kind,
            mode=resource.mode,
            per_gpu=resource.per_gpu,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "cpus_per_task": self.cpus_per_task,
            "memory_mb": self.memory_mb,
            "launcher": self.launcher,
            "mode": self.mode,
            "per_gpu": self.per_gpu,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Allocation:
        expected = {
            "nodes",
            "gpus_per_node",
            "cpus_per_task",
            "memory_mb",
            "launcher",
            "mode",
            "per_gpu",
        }
        if set(value) != expected:
            raise ValueError("allocation evidence shape is invalid")
        return cls(**dict(value))

    @classmethod
    def from_json(cls, value: str) -> Allocation:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("allocation evidence is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("allocation evidence must be an object")
        return cls.from_mapping(payload)


def allocations_by_operation_role(
    profile,
    execution_plans: Mapping[str, Sequence[object]],
) -> dict[str, Allocation]:
    """Project prepared unit bindings into one immutable allocation per role."""

    result: dict[str, Allocation] = {}
    for plans in execution_plans.values():
        for planned in plans:
            role = planned.unit.operation_role
            allocation = Allocation.from_resource_class(
                profile.resource_classes[planned.unit.resource_class]
            )
            existing = result.get(role)
            if existing is not None and existing != allocation:
                raise ValueError(
                    f"operation role has contradictory allocation bindings: {role}"
                )
            result[role] = allocation
    return result


def validate_operation_allocation(operation_role: str, allocation: Allocation) -> None:
    """Fail closed on unsupported GPU topology/backend role combinations."""
    validate_operation_resource_shape(
        operation_role,
        gpus_per_node=allocation.gpus_per_node,
        mode=allocation.mode,
        per_gpu=allocation.per_gpu,
    )


__all__ = [
    "Allocation",
    "allocations_by_operation_role",
    "validate_operation_allocation",
]
