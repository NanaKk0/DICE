from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable

from workflow_runtime.execution.profile import ExecutionProfile, ResourceClass


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class ExecutionUnit:
    unit_id: str
    block_id: str
    operation_role: str
    task_id: str
    resource_class: str
    launcher: str
    resource_shape: dict[str, object]


@dataclass(frozen=True, slots=True)
class PlannedExecution:
    unit: ExecutionUnit
    target_id: str
    target_pool: str
    pool_digest: str
    profile_digest: str
    attempt_id: str


def plan_execution_units(
    *,
    block_id: str,
    operation_role: str,
    resource_class: str,
    task_ids: Iterable[str],
    profile: ExecutionProfile,
    attempt_id: str,
    unavailable_targets: set[str] | None = None,
) -> tuple[PlannedExecution, ...]:
    """Bind a complete canonical unit inventory to a stable ordered pool.

    ``unavailable_targets`` is deliberately admission-only evidence.  It is
    accepted here so callers cannot accidentally implement availability-based
    routing; it never participates in the assignment.
    """
    del unavailable_targets
    if not isinstance(block_id, str) or not block_id or block_id.startswith("/"):
        raise ValueError("execution block id is invalid")
    if not isinstance(operation_role, str) or not operation_role:
        raise ValueError("execution operation role is invalid")
    if not isinstance(attempt_id, str) or not _SAFE_COMPONENT.fullmatch(attempt_id):
        raise ValueError("execution attempt id is invalid")
    ordered_task_ids = tuple(task_ids)
    if not ordered_task_ids or len(set(ordered_task_ids)) != len(ordered_task_ids):
        raise ValueError("execution task inventory must be non-empty and unique")
    for task_id in ordered_task_ids:
        if not isinstance(task_id, str) or not _SAFE_COMPONENT.fullmatch(task_id):
            raise ValueError(f"execution task id is invalid: {task_id!r}")

    binding = profile.binding_for(operation_role, resource_class)
    resource = profile.resource_classes[resource_class]
    pool = profile.target_pools[binding.target_pool]
    units = sorted(
        (
            _execution_unit(
                block_id=block_id,
                operation_role=operation_role,
                task_id=task_id,
                resource=resource,
            )
            for task_id in ordered_task_ids
        ),
        key=lambda unit: unit.unit_id,
    )
    pool_digest = _digest(
        {"pool_id": pool.pool_id, "targets": list(pool.targets)}
    )
    return tuple(
        PlannedExecution(
            unit=unit,
            target_id=pool.targets[index % len(pool.targets)],
            target_pool=pool.pool_id,
            pool_digest=pool_digest,
            profile_digest=profile.digest,
            attempt_id=attempt_id,
        )
        for index, unit in enumerate(units)
    )


def plan_explicit_reroute(
    old_plans: Iterable[PlannedExecution],
    *,
    old_states: dict[str, str],
    new_profile: ExecutionProfile,
    new_attempt_id: str,
) -> tuple[PlannedExecution, ...]:
    old = tuple(old_plans)
    if not old:
        raise ValueError("explicit execution reroute requires the old complete inventory")
    allowed = {"succeeded", "failed", "cancelled", "never_submitted"}
    unit_ids = {item.unit.unit_id for item in old}
    if set(old_states) != unit_ids:
        raise ValueError("explicit execution reroute requires exact old-unit state coverage")
    unsafe = sorted(
        unit_id for unit_id, state in old_states.items() if state not in allowed
    )
    if unsafe:
        raise ValueError(
            "old execution unit is not exactly terminal or never submitted: "
            f"{unsafe[0]}"
        )
    coordinates = {
        (item.unit.block_id, item.unit.operation_role, item.unit.resource_class)
        for item in old
    }
    if len(coordinates) != 1:
        raise ValueError("explicit execution reroute requires one homogeneous unit inventory")
    block_id, operation_role, resource_class = next(iter(coordinates))
    return plan_execution_units(
        block_id=block_id,
        operation_role=operation_role,
        resource_class=resource_class,
        task_ids=tuple(item.unit.task_id for item in old),
        profile=new_profile,
        attempt_id=new_attempt_id,
    )


def _execution_unit(
    *,
    block_id: str,
    operation_role: str,
    task_id: str,
    resource: ResourceClass,
) -> ExecutionUnit:
    shape = {
        "nodes": resource.nodes,
        "processes_per_node": resource.processes_per_node,
        "gpus_per_node": resource.gpus_per_node,
        "cpus_per_process": resource.cpus_per_process,
        "memory_mb": resource.memory_mb,
        "wall_time_seconds": resource.wall_time_seconds,
        "mode": resource.mode,
        "per_gpu": resource.per_gpu,
    }
    identity = {
        "schema": "dice.execution-unit",
        "schema_revision": 1,
        "block_id": block_id,
        "operation_role": operation_role,
        "task_id": task_id,
        "resource_class": resource.resource_class_id,
        "launcher": resource.launcher.kind,
        "resource_shape": shape,
    }
    return ExecutionUnit(
        unit_id=f"unit-{_digest(identity)}",
        block_id=block_id,
        operation_role=operation_role,
        task_id=task_id,
        resource_class=resource.resource_class_id,
        launcher=resource.launcher.kind,
        resource_shape=shape,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
