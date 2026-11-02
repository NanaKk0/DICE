from __future__ import annotations

import os
from pathlib import Path
from contextlib import nullcontext
import json
import threading
import warnings
from typing import Callable, Sequence, TypeVar

from workflow_backends.execution import (
    ProcessGroupRegistry,
    operation_execution_scope,
)
from workflow_runtime.execution.allocation import validate_operation_allocation
from workflow_runtime.execution.task_scheduler import (
    AllocationTask,
    DeviceAssignment,
    DistributedAssignment,
    execute_allocation_tasks,
)
from workflow_runtime.lifecycle.environment import OperationContext


from workflow_backends.common.training_pool import TrainingPoolCapability
from workflow_backends.common.cancellation import handled_sigterm


TaskValue = TypeVar("TaskValue")
ResultValue = TypeVar("ResultValue")


def execute_committee_members(
    members: Sequence[TaskValue],
    *,
    context: OperationContext | None,
    task_id: Callable[[TaskValue], str],
    execute: Callable[[TaskValue], ResultValue],
    capability: TrainingPoolCapability | None = None,
    diagnostic_root: Path | None = None,
    execute_distributed: Callable[
        [TaskValue, DistributedAssignment], ResultValue
    ] | None = None,
) -> tuple[ResultValue, ...]:
    """Run backend-owned member values through the shared allocation protocol."""

    ordered = tuple(members)
    if not ordered:
        raise ValueError("committee execution requires at least one member")
    if context is None:
        return tuple(execute(member) for member in ordered)

    allocation = context.allocation
    validate_operation_allocation(context.role, allocation)
    if allocation.mode == "distributed":
        if context.role == "teacher.mace.train" and len(ordered) > 1:
            raise ValueError(
                "MACE committee with more than one member does not support distributed mode"
            )
        if context.role == "teacher.mace.train" and execute_distributed is None:
            raise ValueError(
                "MACE distributed mode requires the explicit DDP capability"
            )
        if context.role != "teacher.mace.train":
            raise ValueError(
                f"operation role {context.role!r} does not support distributed mode"
            )
        inventory = (AllocationTask(task_id=task_id(ordered[0]), value=ordered[0]),)
        registry = ProcessGroupRegistry()

        def execute_assigned(
            member: TaskValue,
            assignment: DistributedAssignment,
        ) -> ResultValue:
            try:
                with operation_execution_scope(
                    environment_overrides=assignment.environment_override,
                    process_registry=registry,
                ):
                    if execute_distributed is None:
                        raise AssertionError("distributed MACE capability disappeared")
                    return execute_distributed(member, assignment)
            except BaseException:
                registry.cancel_all()
                raise

        return execute_allocation_tasks(
            inventory,
            allocation=allocation,
            visible_devices=_visible_devices(allocation.gpus_per_node),
            execute=execute_assigned,
            cancel_active=registry.cancel_all,
        )
    sharing = allocation.per_gpu != 1
    if sharing and (capability is None or diagnostic_root is None):
        raise ValueError("committee MPS requires an explicit training capability and diagnostic root")
    if allocation.gpus_per_node <= 1 and not sharing:
        return tuple(execute(member) for member in ordered)

    tasks = tuple(
        AllocationTask(task_id=task_id(member), value=member) for member in ordered
    )
    devices = _visible_devices(allocation.gpus_per_node)
    provider = capability.open_sessions(context, Path(diagnostic_root), devices) if sharing else nullcontext({})
    with handled_sigterm("training"), provider as sessions:
        if sharing and (set(sessions) != set(devices) or any(
            sessions[device].assigned_visible_device != device for device in devices
        )):
            raise ValueError("training session provider returned a mismatched device inventory")
        registry = ProcessGroupRegistry()
        journal = _TrainingJournal(diagnostic_root, tuple(task.task_id for task in tasks)) if sharing else None

        def cancel_all():
            errors = []
            for owned in (registry, *(session.process_registry for session in sessions.values())):
                try:
                    owned.cancel_all()
                except BaseException as exc:
                    errors.append(str(exc))
            if errors:
                raise RuntimeError("training cancellation failed: " + "; ".join(errors))

        def execute_assigned(member: TaskValue, assignment: DeviceAssignment) -> ResultValue:
            environment = assignment.environment_override
            task_registry = registry
            if sharing:
                session = sessions[assignment.visible_device]
                environment.update(session)
                task_registry = session.process_registry
            if journal:
                journal.record(assignment, "running", environment)
            try:
                with operation_execution_scope(environment_overrides=environment, process_registry=task_registry):
                    result = execute(member)
            except BaseException:
                if journal:
                    journal.record(assignment, "failed", environment)
                raise
            if journal:
                journal.record(assignment, "completed", environment)
            return result

        return execute_allocation_tasks(
            tasks, allocation=allocation, visible_devices=devices,
            execute=execute_assigned, cancel_active=cancel_all,
        )


class _TrainingJournal:
    """Bounded per-member transition diagnostics, never completion evidence."""
    def __init__(self, root, task_ids):
        self.path = Path(root) / "backend_logs" / "training_pool.jsonl"
        self.states = dict.fromkeys(task_ids, "queued")
        self.lock = threading.Lock()

    def record(self, assignment, state, environment):
        with self.lock:
            self.states[assignment.task_id] = state
            event = dict(member=assignment.task_id, status=state,
                         mps_pipe=environment.get("CUDA_MPS_PIPE_DIRECTORY"),
                         device=assignment.visible_device, slot=assignment.slot_index,
                         assignment_id=assignment.assignment_id,
                         counts={name: sum(value == name for value in self.states.values())
                                 for name in ("queued", "running", "completed", "failed")})
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
            except OSError as exc:
                warnings.warn(f"training pool diagnostic unavailable: {exc}", RuntimeWarning)


def _visible_devices(expected: int) -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        raise ValueError(
            "multi-GPU pool requires scheduler-provided CUDA_VISIBLE_DEVICES"
        )
    devices = tuple(item.strip() for item in raw.split(","))
    if (
        any(not item or item in {"-1", "NoDevFiles", "void"} for item in devices)
        or len(devices) != expected
        or len(set(devices)) != len(devices)
    ):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES does not match the allocated GPUs per node"
        )
    return devices


__all__ = ["execute_committee_members"]
