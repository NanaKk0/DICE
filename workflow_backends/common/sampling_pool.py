from __future__ import annotations

from contextlib import nullcontext
import os
from threading import Lock
from typing import Callable, Mapping, Sequence, TypeVar

from workflow_backends.execution import (
    ProcessGroupRegistry,
    operation_execution_scope,
)
from workflow_runtime.execution.allocation import validate_operation_allocation
from workflow_runtime.execution.task_scheduler import (
    AllocationTask,
    DeviceAssignment,
    PoolDeviceUnavailable,
    execute_allocation_tasks,
)
from workflow_runtime.lifecycle.environment import OperationContext
from workflow_backends.common.device_session import DeviceSessionSupervisor
from workflow_backends.common.device_fault import fault_reporter


TaskValue = TypeVar("TaskValue")
ResultValue = TypeVar("ResultValue")


from workflow_backends.common.device_environment import DeviceSessionEnvironment


def execute_sampling_tasks(
    tasks: Sequence[TaskValue],
    *,
    context: OperationContext | None,
    task_id: Callable[[TaskValue], str],
    execute: Callable[[TaskValue], ResultValue],
    allow_per_gpu: bool = False,
    assignment_environment: Callable[
        [DeviceAssignment], Mapping[str, str]
    ] | None = None,
) -> tuple[ResultValue, ...]:
    """Dispatch backend-owned sampling tasks through the shared GPU pool."""

    ordered = tuple(tasks)
    if not ordered:
        raise ValueError("sampling pool requires at least one task")
    if context is None:
        return tuple(execute(task) for task in ordered)

    allocation = context.allocation
    validate_operation_allocation(context.role, allocation)
    if allocation.mode == "distributed":
        raise ValueError(
            f"operation role {context.role!r} does not support distributed mode"
        )
    if allocation.per_gpu != 1 and not allow_per_gpu:
        if context.role.startswith("sampler.lammps."):
            raise ValueError(
                "LAMMPS pool per_gpu > 1 requires an MPS device-session capability"
            )
        raise ValueError(
            f"sampling operation role {context.role!r} supports pool per_gpu 1 only"
        )
    # A single physical GPU can still expose multiple pool slots when the
    # backend owns an MPS session (LAMMPS `per_gpu > 1`).  Do not collapse
    # that topology into the serial fast path.
    if assignment_environment is None and allocation.gpus_per_node <= 1 and (
        not allow_per_gpu or allocation.per_gpu == 1
    ):
        return tuple(execute(task) for task in ordered)

    registry = ProcessGroupRegistry()
    session_registries = set()
    session_lock = Lock()
    inventory = tuple(
        AllocationTask(task_id=task_id(task), value=task) for task in ordered
    )

    def execute_assigned(task: TaskValue, assignment: DeviceAssignment) -> ResultValue:
        session = assignment_environment(assignment) if assignment_environment else None
        gate = session.admit() if isinstance(session, DeviceSessionSupervisor) else nullcontext(session)
        # Quarantined-device tasks remain unpublished. Drain other GPUs before
        # reporting the incomplete stage, rather than globally killing siblings.
        try:
            current = gate.__enter__()
        except RuntimeError as exc:
            if not isinstance(session, DeviceSessionSupervisor):
                raise
            raise PoolDeviceUnavailable(str(exc)) from exc
        reporter = (lambda kind, detail: session.report(kind, f'{task_id(task)}: {detail}')) if isinstance(session, DeviceSessionSupervisor) else None
        token = fault_reporter.set(reporter)
        try:
            return execute_in_session(task, assignment, current)
        finally:
            fault_reporter.reset(token)
            gate.__exit__(None, None, None)

    def execute_in_session(task, assignment, session_environment):
        environment = assignment.environment_override
        task_registry = registry
        if session_environment is not None:
            extra = dict(session_environment)
            if "CUDA_VISIBLE_DEVICES" in extra:
                if not isinstance(session_environment, DeviceSessionEnvironment):
                    raise ValueError(
                        "pool device session must not replace assignment device visibility"
                    )
                if (
                    session_environment.assigned_visible_device
                    != assignment.visible_device
                ):
                    raise ValueError(
                        "pool device session does not own its physical assignment device"
                    )
            environment.update(extra)
            if isinstance(session_environment, DeviceSessionEnvironment):
                task_registry = session_environment.process_registry
                with session_lock:
                    session_registries.add(task_registry)
        with operation_execution_scope(
            environment_overrides=environment,
            process_registry=task_registry,
        ):
            return execute(task)

    def cancel_all():
        registry.cancel_all()
        with session_lock:
            owned = tuple(session_registries)
        for owned_registry in owned:
            owned_registry.cancel_all()

    results = execute_allocation_tasks(
        inventory,
        allocation=allocation,
        visible_devices=sampling_visible_devices(allocation.gpus_per_node),
        execute=execute_assigned,
        cancel_active=cancel_all,
    )
    return results


def sampling_visible_devices(expected: int) -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        if expected == 1:
            return ("0",)
        raise ValueError(
            "multi-GPU sampling pool requires scheduler-provided CUDA_VISIBLE_DEVICES"
        )
    devices = tuple(item.strip() for item in raw.split(","))
    if (
        any(not item for item in devices)
        or len(devices) != expected
        or len(set(devices)) != len(devices)
    ):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES does not match the allocated GPUs per node"
        )
    return devices


__all__ = [
    "DeviceSessionEnvironment",
    "execute_sampling_tasks",
    "sampling_visible_devices",
]
