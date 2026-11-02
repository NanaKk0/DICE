from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Mapping, Protocol, Sequence

from workflow_runtime.execution.controller import DurableExecutionController
from workflow_runtime.execution.planning import PlannedExecution
from workflow_runtime.execution.protocol import (
    ExecutionAdapter,
    ExternalJobReceipt,
    PreparedExecution,
    TERMINAL_STATES,
)
from workflow_runtime.execution.store import ExecutionStore


class RuntimeExecutionConfig(Protocol):
    """Minimum launch configuration needed by durable execution."""

    iter_index: int


class RuntimeExecutionLaunch(Protocol):
    """Structural boundary accepted from the outer lifecycle composition."""

    run_root: Path
    config: RuntimeExecutionConfig
    execution_profile: object | None
    execution_plans: Mapping[str, Sequence[PlannedExecution]]


@dataclass(frozen=True, slots=True)
class LifecycleExecutionResult:
    receipt: ExternalJobReceipt
    destination: Path
    validation_evidence: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class PlannedUnitWork:
    prepared: PreparedExecution
    destination: Path
    expected_output_inventory: Mapping[str, str]
    backend_validator: Callable[[Path], bool]
    canonical_validator: Callable[[Path], bool]


def execute_planned_block(
    launch: RuntimeExecutionLaunch,
    *,
    block_id: str,
    work_by_task: Mapping[str, PlannedUnitWork],
    adapters: Mapping[str, ExecutionAdapter],
    poll_interval_seconds: float = 1.0,
) -> dict[str, LifecycleExecutionResult]:
    """Execute and validate the complete preplanned block inventory.

    Revision 1 deliberately uses one immutable unit per adapter submission.
    Deterministic target pinning is established by planning and never recomputed
    from the unfinished subset.
    """
    try:
        plans = launch.execution_plans[block_id]
    except KeyError as exc:
        raise ValueError(f"lifecycle block has no execution plan: {block_id!r}") from exc
    expected_tasks = {plan.unit.task_id for plan in plans}
    if set(work_by_task) != expected_tasks:
        raise ValueError("lifecycle block execution requires exact task inventory coverage")
    results: dict[str, LifecycleExecutionResult] = {}
    for planned in plans:
        task_id = planned.unit.task_id
        work = work_by_task[task_id]
        results[task_id] = execute_planned_unit(
            launch,
            block_id=block_id,
            task_id=task_id,
            prepared=work.prepared,
            adapters=adapters,
            destination=work.destination,
            expected_output_inventory=work.expected_output_inventory,
            backend_validator=work.backend_validator,
            canonical_validator=work.canonical_validator,
            poll_interval_seconds=poll_interval_seconds,
        )
    return results


def execute_planned_unit(
    launch: RuntimeExecutionLaunch,
    *,
    block_id: str,
    task_id: str,
    prepared: PreparedExecution,
    adapters: Mapping[str, ExecutionAdapter],
    destination: str | Path,
    expected_output_inventory: Mapping[str, str],
    backend_validator: Callable[[Path], bool],
    canonical_validator: Callable[[Path], bool],
    poll_interval_seconds: float = 1.0,
) -> LifecycleExecutionResult:
    """Run one preplanned unit without taking lifecycle publication authority.

    Backends prepare commands and validate native outputs. The selected unit,
    durable attempt evidence and exact collection remain runtime-owned; the
    outer lifecycle keeps authority over ordinary block publication.
    """
    if launch.execution_profile is None:
        raise ValueError("lifecycle execution bridge requires a selected execution profile")
    if poll_interval_seconds <= 0:
        raise ValueError("execution poll interval must be positive")
    try:
        plans = launch.execution_plans[block_id]
    except KeyError as exc:
        raise ValueError(f"lifecycle block has no execution plan: {block_id!r}") from exc
    matching = tuple(plan for plan in plans if plan.unit.task_id == task_id)
    if len(matching) != 1:
        raise ValueError(
            f"lifecycle execution unit selection is not exact for {block_id!r}/{task_id!r}"
        )
    planned = matching[0]
    controller = DurableExecutionController(
        store=ExecutionStore(launch.run_root, iter_index=launch.config.iter_index),
        adapters=adapters,
    )
    receipt = controller.submit(planned, prepared=prepared)
    deadline = time.monotonic() + float(planned.unit.resource_shape["wall_time_seconds"])
    polled = False
    while receipt.state not in TERMINAL_STATES:
        if time.monotonic() >= deadline:
            controller.cancel(planned)
            raise TimeoutError(
                f"lifecycle execution exceeded its wall-time cap: {planned.unit.unit_id}"
            )
        receipt = controller.poll(planned)
        polled = True
        if receipt.state not in TERMINAL_STATES:
            time.sleep(poll_interval_seconds)
    if receipt.state != "succeeded":
        raise RuntimeError(
            f"lifecycle execution did not succeed: {planned.unit.unit_id} ({receipt.state})"
        )
    # A synchronous adapter can return terminal state from submit; publish its
    # terminal receipt through the ordinary poll path exactly once.
    if not polled:
        receipt = controller.poll(planned)
    output = controller.collect_and_validate(
        planned,
        destination=destination,
        expected_output_inventory=expected_output_inventory,
        backend_validator=backend_validator,
        canonical_validator=canonical_validator,
    )
    return LifecycleExecutionResult(receipt=receipt, destination=output)


__all__ = [
    "LifecycleExecutionResult",
    "PlannedUnitWork",
    "RuntimeExecutionConfig",
    "RuntimeExecutionLaunch",
    "execute_planned_block",
    "execute_planned_unit",
]
