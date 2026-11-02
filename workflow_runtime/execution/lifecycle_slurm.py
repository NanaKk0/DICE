from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Protocol

from workflow_runtime.execution.lifecycle import (
    LifecycleExecutionResult,
    execute_planned_unit,
)
from workflow_runtime.execution.allocation import Allocation
from workflow_runtime.execution.planning import PlannedExecution, plan_execution_units
from workflow_runtime.execution.protocol import PreparedExecution
from workflow_runtime.execution.store import ExecutionStore
from workflow_runtime.execution.slurm import (
    NativeSlurmCommands,
    SlurmCommands,
    SlurmExecutionAdapter,
)


_WORKER_RECEIPT = "scheduled-worker-receipt.json"


class SlurmLifecycleLaunch(Protocol):
    run_root: Path
    config: object
    environment_path: Path | None
    execution_profile: object | None
    execution_plans: dict[str, tuple[PlannedExecution, ...]]
    plan: object


def is_slurm_lifecycle_block(launch: SlurmLifecycleLaunch, block_id: str) -> bool:
    """Return whether one prepared block is bound wholly to Slurm."""

    if launch.execution_profile is None:
        return False
    plans = launch.execution_plans.get(block_id, ())
    if not plans:
        return False
    return all(
        launch.execution_profile.targets[plan.target_id].adapter == "slurm"
        for plan in plans
    )


def execute_slurm_lifecycle_block(
    launch: SlurmLifecycleLaunch,
    *,
    block_id: str,
    canonical_validator: Callable[[Path], bool],
    commands: SlurmCommands | None = None,
    poll_interval_seconds: float = 1.0,
) -> LifecycleExecutionResult:
    """Run one public block as one durable shared-filesystem Slurm job."""
    planned = _select_retry_attempt(launch, _block_plan(launch, block_id))
    prepared, expected = _worker_command(
        launch,
        block_id,
        planned,
    )
    adapter = SlurmExecutionAdapter(
        target=launch.execution_profile.targets[planned.target_id],
        commands=commands if commands is not None else NativeSlurmCommands(),
    )
    destination = (
        Path(launch.run_root)
        / ".dice_state"
        / "executions"
        / "collected-workers"
        / planned.unit.unit_id
        / planned.attempt_id
    )
    result = execute_planned_unit(
        _BlockLaunch(launch, block_id, planned),
        block_id=block_id,
        task_id=planned.unit.task_id,
        prepared=prepared,
        adapters={planned.target_id: adapter},
        destination=destination,
        expected_output_inventory={
            _WORKER_RECEIPT: hashlib.sha256(expected).hexdigest()
        },
        backend_validator=lambda root: _validate_worker_receipt(root, block_id),
        canonical_validator=canonical_validator,
        poll_interval_seconds=poll_interval_seconds,
    )
    run_root = Path(launch.run_root).resolve()
    validation_path = (
        ExecutionStore(run_root, iter_index=launch.config.iter_index)
        .attempt_root(planned)
        / "validation_receipt.json"
    )
    validation_evidence = {
        "block": block_id,
        "unit_id": result.receipt.unit_id,
        "attempt_id": result.receipt.attempt_id,
        "external_id": result.receipt.external_id,
        "validation_receipt_path": validation_path.relative_to(run_root).as_posix(),
        "validation_receipt_digest": hashlib.sha256(
            validation_path.read_bytes()
        ).hexdigest(),
    }
    return LifecycleExecutionResult(
        receipt=result.receipt,
        destination=result.destination,
        validation_evidence=validation_evidence,
    )


class _BlockLaunch:
    def __init__(
        self,
        launch: SlurmLifecycleLaunch,
        block_id: str,
        planned: PlannedExecution,
    ) -> None:
        self.run_root = launch.run_root
        self.config = launch.config
        self.execution_profile = launch.execution_profile
        self.execution_plans = {block_id: (planned,)}


def _block_plan(
    launch: SlurmLifecycleLaunch,
    block_id: str,
) -> PlannedExecution:
    if launch.execution_profile is None:
        raise ValueError("Slurm lifecycle execution requires an execution profile")
    plans = tuple(launch.execution_plans.get(block_id, ()))
    if not plans:
        raise ValueError(f"public lifecycle block has no Slurm binding: {block_id}")
    coordinates = {
        (
            plan.unit.operation_role,
            plan.unit.resource_class,
            plan.target_pool,
            plan.target_id,
            plan.attempt_id,
        )
        for plan in plans
    }
    if len(coordinates) != 1:
        raise ValueError(
            "public lifecycle block requires one unambiguous Slurm allocation binding"
        )
    operation_role, resource_class, _pool, target_id, attempt_id = next(
        iter(coordinates)
    )
    resource = launch.execution_profile.resource_classes[resource_class]
    if resource.mode == "distributed" and len(plans) != 1:
        raise ValueError(
            "public lifecycle distributed allocation requires exactly one task"
        )
    target = launch.execution_profile.targets[target_id]
    if target.adapter != "slurm":
        raise ValueError("public lifecycle block is not bound to a Slurm target")
    planned = plan_execution_units(
        block_id=block_id,
        operation_role=operation_role,
        resource_class=resource_class,
        task_ids=("public-block",),
        profile=launch.execution_profile,
        attempt_id=attempt_id,
    )
    if len(planned) != 1 or planned[0].target_id != target_id:
        raise ValueError("public lifecycle block Slurm target binding is ambiguous")
    return planned[0]

def _select_retry_attempt(
    launch: SlurmLifecycleLaunch,
    planned: PlannedExecution,
) -> PlannedExecution:
    """Select a fresh durable attempt after an exactly terminal failure.

    The execution unit remains stable, while every scheduler submission keeps
    its own attempt identity. An unfinished or successful attempt is always
    adopted; only an exact failed/cancelled terminal receipt advances the
    deterministic retry suffix.
    """
    store = ExecutionStore(launch.run_root, iter_index=launch.config.iter_index)
    base_attempt_id = planned.attempt_id
    candidate = planned
    for retry_number in range(1, 1001):
        receipt = store.read(candidate, "external_job.json")
        if receipt is None:
            return candidate
        if (
            receipt.get("unit_id") != candidate.unit.unit_id
            or receipt.get("attempt_id") != candidate.attempt_id
            or receipt.get("target_id") != candidate.target_id
        ):
            raise ValueError("durable execution retry receipt identity is invalid")
        state = receipt.get("state")
        if state not in {"failed", "cancelled"}:
            return candidate
        terminal = store.read(candidate, "terminal_receipt.json")
        if terminal != receipt:
            raise ValueError("durable execution retry requires exact terminal evidence")
        candidate = replace(
            planned,
            attempt_id=f"{base_attempt_id}-retry-{retry_number}",
        )
    raise RuntimeError("durable execution retry attempt limit exceeded")


def _worker_command(
    launch: SlurmLifecycleLaunch,
    block_id: str,
    planned: PlannedExecution,
) -> tuple[PreparedExecution, bytes]:
    run_root = Path(launch.run_root).resolve()
    run_json = Path(launch.config.path).resolve()
    try:
        relative_run = run_json.relative_to(run_root).as_posix()
    except ValueError as exc:
        raise ValueError("scheduled worker run.json must be inside the shared run root") from exc
    payload = run_json.read_bytes()
    argv = [
        "env",
        "DICE_SCHEDULED_WORKER=1",
        "DICE_EXECUTION=",
        "DICE_ALLOCATION_JSON="
        + Allocation.from_resource_class(
            launch.execution_profile.resource_classes[planned.unit.resource_class]
        ).to_json(),
        sys.executable,
        "-m",
        "workflow_app.scheduled_worker",
        str(run_json),
    ]
    if launch.environment_path is not None:
        argv.extend(("--environment", str(Path(launch.environment_path).resolve())))
    argv.extend(("--scheduled-worker", block_id))
    expected = (
        json.dumps(
            {
                "schema": "dice.scheduled-worker-receipt",
                "schema_revision": 1,
                "block": block_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return (
        PreparedExecution(
            source_root=run_root,
            input_inventory={relative_run: hashlib.sha256(payload).hexdigest()},
            argv=tuple(argv),
            output_paths=(_WORKER_RECEIPT,),
        ),
        expected,
    )


def _validate_worker_receipt(root: Path, block_id: str) -> bool:
    try:
        value = json.loads((Path(root) / _WORKER_RECEIPT).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value == {
        "schema": "dice.scheduled-worker-receipt",
        "schema_revision": 1,
        "block": block_id,
    }


__all__ = [
    "execute_slurm_lifecycle_block",
    "is_slurm_lifecycle_block",
]
