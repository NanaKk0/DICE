from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Queue
import threading
from typing import Callable, Generic, Sequence, TypeVar
import uuid

from workflow_runtime.execution.allocation import Allocation


TaskValue = TypeVar("TaskValue")
ResultValue = TypeVar("ResultValue")


@dataclass(frozen=True, slots=True)
class AllocationTask(Generic[TaskValue]):
    task_id: str
    value: TaskValue

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("allocation task id must be non-empty")


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    assignment_id: str
    task_id: str
    logical_gpu: int
    visible_device: str
    slot_index: int

    @property
    def environment_override(self) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": self.visible_device,
            "DICE_GPU_SLOT": f"{self.logical_gpu}:{self.slot_index}",
            "DICE_ASSIGNMENT_ID": self.assignment_id,
        }


@dataclass(frozen=True, slots=True)
class DistributedAssignment:
    assignment_id: str
    task_id: str
    visible_devices: tuple[str, ...]

    @property
    def environment_override(self) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": ",".join(self.visible_devices),
            "DICE_ASSIGNMENT_ID": self.assignment_id,
        }


@dataclass(frozen=True, slots=True)
class PoolCompletionEvent:
    assignment_id: str
    task_id: str
    logical_gpu: int
    slot_index: int
    status: str


class CompletionProtocolError(RuntimeError):
    pass


class PoolDeviceUnavailable(RuntimeError):
    """Admission failed BEFORE native execution; quarantine only this GPU."""


class PoolAssignmentCoordinator:
    """Single-owner pending, assignment and occupancy state machine."""

    def __init__(
        self,
        task_ids: Sequence[str],
        *,
        allocation: Allocation,
        visible_devices: Sequence[str],
    ) -> None:
        if allocation.mode != "pool" or allocation.per_gpu is None:
            raise ValueError("pool coordinator requires a pool allocation")
        ordered = tuple(task_ids)
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("pool coordinator task ids must be non-empty and unique")
        self.allocation = allocation
        self.devices = _validate_devices(visible_devices, allocation.gpus_per_node)
        self._pending = deque(ordered)
        self._states = {task_id: "pending" for task_id in ordered}
        self._occupied = [
            [False for _ in range(allocation.per_gpu)] for _ in self.devices
        ]
        self._active: dict[str, DeviceAssignment] = {}
        self._terminal_assignments: set[str] = set()
        self._admission_closed = False
        self._quarantined: set[int] = set()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def occupancy(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(tuple(slots) for slots in self._occupied)

    @property
    def active_assignments(self) -> tuple[DeviceAssignment, ...]:
        return tuple(self._active.values())

    @property
    def admission_closed(self) -> bool:
        return self._admission_closed

    def stop_admission(self) -> None:
        self._admission_closed = True

    def claim_next(
        self,
        *,
        preferred_slot: tuple[int, int] | None = None,
    ) -> DeviceAssignment | None:
        if self._admission_closed or not self._pending:
            return None
        if preferred_slot is None:
            if not any(not used for gpu, slots in enumerate(self._occupied)
                       if gpu not in self._quarantined for used in slots):
                return None
            gpu, slot = _least_occupied_slot(self._occupied, self._quarantined)
        else:
            gpu, slot = preferred_slot
            if gpu in self._quarantined:
                return self.claim_next()
            if (
                gpu < 0
                or gpu >= len(self._occupied)
                or slot < 0
                or slot >= len(self._occupied[gpu])
                or self._occupied[gpu][slot]
            ):
                raise RuntimeError("preferred pool refill slot is not free")
        task_id = self._pending.popleft()
        if self._states[task_id] != "pending":
            raise RuntimeError("pool coordinator pending task state is contradictory")
        assignment = DeviceAssignment(
            assignment_id=_assignment_id(),
            task_id=task_id,
            logical_gpu=gpu,
            visible_device=self.devices[gpu],
            slot_index=slot,
        )
        self._states[task_id] = "claimed"
        self._occupied[gpu][slot] = True
        self._active[assignment.assignment_id] = assignment
        return assignment

    def complete(self, event: PoolCompletionEvent) -> tuple[int, int]:
        current = self._active.get(event.assignment_id)
        valid_status = event.status in {"succeeded", "failed", "unstarted"}
        matches = current is not None and (
            current.task_id == event.task_id
            and current.logical_gpu == event.logical_gpu
            and current.slot_index == event.slot_index
        )
        if (
            event.assignment_id in self._terminal_assignments
            or not valid_status
            or not matches
        ):
            self._admission_closed = True
            raise CompletionProtocolError(
                "pool completion event is duplicate, stale, unknown, or contradictory"
            )
        if current is None:
            raise AssertionError("validated completion has no active assignment")
        if not self._occupied[current.logical_gpu][current.slot_index]:
            self._admission_closed = True
            raise CompletionProtocolError("pool completion cannot release an empty slot")
        del self._active[event.assignment_id]
        self._terminal_assignments.add(event.assignment_id)
        self._states[event.task_id] = event.status
        self._occupied[current.logical_gpu][current.slot_index] = False
        if event.status == "failed":
            self._admission_closed = True
        elif event.status == "unstarted":
            self._quarantined.add(current.logical_gpu)
            self._states[event.task_id] = 'pending'
            self._pending.appendleft(event.task_id)
        return current.logical_gpu, current.slot_index


@dataclass(frozen=True, slots=True)
class _Completion(Generic[ResultValue]):
    assignment: DeviceAssignment
    result: ResultValue | None
    error: BaseException | None


class PoolTaskError(RuntimeError):
    def __init__(
        self,
        assignment: DeviceAssignment,
        error: BaseException,
        *,
        cancellation_error: BaseException | None = None,
    ):
        cancellation = (
            ""
            if cancellation_error is None
            else "; coordinated cancellation also failed: "
            f"{type(cancellation_error).__name__}: {cancellation_error}"
        )
        super().__init__(
            f"allocation pool task failed: {assignment.task_id} "
            f"({assignment.assignment_id}): {error}{cancellation}"
        )
        self.assignment = assignment
        self.error = error
        self.cancellation_error = cancellation_error


def execute_allocation_tasks(
    tasks: Sequence[AllocationTask[TaskValue]],
    *,
    allocation: Allocation,
    visible_devices: Sequence[str],
    execute: Callable[
        [TaskValue, DeviceAssignment | DistributedAssignment], ResultValue
    ],
    cancel_active: Callable[[], None] | None = None,
) -> tuple[ResultValue, ...]:
    """Dispatch generic values through one closed allocation topology mode."""

    ordered = tuple(tasks)
    _validate_tasks(ordered)
    devices = _validate_devices(visible_devices, allocation.gpus_per_node)
    if allocation.mode == "distributed":
        if len(ordered) != 1:
            raise ValueError("distributed allocation requires exactly one task")
        task = ordered[0]
        assignment = DistributedAssignment(
            assignment_id=_assignment_id(),
            task_id=task.task_id,
            visible_devices=devices,
        )
        return (execute(task.value, assignment),)
    if allocation.mode != "pool":
        raise ValueError(f"allocation mode is unsupported: {allocation.mode!r}")
    if allocation.capacity <= 0:
        raise ValueError("pool allocation requires at least one GPU slot")
    return _execute_pool(
        ordered,
        allocation,
        devices,
        execute,
        cancel_active=cancel_active,
    )


def _execute_pool(
    tasks: tuple[AllocationTask[TaskValue], ...],
    allocation: Allocation,
    devices: tuple[str, ...],
    execute: Callable[
        [TaskValue, DeviceAssignment | DistributedAssignment], ResultValue
    ],
    *,
    cancel_active: Callable[[], None] | None,
) -> tuple[ResultValue, ...]:
    by_id = {task.task_id: (index, task) for index, task in enumerate(tasks)}
    coordinator = PoolAssignmentCoordinator(
        tuple(by_id),
        allocation=allocation,
        visible_devices=devices,
    )
    completions: Queue[_Completion[ResultValue]] = Queue()
    threads: list[threading.Thread] = []
    results: list[ResultValue | None] = [None] * len(tasks)
    first_failure: tuple[DeviceAssignment, BaseException] | None = None
    cancellation_error: BaseException | None = None
    cancellation_requested = False
    unavailable_devices = []

    def request_cancellation() -> None:
        nonlocal cancellation_error, cancellation_requested
        if cancellation_requested or cancel_active is None:
            return
        cancellation_requested = True
        try:
            cancel_active()
        except BaseException as exc:
            cancellation_error = exc

    def start(assignment: DeviceAssignment) -> None:
        _index, task = by_id[assignment.task_id]

        def worker() -> None:
            try:
                result = execute(task.value, assignment)
            except BaseException as exc:
                completions.put(_Completion(assignment, None, exc))
            else:
                completions.put(_Completion(assignment, result, None))

        thread = threading.Thread(
            target=worker,
            name=f"dice-allocation-{task.task_id}",
        )
        threads.append(thread)
        thread.start()

    try:
        while coordinator.active_count < allocation.capacity:
            assignment = coordinator.claim_next()
            if assignment is None:
                break
            start(assignment)

        while coordinator.active_count:
            completion = completions.get()
            assignment = completion.assignment
            unavailable = isinstance(completion.error, PoolDeviceUnavailable)
            status = 'unstarted' if unavailable else "failed" if completion.error is not None else "succeeded"
            try:
                released = coordinator.complete(
                    PoolCompletionEvent(
                        assignment_id=assignment.assignment_id,
                        task_id=assignment.task_id,
                        logical_gpu=assignment.logical_gpu,
                        slot_index=assignment.slot_index,
                        status=status,
                    )
                )
            except CompletionProtocolError as exc:
                first_failure = first_failure or (
                    assignment,
                    exc,
                )
                request_cancellation()
                continue
            index, _task = by_id[assignment.task_id]
            if unavailable:
                unavailable_devices.append((assignment.visible_device, str(completion.error)))
            elif completion.error is not None:
                first_failure = first_failure or (assignment, completion.error)
                coordinator.stop_admission()
                request_cancellation()
            else:
                results[index] = completion.result
            if first_failure is None:
                next_assignment = coordinator.claim_next(preferred_slot=released)
                if next_assignment is not None:
                    start(next_assignment)

    except BaseException:
        coordinator.stop_admission()
        request_cancellation()
        raise
    finally:
        for thread in threads:
            thread.join()
    if any(any(slots) for slots in coordinator.occupancy) or coordinator.active_count:
        raise RuntimeError("allocation pool leaked active assignment capacity")
    if first_failure is not None:
        if not isinstance(first_failure[1], Exception):
            raise first_failure[1]
        error = PoolTaskError(
            *first_failure,
            cancellation_error=cancellation_error,
        )
        raise error from first_failure[1]
    if coordinator.pending_count:
        raise RuntimeError(f"allocation pool stopped with {coordinator.pending_count} unclaimed tasks; quarantined devices: {unavailable_devices}")
    if unavailable_devices:
        raise RuntimeError(f'allocation pool drained healthy devices after quarantine: {unavailable_devices}')
    return tuple(results)  # type: ignore[return-value]


def _least_occupied_slot(occupied: list[list[bool]], excluded=()) -> tuple[int, int]:
    candidates = []
    for gpu, slots in enumerate(occupied):
        if gpu in excluded:
            continue
        count = sum(slots)
        for slot, in_use in enumerate(slots):
            if not in_use:
                candidates.append((count, gpu, slot))
    if not candidates:
        raise RuntimeError("allocation pool has no free slot")
    _, gpu, slot = min(candidates)
    return gpu, slot


def _validate_tasks(tasks: tuple[AllocationTask[TaskValue], ...]) -> None:
    if not tasks:
        raise ValueError("allocation execution requires at least one task")
    ids = tuple(task.task_id for task in tasks)
    if len(set(ids)) != len(ids):
        raise ValueError("allocation task ids must be unique")


def _validate_devices(values: Sequence[str], expected: int) -> tuple[str, ...]:
    devices = tuple(values)
    if (
        len(devices) != expected
        or len(set(devices)) != len(devices)
        or any(not isinstance(device, str) or not device for device in devices)
    ):
        raise ValueError("allocation visible devices do not match gpus_per_node")
    return devices


def _assignment_id() -> str:
    return "assignment-" + uuid.uuid4().hex


__all__ = [
    "AllocationTask",
    "DeviceAssignment",
    "DistributedAssignment",
    "PoolTaskError",
    "execute_allocation_tasks",
]
