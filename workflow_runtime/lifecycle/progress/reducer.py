from __future__ import annotations

from dataclasses import dataclass, replace

from .types import (
    AttemptStarted,
    BlockFailed,
    BlockInterrupted,
    BlockSummary,
    BlockStarted,
    BlockStatus,
    BlockSucceeded,
    DiagnosticArchived,
    DiagnosticPruned,
    CountChainSnapshot,
    CountFlowSnapshot,
    DatasetSnapshot,
    DescriptorSnapshot,
    GenericWorkSnapshot,
    HeartbeatReported,
    LabelingSnapshot,
    PhaseEntered,
    ProgressEnvelope,
    ProgressSnapshot,
    SnapshotReported,
    ScoreObservationSnapshot,
    ScorecardSnapshot,
    RoutingSnapshot,
    SamplingSnapshot,
    TrainingSnapshot,
)


@dataclass(frozen=True, slots=True)
class BlockProjection:
    block_id: str
    status: BlockStatus
    phase: str | None = None
    active_operation_id: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    snapshots: tuple[ProgressSnapshot, ...] = ()
    summary: BlockSummary | None = None
    message: str | None = None
    diagnostic_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ProgressProjection:
    iter_index: int
    attempt_id: str
    started_at: str
    updated_at: str
    origin: str
    blocks: tuple[BlockProjection, ...]
    last_sequence: int = 0

    def block(self, block_id: str) -> BlockProjection:
        for item in self.blocks:
            if item.block_id == block_id:
                return item
        raise ValueError(f"unknown progress block {block_id!r}")


def initial_projection(
    *,
    iter_index: int,
    plan: tuple[str, ...],
    completed: tuple[str, ...],
    ready: str | None,
    attempt_id: str,
    started_at: str,
) -> ProgressProjection:
    if len(set(plan)) != len(plan):
        raise ValueError("progress plan contains duplicate blocks")
    completed_set = set(completed)
    if any(block not in plan for block in completed_set):
        raise ValueError("completed progress block is outside the plan")
    if ready is not None and ready not in plan:
        raise ValueError("ready progress block is outside the plan")
    items = tuple(
        BlockProjection(
            block_id=block,
            status=(
                BlockStatus.SUCCESS
                if block in completed_set
                else BlockStatus.READY
                if block == ready
                else BlockStatus.WAITING
            ),
        )
        for block in plan
    )
    return ProgressProjection(
        iter_index=iter_index,
        attempt_id=attempt_id,
        started_at=started_at,
        updated_at=started_at,
        origin="reconciled",
        blocks=items,
    )


def reduce_progress(state: ProgressProjection, event: ProgressEnvelope) -> ProgressProjection:
    """Purely project one typed process-local fact into the current iter view."""
    if event.iter_index != state.iter_index or event.attempt_id != state.attempt_id:
        raise ValueError("progress event context does not match projection")
    if event.sequence <= state.last_sequence:
        raise ValueError("progress event sequence must increase monotonically")
    current = state.block(event.block)
    payload = event.payload
    origin = state.origin

    if isinstance(payload, (AttemptStarted, DiagnosticArchived, DiagnosticPruned)):
        updated = current
        if isinstance(payload, AttemptStarted):
            origin = f"resume:{payload.resume_target}" if payload.resume_target else "reconciled"
    elif isinstance(payload, BlockStarted):
        if current.status is not BlockStatus.READY:
            raise ValueError(f"cannot start progress block in {current.status.value} state")
        if any(item.status is BlockStatus.RUNNING for item in state.blocks):
            raise ValueError("only one progress block may be current")
        updated = replace(
            current,
            status=BlockStatus.RUNNING,
            started_at=event.received_at,
            updated_at=event.received_at,
            phase=event.phase,
        )
    elif isinstance(payload, PhaseEntered):
        _require_running(current)
        phase_changed = (current.phase, current.active_operation_id) != (
            payload.phase,
            payload.operation_id,
        )
        updated = replace(
            current,
            phase=payload.phase,
            active_operation_id=payload.operation_id,
            updated_at=event.received_at,
            snapshots=(
                tuple(
                    snapshot
                    for snapshot in current.snapshots
                    if isinstance(snapshot, DatasetSnapshot)
                )
                if phase_changed
                else current.snapshots
            ),
        )
    elif isinstance(payload, SnapshotReported):
        _require_running(current)
        if current.active_operation_id is not None and event.operation_id != current.active_operation_id:
            raise ValueError(f"progress update belongs to closed operation {event.operation_id!r}")
        updated = replace(
            current,
            snapshots=_replace_snapshot(current.snapshots, payload.snapshot),
            updated_at=event.received_at,
        )
    elif isinstance(payload, HeartbeatReported):
        _require_running(current)
        if current.active_operation_id is not None and event.operation_id != current.active_operation_id:
            raise ValueError(f"progress heartbeat belongs to closed operation {event.operation_id!r}")
        updated = replace(current, updated_at=event.received_at)
    elif isinstance(payload, BlockSucceeded):
        _require_running(current)
        updated = replace(
            current,
            status=BlockStatus.SUCCESS,
            phase=None,
            active_operation_id=None,
            updated_at=event.received_at,
            summary=payload.summary,
        )
    elif isinstance(payload, (BlockFailed, BlockInterrupted)):
        _require_running(current)
        updated = replace(
            current,
            status=BlockStatus.FAILED if isinstance(payload, BlockFailed) else BlockStatus.INTERRUPTED,
            active_operation_id=None,
            updated_at=event.received_at,
            message=payload.message,
            diagnostic_ref=payload.diagnostic_ref,
        )
    else:  # pragma: no cover - closed union guard
        raise TypeError(f"unsupported progress payload {type(payload)!r}")

    blocks = tuple(updated if item.block_id == updated.block_id else item for item in state.blocks)
    if isinstance(payload, BlockSucceeded):
        blocks = _ready_next(blocks, updated.block_id)
    return replace(
        state,
        blocks=blocks,
        last_sequence=event.sequence,
        updated_at=event.received_at,
        origin=origin,
    )


def _replace_snapshot(
    snapshots: tuple[ProgressSnapshot, ...], snapshot: ProgressSnapshot
) -> tuple[ProgressSnapshot, ...]:
    key = _snapshot_key(snapshot)
    retained = tuple(item for item in snapshots if _snapshot_key(item) != key)
    return (*retained, snapshot)


def _snapshot_key(snapshot: ProgressSnapshot) -> tuple[type[object], str]:
    if isinstance(snapshot, GenericWorkSnapshot):
        return type(snapshot), snapshot.activity
    if isinstance(snapshot, DatasetSnapshot):
        return type(snapshot), snapshot.label
    if isinstance(snapshot, CountChainSnapshot):
        return type(snapshot), snapshot.label
    if isinstance(snapshot, CountFlowSnapshot):
        return type(snapshot), snapshot.title
    if isinstance(snapshot, TrainingSnapshot):
        return type(snapshot), snapshot.member_id
    if isinstance(snapshot, LabelingSnapshot):
        return type(snapshot), "labeling"
    if isinstance(snapshot, ScoreObservationSnapshot):
        return type(snapshot), snapshot.role
    if isinstance(snapshot, ScorecardSnapshot):
        return type(snapshot), snapshot.role
    if isinstance(snapshot, RoutingSnapshot):
        return type(snapshot), "routing"
    if isinstance(snapshot, SamplingSnapshot):
        return type(snapshot), "sampling"
    if isinstance(snapshot, DescriptorSnapshot):
        return type(snapshot), snapshot.label
    raise TypeError(f"unsupported progress snapshot {type(snapshot)!r}")


def _ready_next(
    blocks: tuple[BlockProjection, ...], completed_block: str
) -> tuple[BlockProjection, ...]:
    index = next(index for index, item in enumerate(blocks) if item.block_id == completed_block)
    if index + 1 >= len(blocks):
        return blocks
    following = blocks[index + 1]
    if following.status is not BlockStatus.WAITING:
        raise ValueError("progress plan successor is not waiting")
    return tuple(
        replace(item, status=BlockStatus.READY) if offset == index + 1 else item
        for offset, item in enumerate(blocks)
    )


def _require_running(block: BlockProjection) -> None:
    if block.status is not BlockStatus.RUNNING:
        raise ValueError(f"progress block {block.block_id} is not running")
