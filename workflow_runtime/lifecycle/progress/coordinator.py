from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable, TextIO
from uuid import uuid4

from .reducer import ProgressProjection, initial_projection, reduce_progress
from .types import (
    AttemptStarted,
    BasicBlockSummary,
    BlockFailed,
    BlockInterrupted,
    BlockStarted,
    BlockSucceeded,
    DiagnosticArchived,
    DiagnosticPruned,
    HeartbeatReported,
    PhaseEntered,
    ProgressEnvelope,
    ProgressPayload,
    BlockSummary,
    ProgressSnapshot,
    SnapshotReported,
    CountChainSnapshot,
    CountFlowSnapshot,
    DatasetSnapshot,
    DescriptorSnapshot,
    GenericWorkSnapshot,
    LabelingSnapshot,
    RoutingSnapshot,
    SamplingSnapshot,
    ScorecardSnapshot,
    ScoreObservationSnapshot,
    TrainingSnapshot,
)


class ProgressCoordinator:
    """The sole lifecycle writer for one invocation's projection and human history."""

    def __init__(
        self,
        *,
        iter_root: str | Path,
        iter_index: int,
        plan: tuple[str, ...],
        completed: tuple[str, ...],
        ready: str | None,
        terminal: TextIO | None = None,
        attempt_id: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.iter_root = Path(iter_root)
        self.status_path = self.iter_root / "status.log"
        self.terminal = terminal if terminal is not None else sys.stdout
        self.attempt_id = attempt_id or _new_attempt_id()
        self.started_at = _now()
        self.state = initial_projection(
            iter_index=iter_index,
            plan=plan,
            completed=completed,
            ready=ready,
            attempt_id=self.attempt_id,
            started_at=self.started_at,
        )
        self._sequence = 0
        self._warned = False
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.RLock()
        self._snapshot_delivery: dict[
            tuple[str, str, str, type[object], str], tuple[float, ProgressSnapshot, str]
        ] = {}

    def start(self, *, resume_target: str | None) -> None:
        block = self._context_block()
        self._emit(block, AttemptStarted(resume_target=resume_target))

    def block_started(self, block: str) -> None:
        self._emit(block, BlockStarted())

    def phase_entered(self, block: str, phase: str, *, operation_id: str | None = None) -> None:
        with self._lock:
            current = self.state.block(block)
            if current.phase == phase and current.active_operation_id == operation_id:
                return
            self._emit(
                block,
                PhaseEntered(phase=phase, operation_id=operation_id),
                phase=phase,
                operation_id=operation_id,
            )

    def snapshot(self, block: str, snapshot: ProgressSnapshot, *, operation_id: str | None = None) -> None:
        with self._lock:
            current = self.state.block(block)
            self._emit(
                block,
                SnapshotReported(snapshot),
                phase=current.phase,
                operation_id=operation_id,
            )

    def heartbeat(self, block: str, *, operation_id: str | None = None) -> None:
        with self._lock:
            current = self.state.block(block)
            self._emit(
                block,
                HeartbeatReported(),
                phase=current.phase,
                operation_id=operation_id,
            )

    def operation_reporter(
        self,
        block: str,
        *,
        operation_id: str,
        progress_scope_id: str | None = None,
    ):
        from .operation import OperationProgressReporter

        event_operation_id = progress_scope_id or operation_id
        return OperationProgressReporter(
            operation_id=operation_id,
            on_snapshot=lambda snapshot: self.snapshot(
                block, snapshot, operation_id=event_operation_id
            ),
            on_heartbeat=lambda: self.heartbeat(
                block, operation_id=event_operation_id
            ),
        )

    def block_succeeded(self, block: str, *, summary: BlockSummary | None = None) -> None:
        self._emit(block, BlockSucceeded(summary=summary or BasicBlockSummary()))

    def block_failed(
        self,
        block: str,
        *,
        message: str | None = None,
        diagnostic_ref: str | None = None,
    ) -> None:
        self._emit(block, BlockFailed(message=message, diagnostic_ref=diagnostic_ref))

    def block_interrupted(
        self,
        block: str,
        *,
        message: str | None = None,
        diagnostic_ref: str | None = None,
    ) -> None:
        self._emit(block, BlockInterrupted(message=message, diagnostic_ref=diagnostic_ref))

    def degradation_warning(self, exc: Exception) -> None:
        self._warn_once(exc)

    def diagnostic_archived(self, block: str, *, attempt_id: str, diagnostic_ref: str) -> None:
        self._emit(
            block,
            DiagnosticArchived(archived_attempt_id=attempt_id, diagnostic_ref=diagnostic_ref),
        )

    def diagnostic_pruned(self, block: str, *, attempt_id: str) -> None:
        self._emit(block, DiagnosticPruned(pruned_attempt_id=attempt_id))

    def _emit(
        self,
        block: str,
        payload: ProgressPayload,
        *,
        phase: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        with self._lock:
            self._emit_unlocked(
                block, payload, phase=phase, operation_id=operation_id
            )

    def _emit_unlocked(
        self,
        block: str,
        payload: ProgressPayload,
        *,
        phase: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        self._sequence += 1
        event = ProgressEnvelope(
            attempt_id=self.attempt_id,
            sequence=self._sequence,
            received_at=_now(),
            iter_index=self.state.iter_index,
            block=block,
            phase=phase,
            operation_id=operation_id,
            payload=payload,
        )
        self.state = reduce_progress(self.state, event)
        from workflow_artifacts.lifecycle.progress import render_history_event

        history = render_history_event(event)
        if isinstance(payload, SnapshotReported) and (
            history is None
            or not self._deliver_snapshot(
                block,
                payload.snapshot,
                history,
                phase=event.phase,
                operation_id=event.operation_id,
            )
        ):
            history = None
        if history is not None:
            try:
                _append_text(self.status_path, history)
            except Exception as exc:
                self._warn_once(exc)
            try:
                self.terminal.write(history)
                self.terminal.flush()
            except Exception as exc:
                self._warn_once(exc)

    def _context_block(self) -> str:
        for block in self.state.blocks:
            if block.status.value == "READY":
                return block.block_id
        return self.state.blocks[-1].block_id

    def _deliver_snapshot(
        self,
        block: str,
        snapshot: ProgressSnapshot,
        history: str,
        *,
        phase: str | None,
        operation_id: str | None,
    ) -> bool:
        key = (
            block,
            phase or "",
            operation_id or "",
            *_snapshot_delivery_key(snapshot),
        )
        now = self._clock()
        previous = self._snapshot_delivery.get(key)
        deliver = previous is None
        if previous is not None:
            previous_time, old, previous_history = previous
            if history == previous_history:
                return False
            deliver = _is_final_snapshot(snapshot)
            if isinstance(snapshot, TrainingSnapshot) and isinstance(old, TrainingSnapshot):
                deliver = deliver or (snapshot.stage, snapshot.series_generation) != (old.stage, old.series_generation)
            elif isinstance(snapshot, SamplingSnapshot) and isinstance(old, SamplingSnapshot):
                deliver = deliver or (
                    snapshot.completed_tasks != old.completed_tasks
                    or snapshot.active_task_id != old.active_task_id
                    or snapshot.mps_phase != old.mps_phase
                    or _sampling_alert_identity(snapshot) != _sampling_alert_identity(old)
                )
            position, total = _snapshot_position(snapshot)
            old_position, old_total = _snapshot_position(old)
            if position is not None and old_position is not None:
                if total is not None and total > 0 and old_total == total:
                    deliver = deliver or position * 10 // total > old_position * 10 // total
                deliver = deliver or (now - previous_time >= 300.0 and position != old_position)
        if deliver:
            self._snapshot_delivery[key] = (now, snapshot, history)
        return deliver

    def _warn_once(self, exc: Exception) -> None:
        if self._warned:
            return
        self._warned = True
        detail = " ".join(str(exc).split())
        if len(detail) > 240:
            detail = detail[:239].rstrip() + "…"
        warning = (
            f"{_now()} iter{self.state.iter_index} [WARNING] "
            f"lifecycle progress display degraded: {detail}\n"
        )
        try:
            _append_text(self.status_path, warning)
        except Exception:
            pass
        try:
            self.terminal.write(warning)
            self.terminal.flush()
        except Exception:
            pass
        try:
            sys.stderr.write(warning)
            sys.stderr.flush()
        except Exception:
            pass


def _append_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _new_attempt_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _snapshot_delivery_key(snapshot: ProgressSnapshot) -> tuple[type[object], str]:
    if isinstance(snapshot, GenericWorkSnapshot):
        identity = snapshot.activity
    elif isinstance(snapshot, DatasetSnapshot):
        identity = snapshot.label
    elif isinstance(snapshot, CountChainSnapshot):
        identity = snapshot.label
    elif isinstance(snapshot, CountFlowSnapshot):
        identity = snapshot.title
    elif isinstance(snapshot, TrainingSnapshot):
        identity = snapshot.member_id
    elif isinstance(snapshot, LabelingSnapshot):
        identity = "labeling"
    elif isinstance(snapshot, (ScoreObservationSnapshot, ScorecardSnapshot)):
        identity = snapshot.role
    elif isinstance(snapshot, DescriptorSnapshot):
        identity = snapshot.label
    elif isinstance(snapshot, RoutingSnapshot):
        identity = "routing"
    elif isinstance(snapshot, SamplingSnapshot):
        identity = "sampling"
    else:  # pragma: no cover - closed typed union
        raise TypeError(f"unsupported snapshot delivery type {type(snapshot)!r}")
    return type(snapshot), identity


def _is_final_snapshot(snapshot: ProgressSnapshot) -> bool:
    if isinstance(snapshot, GenericWorkSnapshot):
        return snapshot.total is not None and snapshot.completed == snapshot.total
    if isinstance(snapshot, TrainingSnapshot):
        return snapshot.position == snapshot.total
    if isinstance(snapshot, (LabelingSnapshot, ScoreObservationSnapshot, DescriptorSnapshot)):
        return snapshot.completed == snapshot.total
    if isinstance(snapshot, SamplingSnapshot):
        return (
            snapshot.completed_tasks == snapshot.total_tasks
            or snapshot.active_step == snapshot.steps_per_task
        )
    return False


def _snapshot_position(snapshot: ProgressSnapshot) -> tuple[int | None, int | None]:
    if isinstance(snapshot, GenericWorkSnapshot):
        return snapshot.completed, snapshot.total
    if isinstance(snapshot, DatasetSnapshot):
        return snapshot.unique, None
    if isinstance(snapshot, TrainingSnapshot):
        return snapshot.position, snapshot.total
    if isinstance(snapshot, (LabelingSnapshot, ScoreObservationSnapshot, DescriptorSnapshot)):
        return snapshot.completed, snapshot.total
    if isinstance(snapshot, SamplingSnapshot):
        active_step = 0
        if snapshot.completed_tasks < snapshot.total_tasks and snapshot.active_step is not None:
            active_step = snapshot.active_step
        return (
            snapshot.completed_tasks * snapshot.steps_per_task + active_step,
            snapshot.total_tasks * snapshot.steps_per_task,
        )
    return None, None


def _sampling_alert_identity(
    snapshot: SamplingSnapshot,
) -> tuple[tuple[str, str, str | None, str | None], ...]:
    return tuple(
        (detail.task_id, detail.state, detail.message, detail.diagnostic_locator)
        for detail in snapshot.task_details
        if detail.state in {"warning", "failed"}
    )
