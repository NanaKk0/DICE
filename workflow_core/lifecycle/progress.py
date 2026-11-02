from __future__ import annotations

from workflow_core.lifecycle.diagnostic_contracts import (
    BlockSummaryFact,
    DiagnosticObserver,
    OperationProgressObserver,
    ProgressFact,
)


class CoreBlockProgress:
    """Core-facing fact reporter without terminal lifecycle authority."""

    def __init__(self, *, coordinator: DiagnosticObserver, block: str) -> None:
        self._coordinator = coordinator
        self.block = block
        self._summary: BlockSummaryFact | None = None

    @property
    def final_summary(self) -> BlockSummaryFact | None:
        return self._summary

    def phase(self, phase: str, *, operation_id: str | None = None) -> None:
        self._coordinator.phase_entered(self.block, phase, operation_id=operation_id)

    def snapshot(self, snapshot: ProgressFact, *, operation_id: str | None = None) -> None:
        self._coordinator.snapshot(self.block, snapshot, operation_id=operation_id)

    def operation(
        self, operation_id: str, *, phase: str
    ) -> OperationProgressObserver:
        self.phase(phase, operation_id=operation_id)
        return self._coordinator.operation_reporter(self.block, operation_id=operation_id)

    def concurrent_operation(
        self,
        operation_id: str,
        *,
        phase: str,
        group_id: str,
    ) -> OperationProgressObserver:
        """Report one member while its concurrent group owns the block phase."""
        self.phase(phase, operation_id=group_id)
        return self._coordinator.operation_reporter(
            self.block,
            operation_id=operation_id,
            progress_scope_id=group_id,
        )

    def summary(self, summary: BlockSummaryFact) -> None:
        self._summary = summary

    def warning(self, exc: Exception) -> None:
        """Report a non-fatal diagnostic failure through operational progress."""
        self._coordinator.degradation_warning(exc)

    def screen_report_noop(self) -> None:
        """Owned seam for a screen route that intentionally has no report yet."""
        return None
