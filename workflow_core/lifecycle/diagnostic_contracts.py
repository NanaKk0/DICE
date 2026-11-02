from __future__ import annotations

from typing import Protocol, TypeAlias


ProgressFact: TypeAlias = object
BlockSummaryFact: TypeAlias = object


class OperationProgressObserver(Protocol):
    @property
    def closed(self) -> bool: ...

    def report(self, fact: ProgressFact) -> None: ...

    def heartbeat(self) -> None: ...

    def close(self) -> None: ...


class DiagnosticObserver(Protocol):
    """Logic-owned non-authoritative diagnostic observation capability."""

    def phase_entered(
        self,
        block: str,
        phase: str,
        *,
        operation_id: str | None = None,
    ) -> None: ...

    def snapshot(
        self,
        block: str,
        fact: ProgressFact,
        *,
        operation_id: str | None = None,
    ) -> None: ...

    def operation_reporter(
        self,
        block: str,
        *,
        operation_id: str,
        progress_scope_id: str | None = None,
    ) -> OperationProgressObserver: ...

    def degradation_warning(self, exc: Exception) -> None: ...


__all__ = [
    "BlockSummaryFact",
    "DiagnosticObserver",
    "OperationProgressObserver",
    "ProgressFact",
]
