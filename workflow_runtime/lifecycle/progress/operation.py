from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .types import ProgressSnapshot


@dataclass(frozen=True, slots=True)
class ProbeUpdate:
    semantic_token: str
    snapshot: ProgressSnapshot
    warning: str | None = None

    def __post_init__(self) -> None:
        if not self.semantic_token:
            raise ValueError("progress probe semantic token must be non-empty")


class ProgressProbe(Protocol):
    def poll(self) -> ProbeUpdate | None: ...


class OperationProgressReporter:
    """A deliberately small reporter whose lifetime is exactly one operation."""

    def __init__(
        self,
        *,
        operation_id: str,
        on_snapshot: Callable[[ProgressSnapshot], None],
        on_heartbeat: Callable[[], None] | None = None,
    ) -> None:
        if not operation_id:
            raise ValueError("operation progress reporter requires an operation_id")
        self.operation_id = operation_id
        self._on_snapshot = on_snapshot
        self._on_heartbeat = on_heartbeat
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def report(self, snapshot: ProgressSnapshot) -> None:
        self._require_open()
        self._on_snapshot(snapshot)

    def heartbeat(self) -> None:
        self._require_open()
        if self._on_heartbeat is not None:
            self._on_heartbeat()

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError(f"operation progress reporter {self.operation_id!r} is closed")
