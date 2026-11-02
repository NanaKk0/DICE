from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

from workflow_runtime.execution.planning import PlannedExecution


EXECUTION_STATES = frozenset(
    {
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "missing",
        "indeterminate",
    }
)
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class PreparedExecution:
    source_root: Path
    input_inventory: Mapping[str, str]
    argv: tuple[str, ...]
    output_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExternalJobReceipt:
    schema: str
    schema_revision: int
    unit_id: str
    attempt_id: str
    target_id: str
    adapter: str
    external_id: str
    state: str
    immutable_spec_digest: str
    allocation_evidence: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ExecutionAdapter(Protocol):
    target_id: str
    adapter_kind: str

    def submit(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt: ...

    def recover_submission(
        self,
        planned: PlannedExecution,
        *,
        launch_intent: Mapping[str, object],
    ) -> ExternalJobReceipt | None: ...

    def poll(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt: ...

    def cancel(self, receipt: ExternalJobReceipt) -> ExternalJobReceipt: ...

    def collect(
        self,
        receipt: ExternalJobReceipt,
        *,
        destination: Path,
    ) -> Mapping[str, str]: ...
