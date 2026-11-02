from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from ase import Atoms

from workflow_core.lifecycle.fp_capabilities import (
    FirstPrinciplesLabelCapability,
    fp_task_id,
)


@dataclass(frozen=True, slots=True)
class FPLabelRequest:
    selected_path: Path
    task_root: Path
    recipe_path: Path
    controller_root: Path
    calculator: str
    operation_role: str
    operation_id: str
    operation_context: Any | None = None


@dataclass(frozen=True, slots=True)
class FPPrepareRequest:
    selected_path: Path
    task_root: Path
    recipe_path: Path
    operation_context: Any


@dataclass(frozen=True, slots=True)
class FPPreparedTask:
    task_id: str
    selected_index: int
    structure_id: str
    system_id: str
    task_path: Path
    input_digest: str
    logical_executable: str
    arguments: tuple[str, ...]
    required_labels: tuple[str, ...] = ("energy", "forces")


@dataclass(frozen=True, slots=True)
class FPParseRequest:
    prepared_task: FPPreparedTask
    completed_task: Any


@dataclass(frozen=True, slots=True)
class FPLabelResult:
    task_id: str
    selected_index: int
    structure_id: str
    system_id: str
    labeled_atoms: Atoms
    input_digest: str
    parser_evidence: Mapping[str, Any]
    required_labels: tuple[str, ...] = ("energy", "forces")


@dataclass(frozen=True, slots=True)
class FPLabelBatchResult:
    results: tuple[FPLabelResult, ...]
    summary_evidence: Mapping[str, Any]


FirstPrinciplesLabelService = FirstPrinciplesLabelCapability


class FirstPrinciplesCalculatorAdapter(Protocol):
    calculator: str
    operation_role: str
    logical_executable: str

    def prepare(self, request: FPPrepareRequest) -> tuple[FPPreparedTask, ...]: ...

    def parse(self, request: FPParseRequest) -> FPLabelResult: ...


__all__ = [
    "FirstPrinciplesLabelService",
    "FirstPrinciplesCalculatorAdapter",
    "FPLabelBatchResult",
    "FPLabelRequest",
    "FPLabelResult",
    "FPParseRequest",
    "FPPrepareRequest",
    "FPPreparedTask",
    "fp_task_id",
]
