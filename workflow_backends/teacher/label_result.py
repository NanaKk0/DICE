from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ase import Atoms

from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter
from workflow_core.lifecycle.teacher_capabilities import (
    TeacherLabelIntent,
    TeacherLabelOutcome,
    TeacherLabelingCapability,
)


@dataclass(slots=True)
class TeacherLabelResult:
    """Backend-neutral output returned by a formal teacher label operation."""

    labeled_structures: list[Any]
    uncertainties: Any


@dataclass(frozen=True, slots=True)
class TeacherLabelRequest:
    models: tuple[Path, ...]
    frames: tuple[Atoms, ...]
    work_dir: Path
    trusted_below: float | None = None
    progress_reporter_factory: (
        Callable[[str, str], OperationProgressReporter] | None
    ) = None

    def __post_init__(self) -> None:
        models = tuple(Path(model).resolve() for model in self.models)
        frames = tuple(self.frames)
        if not models or len(set(models)) != len(models):
            raise ValueError("teacher label requires unique ordered model handles")
        if any(not isinstance(frame, Atoms) for frame in frames):
            raise TypeError("teacher label frames must be ASE Atoms")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "work_dir", Path(self.work_dir).resolve())

    def progress(
        self, operation_id: str, phase: str
    ) -> OperationProgressReporter | None:
        if self.progress_reporter_factory is None:
            return None
        return self.progress_reporter_factory(operation_id, phase)


@dataclass(frozen=True, slots=True)
class TeacherNativeLabelRequest:
    models: tuple[Path, ...]
    frames: tuple[Atoms, ...]
    work_dir: Path
    trusted_below: float | None = None
    progress_reporter: OperationProgressReporter | None = None


@dataclass(frozen=True, slots=True)
class TeacherNativeLabelResult:
    frames: tuple[Atoms, ...]

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if any(not isinstance(frame, Atoms) for frame in frames):
            raise TypeError("teacher native label result must contain ASE Atoms")
        object.__setattr__(self, "frames", frames)


class TeacherLabelOps(Protocol):
    def label(self, request: TeacherNativeLabelRequest) -> TeacherNativeLabelResult: ...


__all__ = [
    "TeacherLabelIntent",
    "TeacherLabelOutcome",
    "TeacherLabelingCapability",
    "TeacherLabelOps",
    "TeacherLabelRequest",
    "TeacherLabelResult",
    "TeacherNativeLabelRequest",
    "TeacherNativeLabelResult",
]
