from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ArtifactPublication:
    source: Path
    target: Path
    allow_empty: bool = False
    validate_extxyz: bool = True
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source).resolve())
        object.__setattr__(self, "target", Path(self.target).resolve())
        object.__setattr__(self, "allow_empty", bool(self.allow_empty))
        object.__setattr__(self, "validate_extxyz", bool(self.validate_extxyz))
        object.__setattr__(self, "required", bool(self.required))


@dataclass(frozen=True, slots=True)
class StagePublishPlan:
    student_round: int
    stage_run_index: int
    artifact_publications: tuple[ArtifactPublication, ...] = ()
    backend_log_publications: tuple[ArtifactPublication, ...] = ()
    stage_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "student_round", int(self.student_round))
        object.__setattr__(self, "stage_run_index", int(self.stage_run_index))
        object.__setattr__(self, "artifact_publications", tuple(self.artifact_publications or ()))
        object.__setattr__(self, "backend_log_publications", tuple(self.backend_log_publications or ()))
        object.__setattr__(self, "stage_payload", MappingProxyType(dict(self.stage_payload or {})))


StageCommitPlan = StagePublishPlan
