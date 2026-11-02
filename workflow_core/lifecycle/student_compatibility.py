from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from workflow_core.lifecycle.model_handles import ModelArtifact, PublishedModelHandle
from workflow_core.lifecycle.student_capabilities import (
    StudentModelOutcome,
    StudentModelRef,
    StudentOutcomeStatus,
    StudentTrainingOutcome,
)


@dataclass(frozen=True, slots=True)
class StudentTrainRequest:
    """Temporary path-backed training wrapper pending shared handle integration."""

    dataset_path: Path
    train_dir: Path
    member_ids: tuple[str, ...]
    progress_reporter_factory: Callable[[str], object] | None = None

    def __post_init__(self) -> None:
        member_ids = _member_ids(self.member_ids)
        object.__setattr__(self, "dataset_path", Path(self.dataset_path).resolve())
        object.__setattr__(self, "train_dir", Path(self.train_dir).resolve())
        object.__setattr__(self, "member_ids", member_ids)


@dataclass(frozen=True, slots=True)
class StudentTrainResult:
    artifacts: tuple[ModelArtifact, ...]

    def __post_init__(self) -> None:
        artifacts = tuple(self.artifacts)
        member_ids = tuple(artifact.member_id for artifact in artifacts)
        if not artifacts or len(set(member_ids)) != len(member_ids):
            raise ValueError("student train result requires ordered unique artifacts")
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def semantic_outcome(self) -> StudentTrainingOutcome:
        return StudentTrainingOutcome(
            status=StudentOutcomeStatus.SUCCEEDED,
            models=tuple(
                StudentModelRef(artifact.member_id, artifact.member_id)
                for artifact in self.artifacts
            ),
        )


@dataclass(frozen=True, slots=True)
class StudentTrainPublicationRequest:
    train: StudentTrainResult
    handles: tuple[PublishedModelHandle, ...]
    stage_root: Path
    published_root: Path
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        member_ids = _member_ids(self.member_ids)
        returned = tuple(item.member_id for item in self.train.artifacts)
        if returned != member_ids:
            raise ValueError(
                "student publication requires the ordered train result members"
            )
        handles = tuple(self.handles)
        if tuple(item.member_id for item in handles) != member_ids:
            raise ValueError(
                "student publication requires the exact ordered published handles"
            )
        object.__setattr__(self, "stage_root", Path(self.stage_root).resolve())
        object.__setattr__(self, "published_root", Path(self.published_root).resolve())
        object.__setattr__(self, "handles", handles)
        object.__setattr__(self, "member_ids", member_ids)


@dataclass(frozen=True, slots=True)
class StudentHistoricalModelRequest:
    member_id: str
    legacy_handle: Path

    def __post_init__(self) -> None:
        member_id = str(self.member_id).strip()
        if not member_id:
            raise ValueError("student historical recovery requires a member identity")
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "legacy_handle", Path(self.legacy_handle).absolute())


@dataclass(frozen=True, slots=True)
class StudentHistoricalModelResult:
    member_id: str
    path: Path

    def __post_init__(self) -> None:
        member_id = str(self.member_id).strip()
        if not member_id:
            raise ValueError("student historical result requires a member identity")
        path = Path(self.path).resolve()
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(
                "student historical result requires a non-empty regular native model"
            )
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "path", path)

    @property
    def semantic_outcome(self) -> StudentModelOutcome:
        return StudentModelOutcome(
            status=StudentOutcomeStatus.VALIDATED_REUSE,
            models=(StudentModelRef(self.member_id, self.member_id),),
        )


def _member_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    member_ids = tuple(str(item).strip() for item in values)
    if not member_ids or len(set(member_ids)) != len(member_ids) or any(
        not item for item in member_ids
    ):
        raise ValueError("student member identities must be ordered and unique")
    return member_ids


__all__ = [
    "StudentHistoricalModelRequest",
    "StudentHistoricalModelResult",
    "StudentTrainPublicationRequest",
    "StudentTrainRequest",
    "StudentTrainResult",
]
