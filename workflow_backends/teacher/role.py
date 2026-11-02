from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from workflow_backends.teacher.label_result import TeacherLabelOps
from workflow_backends.teacher.readout import TeacherReadoutOps
from workflow_backends.teacher.training import TeacherTrainingOps
from workflow_artifacts.lifecycle.model_publication import PublishedModelHandle
from workflow_core.lifecycle.teacher_capabilities import (
    TeacherModelCapability,
    TeacherModelHandle,
)


@dataclass(frozen=True, slots=True)
class TeacherScreenOperationProfiles:
    gmm_score: str | None
    label: str | None


def screen_profile_id(context: object | None) -> str | None:
    value = None if context is None else getattr(context, "profile_id", None)
    if value is None:
        return None
    profile_id = str(value).strip()
    return profile_id or None


@dataclass(frozen=True, slots=True)
class TeacherBackendBinding:
    """Validated route-stable inputs bound once to a selected teacher factory."""

    run_json_path: Path
    iter_index: int
    recipe_path: Path
    type_map: tuple[str, ...]
    model_shape: str
    member_ids: tuple[str, ...]
    policy: str
    foundation_model: Path | None
    train_options: Mapping[str, bool]
    requires_readout: bool

    def __post_init__(self) -> None:
        type_map = tuple(str(symbol).strip() for symbol in self.type_map)
        member_ids = tuple(str(item).strip() for item in self.member_ids)
        if not type_map or any(not symbol for symbol in type_map):
            raise ValueError("teacher backend binding requires a non-empty type map")
        if self.model_shape not in {"single", "committee"}:
            raise ValueError("teacher backend binding has an invalid model shape")
        if (
            not member_ids
            or any(not item for item in member_ids)
            or len(set(member_ids)) != len(member_ids)
        ):
            raise ValueError(
                "teacher backend binding requires ordered unique member identities"
            )
        if self.model_shape == "single" and len(member_ids) != 1:
            raise ValueError("single teacher binding requires exactly one member")
        if self.model_shape == "committee" and len(member_ids) < 2:
            raise ValueError("committee teacher binding requires at least two members")
        if self.policy not in {"from_scratch", "foundation_finetune"}:
            raise ValueError("teacher backend binding has an invalid training policy")
        if self.policy == "foundation_finetune" and self.foundation_model is None:
            raise ValueError(
                "teacher foundation_finetune binding requires a foundation model"
            )
        object.__setattr__(self, "run_json_path", Path(self.run_json_path).resolve())
        object.__setattr__(self, "iter_index", int(self.iter_index))
        object.__setattr__(self, "recipe_path", Path(self.recipe_path).resolve())
        object.__setattr__(self, "type_map", type_map)
        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(
            self,
            "foundation_model",
            None
            if self.foundation_model is None
            else Path(self.foundation_model).resolve(),
        )
        object.__setattr__(
            self, "train_options", MappingProxyType(dict(self.train_options))
        )
        object.__setattr__(self, "requires_readout", bool(self.requires_readout))


@dataclass(frozen=True, slots=True)
class TeacherPublishedTrainingValidationRequest:
    stage_root: Path
    handles: tuple[PublishedModelHandle, ...]
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        handles = tuple(self.handles)
        member_ids = tuple(str(item) for item in self.member_ids)
        if tuple(handle.member_id for handle in handles) != member_ids:
            raise ValueError(
                "teacher completion validation requires ordered published members"
            )
        object.__setattr__(self, "stage_root", Path(self.stage_root).resolve())
        object.__setattr__(self, "handles", handles)
        object.__setattr__(self, "member_ids", member_ids)


@dataclass(frozen=True, slots=True)
class TeacherHistoricalModelRequest:
    member_id: str
    legacy_handle: Path

    def __post_init__(self) -> None:
        member_id = str(self.member_id).strip()
        if not member_id:
            raise ValueError("teacher historical recovery requires a member identity")
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "legacy_handle", Path(self.legacy_handle).absolute())


@dataclass(frozen=True, slots=True)
class TeacherHistoricalModelResult:
    member_id: str
    path: Path

    def __post_init__(self) -> None:
        member_id = str(self.member_id).strip()
        path = Path(self.path).resolve()
        if not member_id:
            raise ValueError("teacher historical recovery result requires a member identity")
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(
                "teacher historical recovery requires a non-empty regular native model"
            )
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "path", path)


class TeacherPublishedTrainingCapability(Protocol):
    def validate_published_training(
        self, request: TeacherPublishedTrainingValidationRequest
    ) -> bool: ...


class TeacherHistoricalModelCapability(Protocol):
    def recover_historical_model(
        self, request: TeacherHistoricalModelRequest
    ) -> TeacherHistoricalModelResult: ...


class TeacherBackend(
    TeacherTrainingOps,
    TeacherLabelOps,
    TeacherPublishedTrainingCapability,
    TeacherHistoricalModelCapability,
    Protocol,
):
    readout_ops: TeacherReadoutOps
    screen_operation_profiles: TeacherScreenOperationProfiles


class TeacherBackendFactory(Protocol):
    def bind(self, binding: TeacherBackendBinding) -> TeacherBackend: ...


__all__ = [
    "TeacherModelCapability",
    "TeacherModelHandle",
    "TeacherBackend",
    "TeacherBackendBinding",
    "TeacherBackendFactory",
    "TeacherHistoricalModelCapability",
    "TeacherHistoricalModelRequest",
    "TeacherHistoricalModelResult",
    "TeacherPublishedTrainingCapability",
    "TeacherPublishedTrainingValidationRequest",
    "TeacherScreenOperationProfiles",
    "screen_profile_id",
]
