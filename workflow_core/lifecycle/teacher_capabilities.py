from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import copy
import math
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

from workflow_core.lifecycle.artifact_handles import (
    DatasetHandle,
    ModelHandle,
    StructureBatchHandle,
)
from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics


TEACHER_TRAINING_ARTIFACT_KIND = "dice.teacher-training-publication"
TEACHER_LABEL_ARTIFACT_KIND = "dice.teacher-label-publication"
TEACHER_READOUT_ARTIFACT_KIND = "dice.teacher-readout-publication"


class TeacherOutcomeStatus(str, Enum):
    """Stable lifecycle-visible outcome classes for teacher operations."""

    SUCCEEDED = "succeeded"
    VALIDATED_REUSE = "validated-reuse"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TeacherDatasetHandle:
    """Path-free identity and ordered membership of one teacher dataset."""

    dataset_id: str
    train_structure_ids: tuple[str, ...]
    validation_structure_ids: tuple[str, ...]
    validation_mode: str

    def __post_init__(self) -> None:
        train_handle = DatasetHandle(
            dataset_id=self.dataset_id,
            ordered_structure_ids=self.train_structure_ids,
        )
        validation_handle = DatasetHandle(
            dataset_id=self.dataset_id,
            ordered_structure_ids=self.validation_structure_ids,
        )
        if self.validation_mode not in {"self", "holdout"}:
            raise ValueError("teacher dataset validation mode must be self or holdout")
        if (
            self.validation_mode == "self"
            and validation_handle.ordered_structure_ids
            != train_handle.ordered_structure_ids
        ):
            raise ValueError(
                "teacher self-validation must preserve the exact train membership and order"
            )
        if self.validation_mode == "holdout" and set(
            train_handle.ordered_structure_ids
        ) & set(validation_handle.ordered_structure_ids):
            raise ValueError("teacher holdout train and validation membership must be disjoint")
        object.__setattr__(
            self,
            "dataset_id",
            train_handle.dataset_id,
        )
        object.__setattr__(
            self,
            "train_structure_ids",
            train_handle.ordered_structure_ids,
        )
        object.__setattr__(
            self,
            "validation_structure_ids",
            validation_handle.ordered_structure_ids,
        )

    @property
    def train(self) -> DatasetHandle:
        return DatasetHandle(self.dataset_id, self.train_structure_ids)

    @property
    def validation(self) -> DatasetHandle:
        return DatasetHandle(self.dataset_id, self.validation_structure_ids)


TeacherModelHandle = ModelHandle
TeacherStructureBatchHandle = StructureBatchHandle


@dataclass(frozen=True, slots=True)
class TeacherTrainingArtifactFacts:
    """Exact dataset and model semantics expected from teacher training."""

    dataset: TeacherDatasetHandle
    models: tuple[ModelHandle, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, TeacherDatasetHandle):
            raise TypeError("teacher training facts require the canonical dataset view")
        models = tuple(self.models)
        if not models or any(not isinstance(model, ModelHandle) for model in models):
            raise TypeError("teacher model artifact facts require shared model handles")
        members = tuple(model.member_id for model in models)
        model_ids = tuple(model.model_id for model in models)
        if len(set(members)) != len(members) or len(set(model_ids)) != len(model_ids):
            raise ValueError("teacher model artifact facts must be ordered and unique")
        object.__setattr__(self, "models", models)


@dataclass(frozen=True, slots=True)
class TeacherTrainingRecoveryFacts:
    """Path-free settings needed to validate one committed teacher train block."""

    backend: str
    cli: str | None
    model_shape: str
    member_ids: tuple[str, ...]
    type_map: tuple[str, ...]
    validation_mode: str

    def __post_init__(self) -> None:
        backend = _identity(self.backend, label="teacher backend")
        cli = None if self.cli is None else _identity(self.cli, label="teacher CLI")
        model_shape = str(self.model_shape).strip()
        member_ids = _ordered_identities(
            tuple(self.member_ids), label="teacher recovery members"
        )
        type_map = _ordered_identities(
            tuple(self.type_map), label="teacher recovery type map"
        )
        if model_shape not in {"single", "committee"}:
            raise ValueError("teacher recovery model shape must be single or committee")
        if (model_shape == "single") != (len(member_ids) == 1):
            raise ValueError("teacher recovery member inventory disagrees with model shape")
        if self.validation_mode not in {"self", "holdout"}:
            raise ValueError("teacher recovery validation mode must be self or holdout")
        if backend == "DeepMD" and cli not in {"tf", "pt"}:
            raise ValueError("DeepMD teacher recovery requires tf or pt CLI")
        if backend != "DeepMD" and cli is not None:
            raise ValueError("non-DeepMD teacher recovery cannot declare a DeepMD CLI")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "cli", cli)
        object.__setattr__(self, "model_shape", model_shape)
        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "type_map", type_map)


@dataclass(frozen=True, slots=True)
class TeacherScreenRecoveryFacts:
    """Path-free teacher-screen publication requirements."""

    iter_index: int
    type_map: tuple[str, ...]
    committee: bool
    teacher_lo: float | None

    def __post_init__(self) -> None:
        if isinstance(self.iter_index, bool) or self.iter_index < 0:
            raise ValueError("teacher screen recovery iter must be non-negative")
        type_map = _ordered_identities(
            tuple(self.type_map), label="teacher screen recovery type map"
        )
        if not isinstance(self.committee, bool):
            raise TypeError("teacher screen recovery committee flag must be boolean")
        if self.committee:
            if self.teacher_lo is None or not math.isfinite(float(self.teacher_lo)):
                raise ValueError("committee teacher screen recovery requires finite lo")
            object.__setattr__(self, "teacher_lo", float(self.teacher_lo))
        elif self.teacher_lo is not None:
            raise ValueError("single teacher screen recovery cannot declare lo")
        object.__setattr__(self, "type_map", type_map)


@dataclass(frozen=True, slots=True)
class TeacherScanRecoveryFacts:
    """Configured ordered teacher elements admitted by a committed router."""

    type_map: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "type_map",
            _ordered_identities(
                tuple(self.type_map), label="teacher scan recovery type map"
            ),
        )


@dataclass(frozen=True, slots=True)
class TeacherStructureArtifactFacts:
    """Exact ordered semantic batch expected from labeling or readout."""

    structures: StructureBatchHandle
    descriptor_schema_name: str | None = None
    descriptor_schema_revision: int | None = None
    emit_frame_means: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.structures, StructureBatchHandle):
            raise TypeError("teacher structure artifact facts require a shared batch handle")
        if (self.descriptor_schema_name is None) != (
            self.descriptor_schema_revision is None
        ):
            raise ValueError("teacher readout schema name and revision must appear together")
        if self.descriptor_schema_name is not None:
            name = str(self.descriptor_schema_name).strip()
            revision = self.descriptor_schema_revision
            if not name or isinstance(revision, bool) or int(revision) <= 0:
                raise ValueError("teacher readout artifact schema is invalid")
            object.__setattr__(self, "descriptor_schema_name", name)
            object.__setattr__(self, "descriptor_schema_revision", int(revision))
        if self.emit_frame_means is not None and not isinstance(
            self.emit_frame_means, bool
        ):
            raise TypeError("teacher readout frame-mean expectation must be boolean")


def require_teacher_models(
    expectation: ExpectedArtifactSemantics[TeacherTrainingArtifactFacts],
    *,
    member_ids: tuple[str, ...],
    dataset: TeacherDatasetHandle | None = None,
) -> tuple[ModelHandle, ...]:
    if expectation.artifact_kind != TEACHER_TRAINING_ARTIFACT_KIND or not isinstance(
        expectation.facts, TeacherTrainingArtifactFacts
    ):
        raise ValueError("teacher training requires canonical artifact semantics")
    if dataset is not None and expectation.facts.dataset != dataset:
        raise ValueError("teacher training expectation changed canonical dataset semantics")
    models = expectation.facts.models
    if tuple(model.member_id for model in models) != tuple(member_ids):
        raise ValueError("teacher model expectation changed ordered members")
    return models


def require_teacher_structures(
    expectation: ExpectedArtifactSemantics[TeacherStructureArtifactFacts],
    *,
    artifact_kind: str,
    structures: StructureBatchHandle,
) -> TeacherStructureArtifactFacts:
    if expectation.artifact_kind != artifact_kind or not isinstance(
        expectation.facts, TeacherStructureArtifactFacts
    ):
        raise ValueError("teacher operation requires canonical artifact semantics")
    if expectation.facts.structures != structures:
        raise ValueError("teacher artifact expectation changed canonical structure order")
    return expectation.facts


@dataclass(frozen=True, slots=True)
class TeacherDescriptorSchema:
    """Logic-owned descriptor compatibility facts, independent of native arrays."""

    name: str
    revision: int
    backend: str
    width: int
    attributes: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        name = _identity(self.name, label="teacher descriptor schema")
        backend = _identity(self.backend, label="teacher descriptor backend")
        if isinstance(self.revision, bool) or self.revision <= 0:
            raise ValueError("teacher descriptor schema revision must be positive")
        if isinstance(self.width, bool) or self.width <= 0:
            raise ValueError("teacher descriptor width must be positive")
        attributes = copy.deepcopy(dict(self.attributes))
        if any(not isinstance(key, str) or not key for key in attributes):
            raise ValueError("teacher descriptor attributes require non-empty string keys")
        if any(
            not _valid_descriptor_fact(value)
            for value in attributes.values()
        ):
            raise ValueError("teacher descriptor attributes must be finite scalar facts")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "attributes", MappingProxyType(attributes))


def _valid_descriptor_fact(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_valid_descriptor_fact(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str)
            and bool(key)
            and _valid_descriptor_fact(item)
            for key, item in value.items()
        )
    return False


@dataclass(frozen=True, slots=True)
class TeacherReferenceArtifactFacts:
    """Expected model, dataset and descriptor identity for teacher readout fit."""

    model: ModelHandle
    reference: TeacherDatasetHandle
    descriptor_schema: TeacherDescriptorSchema

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelHandle):
            raise TypeError("teacher reference facts require a shared model handle")
        if not isinstance(self.reference, TeacherDatasetHandle):
            raise TypeError("teacher reference facts require a canonical dataset view")
        if not isinstance(self.descriptor_schema, TeacherDescriptorSchema):
            raise TypeError("teacher reference facts require descriptor semantics")


def require_teacher_reference(
    expectation: ExpectedArtifactSemantics[TeacherReferenceArtifactFacts],
    *,
    model: ModelHandle,
    reference: TeacherDatasetHandle,
) -> TeacherDescriptorSchema:
    if expectation.artifact_kind != TEACHER_READOUT_ARTIFACT_KIND or not isinstance(
        expectation.facts, TeacherReferenceArtifactFacts
    ):
        raise ValueError("teacher reference fit requires canonical artifact semantics")
    if expectation.facts.model != model or expectation.facts.reference != reference:
        raise ValueError("teacher reference expectation changed model or dataset identity")
    return expectation.facts.descriptor_schema


@dataclass(frozen=True, slots=True)
class TeacherTrainingIntent:
    iter_index: int
    dataset: TeacherDatasetHandle
    model_shape: str
    member_ids: tuple[str, ...]
    type_map: tuple[str, ...]
    policy: str
    foundation_model: TeacherModelHandle | None = None
    train_options: Mapping[str, bool] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        members = _ordered_identities(self.member_ids, label="teacher members")
        type_map = _ordered_identities(self.type_map, label="teacher type map")
        valid_shape = (self.model_shape == "single" and len(members) == 1) or (
            self.model_shape == "committee" and len(members) >= 2
        )
        if not valid_shape:
            raise ValueError("teacher model shape does not match ordered member identities")
        if self.policy not in {"from_scratch", "foundation_finetune"}:
            raise ValueError("teacher training policy is invalid")
        if (self.policy == "foundation_finetune") != (
            self.foundation_model is not None
        ):
            raise ValueError(
                "teacher foundation model presence must match the training policy"
            )
        options = dict(self.train_options)
        if set(options) - {"skip_neighbor_stat"}:
            raise ValueError("teacher train options contain unsupported fields")
        if any(not isinstance(value, bool) for value in options.values()):
            raise ValueError("teacher train options require boolean values")
        object.__setattr__(self, "iter_index", int(self.iter_index))
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "type_map", type_map)
        object.__setattr__(self, "train_options", MappingProxyType(options))


@dataclass(frozen=True, slots=True)
class TeacherTrainingOutcome:
    status: TeacherOutcomeStatus
    models: tuple[TeacherModelHandle, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        status = TeacherOutcomeStatus(self.status)
        models = tuple(self.models)
        member_ids = tuple(model.member_id for model in models)
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("teacher training outcome duplicates a member identity")
        reason = None if self.reason is None else str(self.reason).strip()
        if status is TeacherOutcomeStatus.FAILED and not reason:
            raise ValueError("failed teacher training requires a reason")
        if status is not TeacherOutcomeStatus.FAILED and reason:
            raise ValueError("successful or partial teacher training cannot carry a failure reason")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class TeacherLabelIntent:
    models: tuple[TeacherModelHandle, ...]
    structures: TeacherStructureBatchHandle
    trusted_below: float | None = None

    def __post_init__(self) -> None:
        models = tuple(self.models)
        member_ids = tuple(model.member_id for model in models)
        if not models or len(set(member_ids)) != len(member_ids):
            raise ValueError("teacher label requires ordered unique model members")
        if self.trusted_below is not None and not math.isfinite(
            float(self.trusted_below)
        ):
            raise ValueError("teacher label trust boundary must be finite")
        object.__setattr__(self, "models", models)


@dataclass(frozen=True, slots=True)
class TeacherLabelOutcome:
    status: TeacherOutcomeStatus
    labeled_structure_ids: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        status = TeacherOutcomeStatus(self.status)
        identities = _ordered_identities(
            self.labeled_structure_ids,
            label="teacher labeled structures",
            allow_empty=True,
        )
        reason = None if self.reason is None else str(self.reason).strip()
        if status is TeacherOutcomeStatus.FAILED and not reason:
            raise ValueError("failed teacher labeling requires a reason")
        if status is not TeacherOutcomeStatus.FAILED and reason:
            raise ValueError("successful or partial teacher labeling cannot carry a failure reason")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "labeled_structure_ids", identities)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class TeacherReferenceIntent:
    model: TeacherModelHandle
    reference: TeacherDatasetHandle
    type_map: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "type_map", _ordered_identities(self.type_map, label="teacher type map")
        )


@dataclass(frozen=True, slots=True)
class TeacherReferenceOutcome:
    descriptor_schema: TeacherDescriptorSchema


@dataclass(frozen=True, slots=True)
class TeacherScoreIntent:
    model: TeacherModelHandle
    reference: TeacherDatasetHandle
    candidates: TeacherStructureBatchHandle
    chunk_size: int
    emit_frame_means: bool
    reference_descriptor_schema: TeacherDescriptorSchema

    def __post_init__(self) -> None:
        if isinstance(self.chunk_size, bool) or int(self.chunk_size) <= 0:
            raise ValueError("teacher score chunk size must be positive")
        object.__setattr__(self, "chunk_size", int(self.chunk_size))


@dataclass(frozen=True, slots=True)
class TeacherScoreOutcome:
    ordered_structure_ids: tuple[str, ...]
    frame_probabilities: tuple[float, ...]
    frame_statistics: tuple[float, ...]
    descriptor_schema: TeacherDescriptorSchema
    frame_means: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        identities = _ordered_identities(
            self.ordered_structure_ids,
            label="teacher score structures",
            allow_empty=True,
        )
        probabilities = tuple(float(value) for value in self.frame_probabilities)
        statistics = tuple(float(value) for value in self.frame_statistics)
        if len(probabilities) != len(identities) or len(statistics) != len(identities):
            raise ValueError("teacher score evidence does not cover every ordered structure")
        if not all(math.isfinite(value) for value in (*probabilities, *statistics)):
            raise ValueError("teacher score evidence must be finite")
        means = None
        if self.frame_means is not None:
            means = tuple(tuple(float(value) for value in row) for row in self.frame_means)
            if len(means) != len(identities) or any(
                len(row) != self.descriptor_schema.width for row in means
            ):
                raise ValueError("teacher frame means do not match descriptor width and order")
            if any(not math.isfinite(value) for row in means for value in row):
                raise ValueError("teacher frame means must be finite")
        object.__setattr__(self, "ordered_structure_ids", identities)
        object.__setattr__(self, "frame_probabilities", probabilities)
        object.__setattr__(self, "frame_statistics", statistics)
        object.__setattr__(self, "frame_means", means)


@runtime_checkable
class TeacherTrainingCapability(Protocol):
    def train(self, request: TeacherTrainingIntent) -> TeacherTrainingOutcome: ...


@runtime_checkable
class TeacherLabelingCapability(Protocol):
    def label(self, request: TeacherLabelIntent) -> TeacherLabelOutcome: ...


@runtime_checkable
class TeacherReadoutCapability(Protocol):
    def fit_reference(
        self, request: TeacherReferenceIntent
    ) -> TeacherReferenceOutcome: ...

    def score_candidates(
        self, request: TeacherScoreIntent
    ) -> TeacherScoreOutcome: ...


@runtime_checkable
class TeacherModelCapability(Protocol):
    def validate_published_training(
        self, models: tuple[TeacherModelHandle, ...]
    ) -> bool: ...

    def recover_historical_model(self, member_id: str) -> TeacherModelHandle: ...


def _identity(value: object, *, label: str) -> str:
    identity = str(value).strip()
    if not identity:
        raise ValueError(f"{label} identity must be non-empty")
    return identity


def _ordered_identities(
    values: tuple[str, ...], *, label: str, allow_empty: bool = False
) -> tuple[str, ...]:
    ordered = tuple(_identity(value, label=label) for value in values)
    if (not ordered and not allow_empty) or len(set(ordered)) != len(ordered):
        raise ValueError(f"{label} must be ordered and unique")
    return ordered


__all__ = [
    "TEACHER_LABEL_ARTIFACT_KIND",
    "TEACHER_TRAINING_ARTIFACT_KIND",
    "TEACHER_READOUT_ARTIFACT_KIND",
    "TeacherDatasetHandle",
    "TeacherDescriptorSchema",
    "TeacherLabelIntent",
    "TeacherLabelOutcome",
    "TeacherLabelingCapability",
    "TeacherModelCapability",
    "TeacherModelHandle",
    "TeacherTrainingArtifactFacts",
    "TeacherTrainingRecoveryFacts",
    "TeacherScreenRecoveryFacts",
    "TeacherScanRecoveryFacts",
    "TeacherOutcomeStatus",
    "TeacherReadoutCapability",
    "TeacherReferenceArtifactFacts",
    "TeacherReferenceIntent",
    "TeacherReferenceOutcome",
    "TeacherScoreIntent",
    "TeacherScoreOutcome",
    "TeacherStructureBatchHandle",
    "TeacherStructureArtifactFacts",
    "TeacherTrainingCapability",
    "TeacherTrainingIntent",
    "TeacherTrainingOutcome",
    "require_teacher_models",
    "require_teacher_reference",
    "require_teacher_structures",
]
