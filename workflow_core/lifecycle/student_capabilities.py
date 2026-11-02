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
from workflow_core.lifecycle.state_model import PublicBlockId


class StudentOutcomeStatus(str, Enum):
    """Stable lifecycle-visible student outcome classes."""

    SUCCEEDED = "succeeded"
    VALIDATED_REUSE = "validated-reuse"
    PARTIAL = "partial"
    FAILED = "failed"


# Transitional C02 names remain import-compatible while sharing E01's sole
# semantic handle definitions.  These are aliases, not parallel DTOs.
StudentDatasetRef = DatasetHandle
StudentModelRef = ModelHandle
StudentStructureBatchRef = StructureBatchHandle


@dataclass(frozen=True, slots=True)
class StudentDescriptorSchema:
    """Logic-owned descriptor compatibility facts."""

    name: str
    revision: int
    backend: str
    width: int
    attributes: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        name = _identity(self.name, label="student descriptor schema")
        backend = _identity(self.backend, label="student descriptor backend")
        if isinstance(self.revision, bool) or int(self.revision) <= 0:
            raise ValueError("student descriptor revision must be positive")
        if isinstance(self.width, bool) or int(self.width) <= 0:
            raise ValueError("student descriptor width must be positive")
        attributes = copy.deepcopy(dict(self.attributes))
        if any(not isinstance(key, str) or not key for key in attributes):
            raise ValueError(
                "student descriptor attributes require non-empty string keys"
            )
        if any(
            not _valid_descriptor_fact(value)
            for value in attributes.values()
        ):
            raise ValueError("student descriptor attributes must be finite scalar facts")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "width", int(self.width))
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
class StudentTrainingIntent:
    iter_index: int
    round_index: int
    dataset: StudentDatasetRef
    model_shape: str
    member_ids: tuple[str, ...]
    type_map: tuple[str, ...]
    policy: str
    foundation_model: StudentModelRef | None = None
    train_options: Mapping[str, bool] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        members = _ordered_identities(self.member_ids, label="student members")
        type_map = _ordered_identities(self.type_map, label="student type map")
        valid_shape = (self.model_shape == "single" and len(members) == 1) or (
            self.model_shape == "committee" and len(members) >= 2
        )
        if not valid_shape:
            raise ValueError("student model shape does not match ordered members")
        if self.policy not in {"from_scratch", "foundation_finetune"}:
            raise ValueError("student training policy is invalid")
        if (self.policy == "foundation_finetune") != (
            self.foundation_model is not None
        ):
            raise ValueError(
                "student foundation model presence must match the training policy"
            )
        options = dict(self.train_options)
        if any(not isinstance(key, str) or not key for key in options):
            raise ValueError("student train options require non-empty string keys")
        if any(not isinstance(value, bool) for value in options.values()):
            raise ValueError("student train options require boolean values")
        object.__setattr__(self, "iter_index", int(self.iter_index))
        object.__setattr__(self, "round_index", int(self.round_index))
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "type_map", type_map)
        object.__setattr__(self, "train_options", MappingProxyType(options))


@dataclass(frozen=True, slots=True)
class StudentTrainingOutcome:
    status: StudentOutcomeStatus
    models: tuple[StudentModelRef, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        status = StudentOutcomeStatus(self.status)
        models = tuple(self.models)
        members = tuple(model.member_id for model in models)
        if len(set(members)) != len(members):
            raise ValueError("student training outcome duplicates a member")
        reason = _outcome_reason(status, self.reason)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class StudentTrainingExpectation:
    """Complete path-free train semantics supplied to E03 validation."""

    intent: StudentTrainingIntent
    models: tuple[ModelHandle, ...]

    def __post_init__(self) -> None:
        models = tuple(self.models)
        if not models or any(not isinstance(model, ModelHandle) for model in models):
            raise TypeError("student train expectation requires canonical model handles")
        if tuple(model.member_id for model in models) != self.intent.member_ids:
            raise ValueError(
                "student train expectation changed the ordered model members"
            )
        if len({model.model_id for model in models}) != len(models):
            raise ValueError("student train expectation model identities must be unique")
        object.__setattr__(self, "models", models)

    def for_block(
        self, block: PublicBlockId
    ) -> ExpectedArtifactSemantics[StudentTrainingExpectation]:
        if block != PublicBlockId.round(self.intent.round_index, "train"):
            raise ValueError("student train expectation is bound to another block")
        return ExpectedArtifactSemantics(block, "student-training", self)

    def semantic_fields(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "iter_index": self.intent.iter_index,
                "round_index": self.intent.round_index,
                "dataset": {
                    "dataset_id": self.intent.dataset.dataset_id,
                    "ordered_structure_ids": self.intent.dataset.ordered_structure_ids,
                },
                "models": tuple(
                    {"member_id": model.member_id, "model_id": model.model_id}
                    for model in self.models
                ),
                "model_shape": self.intent.model_shape,
                "type_map": self.intent.type_map,
                "policy": self.intent.policy,
                "foundation_model": (
                    None
                    if self.intent.foundation_model is None
                    else {
                        "member_id": self.intent.foundation_model.member_id,
                        "model_id": self.intent.foundation_model.model_id,
                    }
                ),
                "train_options": dict(self.intent.train_options),
            }
        )


@dataclass(frozen=True, slots=True)
class StudentTrainingRecoveryFacts:
    """Path-free settings needed to validate one committed student train."""

    backend: str
    cli: str | None
    model_shape: str
    member_ids: tuple[str, ...]
    type_map: tuple[str, ...]

    def __post_init__(self) -> None:
        backend = _identity(self.backend, label="student recovery backend")
        cli = None if self.cli is None else _identity(self.cli, label="student recovery CLI")
        model_shape = str(self.model_shape).strip()
        members = _ordered_identities(
            tuple(self.member_ids), label="student recovery members"
        )
        type_map = _ordered_identities(
            tuple(self.type_map), label="student recovery type map"
        )
        if model_shape not in {"single", "committee"}:
            raise ValueError("student recovery shape must be single or committee")
        if (model_shape == "single") != (len(members) == 1):
            raise ValueError("student recovery members disagree with model shape")
        if backend == "DeepMD" and cli not in {"tf", "pt"}:
            raise ValueError("DeepMD student recovery requires tf or pt CLI")
        if backend != "DeepMD" and cli is not None:
            raise ValueError("non-DeepMD student recovery cannot declare CLI")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "cli", cli)
        object.__setattr__(self, "model_shape", model_shape)
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "type_map", type_map)


@dataclass(frozen=True, slots=True)
class StudentReadoutIntent:
    model: StudentModelRef
    reference: StudentDatasetRef
    candidates: StudentStructureBatchRef
    chunk_size: int
    emit_frame_means: bool
    type_map: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.chunk_size, bool) or int(self.chunk_size) <= 0:
            raise ValueError("student readout chunk size must be positive")
        object.__setattr__(self, "chunk_size", int(self.chunk_size))
        object.__setattr__(
            self,
            "type_map",
            _ordered_identities(self.type_map, label="student readout type map"),
        )


@dataclass(frozen=True, slots=True)
class StudentReadoutOutcome:
    status: StudentOutcomeStatus
    ordered_structure_ids: tuple[str, ...]
    frame_probabilities: tuple[float, ...]
    descriptor_schema: StudentDescriptorSchema | None
    frame_means: tuple[tuple[float, ...], ...] | None = None
    reason: str | None = None
    element_frame_means: Mapping[str, Mapping[str, tuple[float, ...]]] | None = None

    def __post_init__(self) -> None:
        status = StudentOutcomeStatus(self.status)
        identities = _ordered_identities(
            self.ordered_structure_ids,
            label="student readout structures",
            allow_empty=True,
        )
        probabilities = tuple(float(value) for value in self.frame_probabilities)
        if len(probabilities) != len(identities):
            raise ValueError("student readout evidence does not cover every structure")
        if not all(math.isfinite(value) for value in probabilities):
            raise ValueError("student readout evidence must be finite")
        if status in {
            StudentOutcomeStatus.SUCCEEDED,
            StudentOutcomeStatus.VALIDATED_REUSE,
        } and self.descriptor_schema is None:
            raise ValueError("successful student readout requires a descriptor schema")
        means = None
        if self.frame_means is not None:
            if self.descriptor_schema is None:
                raise ValueError("student frame means require a descriptor schema")
            means = tuple(tuple(float(value) for value in row) for row in self.frame_means)
            if len(means) != len(identities) or any(
                len(row) != self.descriptor_schema.width for row in means
            ):
                raise ValueError(
                    "student frame means do not match descriptor width and order"
                )
            if any(not math.isfinite(value) for row in means for value in row):
                raise ValueError("student frame means must be finite")
        reason = _outcome_reason(status, self.reason)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "ordered_structure_ids", identities)
        object.__setattr__(self, "frame_probabilities", probabilities)
        object.__setattr__(self, "frame_means", means)
        if self.element_frame_means is not None:
            if self.descriptor_schema is None or means is None:
                raise ValueError("student element means require frame means and a descriptor schema")
            positions = {identity: index for index, identity in enumerate(identities)}
            elements = {}
            for element, raw_rows in self.element_frame_means.items():
                if not isinstance(element, str) or not element:
                    raise ValueError("student element means require element identities")
                rows = {identity: tuple(float(value) for value in row) for identity, row in raw_rows.items()}
                try:
                    ordinals = tuple(positions[identity] for identity in rows)
                except KeyError as exc:
                    raise ValueError("student element means contain an unknown frame") from exc
                if ordinals != tuple(sorted(set(ordinals))) or any(len(row) != self.descriptor_schema.width or not all(math.isfinite(value) for value in row) for row in rows.values()):
                    raise ValueError("student element means changed canonical order or width")
                elements[element] = MappingProxyType(rows)
            object.__setattr__(self, "element_frame_means", MappingProxyType(elements))
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class StudentModelOutcome:
    status: StudentOutcomeStatus
    models: tuple[StudentModelRef, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        status = StudentOutcomeStatus(self.status)
        models = tuple(self.models)
        members = tuple(model.member_id for model in models)
        if len(set(members)) != len(members):
            raise ValueError("student model outcome duplicates a member")
        reason = _outcome_reason(status, self.reason)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class StudentScreenOperationProfiles:
    gmm_reference: str | None
    gmm_score: str | None


@runtime_checkable
class StudentTrainingCapability(Protocol):
    def train(self, request: StudentTrainingIntent) -> StudentTrainingOutcome: ...


@runtime_checkable
class StudentReadoutCapability(Protocol):
    def readout(self, request: StudentReadoutIntent) -> StudentReadoutOutcome: ...


@runtime_checkable
class StudentModelCapability(Protocol):
    def inspect(self, models: tuple[StudentModelRef, ...]) -> StudentModelOutcome: ...


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


def _outcome_reason(
    status: StudentOutcomeStatus, reason: str | None
) -> str | None:
    normalized = None if reason is None else str(reason).strip()
    if status in {StudentOutcomeStatus.PARTIAL, StudentOutcomeStatus.FAILED}:
        if not normalized:
            raise ValueError(f"{status.value} student outcome requires a reason")
    elif normalized:
        raise ValueError("successful or reused student outcome cannot carry a reason")
    return normalized


__all__ = [
    "StudentDatasetRef",
    "StudentDescriptorSchema",
    "StudentModelCapability",
    "StudentModelRef",
    "StudentModelOutcome",
    "StudentOutcomeStatus",
    "StudentReadoutCapability",
    "StudentReadoutIntent",
    "StudentReadoutOutcome",
    "StudentScreenOperationProfiles",
    "StudentStructureBatchRef",
    "StudentTrainingCapability",
    "StudentTrainingIntent",
    "StudentTrainingExpectation",
    "StudentTrainingRecoveryFacts",
    "StudentTrainingOutcome",
]
