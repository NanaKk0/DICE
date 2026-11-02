from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
from workflow_core.lifecycle.artifact_handles import ModelHandle
from workflow_core.lifecycle.state_model import PublicBlockId


class SamplingModelInputs(Protocol):
    """Ordered semantic model identity; physical resolution stays in adapters."""

    @property
    def member_ids(self) -> tuple[str, ...]: ...


class SamplingModelPreparation(Protocol):
    """Semantic identity supplied when preparing or loading sampling models."""

    @property
    def iter_index(self) -> int: ...

    @property
    def round_index(self) -> int: ...

    @property
    def models(self) -> SamplingModelInputs | tuple[ModelHandle, ...]: ...


@dataclass(frozen=True, slots=True)
class SamplingModelPreparationExpectation:
    """Path-free model-preparation facts bound to physical evidence by E03."""

    iter_index: int
    round_index: int
    models: tuple[ModelHandle, ...]
    backend: str
    compress: bool
    attributes: Mapping[str, str | int | bool]

    def __post_init__(self) -> None:
        if isinstance(self.iter_index, bool) or int(self.iter_index) < 0:
            raise ValueError("sampling preparation iter index must be non-negative")
        if isinstance(self.round_index, bool) or int(self.round_index) < 0:
            raise ValueError("sampling preparation round index must be non-negative")
        models = _validated_model_handles(self.models)
        backend = str(self.backend).strip()
        if not backend:
            raise ValueError("sampling preparation backend must be non-empty")
        attributes = dict(self.attributes)
        if any(not isinstance(key, str) or not key for key in attributes):
            raise ValueError("sampling preparation attributes require named facts")
        if any(not isinstance(value, (str, int, bool)) for value in attributes.values()):
            raise TypeError("sampling preparation attributes must be scalar facts")
        object.__setattr__(self, "iter_index", int(self.iter_index))
        object.__setattr__(self, "round_index", int(self.round_index))
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "compress", bool(self.compress))
        object.__setattr__(self, "attributes", MappingProxyType(attributes))

    def for_block(
        self, block: PublicBlockId
    ) -> ExpectedArtifactSemantics[SamplingModelPreparationExpectation]:
        if block != PublicBlockId.round(self.round_index, "sample"):
            raise ValueError("sampling preparation expectation is bound to another block")
        return ExpectedArtifactSemantics(
            block=block,
            artifact_kind="sampling-model-preparation",
            facts=self,
        )

    def semantic_fields(self) -> Mapping[str, object]:
        """Return stable JSON-shaped model preparation facts."""

        return MappingProxyType(
            {
                "iter_index": self.iter_index,
                "round_index": self.round_index,
                "models": tuple(
                    {"member_id": model.member_id, "model_id": model.model_id}
                    for model in self.models
                ),
                "backend": self.backend,
                "compress": self.compress,
                "attributes": dict(self.attributes),
            }
        )


@dataclass(frozen=True, slots=True)
class SamplingModelsResult:
    models: SamplingModelInputs | tuple[ModelHandle, ...]

    def __post_init__(self) -> None:
        _model_member_ids(self.models)


@runtime_checkable
class SamplingModelProvider(Protocol):
    def prepare_sampling_models(
        self, request: SamplingModelPreparation
    ) -> SamplingModelsResult: ...

    def load_sampling_models(
        self, request: SamplingModelPreparation
    ) -> SamplingModelsResult: ...


def require_sampling_model_provider(provider: object) -> SamplingModelProvider:
    if not isinstance(provider, SamplingModelProvider):
        raise TypeError("sampling backend requires a SamplingModelProvider")
    return provider


def validate_sampling_models_result(
    request: SamplingModelPreparation,
    result: SamplingModelsResult,
) -> SamplingModelInputs | tuple[ModelHandle, ...]:
    expected = _model_member_ids(request.models)
    actual = _model_member_ids(result.models)
    if actual != expected:
        raise ValueError(
            "sampling-model provider result changed ordered member identities"
        )
    return result.models


def _model_member_ids(
    models: SamplingModelInputs | tuple[ModelHandle, ...],
) -> tuple[str, ...]:
    if isinstance(models, tuple):
        return tuple(model.member_id for model in _validated_model_handles(models))
    return _validated_member_ids(models.member_ids)


def _validated_model_handles(
    models: tuple[ModelHandle, ...],
) -> tuple[ModelHandle, ...]:
    ordered = tuple(models)
    if not ordered or any(not isinstance(model, ModelHandle) for model in ordered):
        raise TypeError("sampling model handles must be canonical ModelHandle values")
    member_ids = tuple(model.member_id for model in ordered)
    model_ids = tuple(model.model_id for model in ordered)
    _validated_member_ids(member_ids)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("sampling model identities must be ordered and unique")
    return ordered


def _validated_member_ids(member_ids: tuple[str, ...]) -> tuple[str, ...]:
    ordered = tuple(str(item).strip() for item in member_ids)
    if not ordered or any(not item for item in ordered) or len(set(ordered)) != len(
        ordered
    ):
        raise ValueError(
            "sampling model member identities must be ordered, unique, and complete"
        )
    return ordered


__all__ = [
    "SamplingModelInputs",
    "SamplingModelPreparation",
    "SamplingModelPreparationExpectation",
    "SamplingModelProvider",
    "SamplingModelsResult",
    "require_sampling_model_provider",
    "validate_sampling_models_result",
]
