from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from workflow_core.lifecycle.sampling_models import (
    SamplingModelInputs,
    SamplingModelPreparation,
    SamplingModelProvider,
    SamplingModelsResult,
    require_sampling_model_provider,
    validate_sampling_models_result as validate_semantic_sampling_models_result,
)


@dataclass(frozen=True, slots=True)
class OrderedSamplingModels:
    """Ordered sampling inputs paired with complete lifecycle member identities."""

    paths: tuple[Path, ...]
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_sampling_model_inputs(self)
        object.__setattr__(
            self, "paths", tuple(Path(path).resolve() for path in self.paths)
        )
        object.__setattr__(
            self, "member_ids", tuple(str(item).strip() for item in self.member_ids)
        )


@dataclass(frozen=True, slots=True)
class SamplingModelsRequest:
    """Temporary path-backed provider request retained outside lifecycle logic."""

    models: SamplingModelInputs
    sample_dir: Path

    def __post_init__(self) -> None:
        validate_sampling_model_inputs(self.models)
        object.__setattr__(self, "sample_dir", Path(self.sample_dir).resolve())


def validate_sampling_model_inputs(models: SamplingModelInputs) -> None:
    """Validate the shared file and member-identity contract without flattening."""

    raw_paths = tuple(Path(path) for path in models.paths)
    member_ids = tuple(str(item).strip() for item in models.member_ids)
    if not raw_paths:
        raise ValueError("sampling models require at least one ordered model input")
    if any(
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        for path in raw_paths
    ):
        raise ValueError("sampling model inputs must be non-empty regular files")
    if (
        len(member_ids) != len(raw_paths)
        or any(not item for item in member_ids)
        or len(set(member_ids)) != len(member_ids)
    ):
        raise ValueError(
            "sampling model member identities must be ordered, unique, and complete"
        )


def validate_sampling_models_result(
    request: SamplingModelPreparation | SamplingModelsRequest,
    result: SamplingModelsResult,
) -> SamplingModelInputs:
    """Return the concrete result model set after exact identity attestation."""

    validate_sampling_model_inputs(request.models)
    validate_sampling_model_inputs(result.models)
    return validate_semantic_sampling_models_result(request, result)


__all__ = [
    "OrderedSamplingModels",
    "SamplingModelInputs",
    "SamplingModelProvider",
    "SamplingModelsRequest",
    "SamplingModelsResult",
    "require_sampling_model_provider",
    "validate_sampling_model_inputs",
    "validate_sampling_models_result",
]
