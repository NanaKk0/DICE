from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from workflow_backends.sampling_models import SamplingModelProvider
from workflow_backends.student.readout import (
    FusedStudentReadoutOps,
    PhaseStudentReadoutOps,
    StudentReadoutCapability,
)
from workflow_artifacts.lifecycle.model_publication import (
    ModelArtifact,
    ModelPublicationAdapter,
    PublishedModelHandle,
    ResolvedModelArtifact,
)
from workflow_core.lifecycle.artifact_handles import ModelHandle
from workflow_core.lifecycle.state_model import PublicBlockId
from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter
from workflow_runtime.lifecycle.artifact_validation import BoundArtifactValidator
from workflow_core.structures.formats import read_extxyz_structures


@dataclass(frozen=True, slots=True)
class StudentScreenOperationProfiles:
    gmm_reference: str | None
    gmm_score: str | None


def screen_profile_id(context: object | None) -> str | None:
    value = None if context is None else getattr(context, "profile_id", None)
    if value is None:
        return None
    profile_id = str(value).strip()
    return profile_id or None


@dataclass(frozen=True, slots=True)
class StudentBackendBinding:
    """Validated route configuration bound once to the selected student role."""

    run_json_path: Path
    recipe_path: Path
    type_map: tuple[str, ...]
    model_shape: str
    policy: str
    foundation_model: Path | None
    foundation_restart: Path | None
    train_options: Mapping[str, bool]
    compress_sampling_models: bool
    foundation_model_id: str | None = None

    def __post_init__(self) -> None:
        type_map = tuple(str(symbol) for symbol in self.type_map)
        if not type_map:
            raise ValueError("student backend binding requires a non-empty type map")
        if self.model_shape not in {"single", "committee"}:
            raise ValueError("student backend binding has an invalid model shape")
        object.__setattr__(self, "run_json_path", Path(self.run_json_path).resolve())
        object.__setattr__(self, "recipe_path", Path(self.recipe_path).resolve())
        object.__setattr__(self, "type_map", type_map)
        object.__setattr__(
            self,
            "foundation_model",
            None
            if self.foundation_model is None
            else Path(self.foundation_model).resolve(),
        )
        object.__setattr__(
            self,
            "foundation_restart",
            None
            if self.foundation_restart is None
            else Path(self.foundation_restart).resolve(),
        )
        object.__setattr__(self, "train_options", MappingProxyType(dict(self.train_options)))
        foundation_model_id = (
            None
            if self.foundation_model_id is None
            else str(self.foundation_model_id).strip()
        )
        if self.foundation_model_id is not None and not foundation_model_id:
            raise ValueError("student foundation model identity must be non-empty")
        object.__setattr__(self, "foundation_model_id", foundation_model_id)


@dataclass(frozen=True, slots=True)
class StudentTrainRequest:
    dataset_path: Path
    train_dir: Path
    member_ids: tuple[str, ...]
    progress_reporter_factory: Callable[[str], OperationProgressReporter] | None = None

    def __post_init__(self) -> None:
        member_ids = tuple(str(item) for item in self.member_ids)
        if not member_ids or len(set(member_ids)) != len(member_ids) or any(
            not item for item in member_ids
        ):
            raise ValueError("student train member identities must be ordered and unique")
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
    def status(self) -> str:
        return "succeeded"


@dataclass(frozen=True, slots=True)
class StudentTrainPublicationRequest:
    train: StudentTrainResult
    handles: tuple[PublishedModelHandle, ...]
    stage_root: Path
    published_root: Path
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        member_ids = tuple(str(item) for item in self.member_ids)
        if tuple(item.member_id for item in self.train.artifacts) != member_ids:
            raise ValueError("student publication requires ordered train members")
        handles = tuple(self.handles)
        if tuple(item.member_id for item in handles) != member_ids:
            raise ValueError("student publication requires ordered published handles")
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
    def status(self) -> str:
        return "validated-reuse"


class StudentTrainCapability(Protocol):
    def train(self, request: StudentTrainRequest) -> StudentTrainResult: ...

    def finalize_published_training(
        self, request: StudentTrainPublicationRequest
    ) -> None: ...


class StudentHistoricalModelCapability(Protocol):
    def recover_historical_model(
        self, request: StudentHistoricalModelRequest
    ) -> StudentHistoricalModelResult: ...


@dataclass(frozen=True, slots=True)
class _SemanticTrainingResult:
    status: str
    models: tuple[object, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _SemanticReadoutResult:
    status: str
    ordered_structure_ids: tuple[str, ...] = ()
    frame_probabilities: tuple[float, ...] = ()
    descriptor_schema: object | None = None
    frame_means: tuple[tuple[float, ...], ...] | None = None
    reason: str | None = None
    element_frame_means: object | None = None


@dataclass(frozen=True, slots=True)
class _SemanticModelResult:
    status: str
    models: tuple[object, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BoundStudentTrainingCapability:
    """Bind physical training effects outside one path-free semantic intent."""

    service: Any
    intent: object
    request: StudentTrainRequest
    models: tuple[object, ...]
    published_root: Path | None = None
    expectation: object | None = None

    def train(self, intent: object) -> _SemanticTrainingResult:
        if intent != self.intent:
            raise ValueError("student training intent does not match its physical binding")
        if self.expectation is not None and (
            self.expectation.intent != intent
            or self.expectation.models != self.models
        ):
            raise ValueError(
                "student training diverges from its canonical artifact expectation"
            )
        binding = self.service.binding
        semantic_foundation = getattr(intent, "foundation_model")
        physical_foundation = getattr(binding, "foundation_model", None)
        physical_foundation_id = getattr(binding, "foundation_model_id", None)
        if self.expectation is not None and (
            (semantic_foundation is None) != (physical_foundation is None)
            or (
                semantic_foundation is not None
                and (
                    physical_foundation_id is None
                    or getattr(semantic_foundation, "model_id")
                    != physical_foundation_id
                )
            )
        ):
            raise ValueError(
                "student foundation model identity diverges from bound native inputs"
            )
        if (
            tuple(getattr(intent, "member_ids")) != self.request.member_ids
            or getattr(intent, "model_shape") != binding.model_shape
            or tuple(getattr(intent, "type_map")) != binding.type_map
            or getattr(intent, "policy") != binding.policy
            or dict(getattr(intent, "train_options")) != dict(binding.train_options)
            or tuple(getattr(model, "member_id") for model in self.models)
            != self.request.member_ids
        ):
            raise ValueError("student training intent diverges from bound native inputs")
        result = self.service.train(self.request)
        returned = tuple(artifact.member_id for artifact in result.artifacts)
        if returned != self.request.member_ids:
            positions = {member: index for index, member in enumerate(self.request.member_ids)}
            if returned and all(member in positions for member in returned) and tuple(
                positions[member] for member in returned
            ) == tuple(sorted(positions[member] for member in returned)):
                selected = tuple(
                    model for model in self.models if getattr(model, "member_id") in returned
                )
                return _SemanticTrainingResult(
                    "partial", selected, "native training returned a strict member subset"
                )
            return _SemanticTrainingResult(
                "failed", reason="native training changed ordered member identities"
            )
        if self.published_root is not None:
            if any(not isinstance(model, ModelHandle) for model in self.models):
                raise TypeError(
                    "student publication requires canonical ModelHandle expectations"
                )
            resolved = ModelPublicationAdapter("student").publish(
                result.artifacts,
                expected_handles=self.models,
                stage_root=self.request.train_dir,
            )
            if tuple(item.handle for item in resolved) != self.models:
                raise ValueError("student publication changed canonical model handles")
            self.service.finalize_published_training(
                StudentTrainPublicationRequest(
                    train=result,
                    handles=tuple(
                        PublishedModelHandle(item.handle.member_id, item.path)
                        for item in resolved
                    ),
                    stage_root=self.request.train_dir,
                    published_root=self.published_root,
                    member_ids=self.request.member_ids,
                )
            )
        return _SemanticTrainingResult("succeeded", self.models)


@dataclass(frozen=True, slots=True)
class BoundStudentReadoutCapability:
    """Bind physical readout paths while exposing only the path-free port."""

    service: Any
    intent: object
    request: object
    resolved_model: ResolvedModelArtifact | None = None

    def readout(self, intent: object) -> _SemanticReadoutResult:
        if intent != self.intent:
            raise ValueError("student readout intent does not match its physical binding")
        candidates = tuple(getattr(self.request, "candidates"))
        identities = tuple(
            str(frame.info.get("structure_id", "")) for frame in candidates
        )
        if self.resolved_model is not None and (
            self.resolved_model.handle != getattr(intent, "model")
            or self.resolved_model.path != Path(getattr(self.request, "model")).resolve()
        ):
            raise ValueError(
                "student readout model handle diverges from its physical resolution"
            )
        if (
            identities
            != tuple(getattr(getattr(intent, "candidates"), "ordered_structure_ids"))
            or tuple(getattr(self.request, "type_map")) != tuple(getattr(intent, "type_map"))
            or int(getattr(self.request, "chunk_size")) != int(getattr(intent, "chunk_size"))
            or bool(getattr(self.request, "emit_frame_means"))
            != bool(getattr(intent, "emit_frame_means"))
        ):
            raise ValueError("student readout intent diverges from bound native inputs")
        score = self.service.readout.readout_gmm(self.request)
        try:
            actual = tuple(str(identity) for identity in score.values)
            means = None
            if score.frame_means is not None:
                means = tuple(
                    tuple(float(value) for value in score.frame_means[identity])
                    for identity in actual
                )
            status = getattr(score, "status", "succeeded")
            return _SemanticReadoutResult(
                status=str(getattr(status, "value", status)),
                ordered_structure_ids=actual,
                frame_probabilities=tuple(
                    float(score.values[identity]) for identity in actual
                ),
                descriptor_schema=score.descriptor_schema,
                frame_means=means,
                element_frame_means=getattr(score, "element_frame_means", None),
            )
        finally:
            close = getattr(score, "close", None)
            if callable(close):
                close()


@dataclass(frozen=True, slots=True)
class BoundStudentTrainingArtifactValidator:
    """Validate one train artifact against its complete path-free expectation."""

    service: Any
    expectation: object

    def __call__(self, root: Path, expectation: object) -> bool:
        if expectation != self.expectation:
            raise ValueError("student artifact expectation changed after binding")
        intent = getattr(expectation, "intent")
        expected_models = tuple(getattr(expectation, "models"))
        dataset = getattr(intent, "dataset")
        frames = read_extxyz_structures(
            Path(root) / "dataset" / "train.xyz", allow_empty=False
        )
        if tuple(str(frame.info.get("structure_id", "")) for frame in frames) != tuple(
            getattr(dataset, "ordered_structure_ids")
        ):
            return False

        def recover(member_id: str, legacy_handle: Path) -> Path:
            result = self.service.recover_historical_model(
                StudentHistoricalModelRequest(member_id, legacy_handle)
            )
            if result.member_id != member_id:
                raise ValueError("student historical recovery changed member identity")
            return result.path

        resolved = ModelPublicationAdapter("student").resolve(
            expected_handles=expected_models,
            stage_root=root,
            legacy_resolver=recover,
        )
        return tuple(item.handle for item in resolved) == expected_models


def bind_student_training_artifact_validator(
    service: Any, expectation: object
) -> BoundArtifactValidator[object]:
    expected = getattr(expectation, "for_block")(
        PublicBlockId.round(getattr(expectation, "intent").round_index, "train")
    )
    return BoundArtifactValidator(
        expected,
        BoundStudentTrainingArtifactValidator(service, expectation),
    )


def resolve_student_model_expectations(
    service: Any,
    models: tuple[ModelHandle, ...],
    *,
    stage_root: Path,
) -> tuple[ResolvedModelArtifact, ...]:
    expected = tuple(models)

    def recover(member_id: str, legacy_handle: Path) -> Path:
        result = service.recover_historical_model(
            StudentHistoricalModelRequest(member_id, legacy_handle)
        )
        if result.member_id != member_id:
            raise ValueError("student historical recovery changed member identity")
        return result.path

    return ModelPublicationAdapter("student").resolve(
        expected_handles=expected,
        stage_root=stage_root,
        legacy_resolver=recover,
    )


@dataclass(frozen=True, slots=True)
class BoundStudentModelCapability:
    """Classify exact historical model recovery as reuse, partial or failure."""

    service: Any
    models: tuple[object, ...]
    requests: tuple[StudentHistoricalModelRequest, ...]

    def inspect(self, models: tuple[object, ...]) -> _SemanticModelResult:
        if tuple(models) != self.models:
            raise ValueError("student model intent does not match its physical binding")
        if len(self.requests) != len(self.models) or tuple(
            request.member_id for request in self.requests
        ) != tuple(getattr(model, "member_id") for model in self.models):
            raise ValueError("student model refs diverge from bound recovery requests")
        recovered = []
        failures = []
        for model, request in zip(self.models, self.requests, strict=True):
            try:
                result = self.service.recover_historical_model(request)
                if result.member_id != getattr(model, "member_id"):
                    raise ValueError("historical recovery changed member identity")
            except Exception as exc:
                failures.append(str(exc) or type(exc).__name__)
            else:
                recovered.append(model)
        if not failures:
            return _SemanticModelResult("validated-reuse", tuple(recovered))
        if recovered:
            return _SemanticModelResult(
                "partial", tuple(recovered), "; ".join(failures)
            )
        return _SemanticModelResult("failed", reason="; ".join(failures))


class StudentBackend(
    StudentTrainCapability,
    StudentHistoricalModelCapability,
    Protocol,
):
    readout_ops: PhaseStudentReadoutOps | FusedStudentReadoutOps
    readout: StudentReadoutCapability
    sampling_models: SamplingModelProvider
    screen_operation_profiles: StudentScreenOperationProfiles


class StudentBackendFactory(Protocol):
    def bind(self, binding: StudentBackendBinding) -> StudentBackend: ...


__all__ = [
    "StudentBackend",
    "StudentBackendBinding",
    "StudentBackendFactory",
    "StudentHistoricalModelCapability",
    "StudentHistoricalModelRequest",
    "StudentHistoricalModelResult",
    "StudentTrainCapability",
    "StudentTrainPublicationRequest",
    "StudentTrainRequest",
    "StudentTrainResult",
    "StudentScreenOperationProfiles",
    "BoundStudentModelCapability",
    "BoundStudentReadoutCapability",
    "BoundStudentTrainingArtifactValidator",
    "bind_student_training_artifact_validator",
    "BoundStudentTrainingCapability",
    "screen_profile_id",
    "resolve_student_model_expectations",
]
