from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from workflow_core.lifecycle.model_handles import (
    ModelArtifact,
    PublishedModelHandle,
    publish_model_handles,
    resolve_model_handles,
)
from workflow_core.lifecycle.sampling_models import SamplingModelProvider
from workflow_core.lifecycle.student_capabilities import (
    StudentModelCapability,
    StudentModelOutcome,
    StudentModelRef,
    StudentOutcomeStatus,
    StudentReadoutCapability,
    StudentReadoutIntent,
    StudentReadoutOutcome,
    StudentTrainingCapability,
    StudentTrainingIntent,
    StudentTrainingOutcome,
)
from workflow_core.lifecycle.student_compatibility import (
    StudentHistoricalModelRequest,
    StudentTrainPublicationRequest,
    StudentTrainRequest,
)
from workflow_core.lifecycle.student_readout import StudentGMMScore
from workflow_core.lifecycle.student_use_cases import (
    StudentModelUseCase,
    StudentReadoutUseCase,
    StudentTrainingUseCase,
)


class _PathBackedReadoutCapability(Protocol):
    """Temporary adapter wrapper removed when shared handles land in Child E."""

    def readout_gmm(self, request: object) -> StudentGMMScore: ...


class _PathBackedTrainingCapability(Protocol):
    def train(self, request: StudentTrainRequest) -> object: ...

    def finalize_published_training(self, request: StudentTrainPublicationRequest) -> None: ...


class _PathBackedModelCapability(Protocol):
    def recover_historical_model(self, request: StudentHistoricalModelRequest) -> object: ...


@dataclass(frozen=True, slots=True)
class _UnavailableCompatibilityReadout:
    def readout_gmm(self, request: object) -> StudentGMMScore:
        del request
        raise TypeError("bound student role has no executable readout capability")


@dataclass(slots=True)
class LifecycleStudentReadoutCoordinator:
    """Path-free coordinator over the logic-owned readout capability."""

    readout: StudentReadoutCapability

    def execute(self, request: StudentReadoutIntent) -> StudentReadoutOutcome:
        return StudentReadoutUseCase(self.readout).execute(request)


@dataclass(slots=True)
class LifecycleStudentTrainingCoordinator:
    """Path-free coordinator over the logic-owned training capability."""

    training: StudentTrainingCapability

    def execute(self, request: StudentTrainingIntent) -> StudentTrainingOutcome:
        return StudentTrainingUseCase(self.training).execute(request)


@dataclass(slots=True)
class LifecycleStudentModelCoordinator:
    """Path-free coordinator over the logic-owned model capability."""

    models: StudentModelCapability

    def inspect(self, models: tuple[StudentModelRef, ...]) -> StudentModelOutcome:
        return StudentModelUseCase(self.models).inspect(models)


@dataclass(slots=True)
class _PathBackedPublicationCoordinator:
    """Temporary publication shell pending Child E handle integration."""

    training: _PathBackedTrainingCapability

    def train_and_publish(
        self,
        request: StudentTrainRequest,
        *,
        published_root,
    ) -> tuple[PublishedModelHandle, ...]:
        result = self.training.train(request)
        artifacts = tuple(getattr(result, "artifacts", ()))
        if any(not isinstance(artifact, ModelArtifact) for artifact in artifacts):
            raise TypeError(
                "student training must return core-publishable model artifacts"
            )
        if tuple(artifact.member_id for artifact in artifacts) != request.member_ids:
            raise ValueError("student training changed ordered member identities")
        handles = publish_model_handles(
            artifacts,
            expected_member_ids=request.member_ids,
            stage_root=request.train_dir,
            role="student",
        )
        self.training.finalize_published_training(
            StudentTrainPublicationRequest(
                train=result,
                handles=handles,
                stage_root=request.train_dir,
                published_root=published_root,
                member_ids=request.member_ids,
            )
        )
        return handles


@dataclass(slots=True)
class LifecycleStudentFacade:
    """Lifecycle student use cases composed from separate narrow capabilities."""

    delegate: object
    _path_readout: _PathBackedReadoutCapability = field(init=False, repr=False)
    _publication: _PathBackedPublicationCoordinator = field(init=False, repr=False)
    _models: _PathBackedModelCapability = field(init=False, repr=False)
    _sampling_models: SamplingModelProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        training = getattr(self.delegate, "training", self.delegate)
        models = getattr(self.delegate, "models", self.delegate)
        for capability, method_name in (
            (training, "train"),
            (training, "finalize_published_training"),
            (models, "recover_historical_model"),
        ):
            if not callable(getattr(capability, method_name, None)):
                raise TypeError(
                    f"bound student role is missing required capability {method_name!r}"
                )
        required: dict[str, Any] = {}
        for capability_name in ("sampling_models",):
            try:
                capability = getattr(self.delegate, capability_name)
            except AttributeError as exc:
                raise TypeError(
                    f"bound student role is missing required capability {capability_name!r}"
                ) from exc
            if capability is None:
                raise TypeError(
                    f"bound student role is missing required capability {capability_name!r}"
                )
            required[capability_name] = capability
        try:
            native_readout = getattr(self.delegate, "readout_ops")
        except AttributeError as exc:
            raise TypeError(
                "bound student role is missing required capability 'readout_ops'"
            ) from exc
        if native_readout is None:
            raise TypeError(
                "bound student role is missing required capability 'readout_ops'"
            )
        readout = getattr(self.delegate, "readout", None)
        if readout is None:
            readout = (
                native_readout
                if callable(getattr(native_readout, "readout_gmm", None))
                else _UnavailableCompatibilityReadout()
            )
        if not callable(getattr(readout, "readout_gmm", None)):
            raise TypeError("bound student readout capability lacks 'readout_gmm'")
        self._path_readout = readout
        self._publication = _PathBackedPublicationCoordinator(training)
        self._models = models
        self._sampling_models = required["sampling_models"]

    @property
    def readout_ops(self) -> object:
        """Temporary native-operation compatibility view removed by Child G."""

        return getattr(self.delegate, "readout_ops")

    @property
    def screen_operation_profiles(self):
        return getattr(self.delegate, "screen_operation_profiles")

    def readout_gmm(self, request: object) -> StudentGMMScore:
        score = self._path_readout.readout_gmm(request)
        if not hasattr(score, "values") or not hasattr(score, "descriptor_schema"):
            raise TypeError("student readout must return an ordered GMM score")
        status = StudentOutcomeStatus(getattr(score, "status", "succeeded"))
        if status not in {
            StudentOutcomeStatus.SUCCEEDED,
            StudentOutcomeStatus.VALIDATED_REUSE,
        }:
            raise ValueError(f"student readout did not complete: {status.value}")
        return score

    def _read_structure_vectors(self, **kwargs):
        capability = getattr(self.delegate, "structure_vectors", None)
        if not callable(capability):
            raise TypeError(
                "bound student role has no structural-vector readout capability"
            )
        return capability(**kwargs)

    def execute_training(
        self,
        request: StudentTrainingIntent,
        *,
        physical_request: StudentTrainRequest,
        models: tuple[StudentModelRef, ...],
        published_root: Path | None = None,
    ) -> StudentTrainingOutcome:
        bind = getattr(self.delegate, "bind_training_capability")
        capability = (
            bind(request, physical_request, models)
            if published_root is None
            else bind(request, physical_request, models, Path(published_root))
        )
        return LifecycleStudentTrainingCoordinator(capability).execute(request)

    def execute_training_expectation(
        self,
        expectation: object,
        *,
        physical_request: StudentTrainRequest,
        published_root: Path,
    ) -> StudentTrainingOutcome:
        bind = getattr(self.delegate, "bind_training_expectation")
        capability = bind(
            expectation,
            physical_request,
            published_root=Path(published_root),
        )
        return LifecycleStudentTrainingCoordinator(capability).execute(
            getattr(expectation, "intent")
        )

    def bind_training_artifact_validator(self, expectation: object) -> object:
        return self.delegate.bind_training_artifact_validator(expectation)

    def resolve_model_expectations(
        self, models: tuple[StudentModelRef, ...], *, stage_root: Path
    ) -> tuple[object, ...]:
        return tuple(
            self.delegate.resolve_model_expectations(
                tuple(models), stage_root=Path(stage_root)
            )
        )

    def execute_readout(
        self,
        request: StudentReadoutIntent,
        *,
        physical_request: object,
        resolved_model: object | None = None,
    ) -> StudentReadoutOutcome:
        bind = getattr(self.delegate, "bind_readout_capability")
        capability = (
            bind(request, physical_request)
            if resolved_model is None
            else bind(request, physical_request, resolved_model)
        )
        return LifecycleStudentReadoutCoordinator(capability).execute(request)

    def inspect_models(
        self,
        models: tuple[StudentModelRef, ...],
        *,
        physical_requests: tuple[StudentHistoricalModelRequest, ...],
    ) -> StudentModelOutcome:
        bind = getattr(self.delegate, "bind_model_capability")
        capability = bind(models, physical_requests)
        return LifecycleStudentModelCoordinator(capability).inspect(models)

    @property
    def sampling_models(self) -> SamplingModelProvider:
        return self._sampling_models

    def train_and_publish(
        self,
        request: StudentTrainRequest,
        *,
        published_root,
    ) -> tuple[PublishedModelHandle, ...]:
        return self._publication.train_and_publish(request, published_root=published_root)

    def resolve_published_models(
        self,
        *,
        stage_root,
        member_ids: tuple[str, ...],
    ) -> tuple[PublishedModelHandle, ...]:
        def recover(member_id, legacy_handle):
            result = self._models.recover_historical_model(
                StudentHistoricalModelRequest(member_id, legacy_handle)
            )
            if result.member_id != member_id:
                raise ValueError(
                    "student historical recovery changed the requested member identity"
                )
            return result.path

        return resolve_model_handles(
            stage_root=stage_root,
            role="student",
            expected_member_ids=member_ids,
            legacy_resolver=recover,
        )

    def validate_published_training(
        self,
        *,
        stage_root,
        member_ids: tuple[str, ...],
    ) -> bool:
        try:
            return bool(
                self.resolve_published_models(
                    stage_root=stage_root,
                    member_ids=member_ids,
                )
            )
        except (OSError, TypeError, ValueError):
            return False


__all__ = [
    "LifecycleStudentFacade",
    "LifecycleStudentReadoutCoordinator",
    "LifecycleStudentModelCoordinator",
    "LifecycleStudentTrainingCoordinator",
]
