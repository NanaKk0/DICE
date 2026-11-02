from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from workflow_artifacts.lifecycle.model_publication import (
    ModelPublicationAdapter,
    PublishedModelHandle,
    ResolvedModelArtifact,
)
from workflow_backends.teacher.role import (
    TeacherHistoricalModelRequest,
    TeacherPublishedTrainingValidationRequest,
)
from workflow_backends.teacher.training import (
    TeacherTrainPublicationRequest,
    TeacherTrainRequest,
)
from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
from workflow_core.lifecycle.teacher_capabilities import (
    TeacherTrainingArtifactFacts,
    TeacherModelHandle,
    TeacherOutcomeStatus,
    TeacherTrainingIntent,
    TeacherTrainingOutcome,
    require_teacher_models,
)


class _TeacherArtifactValidator(Protocol):
    expectation: ExpectedArtifactSemantics[TeacherTrainingArtifactFacts]

    def __call__(self, root: Path) -> bool: ...


@dataclass(slots=True)
class BoundTeacherTrainingCapability:
    """Bind physical teacher training/publication around one semantic intent.

    The lifecycle handler supplies a fresh staging root.  This adapter owns all
    recipe and dataset locations and reuses the shared model publication and
    artifact validation adapters; it never creates another model-handle format.
    """

    delegate: Any
    semantic_intent: TeacherTrainingIntent
    physical_request: TeacherTrainRequest
    stage_root: Path
    expected_artifacts: ExpectedArtifactSemantics[TeacherTrainingArtifactFacts]
    artifact_validator: _TeacherArtifactValidator
    foundation_artifact: ResolvedModelArtifact | None = None

    def __post_init__(self) -> None:
        self.stage_root = Path(self.stage_root).resolve()
        expected = require_teacher_models(
            self.expected_artifacts,
            member_ids=self.semantic_intent.member_ids,
            dataset=self.semantic_intent.dataset,
        )
        if self.artifact_validator.expectation != self.expected_artifacts:
            raise ValueError(
                "teacher training must reuse the lifecycle-bound artifact expectation"
            )
        request = self.physical_request
        if (
            request.iter_index != self.semantic_intent.iter_index
            or request.member_ids != self.semantic_intent.member_ids
            or request.model_shape != self.semantic_intent.model_shape
            or request.type_map != self.semantic_intent.type_map
            or request.policy != self.semantic_intent.policy
            or dict(request.train_options) != dict(self.semantic_intent.train_options)
            or request.validation_mode != self.semantic_intent.dataset.validation_mode
            or request.train_dir != self.stage_root
            or tuple(model.member_id for model in expected) != request.member_ids
        ):
            raise ValueError(
                "bound teacher physical request changed canonical training intent"
            )
        semantic_foundation = self.semantic_intent.foundation_model
        physical_foundation = request.foundation_asset
        if semantic_foundation is None:
            if physical_foundation is not None or self.foundation_artifact is not None:
                raise ValueError(
                    "scratch teacher training cannot bind a foundation artifact"
                )
        elif (
            physical_foundation is None
            or self.foundation_artifact is None
            or self.foundation_artifact.handle != semantic_foundation
            or self.foundation_artifact.path != physical_foundation.path
        ):
            raise ValueError(
                "teacher foundation model identity does not match its physical asset"
            )

    def train(self, request: TeacherTrainingIntent) -> TeacherTrainingOutcome:
        if request != self.semantic_intent:
            raise ValueError("teacher training request does not match its bound intent")
        expected = require_teacher_models(
            self.expected_artifacts,
            member_ids=request.member_ids,
            dataset=request.dataset,
        )
        publication = ModelPublicationAdapter("teacher")
        if self.artifact_validator(self.stage_root):
            resolved = publication.resolve(
                expected_handles=expected,
                stage_root=self.stage_root,
                legacy_resolver=self._recover_legacy,
            )
            if not self._validate_backend(resolved):
                raise ValueError(
                    "teacher model artifacts passed schema validation but failed "
                    "backend validation"
                )
            return TeacherTrainingOutcome(
                TeacherOutcomeStatus.VALIDATED_REUSE,
                tuple(item.handle for item in resolved),
            )

        result = self.delegate.train(self.physical_request)
        resolved = publication.publish(
            tuple(result.member_artifacts),
            expected_handles=expected,
            stage_root=self.stage_root,
        )
        physical = tuple(
            PublishedModelHandle(item.handle.member_id, item.path)
            for item in resolved
        )
        self.delegate.finalize_published_training(
            TeacherTrainPublicationRequest(
                train=result,
                handles=physical,
                stage_root=self.stage_root,
                member_ids=request.member_ids,
            )
        )
        if not self._validate_backend(resolved) or not self.artifact_validator(
            self.stage_root
        ):
            raise ValueError("teacher model publication failed bound validation")
        return TeacherTrainingOutcome(
            TeacherOutcomeStatus.SUCCEEDED,
            tuple(item.handle for item in resolved),
        )

    def validate_published_training(
        self, models: tuple[TeacherModelHandle, ...]
    ) -> bool:
        expected = require_teacher_models(
            self.expected_artifacts,
            member_ids=tuple(model.member_id for model in models),
        )
        if tuple(models) != expected or not self.artifact_validator(self.stage_root):
            return False
        try:
            resolved = ModelPublicationAdapter("teacher").resolve(
                expected_handles=expected,
                stage_root=self.stage_root,
                legacy_resolver=self._recover_legacy,
            )
        except (OSError, TypeError, ValueError):
            return False
        return self._validate_backend(resolved)

    def recover_historical_model(self, member_id: str) -> TeacherModelHandle:
        expected = require_teacher_models(
            self.expected_artifacts,
            member_ids=self.semantic_intent.member_ids,
        )
        resolved = ModelPublicationAdapter("teacher").resolve(
            expected_handles=expected,
            stage_root=self.stage_root,
            legacy_resolver=self._recover_legacy,
        )
        for item in resolved:
            if item.handle.member_id == member_id:
                return item.handle
        raise ValueError("teacher historical recovery requested an unknown member")

    def _validate_backend(self, resolved: tuple[Any, ...]) -> bool:
        handles = tuple(
            PublishedModelHandle(item.handle.member_id, item.path)
            for item in resolved
        )
        return bool(
            self.delegate.validate_published_training(
                TeacherPublishedTrainingValidationRequest(
                    stage_root=self.stage_root,
                    handles=handles,
                    member_ids=self.semantic_intent.member_ids,
                )
            )
        )

    def _recover_legacy(self, member_id: str, legacy_handle: Path) -> Path:
        recovered = self.delegate.recover_historical_model(
            TeacherHistoricalModelRequest(member_id, legacy_handle)
        )
        if recovered.member_id != member_id:
            raise ValueError("teacher historical recovery changed member identity")
        return recovered.path


__all__ = ["BoundTeacherTrainingCapability"]
