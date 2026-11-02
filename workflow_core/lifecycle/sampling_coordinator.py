from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from workflow_core.lifecycle.sampling import (
    SamplingEvidence,
    SamplingEvidenceFailure,
    SamplingEvidenceObserver,
    SamplingEvidenceRebuildRequired,
    SamplingEvidenceState,
    SamplingExecutionCommand,
    SamplingExecutionOutcome,
    SamplingNativeOperations,
    SamplingObservationResult,
    SamplingValidationResult,
    classify_sampling_evidence,
)
from workflow_core.lifecycle.sampling_models import (
    SamplingModelPreparation,
    SamplingModelProvider,
    SamplingModelsResult,
    require_sampling_model_provider,
    validate_sampling_models_result,
)


@dataclass(frozen=True, slots=True)
class LifecycleSamplingCoordinator:
    """Own lifecycle sequencing around one native sampling capability."""

    native_ops: SamplingNativeOperations
    model_provider: SamplingModelProvider
    evidence_observer: SamplingEvidenceObserver | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_provider",
            require_sampling_model_provider(self.model_provider),
        )
        if self.evidence_observer is None:
            object.__setattr__(self, "evidence_observer", self.native_ops)

    def execute(self, request: SamplingExecutionCommand) -> SamplingExecutionOutcome:
        evidence = self._resolved_evidence(
            request.evidence,
            self.model_provider.prepare_sampling_models,
        )
        return self.native_ops.execute(replace(request, evidence=evidence))

    def validate_completion(
        self, request: SamplingEvidence
    ) -> SamplingValidationResult:
        assert self.evidence_observer is not None
        evidence_state = classify_sampling_evidence(
            self.evidence_observer.observe_evidence(request)
        )
        if evidence_state is SamplingEvidenceState.REBUILD_REQUIRED:
            return SamplingValidationResult(False)
        if evidence_state is SamplingEvidenceState.LEGACY_COMPATIBLE:
            try:
                evidence = self._resolved_evidence(
                    request,
                    self.model_provider.load_sampling_models,
                )
                observation = self.native_ops.reconstruct_compatible_legacy(evidence)
                observation.source.catalog()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SamplingEvidenceRebuildRequired(
                    "legacy sample evidence is incompatible; "
                    f"rebuild from r{request.round_index}/sample",
                    failure=SamplingEvidenceFailure.HISTORICAL_INCOMPATIBLE,
                    round_index=request.round_index,
                ) from exc
            return SamplingValidationResult(True)
        evidence = self._resolved_evidence(
            request,
            self.model_provider.load_sampling_models,
        )
        return self.native_ops.validate_current(evidence)

    def open_observations(
        self, request: SamplingEvidence
    ) -> SamplingObservationResult:
        assert self.evidence_observer is not None
        evidence_state = classify_sampling_evidence(
            self.evidence_observer.observe_evidence(request)
        )
        if evidence_state is SamplingEvidenceState.REBUILD_REQUIRED:
            raise SamplingEvidenceRebuildRequired(
                "sample artifact evidence is partial or ambiguous; "
                f"rebuild from r{request.round_index}/sample",
                failure=SamplingEvidenceFailure.PARTIAL_OR_AMBIGUOUS,
                round_index=request.round_index,
            )
        if evidence_state is SamplingEvidenceState.LEGACY_COMPATIBLE:
            try:
                evidence = self._resolved_evidence(
                    request,
                    self.model_provider.load_sampling_models,
                )
                return self.native_ops.reconstruct_compatible_legacy(evidence)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SamplingEvidenceRebuildRequired(
                    "legacy sample evidence is incompatible; "
                    f"rebuild from r{request.round_index}/sample",
                    failure=SamplingEvidenceFailure.HISTORICAL_INCOMPATIBLE,
                    round_index=request.round_index,
                ) from exc
        return self.native_ops.open_current(request)

    def _resolved_evidence(
        self,
        request: SamplingEvidence,
        resolve: Callable[[SamplingModelPreparation], SamplingModelsResult],
    ) -> SamplingEvidence:
        result = resolve(request)
        return replace(
            request,
            models=validate_sampling_models_result(request, result),
        )


__all__ = [
    "LifecycleSamplingCoordinator",
    "SamplingEvidenceRebuildRequired",
]
