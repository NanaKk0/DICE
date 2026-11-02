from __future__ import annotations

from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
from workflow_core.lifecycle.fp_capabilities import (
    FirstPrinciplesLabelArtifactFacts,
    FirstPrinciplesLabelCapability,
    FirstPrinciplesLabelIntent,
    FirstPrinciplesLabelOutcome,
    require_fp_label_artifacts,
    validate_fp_label_outcome,
)


class FirstPrinciplesLabelUseCase:
    """Validate one calculator-neutral result against exact selected order."""

    def __init__(self, capability: FirstPrinciplesLabelCapability) -> None:
        self.capability = capability

    def execute(
        self,
        request: FirstPrinciplesLabelIntent,
        expected_artifacts: ExpectedArtifactSemantics[
            FirstPrinciplesLabelArtifactFacts
        ] | None = None,
    ) -> FirstPrinciplesLabelOutcome:
        if expected_artifacts is not None:
            require_fp_label_artifacts(expected_artifacts, intent=request)
        return validate_fp_label_outcome(request, self.capability.label(request))


__all__ = ["FirstPrinciplesLabelUseCase"]
