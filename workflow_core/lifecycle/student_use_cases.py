from __future__ import annotations

from collections.abc import Mapping

from workflow_core.lifecycle.student_capabilities import (
    StudentDescriptorSchema,
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


class StudentTrainingUseCase:
    """Execute path-free training and enforce exact ordered members."""

    def __init__(self, capability: StudentTrainingCapability) -> None:
        self.capability = capability

    def execute(self, request: StudentTrainingIntent) -> StudentTrainingOutcome:
        outcome = _training_outcome(self.capability.train(request))
        expected = request.member_ids
        actual = tuple(model.member_id for model in outcome.models)
        _validate_ordered_status(outcome.status, expected, actual, label="training")
        return outcome


class StudentReadoutUseCase:
    """Execute path-free readout and preserve exact candidate order."""

    def __init__(self, capability: StudentReadoutCapability) -> None:
        self.capability = capability

    def execute(self, request: StudentReadoutIntent) -> StudentReadoutOutcome:
        outcome = _readout_outcome(self.capability.readout(request))
        expected = request.candidates.ordered_structure_ids
        _validate_ordered_status(
            outcome.status,
            expected,
            outcome.ordered_structure_ids,
            label="readout",
        )
        if (
            request.emit_frame_means
            and outcome.status
            in {StudentOutcomeStatus.SUCCEEDED, StudentOutcomeStatus.VALIDATED_REUSE}
            and outcome.frame_means is None
        ):
            raise ValueError("successful student readout omitted requested frame means")
        return outcome


class StudentModelUseCase:
    """Inspect path-free model references and preserve ordered membership."""

    def __init__(self, capability: StudentModelCapability) -> None:
        self.capability = capability

    def inspect(self, models: tuple[StudentModelRef, ...]) -> StudentModelOutcome:
        expected = tuple(model.member_id for model in models)
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("student model inspection requires ordered unique members")
        outcome = _model_outcome(self.capability.inspect(models))
        actual = tuple(model.member_id for model in outcome.models)
        _validate_ordered_status(outcome.status, expected, actual, label="model")
        return outcome


def _validate_ordered_status(
    status: StudentOutcomeStatus,
    expected: tuple[str, ...],
    actual: tuple[str, ...],
    *,
    label: str,
) -> None:
    if status in {StudentOutcomeStatus.SUCCEEDED, StudentOutcomeStatus.VALIDATED_REUSE}:
        if actual != expected:
            raise ValueError(f"successful student {label} changed exact member order")
        return
    if status is StudentOutcomeStatus.PARTIAL:
        expected_positions = {identity: index for index, identity in enumerate(expected)}
        try:
            positions = tuple(expected_positions[identity] for identity in actual)
        except KeyError as exc:
            raise ValueError(f"partial student {label} returned an unknown identity") from exc
        if not actual or positions != tuple(sorted(set(positions))) or len(actual) >= len(expected):
            raise ValueError(
                f"partial student {label} must retain a strict ordered subset"
            )
        return
    if actual:
        raise ValueError(f"failed student {label} cannot publish successful members")


def _training_outcome(value: object) -> StudentTrainingOutcome:
    if isinstance(value, StudentTrainingOutcome):
        return value
    return StudentTrainingOutcome(
        status=StudentOutcomeStatus(getattr(value, "status")),
        models=tuple(_model_ref(model) for model in getattr(value, "models", ())),
        reason=getattr(value, "reason", None),
    )


def _model_outcome(value: object) -> StudentModelOutcome:
    if isinstance(value, StudentModelOutcome):
        return value
    return StudentModelOutcome(
        status=StudentOutcomeStatus(getattr(value, "status")),
        models=tuple(_model_ref(model) for model in getattr(value, "models", ())),
        reason=getattr(value, "reason", None),
    )


def _readout_outcome(value: object) -> StudentReadoutOutcome:
    if isinstance(value, StudentReadoutOutcome):
        return value
    schema = getattr(value, "descriptor_schema", None)
    return StudentReadoutOutcome(
        status=StudentOutcomeStatus(getattr(value, "status")),
        ordered_structure_ids=tuple(getattr(value, "ordered_structure_ids", ())),
        frame_probabilities=tuple(getattr(value, "frame_probabilities", ())),
        descriptor_schema=None if schema is None else _descriptor_schema(schema),
        frame_means=getattr(value, "frame_means", None),
        element_frame_means=getattr(value, "element_frame_means", None),
        reason=getattr(value, "reason", None),
    )


def _model_ref(value: object) -> StudentModelRef:
    if isinstance(value, StudentModelRef):
        return value
    return StudentModelRef(
        member_id=str(getattr(value, "member_id")),
        model_id=str(getattr(value, "model_id")),
    )


def _descriptor_schema(value: object) -> StudentDescriptorSchema:
    if isinstance(value, StudentDescriptorSchema):
        return value
    if isinstance(value, Mapping):
        known = {"name", "revision", "backend", "width"}
        return StudentDescriptorSchema(
            name=str(value["name"]),
            revision=int(value["revision"]),
            backend=str(value["backend"]),
            width=int(value["width"]),
            attributes={str(key): item for key, item in value.items() if key not in known},
        )
    return StudentDescriptorSchema(
        name=str(getattr(value, "name")),
        revision=int(getattr(value, "revision")),
        backend=str(getattr(value, "backend")),
        width=int(getattr(value, "width")),
        attributes=dict(getattr(value, "attributes", {})),
    )


__all__ = [
    "StudentModelUseCase",
    "StudentReadoutUseCase",
    "StudentTrainingUseCase",
]
