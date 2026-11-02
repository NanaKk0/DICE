from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
from workflow_core.lifecycle.artifact_handles import StructureBatchHandle
from workflow_core.lifecycle.sampling_plan import SelectionPlan


FP_LABEL_ARTIFACT_KIND = "dice.fp-label-publication"


_CALCULATOR_ROLES = {
    "CP2K": "first_principles.cp2k.label",
    "ABACUS": "first_principles.abacus.label",
}
_KNOWN_LABELS = {"energy", "forces", "stress", "virial"}


@dataclass(frozen=True, slots=True)
class FinalFPRecoveryFacts:
    """Normalized path-free settings for terminal FP publication recovery."""

    keep: int
    policy: str
    all_rn: bool
    round_count: int
    iter_index: int
    system_ids: tuple[str, ...]
    requires_frame_mean_evidence: bool = False
    selection_plan: SelectionPlan | None = None
    require_selection_group_audit: bool = False
    joint_model_identity: str | None = None
    joint_dataset_identity: str | None = None
    element_type_map: tuple[str, ...] | None = None
    effective_policy: str | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.keep, bool) or int(self.keep) <= 0:
            raise ValueError("terminal FP recovery keep must be positive")
        if self.policy not in {
            "DIRECT", "FPS", "random", "JOINT_DIRECT",
            "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT",
        }:
            raise ValueError("terminal FP recovery policy is invalid")
        if not isinstance(self.all_rn, bool):
            raise TypeError("terminal FP all_rn must be boolean")
        if isinstance(self.round_count, bool) or int(self.round_count) <= 0:
            raise ValueError("terminal FP round count must be positive")
        if isinstance(self.iter_index, bool) or int(self.iter_index) < 0:
            raise ValueError("terminal FP iter index must be non-negative")
        systems = tuple(
            _identity(item, label="terminal FP system") for item in self.system_ids
        )
        if len(systems) != len(set(systems)):
            raise ValueError("terminal FP systems must be ordered unique")
        if not isinstance(self.requires_frame_mean_evidence, bool):
            raise TypeError("terminal FP frame-mean mode must be boolean")
        if self.requires_frame_mean_evidence and self.policy not in {"DIRECT", "FPS"}:
            raise ValueError("terminal FP frame means require DIRECT or FPS")
        if not isinstance(self.require_selection_group_audit, bool):
            raise TypeError("terminal FP grouped audit mode must be boolean")
        if self.require_selection_group_audit and (
            not isinstance(self.selection_plan, SelectionPlan)
            or self.selection_plan.group_ids != systems
        ):
            raise ValueError(
                "terminal FP grouped audit requires the exact selection plan"
            )
        if not self.require_selection_group_audit and self.selection_plan is not None:
            raise ValueError(
                "legacy terminal FP recovery cannot contain a selection plan"
            )
        effective = self.policy if self.effective_policy is None else str(
            self.effective_policy
        )
        if self.policy == "ELEMENT_DIRECT" and effective == "random":
            if self.fallback_reason != "committee_descriptor_unavailable":
                raise ValueError(
                    "terminal committee fallback requires its exact reason"
                )
        elif effective != self.policy or self.fallback_reason is not None:
            raise ValueError("terminal FP effective policy audit is invalid")
        descriptor_identity_policies = {
            "JOINT_DIRECT", "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT",
        }
        descriptor_execution = effective in descriptor_identity_policies
        if descriptor_execution and (
            not isinstance(self.joint_model_identity, str)
            or not self.joint_model_identity
            or not isinstance(self.joint_dataset_identity, str)
            or not self.joint_dataset_identity
        ):
            raise ValueError(
                "joint DIRECT terminal recovery requires model and dataset identities"
            )
        if not descriptor_execution and (
            self.joint_model_identity is not None
            or self.joint_dataset_identity is not None
        ):
            raise ValueError(
                "non-joint terminal recovery cannot contain joint identities"
            )
        if effective in {"ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT"}:
            if (
                not isinstance(self.element_type_map, tuple)
                or not self.element_type_map
                or any(not isinstance(item, str) or not item for item in self.element_type_map)
                or len(self.element_type_map) != len(set(self.element_type_map))
            ):
                raise ValueError(
                    "terminal element DIRECT recovery requires ordered type_map"
                )
        elif self.element_type_map is not None:
            raise ValueError(
                "non-element terminal recovery cannot contain element type_map"
            )
        object.__setattr__(self, "keep", int(self.keep))
        object.__setattr__(self, "round_count", int(self.round_count))
        object.__setattr__(self, "iter_index", int(self.iter_index))
        object.__setattr__(self, "system_ids", systems)
        object.__setattr__(self, "effective_policy", effective)


def fp_task_id(selected_index: int) -> str:
    """Return the stable task identity for one selected-order position."""

    if (
        not isinstance(selected_index, int)
        or isinstance(selected_index, bool)
        or selected_index < 0
    ):
        raise ValueError("first-principles selected index must be non-negative")
    return f"task_{selected_index:06d}"


@dataclass(frozen=True, slots=True)
class FirstPrinciplesSelectedStructure:
    """Path-free selected structure identity at one canonical order position."""

    selected_index: int
    structure_id: str
    system_id: str

    def __post_init__(self) -> None:
        fp_task_id(self.selected_index)
        object.__setattr__(
            self,
            "structure_id",
            _identity(self.structure_id, label="first-principles structure"),
        )
        object.__setattr__(
            self,
            "system_id",
            _identity(self.system_id, label="first-principles system"),
        )


@dataclass(frozen=True, slots=True, init=False)
class FirstPrinciplesSelection:
    """Shared selected-structure handle plus FP-specific ordered system facts."""

    batch: StructureBatchHandle
    system_ids: tuple[str, ...]

    def __init__(
        self,
        batch: StructureBatchHandle | tuple[FirstPrinciplesSelectedStructure, ...],
        system_ids: tuple[str, ...] | None = None,
    ) -> None:
        if isinstance(batch, StructureBatchHandle):
            canonical = batch
            if system_ids is None:
                raise ValueError(
                    "first-principles shared selection requires ordered system facts"
                )
            systems = tuple(
                _identity(value, label="first-principles system")
                for value in system_ids
            )
        else:
            legacy = tuple(batch)
            if any(
                not isinstance(item, FirstPrinciplesSelectedStructure)
                for item in legacy
            ):
                raise TypeError(
                    "first-principles selection requires a shared batch handle"
                )
            if tuple(item.selected_index for item in legacy) != tuple(
                range(len(legacy))
            ):
                raise ValueError(
                    "first-principles selection indices must match canonical order"
                )
            canonical = StructureBatchHandle(
                "fp-selected",
                tuple(item.structure_id for item in legacy),
            )
            systems = tuple(item.system_id for item in legacy)
        if len(systems) != len(canonical.ordered_structure_ids):
            raise ValueError(
                "first-principles system facts must cover the shared selection order"
            )
        object.__setattr__(self, "batch", canonical)
        object.__setattr__(self, "system_ids", systems)

    @property
    def structures(self) -> tuple[FirstPrinciplesSelectedStructure, ...]:
        return tuple(
            FirstPrinciplesSelectedStructure(index, structure_id, system_id)
            for index, (structure_id, system_id) in enumerate(
                zip(
                    self.batch.ordered_structure_ids,
                    self.system_ids,
                    strict=True,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class FirstPrinciplesLabelArtifactFacts:
    """Expected canonical FP label inventory without physical locations."""

    selection: StructureBatchHandle
    system_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    calculator: str

    def __post_init__(self) -> None:
        if not isinstance(self.selection, StructureBatchHandle):
            raise TypeError("first-principles artifacts require a shared batch handle")
        systems = tuple(
            _identity(value, label="first-principles system")
            for value in self.system_ids
        )
        tasks = tuple(self.task_ids)
        expected_tasks = tuple(
            fp_task_id(index)
            for index in range(len(self.selection.ordered_structure_ids))
        )
        if len(systems) != len(expected_tasks) or tasks != expected_tasks:
            raise ValueError(
                "first-principles artifact facts changed selected task/system order"
            )
        calculator = _identity(
            self.calculator, label="first-principles calculator"
        ).upper()
        if calculator not in _CALCULATOR_ROLES:
            raise ValueError("first-principles artifact calculator is invalid")
        object.__setattr__(self, "system_ids", systems)
        object.__setattr__(self, "task_ids", tasks)
        object.__setattr__(self, "calculator", calculator)


def require_fp_label_artifacts(
    expectation: ExpectedArtifactSemantics[FirstPrinciplesLabelArtifactFacts],
    *,
    intent: FirstPrinciplesLabelIntent,
) -> FirstPrinciplesLabelArtifactFacts:
    if expectation.artifact_kind != FP_LABEL_ARTIFACT_KIND or not isinstance(
        expectation.facts, FirstPrinciplesLabelArtifactFacts
    ):
        raise ValueError(
            "first-principles labeling requires canonical artifact semantics"
        )
    facts = expectation.facts
    if (
        facts.selection != intent.selection.batch
        or facts.system_ids != intent.selection.system_ids
        or facts.calculator != intent.calculator
    ):
        raise ValueError(
            "first-principles artifact expectation changed canonical selection"
        )
    return facts


@dataclass(frozen=True, slots=True)
class FirstPrinciplesLabelIntent:
    """Calculator-neutral lifecycle intent with no provider/runtime locations."""

    selection: FirstPrinciplesSelection
    calculator: str
    operation_role: str
    operation_id: str

    def __post_init__(self) -> None:
        calculator = _identity(
            self.calculator, label="first-principles calculator"
        ).upper()
        if calculator not in _CALCULATOR_ROLES:
            raise ValueError("first-principles calculator must be CP2K or ABACUS")
        if self.operation_role != _CALCULATOR_ROLES[calculator]:
            raise ValueError(
                "first-principles operation role does not match calculator"
            )
        object.__setattr__(
            self,
            "operation_id",
            _identity(self.operation_id, label="first-principles operation"),
        )
        object.__setattr__(self, "calculator", calculator)


@dataclass(frozen=True, slots=True)
class FirstPrinciplesLabelFact:
    """Stable identity and completeness facts for one validated static label."""

    task_id: str
    selected_index: int
    structure_id: str
    system_id: str
    input_digest: str
    required_labels: tuple[str, ...] = ("energy", "forces")
    scf_converged: bool | None = None

    def __post_init__(self) -> None:
        if self.task_id != fp_task_id(self.selected_index):
            raise ValueError(
                "first-principles task identity does not match selected order"
            )
        object.__setattr__(
            self,
            "structure_id",
            _identity(self.structure_id, label="first-principles structure"),
        )
        object.__setattr__(
            self,
            "system_id",
            _identity(self.system_id, label="first-principles system"),
        )
        object.__setattr__(
            self,
            "input_digest",
            _identity(self.input_digest, label="first-principles input digest"),
        )
        required = tuple(self.required_labels)
        if (
            len(required) != len(set(required))
            or required[:2] != ("energy", "forces")
            or any(label not in _KNOWN_LABELS for label in required)
        ):
            raise ValueError("invalid first-principles label contract")
        if self.scf_converged is not None and not isinstance(
            self.scf_converged, bool
        ):
            raise ValueError(
                "first-principles SCF convergence evidence must be boolean or absent"
            )
        object.__setattr__(self, "required_labels", required)


@dataclass(frozen=True, slots=True)
class FirstPrinciplesLabelOutcome:
    """Complete canonical label inventory; convergence remains diagnostic only."""

    labels: tuple[FirstPrinciplesLabelFact, ...]

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        task_ids = tuple(label.task_id for label in labels)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("first-principles label outcome duplicates a task")
        object.__setattr__(self, "labels", labels)

    @property
    def scf_converged(self) -> bool | None:
        convergence = tuple(
            label.scf_converged
            for label in self.labels
            if label.scf_converged is not None
        )
        if not convergence:
            return None
        return all(convergence)


def validate_fp_label_outcome(
    intent: FirstPrinciplesLabelIntent,
    outcome: FirstPrinciplesLabelOutcome,
) -> FirstPrinciplesLabelOutcome:
    expected = tuple(
        (
            fp_task_id(item.selected_index),
            item.selected_index,
            item.structure_id,
            item.system_id,
        )
        for item in intent.selection.structures
    )
    actual = tuple(
        (
            label.task_id,
            label.selected_index,
            label.structure_id,
            label.system_id,
        )
        for label in outcome.labels
    )
    if actual != expected:
        raise ValueError(
            "first-principles labels do not preserve exact selected/task/structure/system order"
        )
    return outcome


@runtime_checkable
class FirstPrinciplesLabelCapability(Protocol):
    def label(
        self, request: FirstPrinciplesLabelIntent
    ) -> FirstPrinciplesLabelOutcome: ...


def _identity(value: object, *, label: str) -> str:
    identity = str(value).strip()
    if not identity:
        raise ValueError(f"{label} identity must be non-empty")
    return identity


__all__ = [
    "FinalFPRecoveryFacts",
    "FP_LABEL_ARTIFACT_KIND",
    "FirstPrinciplesLabelCapability",
    "FirstPrinciplesLabelArtifactFacts",
    "FirstPrinciplesLabelFact",
    "FirstPrinciplesLabelIntent",
    "FirstPrinciplesLabelOutcome",
    "FirstPrinciplesSelectedStructure",
    "FirstPrinciplesSelection",
    "fp_task_id",
    "require_fp_label_artifacts",
    "validate_fp_label_outcome",
]
