from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Any, Protocol

from workflow_core.lifecycle.state_model import PublicBlockId
from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
from workflow_core.lifecycle.state_model import (
    CompletionEvidence,
    CompletionObservation,
    CompletionStatus,
    PublicLifecyclePlan,
    ReconciliationDecision,
    ReconciliationDisposition,
)
from workflow_core.lifecycle.use_cases import (
    PublicBlockInput,
    PublicBlockOutcome,
)
from workflow_runtime.lifecycle.state import LifecycleState
from workflow_runtime.lifecycle.artifact_validation import BoundArtifactValidator


BlockHandler = Callable[[Path], object]
BlockPhaseCallback = Callable[[str], None]
BlockFailureCallback = Callable[[Path, BaseException], None]
BlockReporterFactory = Callable[[PublicBlockId], Any]
ManagedBlockPhaseCallback = Callable[[PublicBlockId, str], None]
ManagedBlockFailureCallback = Callable[[PublicBlockId, Path, BaseException], None]
ManagedBlockSuccessCallback = Callable[[PublicBlockOutcome], None]


class LifecycleBindingFactory(Protocol):
    def __call__(
        self,
        completed_prefix: tuple[PublicBlockId, ...],
    ) -> tuple[LifecycleBlockRuntimeBinding, ...]: ...


class _RuntimeArtifactValidator(Protocol):
    expectation: ExpectedArtifactSemantics[Any]

    def __call__(self, root: Path) -> bool: ...


@dataclass(frozen=True, slots=True)
class _CompatibilityArtifactValidator:
    """Bind a legacy check without changing its typed failure semantics."""

    expectation: ExpectedArtifactSemantics[Any]
    validator: Callable[[Path], bool]

    def __call__(self, root: Path) -> bool:
        accepted = self.validator(Path(root))
        if not isinstance(accepted, bool):
            raise TypeError("lifecycle artifact validator must return a boolean")
        return accepted


@dataclass(frozen=True, slots=True)
class LifecycleBlockRuntimeBinding:
    """Bind one public block to its consumer-runnable result contract."""

    block: PublicBlockId
    artifact_validator: _RuntimeArtifactValidator

    def __post_init__(self) -> None:
        if self.artifact_validator.expectation.block != self.block:
            raise ValueError(
                "lifecycle runtime binding block and artifact contract differ"
            )


def bind_lifecycle_runtime_block(
    *,
    block: PublicBlockId,
    artifact_validator: BoundArtifactValidator[Any] | Callable[[Path], bool],
) -> LifecycleBlockRuntimeBinding:
    """Bind one public block to the validator its consumer actually needs."""

    if isinstance(artifact_validator, BoundArtifactValidator):
        validator = artifact_validator
    else:
        if not callable(artifact_validator):
            raise TypeError("lifecycle runtime binding requires an artifact validator")
        expectation = ExpectedArtifactSemantics(
            block,
            "dice.lifecycle.compatibility-artifact",
            str(block),
        )
        validator = _CompatibilityArtifactValidator(
            expectation,
            artifact_validator,
        )
    if validator.expectation.block != block:
        raise ValueError(
            "lifecycle runtime artifact validator belongs to another block"
        )
    return LifecycleBlockRuntimeBinding(block, validator)


@dataclass(frozen=True, slots=True)
class LifecycleStateBindingProvider:
    """Rebind one full plan against the assets in an exact completed prefix."""

    plan: PublicLifecyclePlan
    binding_factory: LifecycleBindingFactory

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PublicLifecyclePlan):
            raise TypeError("lifecycle binding provider requires a public plan")
        if not callable(self.binding_factory):
            raise TypeError("lifecycle binding provider requires a binding factory")

    def bindings_for(
        self,
        completed_prefix: tuple[PublicBlockId, ...],
    ) -> tuple[LifecycleBlockRuntimeBinding, ...]:
        prefix = tuple(completed_prefix)
        if prefix != self.plan.blocks[: len(prefix)]:
            raise ValueError(
                "lifecycle binding provider requires an exact completed plan prefix"
            )
        return tuple(self.binding_factory(prefix))


@dataclass(slots=True)
class LifecycleStateBindingController:
    """Refresh runtime bindings before one block observes or executes."""

    root: str | Path
    iter_index: int
    plan: PublicLifecyclePlan
    provider: LifecycleStateBindingProvider
    _state: LifecycleState | None = field(init=False, default=None)
    _refreshed_for: PublicBlockId | None = field(init=False, default=None)
    _completed_prefix: tuple[PublicBlockId, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PublicLifecyclePlan):
            raise TypeError("lifecycle binding controller requires a public plan")
        if not isinstance(self.provider, LifecycleStateBindingProvider):
            raise TypeError("lifecycle binding controller requires a binding provider")
        if self.provider.plan != self.plan:
            raise ValueError("lifecycle binding provider plan does not match controller")
        self.root = Path(self.root).resolve()

    @property
    def current_state(self) -> LifecycleState:
        if self._state is None:
            raise ValueError("lifecycle state bindings have not been refreshed")
        return self._state

    @property
    def refreshed_for(self) -> PublicBlockId | None:
        return self._refreshed_for

    @property
    def completed_prefix(self) -> tuple[PublicBlockId, ...]:
        return self._completed_prefix

    def refresh_for(self, block: PublicBlockId) -> LifecycleState:
        index = self.plan.index(block)
        return self._refresh(
            block=block,
            completed_prefix=self.plan.blocks[:index],
        )

    def refresh_after_success(self, block: PublicBlockId) -> LifecycleState:
        """Rebind after the current process has validated and published a block."""

        if self._state is None or self._refreshed_for != block:
            raise ValueError(
                "lifecycle downstream refresh requires the current block binding"
            )
        index = self.plan.index(block)
        prefix = self.plan.blocks[: index + 1]
        next_block = (
            self.plan.blocks[index + 1]
            if index + 1 < len(self.plan.blocks)
            else block
        )
        return self._refresh(block=next_block, completed_prefix=prefix)

    def _refresh(
        self,
        *,
        block: PublicBlockId,
        completed_prefix: tuple[PublicBlockId, ...],
    ) -> LifecycleState:
        bindings = self.provider.bindings_for(completed_prefix)
        self._validate_bindings(bindings)
        state = adapt_lifecycle_state(
            root=self.root,
            iter_index=self.iter_index,
            bindings=bindings,
        )
        self._state = state
        self._refreshed_for = block
        self._completed_prefix = completed_prefix
        return state

    def _validate_bindings(
        self,
        bindings: tuple[LifecycleBlockRuntimeBinding, ...],
    ) -> None:
        if len(bindings) != len(self.plan.blocks) or any(
            not isinstance(binding, LifecycleBlockRuntimeBinding)
            for binding in bindings
        ):
            raise ValueError(
                "lifecycle binding provider must return one binding for every plan block"
            )
        for expected_block, binding in zip(self.plan.blocks, bindings, strict=True):
            expectation_block = binding.artifact_validator.expectation.block
            if binding.block != expected_block:
                raise ValueError(
                    "lifecycle binding provider changed the public plan block order"
                )
            if expectation_block != expected_block:
                raise ValueError(
                    "lifecycle artifact expectation does not match its public plan block"
                )


def adapt_lifecycle_state(
    *,
    root: str | Path,
    iter_index: int,
    bindings: tuple[LifecycleBlockRuntimeBinding, ...],
) -> LifecycleState:
    """Construct the existing state authority from already bound runtime evidence."""

    ordered = tuple(bindings)
    blocks = tuple(str(binding.block) for binding in ordered)
    if not ordered or len(set(blocks)) != len(blocks):
        raise ValueError("lifecycle state bindings require ordered unique blocks")
    return LifecycleState(
        root=root,
        iter_index=iter_index,
        blocks=blocks,
        artifact_checks={
            block: binding.artifact_validator
            for block, binding in zip(blocks, ordered, strict=True)
        },
    )


@dataclass(frozen=True, slots=True)
class StateLifecycleRuntime:
    """Imperative-shell adapter for existing path-backed block handlers."""

    state: LifecycleState
    handlers: Mapping[str, BlockHandler]
    reporter: Any
    phase_callback: BlockPhaseCallback | None = None
    failure_callback: BlockFailureCallback | None = None

    def execute(self, request: PublicBlockInput) -> PublicBlockOutcome:
        block = str(request.block)
        handler = self.handlers.get(block)
        if handler is None:
            raise ValueError(f"lifecycle runtime has no handler for {block}")
        try:
            self.state.publish(
                block,
                lambda staging: _invoke_handler(handler, staging, self.reporter),
                phase_callback=self.phase_callback,
                failure_callback=self.failure_callback,
            )
        except KeyboardInterrupt as exc:
            return PublicBlockOutcome.interrupted(request.block, exc)
        except Exception as exc:
            return PublicBlockOutcome.failed(request.block, exc)
        return PublicBlockOutcome.succeeded(request.block)


def _invoke_handler(handler: BlockHandler, staging: Path, reporter: Any) -> object:
    """Preserve the existing one-argument test and two-argument reporter seams."""

    parameters = tuple(inspect.signature(handler).parameters.values())
    if len(parameters) >= 2 and parameters[1].name == "report":
        return handler(staging, reporter)
    return handler(staging)


@dataclass(frozen=True, slots=True)
class ManagedStateLifecycleRuntime:
    """Full-public runtime adapter controlled only through semantic decisions."""

    state: LifecycleState | None
    handlers: Mapping[str, BlockHandler]
    reporter_factory: BlockReporterFactory
    phase_callback: ManagedBlockPhaseCallback
    failure_callback: ManagedBlockFailureCallback
    success_callback: ManagedBlockSuccessCallback
    state_controller: LifecycleStateBindingController | None = None

    def __post_init__(self) -> None:
        if (self.state is None) == (self.state_controller is None):
            raise ValueError(
                "managed lifecycle runtime requires exactly one state authority"
            )

    @property
    def current_state(self) -> LifecycleState:
        if self.state_controller is not None:
            return self.state_controller.current_state
        if self.state is None:
            raise AssertionError("managed lifecycle runtime has no state authority")
        return self.state

    def _state_for(self, block: PublicBlockId) -> LifecycleState:
        if self.state_controller is not None:
            return self.state_controller.refresh_for(block)
        return self.current_state

    def observe_result(self, block: PublicBlockId) -> CompletionObservation:
        raw = self._state_for(block).reconciliation_observation(str(block))
        return CompletionObservation(block, CompletionStatus(raw.status), raw.reason,
                                     CompletionEvidence(raw.evidence))

    def observe(
        self,
        plan: PublicLifecyclePlan,
        *,
        resume_target: PublicBlockId | None = None,
    ) -> tuple[CompletionObservation, ...]:
        blocks = tuple(str(block) for block in plan.blocks)
        if self.state_controller is not None:
            if plan != self.state_controller.plan:
                raise ValueError("runtime controller does not match the public lifecycle plan")
        elif blocks != self.current_state.blocks:
            raise ValueError("runtime state does not match the public lifecycle plan")
        if resume_target is not None:
            plan.index(resume_target)
        from workflow_core.lifecycle.process_manager import ProcessManager
        return ProcessManager(plan).observe_required(self.observe_result,
                                                     resume_target=resume_target)

    def prepare(
        self,
        decisions: tuple[ReconciliationDecision, ...],
    ) -> None:
        first_run = next(
            (
                str(decision.block)
                for decision in decisions
                if decision.disposition is ReconciliationDisposition.RUN
            ),
            None,
        )
        if first_run is None:
            return
        block = PublicBlockId(first_run)
        self._state_for(block).prepare_reconciliation(first_run)

    def execute(self, request: PublicBlockInput) -> PublicBlockOutcome:
        block = str(request.block)
        handler = self.handlers.get(block)
        if handler is None:
            raise ValueError(f"lifecycle runtime has no handler for {block}")
        reporter = self.reporter_factory(request.block)
        state = self._state_for(request.block)
        try:
            state.publish(
                block,
                lambda staging: _invoke_handler(handler, staging, reporter),
                phase_callback=lambda phase: self.phase_callback(request.block, phase),
                failure_callback=lambda staging, error: self.failure_callback(
                    request.block, staging, error
                ),
            )
        except KeyboardInterrupt as exc:
            return PublicBlockOutcome.interrupted(request.block, exc)
        except Exception as exc:
            return PublicBlockOutcome.failed(request.block, exc)
        return PublicBlockOutcome.succeeded(request.block)

    def succeeded(self, outcome: PublicBlockOutcome) -> None:
        self.success_callback(outcome)
        if self.state_controller is not None:
            self.state_controller.refresh_after_success(outcome.block)
