from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from workflow_core.lifecycle.state_model import PublicBlockId, PublicLifecyclePlan
from workflow_core.lifecycle.state_model import (
    CompletionObservation,
    ReconciliationDecision,
)


@dataclass(frozen=True, slots=True)
class PublicBlockInput:
    """Semantic request to execute exactly one public lifecycle block."""

    block: PublicBlockId


class PublicBlockOutcomeStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class PublicBlockOutcome:
    """Explicit one-block terminal outcome preserving the original failure."""

    block: PublicBlockId
    status: PublicBlockOutcomeStatus
    error: BaseException | None = None

    def __post_init__(self) -> None:
        if self.status is PublicBlockOutcomeStatus.SUCCEEDED:
            if self.error is not None:
                raise ValueError("a succeeded public block cannot carry an error")
            return
        if self.error is None:
            raise ValueError("a failed or interrupted public block requires an error")
        if self.status is PublicBlockOutcomeStatus.INTERRUPTED:
            if not isinstance(self.error, KeyboardInterrupt):
                raise ValueError("an interrupted public block requires KeyboardInterrupt")
        elif isinstance(self.error, KeyboardInterrupt):
            raise ValueError("KeyboardInterrupt must remain an interrupted outcome")

    @classmethod
    def succeeded(cls, block: PublicBlockId) -> PublicBlockOutcome:
        return cls(block, PublicBlockOutcomeStatus.SUCCEEDED)

    @classmethod
    def failed(
        cls,
        block: PublicBlockId,
        error: Exception,
    ) -> PublicBlockOutcome:
        return cls(block, PublicBlockOutcomeStatus.FAILED, error)

    @classmethod
    def interrupted(
        cls,
        block: PublicBlockId,
        error: KeyboardInterrupt,
    ) -> PublicBlockOutcome:
        return cls(block, PublicBlockOutcomeStatus.INTERRUPTED, error)


class LifecycleRuntimeCapability(Protocol):
    """Logic-owned semantic capability implemented by the imperative shell."""

    def observe(
        self,
        plan: PublicLifecyclePlan,
        *,
        resume_target: PublicBlockId | None = None,
    ) -> tuple[CompletionObservation, ...]:
        """Return only the durable classifications required for reconciliation."""

    def prepare(
        self,
        decisions: tuple[ReconciliationDecision, ...],
    ) -> None:
        """Apply the already validated run-suffix decision."""

    def execute(self, request: PublicBlockInput) -> PublicBlockOutcome:
        """Execute and publish one block, preserving its terminal outcome."""

    def succeeded(self, outcome: PublicBlockOutcome) -> None:
        """Apply runtime-only post-publication hooks before the successor."""


@dataclass(frozen=True, slots=True)
class PublicBlockUseCase:
    """One explicit application use case bound to one public block identity."""

    block: PublicBlockId

    def execute(self, runtime: LifecycleRuntimeCapability) -> PublicBlockOutcome:
        outcome = runtime.execute(PublicBlockInput(self.block))
        if outcome.block != self.block:
            raise ValueError(
                f"public block runtime returned {outcome.block} for {self.block}"
            )
        return outcome


@dataclass(frozen=True, slots=True)
class PublicBlockUseCases:
    """Exact, immutable one-use-case-per-public-block coverage."""

    plan: PublicLifecyclePlan
    values: tuple[PublicBlockUseCase, ...]

    def __post_init__(self) -> None:
        actual = tuple(use_case.block for use_case in self.values)
        if len(set(actual)) != len(actual):
            raise ValueError("public lifecycle use cases contain duplicate blocks")
        if actual != self.plan.blocks:
            raise ValueError(
                "public lifecycle use cases must cover the fixed plan exactly and in order"
            )

    @classmethod
    def for_plan(cls, plan: PublicLifecyclePlan) -> PublicBlockUseCases:
        return cls(plan, tuple(PublicBlockUseCase(block) for block in plan.blocks))

    def for_block(self, block: PublicBlockId) -> PublicBlockUseCase:
        self.plan.index(block)
        return self.values[self.plan.index(block)]
