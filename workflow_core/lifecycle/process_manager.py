from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from workflow_core.lifecycle.state_model import (
    CompletionObservation,
    CompletionStatus,
    CompletionEvidence,
    PublicBlockId,
    PublicLifecyclePlan,
    PublicResultDependencies,
    ReconciliationDecision,
    ReconciliationDisposition,
    InvocationState,
    TransitionEvent,
)
from workflow_core.lifecycle.use_cases import (
    LifecycleRuntimeCapability,
    PublicBlockOutcome,
    PublicBlockOutcomeStatus,
    PublicBlockUseCases,
)


@dataclass(frozen=True, slots=True)
class ReconciliationPlan:
    decisions: tuple[ReconciliationDecision, ...]

    @property
    def blocked(self) -> bool:
        return any(
            decision.disposition is ReconciliationDisposition.BLOCKED
            for decision in self.decisions
        )

    @property
    def blocks_to_run(self) -> tuple[PublicBlockId, ...]:
        return tuple(
            decision.block
            for decision in self.decisions
            if decision.disposition is ReconciliationDisposition.RUN
        )


@dataclass(frozen=True, slots=True)
class ResumeFeasibility:
    target: PublicBlockId
    status: str
    blocking_result: PublicBlockId | None = None
    consuming_block: PublicBlockId | None = None
    recommended_resume: PublicBlockId | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "dice.lifecycle.resume-feasibility",
            "schema_revision": 1,
            "status": self.status,
            "target": str(self.target),
        }
        if self.blocking_result is not None:
            payload.update(
                {
                    "blocking_result": str(self.blocking_result),
                    "consuming_block": str(self.consuming_block),
                    "recommended_resume": str(self.recommended_resume),
                }
            )
        payload["reason"] = self.reason
        return payload

@dataclass(frozen=True, slots=True)
class ProcessManager:
    """Pure reconciliation policy for one prepared public lifecycle plan."""

    plan: PublicLifecyclePlan
    result_dependencies: PublicResultDependencies | None = None

    def __post_init__(self) -> None:
        dependencies = self.result_dependencies
        if dependencies is None:
            dependencies = PublicResultDependencies.for_plan(self.plan)
            object.__setattr__(self, "result_dependencies", dependencies)
        elif dependencies.plan != self.plan:
            raise ValueError("result dependencies belong to a different public plan")

    def prepare(
        self,
        runtime: LifecycleRuntimeCapability,
        *,
        resume_target: PublicBlockId | None = None,
    ) -> ReconciliationPlan:
        """Validate the required durable prefix before requesting any cleanup."""

        observer = getattr(runtime, 'observe_result', None)
        observations = (
            self.observe_required(observer, resume_target=resume_target)
            if callable(observer) else runtime.observe(self.plan, resume_target=resume_target)
        )
        reconciliation = self.reconcile(
            observations,
            resume_target=resume_target,
        )
        if reconciliation.blocked:
            blocked = next(
                decision
                for decision in reconciliation.decisions
                if decision.disposition is ReconciliationDisposition.BLOCKED
            )
            raise ValueError(blocked.reason)
        runtime.prepare(reconciliation.decisions)
        return reconciliation

    def observe_required(
        self, observer: Callable[[PublicBlockId], CompletionObservation], *,
        resume_target: PublicBlockId | None = None,
    ) -> tuple[CompletionObservation, ...]:
        """Inspect result boundaries lazily; a valid result hides its producers."""
        dependencies = self.result_dependencies
        assert dependencies is not None
        observed: dict[PublicBlockId, CompletionObservation] = {}

        def inspect(block):
            if block not in observed:
                row = observer(block)
                if row.block != block:
                    raise ValueError('result observer changed block identity')
                observed[block] = row
            return observed[block]

        if resume_target is not None:
            boundary = self.plan.index(resume_target)
            required = {producer for consumer in self.plan.blocks[boundary:]
                        for producer in dependencies.for_consumer(consumer)
                        if self.plan.index(producer) < boundary}
            for block in self.plan.blocks:
                if block in required:
                    inspect(block)
        else:
            def require(block):
                if block in observed:
                    return
                if inspect(block).status is not CompletionStatus.COMPLETE:
                    for producer in dependencies.for_consumer(block):
                        require(producer)
            require(self.plan.blocks[-1])
            # The workflow executes a contiguous suffix, so its other external
            # inputs must also be checked even if absent from the terminal tree.
            while True:
                missing = [self.plan.index(b) for b, row in observed.items()
                           if row.status is not CompletionStatus.COMPLETE]
                if not missing:
                    break
                boundary = min(missing)
                count = len(observed)
                for consumer in self.plan.blocks[boundary:]:
                    for producer in dependencies.for_consumer(consumer):
                        if self.plan.index(producer) < boundary:
                            require(producer)
                if len(observed) == count:
                    break
        return tuple(observed.get(block, CompletionObservation(
            block, CompletionStatus.REBUILDABLE_INCOMPLETE,
            'outside required observation scope', CompletionEvidence.NOT_INSPECTED,
        )) for block in self.plan.blocks)

    def probe(
        self,
        observations: Iterable[CompletionObservation],
        *,
        resume_target: PublicBlockId,
    ) -> ResumeFeasibility:
        """Report exact-resume feasibility without requesting cleanup."""

        observed = self._index_observations(observations)
        target_index = self.plan.index(resume_target)
        blocker = self._resume_blocker(observed, target_index)
        if blocker is None:
            return ResumeFeasibility(
                resume_target,
                "runnable",
                reason="all persisted dependencies required by the suffix are runnable",
            )
        producer, consumer = blocker
        return ResumeFeasibility(
            resume_target,
            "blocked",
            blocking_result=producer,
            consuming_block=consumer,
            recommended_resume=producer,
            reason=(
                f"required result {producer} for {consumer} is not consumer-runnable"
            ),
        )

    def run_prepared(
        self,
        reconciliation: ReconciliationPlan,
        *,
        use_cases: PublicBlockUseCases,
        runtime: LifecycleRuntimeCapability,
    ) -> InvocationState:
        """Own ordered dispatch, validated continuation and terminal propagation."""

        if use_cases.plan != self.plan:
            raise ValueError("public lifecycle use cases belong to a different plan")
        if tuple(decision.block for decision in reconciliation.decisions) != self.plan.blocks:
            raise ValueError("reconciliation decisions do not cover the process-manager plan")
        state = InvocationState.initial(self.plan)
        for decision in reconciliation.decisions:
            if decision.disposition is ReconciliationDisposition.COMPLETE:
                state = state.transition(
                    decision.block,
                    TransitionEvent.REUSE_COMPLETE,
                )
                continue
            if decision.disposition is ReconciliationDisposition.BLOCKED:
                raise ValueError(decision.reason)
            state = state.transition(decision.block, TransitionEvent.START)
            outcome = self.dispatch(
                decision.block,
                use_cases=use_cases,
                runtime=runtime,
            )
            if outcome.status is PublicBlockOutcomeStatus.FAILED:
                state = state.transition(decision.block, TransitionEvent.FAIL)
                if outcome.error is None:
                    raise AssertionError("failed public block outcome requires an error")
                raise outcome.error
            if outcome.status is PublicBlockOutcomeStatus.INTERRUPTED:
                state = state.transition(decision.block, TransitionEvent.INTERRUPT)
                if outcome.error is None:
                    raise AssertionError("interrupted public block outcome requires an error")
                raise outcome.error
            state = state.transition(decision.block, TransitionEvent.SUCCEED)
            runtime.succeeded(outcome)
        return state

    def run(
        self,
        *,
        use_cases: PublicBlockUseCases,
        runtime: LifecycleRuntimeCapability,
        resume_target: PublicBlockId | None = None,
    ) -> InvocationState:
        reconciliation = self.prepare(runtime, resume_target=resume_target)
        return self.run_prepared(
            reconciliation,
            use_cases=use_cases,
            runtime=runtime,
        )

    def run_one(
        self,
        block: PublicBlockId,
        *,
        use_cases: PublicBlockUseCases,
        runtime: LifecycleRuntimeCapability,
    ) -> PublicBlockOutcome:
        """Validate and execute exactly one requested public lifecycle block."""

        reconciliation = self.prepare(runtime, resume_target=block)
        return self.run_one_prepared(
            reconciliation,
            block=block,
            use_cases=use_cases,
            runtime=runtime,
        )

    def run_one_prepared(
        self,
        reconciliation: ReconciliationPlan,
        *,
        block: PublicBlockId,
        use_cases: PublicBlockUseCases,
        runtime: LifecycleRuntimeCapability,
    ) -> PublicBlockOutcome:
        """Execute one block after its normal reconciliation has been prepared."""

        if not reconciliation.blocks_to_run or reconciliation.blocks_to_run[0] != block:
            raise ValueError(
                f"scheduled public block is not the prepared lifecycle target: {block}"
            )
        outcome = self.dispatch(
            block,
            use_cases=use_cases,
            runtime=runtime,
        )
        if outcome.status is PublicBlockOutcomeStatus.FAILED:
            if outcome.error is None:
                raise AssertionError("failed public block outcome requires an error")
            raise outcome.error
        if outcome.status is PublicBlockOutcomeStatus.INTERRUPTED:
            if outcome.error is None:
                raise AssertionError("interrupted public block outcome requires an error")
            raise outcome.error
        runtime.succeeded(outcome)
        return outcome

    def dispatch(
        self,
        block: PublicBlockId,
        *,
        use_cases: PublicBlockUseCases,
        runtime: LifecycleRuntimeCapability,
    ) -> PublicBlockOutcome:
        """Request one semantic block without selecting or inspecting its provider."""

        if use_cases.plan != self.plan:
            raise ValueError("public lifecycle use cases belong to a different plan")
        return use_cases.for_block(block).execute(runtime)

    def reconcile(
        self,
        observations: Iterable[CompletionObservation],
        *,
        resume_target: PublicBlockId | None = None,
    ) -> ReconciliationPlan:
        observed = self._index_observations(observations)
        target_index = (
            None if resume_target is None else self.plan.index(resume_target)
        )
        decisions: list[ReconciliationDecision] = []

        if target_index is not None:
            blocker = self._resume_blocker(observed, target_index)
            if blocker is not None:
                producer, consumer = blocker
                reason = (
                    f"cannot resume {resume_target}: required result {producer} "
                    f"for {consumer} is not consumer-runnable; resume {producer} "
                    "to rebuild it"
                )
                return ReconciliationPlan(
                    tuple(
                        ReconciliationDecision(
                            block,
                            ReconciliationDisposition.BLOCKED,
                            reason,
                        )
                        for block in self.plan.blocks
                    )
                )
            for index, block in enumerate(self.plan.blocks):
                if index < target_index:
                    decisions.append(
                        ReconciliationDecision(
                            block,
                            ReconciliationDisposition.COMPLETE,
                            "outside the requested suffix dependency closure",
                        )
                    )
                    continue
                decisions.append(
                    ReconciliationDecision(
                        block,
                        ReconciliationDisposition.RUN,
                        "targeted resume invalidates target and successor suffix",
                    )
                )
            return ReconciliationPlan(tuple(decisions))

        required_rebuilds: set[PublicBlockId] = set()
        dependencies = self.result_dependencies
        if dependencies is None:  # pragma: no cover - established in __post_init__
            raise AssertionError("process manager has no result dependency map")

        def require_result(block: PublicBlockId) -> None:
            if observed[block].status is CompletionStatus.COMPLETE:
                return
            if block in required_rebuilds:
                return
            required_rebuilds.add(block)
            for producer in dependencies.for_consumer(block):
                require_result(producer)

        require_result(self.plan.blocks[-1])
        while required_rebuilds:
            boundary = min(self.plan.index(block) for block in required_rebuilds)
            count = len(required_rebuilds)
            for consumer in self.plan.blocks[boundary:]:
                for producer in dependencies.for_consumer(consumer):
                    if self.plan.index(producer) < boundary:
                        require_result(producer)
            if len(required_rebuilds) == count:
                break
        if not required_rebuilds:
            target_index = len(self.plan.blocks)
        else:
            target_index = min(self.plan.index(block) for block in required_rebuilds)
        target = None if target_index == len(self.plan.blocks) else self.plan.blocks[target_index]
        return ReconciliationPlan(
            tuple(
                ReconciliationDecision(
                    block,
                    (
                        ReconciliationDisposition.COMPLETE
                        if index < target_index
                        else ReconciliationDisposition.RUN
                    ),
                    (
                        "outside the terminal result dependency closure"
                        if index < target_index
                        else (
                            "earliest producer required by the terminal result"
                            if block == target
                            else f"contiguous successor of required producer {target}"
                        )
                    ),
                )
                for index, block in enumerate(self.plan.blocks)
            )
        )

    def _index_observations(
        self,
        observations: Iterable[CompletionObservation],
    ) -> dict[PublicBlockId, CompletionObservation]:
        indexed: dict[PublicBlockId, CompletionObservation] = {}
        for observation in observations:
            if observation.block in indexed:
                raise ValueError(
                    f"duplicate completion observation for {observation.block}"
                )
            indexed[observation.block] = observation
        expected = set(self.plan.blocks)
        actual = set(indexed)
        if actual != expected:
            missing = ", ".join(str(block) for block in self.plan.blocks if block not in actual)
            unexpected = ", ".join(str(block) for block in actual if block not in expected)
            details = []
            if missing:
                details.append(f"missing: {missing}")
            if unexpected:
                details.append(f"unexpected: {unexpected}")
            raise ValueError(
                "completion observations must cover the public lifecycle plan exactly"
                + (f" ({'; '.join(details)})" if details else "")
            )
        return indexed

    def _resume_blocker(
        self,
        observed: dict[PublicBlockId, CompletionObservation],
        target_index: int,
    ) -> tuple[PublicBlockId, PublicBlockId] | None:
        dependencies = self.result_dependencies
        if dependencies is None:  # pragma: no cover - established in __post_init__
            raise AssertionError("process manager has no result dependency map")
        for consumer in self.plan.blocks[target_index:]:
            for producer in dependencies.for_consumer(consumer):
                if self.plan.index(producer) >= target_index:
                    continue
                if observed[producer].status is not CompletionStatus.COMPLETE:
                    return producer, consumer
        return None

    def _blocked_suffix(
        self,
        prefix: list[ReconciliationDecision],
        start: int,
        reason: str,
    ) -> ReconciliationPlan:
        decisions = list(prefix)
        decisions.extend(
            ReconciliationDecision(
                block,
                ReconciliationDisposition.BLOCKED,
                reason if index == start else f"blocked by {self.plan.blocks[start]}",
            )
            for index, block in enumerate(self.plan.blocks[start:], start=start)
        )
        return ReconciliationPlan(tuple(decisions))
