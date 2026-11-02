from __future__ import annotations

import pytest

from workflow_core.lifecycle.process_manager import ProcessManager
from workflow_core.lifecycle.state_model import (
    CompletionObservation,
    CompletionStatus,
    InvocationOutcome,
    InvocationState,
    LifecycleTransitionError,
    PublicBlockId,
    PublicBlockState,
    PublicLifecyclePlan,
    ReconciliationDisposition,
    TransitionEvent,
)
from workflow_core.lifecycle.use_cases import (
    PublicBlockOutcome,
    PublicBlockOutcomeStatus,
    PublicBlockUseCase,
    PublicBlockUseCases,
)


def _plan(*, label: bool = False) -> PublicLifecyclePlan:
    return PublicLifecyclePlan.fixed(
        2,
        include_teach_scan=True,
        include_fp_label=label,
    )


def _observations(
    plan: PublicLifecyclePlan,
    overrides: dict[str, CompletionStatus] | None = None,
) -> tuple[CompletionObservation, ...]:
    overrides = overrides or {}
    return tuple(
        CompletionObservation(
            block,
            overrides.get(str(block), CompletionStatus.COMPLETE),
        )
        for block in plan.blocks
    )


def test_fixed_public_plan_preserves_manual_fp_and_optional_single_label_terminal() -> None:
    manual = _plan()
    labeled = _plan(label=True)

    assert tuple(map(str, manual.blocks)) == (
        "teach/train",
        "teach/screen",
        "teach/scan",
        "r0/train",
        "r0/sample",
        "r0/screen",
        "r1/train",
        "r1/sample",
        "r1/screen",
        "fp",
    )
    assert tuple(map(str, labeled.blocks[-2:])) == ("fp", "fp/label")
    assert sum(block == PublicBlockId("fp/label") for block in labeled.blocks) == 1


def test_public_plan_owns_prefix_and_successor_semantics() -> None:
    plan = _plan()
    current = PublicBlockId("r0/train")

    assert plan.prefix_before(current) == plan.blocks[:3]
    assert plan.prefix_through(current) == plan.blocks[:4]
    assert plan.successor(current) == PublicBlockId("r0/sample")
    assert plan.successor(plan.blocks[-1]) is None


def test_public_plan_rejects_unknown_blocks_and_nonfixed_order() -> None:
    with pytest.raises(ValueError, match="invalid public lifecycle block"):
        PublicBlockId("r0/native")
    with pytest.raises(ValueError, match="fixed block order"):
        PublicLifecyclePlan(
            (
                PublicBlockId("teach/screen"),
                PublicBlockId("teach/train"),
                PublicBlockId.round(0, "train"),
                PublicBlockId.round(0, "sample"),
                PublicBlockId.round(0, "screen"),
                PublicBlockId("fp"),
            )
        )


def test_all_complete_reconciliation_is_a_no_op() -> None:
    plan = _plan()
    result = ProcessManager(plan).reconcile(_observations(plan))

    assert not result.blocked
    assert result.blocks_to_run == ()
    assert all(
        decision.disposition is ReconciliationDisposition.COMPLETE
        for decision in result.decisions
    )


def test_reusable_terminal_cuts_off_an_incomplete_prefix_result() -> None:
    plan = _plan()
    result = ProcessManager(plan).reconcile(
        _observations(
            plan,
            {"r0/sample": CompletionStatus.REBUILDABLE_INCOMPLETE},
        )
    )

    assert result.blocks_to_run == ()
    assert not result.blocked


def test_incompatible_irrelevant_evidence_does_not_block_reusable_terminal() -> None:
    plan = _plan()
    result = ProcessManager(plan).reconcile(
        _observations(
            plan,
            {"r0/train": CompletionStatus.INCOMPATIBLE},
        )
    )

    assert not result.blocked
    assert result.blocks_to_run == ()


def test_targeted_resume_requires_only_suffix_consumed_predecessors() -> None:
    plan = _plan()
    manager = ProcessManager(plan)
    target = PublicBlockId("r1/sample")

    resumed = manager.reconcile(_observations(plan), resume_target=target)
    assert tuple(map(str, resumed.blocks_to_run)) == (
        "r1/sample",
        "r1/screen",
        "fp",
    )

    blocked = manager.reconcile(
        _observations(
            plan,
            {"r0/screen": CompletionStatus.REBUILDABLE_INCOMPLETE},
        ),
        resume_target=target,
    )
    assert not blocked.blocked
    assert tuple(map(str, blocked.blocks_to_run)) == (
        "r1/sample",
        "r1/screen",
        "fp",
    )


def test_completion_observations_require_exact_unique_plan_coverage() -> None:
    plan = _plan()
    observations = _observations(plan)
    manager = ProcessManager(plan)

    with pytest.raises(ValueError, match="cover.*exactly"):
        manager.reconcile(observations[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        manager.reconcile(observations + (observations[0],))


def test_invocation_state_allows_only_ordered_successful_transitions() -> None:
    plan = PublicLifecyclePlan.fixed(
        1,
        include_teach_scan=False,
        include_fp_label=False,
    )
    state = InvocationState.initial(plan)

    for block in plan.blocks[:-1]:
        state = state.transition(block, TransitionEvent.REUSE_COMPLETE)
    terminal = plan.blocks[-1]
    state = state.transition(terminal, TransitionEvent.START)
    assert state.state_of(terminal) is PublicBlockState.RUNNING
    state = state.transition(terminal, TransitionEvent.SUCCEED)

    assert state.outcome is InvocationOutcome.SUCCEEDED
    with pytest.raises(LifecycleTransitionError, match="terminal succeeded"):
        state.transition(terminal, TransitionEvent.START)


@pytest.mark.parametrize(
    ("event", "expected_state", "expected_outcome", "resumable"),
    (
        (
            TransitionEvent.FAIL,
            PublicBlockState.FAILED,
            InvocationOutcome.FAILED,
            False,
        ),
        (
            TransitionEvent.INTERRUPT,
            PublicBlockState.INTERRUPTED,
            InvocationOutcome.INTERRUPTED,
            True,
        ),
    ),
)
def test_failure_and_interruption_are_terminal_invocation_outcomes(
    event: TransitionEvent,
    expected_state: PublicBlockState,
    expected_outcome: InvocationOutcome,
    resumable: bool,
) -> None:
    plan = PublicLifecyclePlan.fixed(
        1,
        include_teach_scan=False,
        include_fp_label=False,
    )
    first, successor = plan.blocks[:2]
    state = InvocationState.initial(plan).transition(first, TransitionEvent.START)
    state = state.transition(first, event)

    assert state.state_of(first) is expected_state
    assert state.outcome is expected_outcome
    assert state.outcome.resumable is resumable
    with pytest.raises(LifecycleTransitionError, match="terminal"):
        state.transition(successor, TransitionEvent.START)


def test_illegal_transition_and_out_of_order_start_fail_explicitly() -> None:
    plan = PublicLifecyclePlan.fixed(
        1,
        include_teach_scan=False,
        include_fp_label=False,
    )
    first, successor = plan.blocks[:2]
    state = InvocationState.initial(plan)

    with pytest.raises(LifecycleTransitionError, match="illegal"):
        state.transition(first, TransitionEvent.SUCCEED)
    with pytest.raises(LifecycleTransitionError, match="predecessor"):
        state.transition(successor, TransitionEvent.START)


class _RecordingRuntime:
    def __init__(self, outcome: PublicBlockOutcome) -> None:
        self.outcome = outcome
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return self.outcome


class _ScheduledBlockRuntime:
    def __init__(self, plan: PublicLifecyclePlan, block: PublicBlockId) -> None:
        self.plan = plan
        self.block = block
        self.prepared = []
        self.requests = []
        self.successes = []

    def observe(self, plan, *, resume_target=None):
        assert plan == self.plan
        assert resume_target == self.block
        target_index = plan.index(self.block)
        return tuple(
            CompletionObservation(
                candidate,
                (
                    CompletionStatus.COMPLETE
                    if index < target_index
                    else CompletionStatus.REBUILDABLE_INCOMPLETE
                ),
            )
            for index, candidate in enumerate(plan.blocks)
        )

    def prepare(self, decisions):
        self.prepared.append(decisions)

    def execute(self, request):
        self.requests.append(request)
        return PublicBlockOutcome.succeeded(request.block)

    def succeeded(self, outcome):
        self.successes.append(outcome)


def test_process_manager_dispatches_one_logic_owned_use_case() -> None:
    plan = _plan()
    use_cases = PublicBlockUseCases.for_plan(plan)
    block = PublicBlockId("r0/sample")
    runtime = _RecordingRuntime(PublicBlockOutcome.succeeded(block))

    outcome = ProcessManager(plan).dispatch(
        block,
        use_cases=use_cases,
        runtime=runtime,
    )

    assert outcome.status is PublicBlockOutcomeStatus.SUCCEEDED
    assert [request.block for request in runtime.requests] == [block]


def test_process_manager_runs_exactly_one_scheduled_public_block() -> None:
    plan = _plan()
    block = PublicBlockId("r0/sample")
    runtime = _ScheduledBlockRuntime(plan, block)

    outcome = ProcessManager(plan).run_one(
        block,
        use_cases=PublicBlockUseCases.for_plan(plan),
        runtime=runtime,
    )

    assert outcome == PublicBlockOutcome.succeeded(block)
    assert len(runtime.prepared) == 1
    assert [request.block for request in runtime.requests] == [block]
    assert runtime.successes == [outcome]


def test_use_case_coverage_rejects_missing_or_duplicate_blocks_before_dispatch() -> None:
    plan = _plan()
    exact = tuple(PublicBlockUseCase(block) for block in plan.blocks)

    with pytest.raises(ValueError, match="exactly"):
        PublicBlockUseCases(plan, exact[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        PublicBlockUseCases(plan, exact[:-1] + (exact[0], exact[0]))


@pytest.mark.parametrize(
    ("error", "status"),
    (
        (RuntimeError("handler failed"), PublicBlockOutcomeStatus.FAILED),
        (KeyboardInterrupt(), PublicBlockOutcomeStatus.INTERRUPTED),
    ),
)
def test_one_block_failure_and_interruption_outcomes_preserve_original_error(
    error: BaseException,
    status: PublicBlockOutcomeStatus,
) -> None:
    plan = _plan()
    use_cases = PublicBlockUseCases.for_plan(plan)
    block = PublicBlockId("teach/train")
    outcome = (
        PublicBlockOutcome.interrupted(block, error)
        if isinstance(error, KeyboardInterrupt)
        else PublicBlockOutcome.failed(block, error)
    )
    runtime = _RecordingRuntime(outcome)

    observed = ProcessManager(plan).dispatch(
        block,
        use_cases=use_cases,
        runtime=runtime,
    )

    assert observed.status is status
    assert observed.error is error
