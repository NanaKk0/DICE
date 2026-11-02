from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class LifecycleTransitionError(ValueError):
    """Raised when an invocation-local lifecycle transition is illegal."""


@dataclass(frozen=True, order=True, slots=True)
class PublicBlockId:
    """One semantic public lifecycle block, with no durable location attached."""

    value: str

    def __post_init__(self) -> None:
        if self.value in {
            "teach/train",
            "teach/screen",
            "teach/scan",
            "fp",
            "fp/label",
        }:
            return
        prefix, separator, phase = self.value.partition("/")
        round_text = prefix[1:] if prefix.startswith("r") else ""
        if (
            separator != "/"
            or phase not in {"train", "sample", "screen"}
            or not round_text.isdigit()
            or str(int(round_text)) != round_text
        ):
            raise ValueError(f"invalid public lifecycle block id {self.value!r}")

    @classmethod
    def round(cls, index: int, phase: str) -> PublicBlockId:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("public lifecycle round index must be a non-negative integer")
        if phase not in {"train", "sample", "screen"}:
            raise ValueError(f"invalid public lifecycle round phase {phase!r}")
        return cls(f"r{index}/{phase}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class PublicLifecyclePlan:
    """The sole ordered iter/round lifecycle shape accepted by core logic."""

    blocks: tuple[PublicBlockId, ...]

    def __post_init__(self) -> None:
        if not self.blocks or len(set(self.blocks)) != len(self.blocks):
            raise ValueError("public lifecycle plan requires unique blocks")
        include_teach_scan = PublicBlockId("teach/scan") in self.blocks
        include_fp_label = PublicBlockId("fp/label") in self.blocks
        round_blocks = tuple(
            block for block in self.blocks if block.value.startswith("r")
        )
        if len(round_blocks) % 3:
            raise ValueError("public lifecycle plan has an incomplete round")
        expected = _fixed_blocks(
            len(round_blocks) // 3,
            include_teach_scan=include_teach_scan,
            include_fp_label=include_fp_label,
        )
        if self.blocks != expected:
            raise ValueError("public lifecycle plan does not match the fixed block order")

    @classmethod
    def fixed(
        cls,
        round_count: int,
        *,
        include_teach_scan: bool,
        include_fp_label: bool = False,
    ) -> PublicLifecyclePlan:
        return cls(
            _fixed_blocks(
                round_count,
                include_teach_scan=include_teach_scan,
                include_fp_label=include_fp_label,
            )
        )

    def index(self, block: PublicBlockId) -> int:
        try:
            return self.blocks.index(block)
        except ValueError as exc:
            raise ValueError(f"public lifecycle block {block} is not in the plan") from exc

    def prefix_before(self, block: PublicBlockId) -> tuple[PublicBlockId, ...]:
        """Return the exact semantic predecessor prefix for one block."""

        return self.blocks[: self.index(block)]

    def prefix_through(self, block: PublicBlockId) -> tuple[PublicBlockId, ...]:
        """Return the exact semantic prefix after one block has succeeded."""

        return self.blocks[: self.index(block) + 1]

    def successor(self, block: PublicBlockId) -> PublicBlockId | None:
        """Return the next fixed public block, or ``None`` at the terminal block."""

        index = self.index(block)
        return None if index + 1 == len(self.blocks) else self.blocks[index + 1]


@dataclass(frozen=True, slots=True)
class PublicResultDependencies:
    """Route-specific, path-free result dependencies for one public plan."""

    plan: PublicLifecyclePlan
    direct: tuple[tuple[PublicBlockId, tuple[PublicBlockId, ...]], ...]

    def __post_init__(self) -> None:
        if tuple(block for block, _dependencies in self.direct) != self.plan.blocks:
            raise ValueError("result dependencies must cover the public plan in order")
        for consumer, dependencies in self.direct:
            if len(set(dependencies)) != len(dependencies):
                raise ValueError(f"duplicate result dependency for {consumer}")
            consumer_index = self.plan.index(consumer)
            if any(self.plan.index(producer) >= consumer_index for producer in dependencies):
                raise ValueError(
                    f"result dependency for {consumer} must be produced earlier"
                )

    @classmethod
    def for_plan(
        cls,
        plan: PublicLifecyclePlan,
        *,
        fp_all_rounds: bool = False,
        fp_requires_teacher_train: bool = False,
        fp_requires_teacher_scan: bool = False,
    ) -> PublicResultDependencies:
        """Build the supported route shape without persisting a second plan."""

        teach_train = PublicBlockId("teach/train")
        teach_screen = PublicBlockId("teach/screen")
        teach_scan = PublicBlockId("teach/scan")
        has_scan = teach_scan in plan.blocks
        round_count = sum(
            block.value.endswith("/train") and block.value.startswith("r")
            for block in plan.blocks
        )
        direct: list[tuple[PublicBlockId, tuple[PublicBlockId, ...]]] = []
        for block in plan.blocks:
            value = str(block)
            dependencies: tuple[PublicBlockId, ...]
            if value == "teach/train":
                dependencies = ()
            elif value == "teach/screen":
                dependencies = (teach_train,)
            elif value == "teach/scan":
                dependencies = (teach_train,)
            elif value.startswith("r") and value.endswith("/train"):
                round_index = int(value.split("/", 1)[0][1:])
                if round_index == 0:
                    dependencies = (teach_screen,) + ((teach_scan,) if has_scan else ())
                else:
                    dependencies = (
                        PublicBlockId.round(round_index - 1, "train"),
                        PublicBlockId.round(round_index - 1, "screen"),
                    )
            elif value.startswith("r") and value.endswith("/sample"):
                round_index = int(value.split("/", 1)[0][1:])
                dependencies = (PublicBlockId.round(round_index, "train"),)
            elif value.startswith("r") and value.endswith("/screen"):
                round_index = int(value.split("/", 1)[0][1:])
                dependencies = (
                    teach_train,
                    *((teach_scan,) if has_scan else ()),
                    PublicBlockId.round(round_index, "train"),
                    PublicBlockId.round(round_index, "sample"),
                )
            elif value == "fp":
                candidate_rounds = (
                    range(round_count)
                    if fp_all_rounds
                    else (round_count - 1,)
                )
                dependencies = tuple(
                    PublicBlockId.round(index, "screen")
                    for index in candidate_rounds
                )
                if fp_requires_teacher_train and teach_train not in dependencies:
                    dependencies += (teach_train,)
                if fp_requires_teacher_scan:
                    if not has_scan:
                        raise ValueError("terminal FP requires unavailable teach/scan result")
                    dependencies += (teach_scan,)
                dependencies = tuple(
                    sorted(dependencies, key=plan.index)
                )
            elif value == "fp/label":
                dependencies = (PublicBlockId("fp"),)
            else:  # pragma: no cover - PublicBlockId closes this surface
                raise ValueError(f"unsupported public result consumer {block}")
            direct.append((block, dependencies))
        return cls(plan, tuple(direct))

    def for_consumer(self, block: PublicBlockId) -> tuple[PublicBlockId, ...]:
        self.plan.index(block)
        return self.direct[self.plan.index(block)][1]


def _fixed_blocks(
    round_count: int,
    *,
    include_teach_scan: bool,
    include_fp_label: bool,
) -> tuple[PublicBlockId, ...]:
    if (
        not isinstance(round_count, int)
        or isinstance(round_count, bool)
        or round_count < 1
    ):
        raise ValueError("public lifecycle plan requires at least one round")
    if not isinstance(include_teach_scan, bool) or not isinstance(include_fp_label, bool):
        raise ValueError("public lifecycle optional-block flags must be booleans")
    blocks = [PublicBlockId("teach/train"), PublicBlockId("teach/screen")]
    if include_teach_scan:
        blocks.append(PublicBlockId("teach/scan"))
    for index in range(round_count):
        blocks.extend(
            PublicBlockId.round(index, phase)
            for phase in ("train", "sample", "screen")
        )
    blocks.append(PublicBlockId("fp"))
    if include_fp_label:
        blocks.append(PublicBlockId("fp/label"))
    return tuple(blocks)


class CompletionStatus(str, Enum):
    COMPLETE = "complete"
    REBUILDABLE_INCOMPLETE = "rebuildable-incomplete"
    INCOMPATIBLE = "incompatible"


class CompletionEvidence(str, Enum):
    """Path-free durable evidence understood by lifecycle application logic."""

    UNSPECIFIED = "unspecified"
    ARTIFACT_INVALID = "artifact-invalid"
    CANONICAL_COMPLETE = "canonical-complete"
    NOT_INSPECTED = "not-inspected"


@dataclass(frozen=True, slots=True)
class CompletionObservation:
    block: PublicBlockId
    status: CompletionStatus
    reason: str = ""
    evidence: CompletionEvidence = CompletionEvidence.UNSPECIFIED


class ReconciliationDisposition(str, Enum):
    COMPLETE = "complete"
    RUN = "run"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ReconciliationDecision:
    block: PublicBlockId
    disposition: ReconciliationDisposition
    reason: str


class PublicBlockState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class TransitionEvent(str, Enum):
    REUSE_COMPLETE = "reuse-complete"
    START = "start"
    SUCCEED = "succeed"
    FAIL = "fail"
    INTERRUPT = "interrupt"


class InvocationOutcome(str, Enum):
    IN_PROGRESS = "in-progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    @property
    def resumable(self) -> bool:
        return self is InvocationOutcome.INTERRUPTED


@dataclass(frozen=True, slots=True)
class InvocationState:
    """Immutable invocation-local state; durable evidence remains adapter-owned."""

    plan: PublicLifecyclePlan
    block_states: tuple[PublicBlockState, ...]
    outcome: InvocationOutcome = InvocationOutcome.IN_PROGRESS

    def __post_init__(self) -> None:
        if len(self.block_states) != len(self.plan.blocks):
            raise ValueError("invocation state must cover every public lifecycle block")

    @classmethod
    def initial(cls, plan: PublicLifecyclePlan) -> InvocationState:
        return cls(plan, (PublicBlockState.PENDING,) * len(plan.blocks))

    def state_of(self, block: PublicBlockId) -> PublicBlockState:
        return self.block_states[self.plan.index(block)]

    def transition(
        self,
        block: PublicBlockId,
        event: TransitionEvent,
    ) -> InvocationState:
        if self.outcome is not InvocationOutcome.IN_PROGRESS:
            raise LifecycleTransitionError(
                f"cannot transition a terminal {self.outcome.value} invocation"
            )
        index = self.plan.index(block)
        current = self.block_states[index]
        if any(
            state is not PublicBlockState.SUCCEEDED
            for state in self.block_states[:index]
        ):
            raise LifecycleTransitionError(
                f"cannot transition {block}: a predecessor has not succeeded"
            )
        if any(
            state is PublicBlockState.RUNNING
            for position, state in enumerate(self.block_states)
            if position != index
        ):
            raise LifecycleTransitionError(
                f"cannot transition {block}: another block is running"
            )

        next_state, next_outcome = _transition(current, event)
        states = list(self.block_states)
        states[index] = next_state
        if (
            next_state is PublicBlockState.SUCCEEDED
            and index == len(states) - 1
        ):
            next_outcome = InvocationOutcome.SUCCEEDED
        return replace(self, block_states=tuple(states), outcome=next_outcome)


def _transition(
    current: PublicBlockState,
    event: TransitionEvent,
) -> tuple[PublicBlockState, InvocationOutcome]:
    if current is PublicBlockState.PENDING and event is TransitionEvent.REUSE_COMPLETE:
        return PublicBlockState.SUCCEEDED, InvocationOutcome.IN_PROGRESS
    if current is PublicBlockState.PENDING and event is TransitionEvent.START:
        return PublicBlockState.RUNNING, InvocationOutcome.IN_PROGRESS
    if current is PublicBlockState.RUNNING and event is TransitionEvent.SUCCEED:
        return PublicBlockState.SUCCEEDED, InvocationOutcome.IN_PROGRESS
    if current is PublicBlockState.RUNNING and event is TransitionEvent.FAIL:
        return PublicBlockState.FAILED, InvocationOutcome.FAILED
    if current is PublicBlockState.RUNNING and event is TransitionEvent.INTERRUPT:
        return PublicBlockState.INTERRUPTED, InvocationOutcome.INTERRUPTED
    raise LifecycleTransitionError(
        f"illegal public lifecycle transition {current.value} + {event.value}"
    )
