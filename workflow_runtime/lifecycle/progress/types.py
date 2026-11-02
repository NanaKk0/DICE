from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class BlockStatus(str, Enum):
    WAITING = "WAITING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True, slots=True)
class GenericWorkSnapshot:
    """Absolute backend-neutral position for work without a richer view yet."""

    activity: str
    completed: int | None = None
    total: int | None = None
    unit: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.activity:
            raise ValueError("progress activity must be non-empty")
        if self.completed is not None and self.completed < 0:
            raise ValueError("progress completed must be non-negative")
        if self.total is not None and self.total < 0:
            raise ValueError("progress total must be non-negative")
        if self.completed is not None and self.total is not None and self.completed > self.total:
            raise ValueError("progress completed cannot exceed total")


@dataclass(frozen=True, slots=True)
class DatasetContribution:
    label: str
    count: int

    def __post_init__(self) -> None:
        if not self.label or self.count < 0:
            raise ValueError("dataset contribution requires a label and non-negative count")


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    label: str
    unique: int
    train: int | None = None
    valid: int | None = None
    systems: int | None = None
    contributions: tuple[DatasetContribution, ...] = ()
    deduplicated: int = 0

    def __post_init__(self) -> None:
        values = (self.unique, self.deduplicated, *(item for item in (self.train, self.valid, self.systems) if item is not None))
        if not self.label or any(value < 0 for value in values):
            raise ValueError("dataset snapshot counts must be non-negative")


@dataclass(frozen=True, slots=True)
class CountStep:
    label: str
    value: int | None

    def __post_init__(self) -> None:
        if not self.label or (self.value is not None and self.value < 0):
            raise ValueError("count step requires a label and a non-negative completed value")


@dataclass(frozen=True, slots=True)
class CountChainSnapshot:
    label: str
    steps: tuple[CountStep, ...]

    def __post_init__(self) -> None:
        if not self.label or not self.steps:
            raise ValueError("count chain requires a label and at least one step")


@dataclass(frozen=True, slots=True)
class SystemCountRow:
    system_id: str
    values: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if not self.system_id or any(value is not None and value < 0 for value in self.values):
            raise ValueError("system count row contains an invalid value")


@dataclass(frozen=True, slots=True)
class CountFlowSnapshot:
    title: str
    columns: tuple[str, ...]
    rows: tuple[SystemCountRow, ...]

    def __post_init__(self) -> None:
        if not self.title or not self.columns or not self.rows:
            raise ValueError("count flow requires title, columns, and rows")
        if any(len(row.values) != len(self.columns) for row in self.rows):
            raise ValueError("count flow rows must match its columns")


@dataclass(frozen=True, slots=True)
class TrainingMetricTrend:
    """One comparable, bounded native metric series normalized by a backend."""

    label: str
    semantic_quantity: str
    split: str
    unit: str
    values: tuple[float, ...]
    series_id: str

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if not all((self.label, self.semantic_quantity, self.split, self.unit, self.series_id)):
            raise ValueError("training metric trend identity fields must be non-empty")
        if not values or len(values) > 4:
            raise ValueError("training metric trend requires one to four comparable values")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class TrainingSnapshot:
    """Backend-neutral absolute progress for one native training member."""

    member_id: str
    member_index: int
    total_members: int
    unit: str
    position: int
    total: int
    stage: str
    trends: tuple[TrainingMetricTrend, ...] = ()
    learning_rate: float | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    series_generation: int = 0

    def __post_init__(self) -> None:
        if not all((self.member_id, self.unit, self.stage)):
            raise ValueError("training snapshot identity fields must be non-empty")
        if self.member_index < 1 or self.total_members < self.member_index:
            raise ValueError("training snapshot member position is invalid")
        if self.position < 0 or self.total < self.position:
            raise ValueError("training snapshot native position is invalid")
        if self.series_generation < 0:
            raise ValueError("training snapshot series generation must be non-negative")
        if self.learning_rate is not None and self.learning_rate < 0:
            raise ValueError("training snapshot learning rate must be non-negative")
        if self.elapsed_seconds is not None and self.elapsed_seconds < 0:
            raise ValueError("training snapshot elapsed time must be non-negative")
        if self.eta_seconds is not None and self.eta_seconds < 0:
            raise ValueError("training snapshot ETA must be non-negative")
        object.__setattr__(self, "trends", tuple(self.trends))


@dataclass(frozen=True, slots=True)
class LabelSystemProgress:
    system_id: str
    completed: int
    trusted: int | None = None
    total: int | None = None


@dataclass(frozen=True, slots=True)
class LabelingSnapshot:
    completed: int
    total: int
    rows: tuple[LabelSystemProgress, ...]
    trusted_below: float | None = None


@dataclass(frozen=True, slots=True)
class ScoreSystemProgress:
    system_id: str
    scored: int
    p_mean: float | None


@dataclass(frozen=True, slots=True)
class ScoreObservationSnapshot:
    role: str
    completed: int
    total: int
    rows: tuple[ScoreSystemProgress, ...]


@dataclass(frozen=True, slots=True)
class ScorecardRow:
    system_id: str
    values: tuple[int | None, ...]
    p_mean: float | None

    def __post_init__(self) -> None:
        if not self.system_id or any(value is not None and value < 0 for value in self.values):
            raise ValueError("scorecard row contains an invalid value")


@dataclass(frozen=True, slots=True)
class ScorecardSnapshot:
    role: str
    columns: tuple[str, ...]
    rows: tuple[ScorecardRow, ...]

    def __post_init__(self) -> None:
        if self.role not in {"teacher", "student"} or not self.rows:
            raise ValueError("scorecard requires a role, columns, and rows")
        if any(len(row.values) != len(self.columns) for row in self.rows):
            raise ValueError("scorecard rows must match its dynamic columns")


@dataclass(frozen=True, slots=True)
class RoutingSystemRow:
    system_id: str
    teacher: tuple[int, int]
    student: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class RoutingSnapshot:
    rows: tuple[RoutingSystemRow, ...]

    def __post_init__(self) -> None:
        if not self.rows or any(
            not row.system_id or any(value < 0 for value in (*row.teacher, *row.student))
            for row in self.rows
        ):
            raise ValueError("routing snapshot requires non-negative system rows")


@dataclass(frozen=True, slots=True)
class SamplingSystemProgress:
    system_id: str
    completed: int
    total: int


@dataclass(frozen=True, slots=True)
class SamplingTaskDetail:
    task_id: str
    system_id: str
    temperature_k: float | None = None
    step: int | None = None
    speed: float | None = None
    speed_unit: str | None = None
    diagnostic_locator: str | None = None
    state: str = "active"
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.system_id:
            raise ValueError("sampling task detail requires task and system identity")
        if self.step is not None and self.step < 0:
            raise ValueError("sampling task detail step must be non-negative")
        if self.speed is not None and self.speed < 0:
            raise ValueError("sampling task detail speed must be non-negative")
        if self.state not in {"active", "slow", "warning", "failed", "cancelled"}:
            raise ValueError("sampling task detail state is invalid")


@dataclass(frozen=True, slots=True)
class SamplingSnapshot:
    backend: str
    model_shape: str
    completed_tasks: int
    total_tasks: int
    systems: int
    temperatures: int
    ensemble: str
    steps_per_task: int
    rows: tuple[SamplingSystemProgress, ...]
    pressure_bar: float | None = None
    plumed: bool = False
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    active_task_id: str | None = None
    active_system_id: str | None = None
    active_temperature_k: float | None = None
    active_step: int | None = None
    speed: float | None = None
    speed_unit: str | None = None
    task_details: tuple[SamplingTaskDetail, ...] = ()
    omitted_task_details: int = 0
    mps_phase: str | None = None
    task_step_totals: tuple[int, ...] = ()
    sampling_role_totals: tuple[tuple[str, int, int], ...] = ()
    condition_group_totals: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.backend or self.model_shape not in {"single", "committee"}:
            raise ValueError("sampling snapshot backend or model shape is invalid")
        values = (self.completed_tasks, self.total_tasks, self.systems, self.temperatures, self.steps_per_task)
        if any(value < 0 for value in values) or self.completed_tasks > self.total_tasks:
            raise ValueError("sampling snapshot counts are invalid")
        if not self.ensemble or not self.rows:
            raise ValueError("sampling snapshot requires matrix and system rows")
        if any(row.completed < 0 or row.total < row.completed for row in self.rows):
            raise ValueError("sampling system progress is invalid")
        if self.active_step is not None and self.active_step < 0:
            raise ValueError("sampling active step is invalid")
        if self.omitted_task_details < 0:
            raise ValueError("sampling omitted task detail count is invalid")
        object.__setattr__(self, "task_details", tuple(self.task_details))
        task_steps = tuple(int(item) for item in self.task_step_totals)
        if any(item <= 0 for item in task_steps):
            raise ValueError("sampling task step totals must be positive")
        for label, rows in (
            ("role", self.sampling_role_totals),
            ("condition group", self.condition_group_totals),
        ):
            if any(
                not str(identity).strip() or int(tasks) <= 0 or int(steps) <= 0
                for identity, tasks, steps in rows
            ):
                raise ValueError(f"sampling {label} totals are invalid")
        object.__setattr__(self, "task_step_totals", task_steps)
        object.__setattr__(self, "sampling_role_totals", tuple(self.sampling_role_totals))
        object.__setattr__(self, "condition_group_totals", tuple(self.condition_group_totals))


@dataclass(frozen=True, slots=True)
class DescriptorSnapshot:
    label: str
    completed: int
    total: int
    width: int | None = None
    elements_completed: int | None = None
    elements_total: int | None = None


ProgressSnapshot: TypeAlias = (
    GenericWorkSnapshot
    | DatasetSnapshot
    | CountChainSnapshot
    | CountFlowSnapshot
    | TrainingSnapshot
    | LabelingSnapshot
    | ScoreObservationSnapshot
    | ScorecardSnapshot
    | RoutingSnapshot
    | SamplingSnapshot
    | DescriptorSnapshot
)


@dataclass(frozen=True, slots=True)
class BasicBlockSummary:
    """Compact lifecycle-only summary used before block-family enrichment."""

    text: str | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class TrainBlockSummary:
    backend: str
    models: int
    dataset_unique: int
    train: int | None = None
    valid: int | None = None
    evaluation_same_data: bool = False
    elapsed_seconds: float | None = None

    def __post_init__(self) -> None:
        values = (self.models, self.dataset_unique, *(item for item in (self.train, self.valid) if item is not None))
        if not self.backend or any(value < 0 for value in values):
            raise ValueError("train summary requires non-negative counts and backend")


@dataclass(frozen=True, slots=True)
class ScreenBlockSummary:
    observations: int
    train: int
    fp_candidates: int
    elapsed_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class FPBlockSummary:
    candidates: int
    unique: int
    selected: int
    handoff_path: str
    legal_empty: bool
    elapsed_seconds: float | None = None


# The screen-specific progress vocabulary is logic-owned.  These aliases keep
# historical runtime imports class-identical while renderers remain outward.
from workflow_core.lifecycle.screen_progress import (  # noqa: E402
    CountChainSnapshot as CountChainSnapshot,
    CountFlowSnapshot as CountFlowSnapshot,
    CountStep as CountStep,
    GenericWorkSnapshot as GenericWorkSnapshot,
    RoutingSnapshot as RoutingSnapshot,
    RoutingSystemRow as RoutingSystemRow,
    ScreenBlockSummary as ScreenBlockSummary,
    ScorecardRow as ScorecardRow,
    ScorecardSnapshot as ScorecardSnapshot,
    SystemCountRow as SystemCountRow,
)


@dataclass(frozen=True, slots=True)
class SamplingBlockSummary:
    backend: str
    model_shape: str
    tasks: int
    systems: int
    temperatures: int
    total_steps: int
    elapsed_seconds: float | None = None
    kind_counts: tuple[tuple[str, int], ...] = ()
    mc_attempts: int = 0
    failed_tasks: int = 0

    def __post_init__(self) -> None:
        if (
            not self.backend
            or self.model_shape not in {"single", "committee"}
            or any(
                value < 0
                for value in (
                    self.tasks,
                    self.systems,
                    self.temperatures,
                    self.total_steps,
                    self.mc_attempts,
                    self.failed_tasks,
                )
            )
        ):
            raise ValueError("sampling summary fields are invalid")
        kinds = tuple((str(kind), int(count)) for kind, count in self.kind_counts)
        if (
            any(not kind or count <= 0 for kind, count in kinds)
            or len({kind for kind, _count in kinds}) != len(kinds)
            or (kinds and sum(count for _kind, count in kinds) != self.tasks)
        ):
            raise ValueError("sampling summary kind counts are invalid")
        object.__setattr__(self, "kind_counts", kinds)


BlockSummary: TypeAlias = (
    BasicBlockSummary | TrainBlockSummary | ScreenBlockSummary | FPBlockSummary | SamplingBlockSummary
)


@dataclass(frozen=True, slots=True)
class AttemptStarted:
    resume_target: str | None = None


@dataclass(frozen=True, slots=True)
class BlockStarted:
    pass


@dataclass(frozen=True, slots=True)
class PhaseEntered:
    phase: str
    operation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("progress phase must be non-empty")


@dataclass(frozen=True, slots=True)
class SnapshotReported:
    snapshot: ProgressSnapshot


@dataclass(frozen=True, slots=True)
class HeartbeatReported:
    pass


@dataclass(frozen=True, slots=True)
class BlockSucceeded:
    summary: BlockSummary = BasicBlockSummary()


@dataclass(frozen=True, slots=True)
class BlockFailed:
    message: str | None = None
    diagnostic_ref: str | None = None


@dataclass(frozen=True, slots=True)
class BlockInterrupted:
    message: str | None = None
    diagnostic_ref: str | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticArchived:
    archived_attempt_id: str
    diagnostic_ref: str


@dataclass(frozen=True, slots=True)
class DiagnosticPruned:
    pruned_attempt_id: str


ProgressPayload: TypeAlias = (
    AttemptStarted
    | BlockStarted
    | PhaseEntered
    | SnapshotReported
    | HeartbeatReported
    | BlockSucceeded
    | BlockFailed
    | BlockInterrupted
    | DiagnosticArchived
    | DiagnosticPruned
)


@dataclass(frozen=True, slots=True)
class ProgressEnvelope:
    attempt_id: str
    sequence: int
    received_at: str
    iter_index: int
    block: str
    phase: str | None
    operation_id: str | None
    payload: ProgressPayload

    def __post_init__(self) -> None:
        if not self.attempt_id or not self.block:
            raise ValueError("progress envelope requires attempt_id and block")
        if self.sequence < 1:
            raise ValueError("progress sequence starts at one")
