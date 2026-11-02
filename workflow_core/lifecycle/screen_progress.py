from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenericWorkSnapshot:
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
        if (
            self.completed is not None
            and self.total is not None
            and self.completed > self.total
        ):
            raise ValueError("progress completed cannot exceed total")


@dataclass(frozen=True, slots=True)
class CountStep:
    label: str
    value: int | None

    def __post_init__(self) -> None:
        if not self.label or (self.value is not None and self.value < 0):
            raise ValueError(
                "count step requires a label and a non-negative completed value"
            )


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
        if not self.system_id or any(
            value is not None and value < 0 for value in self.values
        ):
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
class ScorecardRow:
    system_id: str
    values: tuple[int | None, ...]
    p_mean: float | None

    def __post_init__(self) -> None:
        if not self.system_id or any(
            value is not None and value < 0 for value in self.values
        ):
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
            not row.system_id
            or any(value < 0 for value in (*row.teacher, *row.student))
            for row in self.rows
        ):
            raise ValueError("routing snapshot requires non-negative system rows")


@dataclass(frozen=True, slots=True)
class ScreenBlockSummary:
    observations: int
    train: int
    fp_candidates: int
    elapsed_seconds: float | None = None


__all__ = [
    "CountChainSnapshot",
    "CountFlowSnapshot",
    "CountStep",
    "GenericWorkSnapshot",
    "RoutingSnapshot",
    "RoutingSystemRow",
    "ScreenBlockSummary",
    "ScorecardRow",
    "ScorecardSnapshot",
    "SystemCountRow",
]
