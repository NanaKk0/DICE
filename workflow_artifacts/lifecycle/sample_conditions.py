from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingTaskCondition:
    system_id: str
    ensemble: str
    temperature_k: float
    pressure_bar: float | None


@dataclass(frozen=True, slots=True)
class SystemSamplingCondition:
    system_id: str
    ensemble: str
    temperatures_k: tuple[float, ...]
    pressures_bar: tuple[float, ...]

    @property
    def task_count(self) -> int:
        return len(self.temperatures_k) * (len(self.pressures_bar) if self.ensemble == "npt" else 1)


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    systems: tuple[SystemSamplingCondition, ...]
    tasks: tuple[SamplingTaskCondition, ...]

    @classmethod
    def create(cls, systems: tuple[SystemSamplingCondition, ...]) -> "SamplingPlan":
        tasks: list[SamplingTaskCondition] = []
        for system in systems:
            if system.ensemble == "nvt":
                tasks.extend(
                    SamplingTaskCondition(system.system_id, system.ensemble, temperature, None)
                    for temperature in system.temperatures_k
                )
            else:
                tasks.extend(
                    SamplingTaskCondition(system.system_id, system.ensemble, temperature, pressure)
                    for temperature in system.temperatures_k
                    for pressure in system.pressures_bar
                )
        return cls(systems=systems, tasks=tuple(tasks))

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    @property
    def system_task_counts(self) -> dict[str, int]:
        return {system.system_id: system.task_count for system in self.systems}

    @property
    def ensembles(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(system.ensemble for system in self.systems))

    @property
    def total_temperature_values(self) -> int:
        return sum(len(system.temperatures_k) for system in self.systems)

__all__ = [
    "SamplingPlan",
    "SamplingTaskCondition",
    "SystemSamplingCondition",
]
