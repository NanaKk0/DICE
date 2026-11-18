"""Explicit scientific requests; omission delegates that setting to native input."""
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class NativeProtocolRequest:
    ensemble: str | None = None
    temperature_k: float | None = None
    pressure_bar: float | None = None
    steps: int | None = None
    timestep_ps: float | None = None
    trajectory_every_steps: int | None = None
    group: str | None = None
    temperature_damping_ps: float | None = None
    pressure_damping_ps: float | None = None
    pressure_coupling: str | None = None
    thermostat: str | None = None
    barostat: str | None = None
    run: int | None = None

    def __post_init__(self):
        if self.ensemble not in {None, 'nvt', 'npt', 'nve'}:
            raise ValueError('ensemble request must be nvt, npt, or nve')
        for name in ('temperature_k', 'timestep_ps', 'temperature_damping_ps', 'pressure_damping_ps'):
            value = getattr(self, name)
            if value is not None and (type(value) not in {float, int} or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        if self.pressure_bar is not None and (type(self.pressure_bar) not in {float, int} or not math.isfinite(self.pressure_bar)):
            raise ValueError('pressure_bar must be finite')
        for name in ('steps', 'trajectory_every_steps', 'run'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f'{name} must be a positive integer')
        if self.pressure_coupling not in {None, 'iso', 'aniso', 'tri'}:
            raise ValueError('pressure_coupling must be iso, aniso, or tri')
        for name in ('thermostat', 'barostat', 'group'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{name} must identify a native fix')
