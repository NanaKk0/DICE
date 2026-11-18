"""Typed scientific settings for the bounded LAMMPS control surface."""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True, slots=True)
class LammpsSamplingOptions:
    mode: str = 'resolved'
    group: str | None = None
    fix_id: str | None = None
    temperature_damping_ps: float | None = None
    pressure_damping_ps: float | None = None
    pressure_coupling: str | None = None
    timestep_ps: float | None = None
    trajectory_every_steps: int | None = None

    @classmethod
    def from_payload(cls, payload, *, prefix='lammps'):
        def mapping(value, allowed, path):
            if not isinstance(value, dict) or set(value) - allowed:
                raise ValueError(f'{path} contains unsupported fields or must be an object')
        mapping(payload, {'control', 'timestep_ps', 'trajectory_every_steps'}, prefix)
        control = payload.get('control', {})
        mapping(control, {'mode', 'group', 'fix_id', 'temperature_damping_ps',
                          'pressure_damping_ps', 'pressure_coupling'}, f'{prefix}.control')
        if 'mode' in control and (not isinstance(control.get('mode'), str) or control.get('mode') not in {'managed', 'native'}):
            raise ValueError(f'{prefix}.control.mode must be managed or native')
        if control.get('mode') == 'native' and set(control) != {'mode'}:
            raise ValueError(f'{prefix}.control native settings belong in the recipe')
        values = dict(control)
        values.update({k: v for k, v in payload.items() if k != 'control'})
        for field in ('temperature_damping_ps', 'pressure_damping_ps', 'timestep_ps'):
            if field in values:
                value = values[field]
                if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{prefix}.{field} must be a positive finite number')
        for field in ('group', 'fix_id'):
            if field in values and (not isinstance(values[field], str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', values[field])):
                raise ValueError(f'{prefix}.control.{field} must be a literal identifier')
        if 'pressure_coupling' in values and (not isinstance(values['pressure_coupling'], str) or values['pressure_coupling'] not in {'iso', 'aniso', 'tri'}):
            raise ValueError(f'{prefix}.control.pressure_coupling must be iso, aniso or tri')
        if 'trajectory_every_steps' in values:
            value = values['trajectory_every_steps']
            if type(value) is not int or value <= 0:
                raise ValueError(f'{prefix}.trajectory_every_steps must be a positive integer')
        return cls(**values)

    def to_payload(self):
        control = {name: getattr(self, name) for name in (
            'mode', 'group', 'fix_id', 'temperature_damping_ps',
            'pressure_damping_ps', 'pressure_coupling') if getattr(self, name) is not None}
        if self.mode == 'resolved':
            del control['mode']
        return {'control': control, **{name: getattr(self, name) for name in (
            'timestep_ps', 'trajectory_every_steps') if getattr(self, name) is not None}}
