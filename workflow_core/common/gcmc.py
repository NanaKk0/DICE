"""Closed, path-free acquisition intent; not a native-engine keyword map."""
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class GcmcMdParameters:
    ensemble: str
    steps: int
    every: int
    pressure_bar: float | None = None

    def __post_init__(self):
        if self.ensemble not in {'nvt', 'nve', 'npt'}:
            raise ValueError('md.ensemble must be nvt, nve, or npt')
        for name in ('steps', 'every'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'md.{name} must be a positive integer')
        if self.ensemble == 'npt':
            if (isinstance(self.pressure_bar, bool) or not isinstance(self.pressure_bar, (int, float))
                or not math.isfinite(self.pressure_bar)):
                raise ValueError('md.pressure_bar must be explicit and finite for npt')
        elif self.pressure_bar is not None:
            raise ValueError('md.pressure_bar is valid only for npt')


@dataclass(frozen=True, slots=True)
class GcmcSamplingParameters:
    cycles_by_round: tuple[int, ...]
    moves: int
    every: int = 100
    translate_a: float = .75
    rotate_deg: float = 90.
    eos_kind: str = 'ideal'
    phi: float = 1.
    weights: tuple[float, ...] = (.25, .25, .25, .25, 0.)
    overlap_scale: float = .6
    radii_a: tuple[tuple[str, float], ...] = ()
    de_max_ev: float | None = None
    max_guests: int | None = None
    random_seed: int | None = None
    md: GcmcMdParameters | None = None
    tc_k: float | None = None
    pc_bar: float | None = None
    omega: float | None = None

    def __post_init__(self):
        if self.md is not None and not isinstance(self.md, GcmcMdParameters):
            raise ValueError('md requires closed MD parameters')
        for name, values in [('cycles', self.cycles_by_round), ('moves', (self.moves,)), ('every', (self.every,))]:
            if not values or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
                raise ValueError(f'{name} must contain positive integers')
        for name in ('translate_a', 'rotate_deg', 'phi', 'overlap_scale'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.rotate_deg > 180:
            raise ValueError('rotate_deg must not exceed 180')
        if self.eos_kind not in {'ideal', 'fixed_phi', 'pr'} or (self.eos_kind != 'fixed_phi' and self.phi != 1):
            raise ValueError('eos requires ideal, fixed_phi or pr with consistent phi')
        for name in ('tc_k', 'pc_bar', 'omega'):
            value = getattr(self, name)
            if self.eos_kind == 'pr':
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or (name != 'omega' and value <= 0)):
                    raise ValueError(f'PR {name} must be explicit and finite (critical values positive)')
            elif value is not None:
                raise ValueError(f'{name} is valid only for PR eos')
        if len(self.weights) != 5 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in self.weights):
            raise ValueError('weights must contain five finite nonnegative values')
        if abs(sum(self.weights)-1) > 1e-12 or self.weights[0] <= 0 or self.weights[0] != self.weights[1] or self.weights[4] != 0:
            raise ValueError('weights require equal positive exchange, sum one and zero reinsertion')
        if self.de_max_ev is not None and (isinstance(self.de_max_ev, bool) or not isinstance(self.de_max_ev, (int, float)) or not math.isfinite(self.de_max_ev) or self.de_max_ev <= 0):
            raise ValueError('de_max_ev must be null or finite and positive')
        for name, value, minimum in [('max_guests', self.max_guests, 1), ('random_seed', self.random_seed, 0)]:
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if len(dict(self.radii_a)) != len(self.radii_a) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for _, v in self.radii_a
        ):
            raise ValueError('radii_a requires unique symbols and finite positive radii')
