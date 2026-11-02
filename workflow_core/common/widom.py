"""Closed scientific intent for independent molecular insertion."""
from dataclasses import dataclass
import math

@dataclass(frozen=True, slots=True)
class WidomSamplingParameters:
    frames: int
    max_attempts: int
    random_seed: int = 0
    overlap_scale: float | None = 0.6
    radii_a: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        for name in ('frames', 'max_attempts', 'random_seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'random_seed' else 1):
                raise ValueError(f'widom {name} must be a valid integer budget/seed')
        if self.max_attempts < self.frames:
            raise ValueError('widom max_attempts must be >= frames')
        if self.overlap_scale is not None:
            self._positive(self.overlap_scale, 'overlap.scale')
        radii = tuple(tuple(item) for item in self.radii_a)
        if len(dict(radii)) != len(radii):
            raise ValueError('widom radii_a must have unique elements')
        for _, value in radii:
            self._positive(value, 'overlap.radii_a')
        object.__setattr__(self, 'radii_a', radii)

    @staticmethod
    def _positive(value, name):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'widom {name} must be finite and positive')
