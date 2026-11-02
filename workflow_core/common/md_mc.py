"""Closed, backend-neutral fixed-composition identity-swap protocol."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IdentitySwapParameters:
    elements: tuple[str, str]
    every_md_steps: int
    attempts_per_event: int
    move: str = 'identity_swap'

    def __post_init__(self):
        elements = tuple(self.elements)
        if len(elements) != 2 or any(not isinstance(s, str) or not s for s in elements) or elements[0] == elements[1]:
            raise ValueError('md_mc elements must be two different element symbols')
        if self.move != 'identity_swap':
            raise ValueError('md_mc move must be identity_swap')
        for field in ('every_md_steps', 'attempts_per_event'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'md_mc {field} must be a positive integer')
        object.__setattr__(self, 'elements', elements)

    def planned_attempts(self, steps: int) -> int:
        return steps // self.every_md_steps * self.attempts_per_event
