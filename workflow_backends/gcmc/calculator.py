"""Copy-safe ASE view of one sequential worker-local evaluator."""
from __future__ import annotations

import numpy as np

from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes


class SharedCalculator(Calculator):
    """Share inference, never ASE result caches or model-selection authority.

    The caller owns evaluator construction and ordered driving-member binding.
    Copies are for sequential calls in the same worker, not concurrent use or
    serializable checkpoints. No energy/reference/force correction lives here.
    """

    def __init__(self, evaluator: Calculator, *, context: str):
        super().__init__()
        self._evaluator = evaluator
        self._context = str(context)
        self.implemented_properties = list(evaluator.implemented_properties)

    def __deepcopy__(self, memo):
        result = type(self)(self._evaluator, context=self._context)
        memo[id(self)] = result
        return result

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {}
        values = {}
        for name in properties:
            try:
                value = self._evaluator.get_property(name, self.atoms)
            except PropertyNotImplementedError:
                raise
            except Exception as exc:
                raise RuntimeError(f'{self._context}: {name} inference failed') from exc
            if not np.all(np.isfinite(value)):
                raise ValueError(f'{self._context}: {name} inference returned nonfinite values')
            allowed_shapes = {
                'energy': ((),), 'free_energy': ((),),
                'forces': ((len(self.atoms), 3),), 'stress': ((6,), (3, 3)),
            }
            if name in allowed_shapes and np.shape(value) not in allowed_shapes[name]:
                raise ValueError(f'{self._context}: {name} inference returned an invalid shape')
            values[name] = value.copy() if isinstance(value, np.ndarray) else value
        self.results = values
