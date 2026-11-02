"""Provider-internal model capability; geometry never depends on a model family."""
from typing import Protocol
import numpy as np


class Evaluator(Protocol):
    def evaluate(self, frames): ...
    def close(self): ...


class EvaluationProvider(Protocol):
    def preflight(self, paths, symbols): ...
    def open(self, paths, symbols) -> Evaluator: ...


def checked_evaluation(evaluator, frame, member_count):
    result = tuple(evaluator.evaluate((frame,)))
    if len(result) != 1:
        raise ValueError('widom evaluator changed frame count')
    energies, forces = (np.asarray(v, dtype=float) for v in result[0])
    if (energies.shape != (member_count,) or forces.shape != (member_count, len(frame), 3)
        or not np.isfinite(energies).all() or not np.isfinite(forces).all()):
        raise ValueError('widom evaluator returned invalid/nonfinite member evidence')
    return energies, forces


class DeepMDEvaluationProvider:
    def preflight(self, paths, symbols):
        if not paths or any(not p.is_file() or p.is_symlink() for p in paths):
            raise ValueError('widom requires prepared regular DeepMD models')

    def open(self, paths, symbols):
        from deepmd.infer import DeepPot
        models = [DeepPot(str(p)) for p in paths]
        for model in models:
            if not set(symbols) <= set(model.get_type_map()):
                raise ValueError('widom species outside DeepMD native type map')
            if getattr(model, 'get_dim_fparam', lambda: 0)() or getattr(model, 'get_dim_aparam', lambda: 0)():
                raise ValueError('widom does not supply model fparam/aparam inputs')
        return _DeepMDEvaluator(models)


class _DeepMDEvaluator:
    def __init__(self, models):
        self.models = models

    def evaluate(self, frames):
        for frame in frames:
            energies, forces = [], []
            for model in self.models:
                types = [model.get_type_map().index(s) for s in frame.get_chemical_symbols()]
                e, f, _ = model.eval(frame.positions.reshape(1, -1), frame.cell.array.reshape(1, 9), types)[:3]
                energies.append(float(np.asarray(e).reshape(-1)[0]))
                forces.append(np.asarray(f).reshape(len(frame), 3))
            yield np.asarray(energies), np.asarray(forces)

    def close(self):
        self.models.clear()
