"""Scientific initial-seed invariants for composition-preserving NVT/NPT swaps."""
import numpy as np
from ase.data import atomic_masses


def validate_seed(seed, parameters, type_map):
    if (not len(seed) or seed.constraints or not np.all(seed.pbc)
        or seed.cell.rank != 3 or np.linalg.det(seed.cell.array) <= 0
        or not np.isfinite(seed.positions).all() or not np.isfinite(seed.cell.array).all()
        or not np.isfinite(seed.get_masses()).all() or np.any(seed.get_masses() <= 0)):
        raise ValueError('md_mc requires an unconstrained finite full-rank periodic seed')
    if not np.allclose(seed.get_masses(), atomic_masses[seed.numbers], rtol=0, atol=1e-10):
        raise ValueError('md_mc currently requires standard elemental masses')
    symbols = set(seed.get_chemical_symbols())
    if not set(parameters.elements) <= symbols or not symbols <= set(type_map):
        raise ValueError('md_mc seed and active type map must cover both exchange elements')
