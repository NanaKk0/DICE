from __future__ import annotations

import numpy as np
from ase import Atoms


def compute_rmsd(atoms_a: Atoms, atoms_b: Atoms | None) -> float:
    if atoms_b is None or len(atoms_a) != len(atoms_b):
        return float("inf")
    if atoms_a.get_chemical_symbols() != atoms_b.get_chemical_symbols():
        return float("inf")

    pos_a = atoms_a.get_positions()
    pos_b = atoms_b.get_positions()
    return float(np.sqrt(np.mean(np.sum((pos_a - pos_b) ** 2, axis=1))))
