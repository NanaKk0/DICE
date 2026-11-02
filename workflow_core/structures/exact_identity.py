from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


EXACT_STRUCTURE_DECIMALS = 8


def exact_structure_hash(atoms: Any) -> str:
    numbers = np.asarray(atoms.numbers, dtype=np.int16)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    payload = [
        numbers.tobytes(),
        np.round(positions, EXACT_STRUCTURE_DECIMALS).tobytes(),
        np.round(cell, EXACT_STRUCTURE_DECIMALS).tobytes(),
        np.asarray(atoms.pbc, dtype=np.int8).tobytes(),
    ]
    return hashlib.sha256(b"".join(payload)).hexdigest()
