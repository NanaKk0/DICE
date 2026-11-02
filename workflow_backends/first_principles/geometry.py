from __future__ import annotations

import math

import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell


def reciprocal_spacing_mesh(
    atoms: Atoms, spacing_inv_angstrom: float
) -> tuple[int, int, int]:
    if (
        isinstance(spacing_inv_angstrom, bool)
        or not isinstance(spacing_inv_angstrom, (int, float))
        or not math.isfinite(float(spacing_inv_angstrom))
        or float(spacing_inv_angstrom) <= 0.0
    ):
        raise ValueError("reciprocal spacing must be a positive finite number")
    cell = np.asarray(atoms.cell.array, dtype=float)
    volume = float(np.linalg.det(cell))
    if not math.isfinite(volume) or abs(volume) < 1.0e-12:
        raise ValueError("cannot derive reciprocal mesh from zero-volume cell")
    reciprocal = 2.0 * math.pi * np.linalg.inv(cell).T
    spacing = float(spacing_inv_angstrom)
    return tuple(
        max(1, int(math.ceil(float(np.linalg.norm(vector)) / spacing)))
        for vector in reciprocal
    )


def length_angle_cellpar(atoms: Atoms) -> tuple[float, float, float, float, float, float]:
    cell = np.asarray(atoms.cell.array, dtype=float)
    cellpar = np.asarray(atoms.cell.cellpar(), dtype=float)
    if (
        cell.shape != (3, 3)
        or not np.all(np.isfinite(cell))
        or not np.all(np.isfinite(cellpar))
        or abs(float(np.linalg.det(cell))) < 1.0e-12
    ):
        raise ValueError("CP2K length/angle cell has invalid or zero volume")
    reconstructed = np.asarray(cellpar_to_cell(cellpar), dtype=float)
    if not np.allclose(cell, reconstructed, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(
            "CP2K length/angle rendering cannot preserve the selected cell representation"
        )
    return tuple(float(value) for value in cellpar)


__all__ = ["length_angle_cellpar", "reciprocal_spacing_mesh"]
