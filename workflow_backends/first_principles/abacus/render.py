from __future__ import annotations

import math

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.units import Bohr

from workflow_backends.first_principles.abacus.recipe import (
    ABACUSElementAsset,
)
from workflow_backends.first_principles.geometry import reciprocal_spacing_mesh


def abacus_species_and_atom_order(
    atoms: Atoms,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    symbols = tuple(atoms.get_chemical_symbols())
    species = tuple(dict.fromkeys(symbols))
    atom_order = tuple(
        index
        for element in species
        for index, symbol in enumerate(symbols)
        if symbol == element
    )
    return species, atom_order


def render_abacus_stru(
    atoms: Atoms,
    *,
    pseudopotentials: tuple[ABACUSElementAsset, ...],
    orbitals: tuple[ABACUSElementAsset, ...],
) -> tuple[str, tuple[int, ...]]:
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    if (
        cell.shape != (3, 3)
        or not np.all(np.isfinite(cell))
        or not np.all(np.isfinite(positions))
        or abs(float(np.linalg.det(cell))) < 1.0e-12
    ):
        raise ValueError("ABACUS STRU rendering requires finite nonzero cell geometry")
    if not bool(np.all(atoms.pbc)):
        raise ValueError("ABACUS STRU rendering requires three-dimensional periodicity")
    species, atom_order = abacus_species_and_atom_order(atoms)
    pseudo_by_element = {asset.element: asset for asset in pseudopotentials}
    orbital_by_element = {asset.element: asset for asset in orbitals}
    if tuple(pseudo_by_element) != species:
        raise ValueError("ABACUS pseudopotential order does not match selected species")
    if orbital_by_element and tuple(orbital_by_element) != species:
        raise ValueError("ABACUS orbital order does not match selected species")
    lines = ["ATOMIC_SPECIES"]
    for element in species:
        mass = float(atomic_masses[atomic_numbers[element]])
        lines.append(
            f"{element} {mass:.10f} "
            f"{pseudo_by_element[element].destination.as_posix()}"
        )
    if orbital_by_element:
        lines.extend(["", "NUMERICAL_ORBITAL"])
        lines.extend(
            orbital_by_element[element].destination.as_posix() for element in species
        )
    lines.extend(
        [
            "",
            "LATTICE_CONSTANT",
            f"{1.0 / Bohr:.16f}",
            "",
            "LATTICE_VECTORS",
        ]
    )
    lines.extend(" ".join(f"{float(value):.16f}" for value in row) for row in cell)
    scaled = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("ABACUS STRU rendering produced non-finite direct coordinates")
    symbols = tuple(atoms.get_chemical_symbols())
    lines.extend(["", "ATOMIC_POSITIONS", "Direct"])
    for element in species:
        indices = [index for index, symbol in enumerate(symbols) if symbol == element]
        lines.extend([element, "0.0", str(len(indices))])
        for index in indices:
            lines.append(
                " ".join(f"{float(value):.16f}" for value in scaled[index])
                + " 0 0 0"
            )
    return "\n".join(lines) + "\n", atom_order


def render_abacus_kpt(
    atoms: Atoms, spacing_inv_angstrom: float
) -> tuple[str, tuple[int, int, int]]:
    mesh = reciprocal_spacing_mesh(atoms, spacing_inv_angstrom)
    text = "K_POINTS\n0\nGamma\n" + " ".join(map(str, mesh)) + " 0 0 0\n"
    return text, mesh


__all__ = [
    "abacus_species_and_atom_order",
    "render_abacus_kpt",
    "render_abacus_stru",
]
