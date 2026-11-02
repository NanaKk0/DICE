from __future__ import annotations

import re

from ase import Atoms

from workflow_backends.first_principles.geometry import (
    length_angle_cellpar,
    reciprocal_spacing_mesh,
)
from workflow_backends.first_principles.cp2k.recipe import CP2KRecipe


_KPOINT_NAMES = ("KPOINTS_A", "KPOINTS_B", "KPOINTS_C")


def render_cp2k_input(
    recipe: CP2KRecipe,
    atoms: Atoms,
    *,
    coordinate_name: str = "coordinates.xyz",
) -> tuple[str, tuple[int, int, int] | None]:
    a, b, c, alpha, beta, gamma = length_angle_cellpar(atoms)
    values = {
        "COORDFILE": coordinate_name,
        "CELL_A": f"{a:.10f}",
        "CELL_B": f"{b:.10f}",
        "CELL_C": f"{c:.10f}",
        "ANGLE_A": f"{alpha:.8f}",
        "ANGLE_B": f"{beta:.8f}",
        "ANGLE_C": f"{gamma:.8f}",
    }
    mesh = None
    if recipe.spacing_inv_angstrom is not None:
        mesh = reciprocal_spacing_mesh(atoms, recipe.spacing_inv_angstrom)
        values.update(dict(zip(_KPOINT_NAMES, map(str, mesh), strict=True)))
        present = set(
            re.findall(
                r"^\s*@SET\s+([A-Z_]+)\b",
                recipe.template.upper(),
                flags=re.MULTILINE,
            )
        )
        missing = sorted(set(_KPOINT_NAMES) - present)
        if missing:
            raise ValueError(
                f"CP2K typed kpoints need template @SET variables: {missing}"
            )
    rendered: list[str] = []
    for line in recipe.template.splitlines():
        match = re.match(r"^(\s*)@SET\s+([A-Za-z_]+)\b", line)
        name = match.group(2).upper() if match else None
        if match and name in values:
            rendered.append(f"{match.group(1)}@SET {match.group(2)} {values[name]}")
        else:
            rendered.append(line)
    return "\n".join(rendered) + "\n", mesh


def render_cp2k_coordinates(atoms: Atoms) -> str:
    lines = [str(len(atoms)), "DICE static CP2K coordinates"]
    for symbol, position in zip(
        atoms.get_chemical_symbols(), atoms.positions, strict=True
    ):
        lines.append(
            f"{symbol} " + " ".join(f"{float(value):.16f}" for value in position)
        )
    return "\n".join(lines) + "\n"


__all__ = ["render_cp2k_coordinates", "render_cp2k_input"]
