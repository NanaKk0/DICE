from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


def read_extxyz_frame(path: str | Path, *, index: int = 0) -> Atoms:
    return read(str(Path(path).resolve()), index=index)


def read_extxyz_structures(path: str | Path) -> list[Atoms]:
    return list(read(str(Path(path).resolve()), index=":"))


def write_extxyz_structures(path: str | Path, atoms_list: list[Atoms]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not atoms_list:
        output.write_text("", encoding="utf-8")
        return output
    write(output, atoms_list, format="extxyz")
    return output


def atoms_to_lammps_data(atoms: Atoms, output: str | Path, *, type_map: list[str]) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, atoms, format="lammps-data", atom_style="atomic", specorder=list(type_map))
    return output


def write_lammps_seed_data_from_paths(
    initial_structure_paths: list[str],
    conf_dir: str | Path,
    *,
    type_map: list[str],
) -> dict[str, Path]:
    conf_dir = Path(conf_dir)
    conf_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for index, structure_path in enumerate(initial_structure_paths):
        resolved = str(Path(structure_path).resolve())
        atoms = read_extxyz_frame(resolved, index=0)
        output = conf_dir / f"seed_{index:06d}.lmp"
        atoms_to_lammps_data(atoms, output, type_map=type_map)
        mapping[resolved] = output.resolve()
    return mapping


def _parse_lammpstrj_box(lines: list[str], index: int, step: int) -> tuple[np.ndarray, int]:
    box_header = lines[index + 4].strip()
    if not box_header.startswith("ITEM: BOX BOUNDS"):
        raise ValueError(f"Unexpected lammpstrj format near timestep {step}")
    bounds_tokens = box_header.split()[3:]
    has_tilt = bounds_tokens[:3] == ["xy", "xz", "yz"]

    bound_x = [float(item) for item in lines[index + 5].split()[: 3 if has_tilt else 2]]
    bound_y = [float(item) for item in lines[index + 6].split()[: 3 if has_tilt else 2]]
    bound_z = [float(item) for item in lines[index + 7].split()[: 3 if has_tilt else 2]]

    if has_tilt:
        xlo_bound, xhi_bound, xy = bound_x
        ylo_bound, yhi_bound, xz = bound_y
        zlo_bound, zhi_bound, yz = bound_z

        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        zlo = zlo_bound
        zhi = zhi_bound

        lx = xhi - xlo
        ly = yhi - ylo
        lz = zhi - zlo
        cell = np.array(
            [
                [lx, 0.0, 0.0],
                [xy, ly, 0.0],
                [xz, yz, lz],
            ],
            dtype=float,
        )
    else:
        xlo, xhi = bound_x[:2]
        ylo, yhi = bound_y[:2]
        zlo, zhi = bound_z[:2]
        cell = np.array(
            [
                [xhi - xlo, 0.0, 0.0],
                [0.0, yhi - ylo, 0.0],
                [0.0, 0.0, zhi - zlo],
            ],
            dtype=float,
        )
    return cell, 9


def _build_atoms_from_lammpstrj_frame(
    *,
    atom_header: list[str],
    atom_rows: list[str],
    natoms: int,
    type_map: list[str],
    cell: np.ndarray,
    step: int,
    include_forces: bool,
) -> Atoms:
    col_index = {name: idx for idx, name in enumerate(atom_header)}
    positions = np.zeros((natoms, 3), dtype=float)
    forces = np.zeros((natoms, 3), dtype=float) if include_forces else None
    symbols_out: list[str] = ["X"] * natoms
    for row in atom_rows:
        fields = row.split()
        atom_id = int(fields[col_index["id"]]) - 1
        type_idx = int(fields[col_index["type"]]) - 1
        positions[atom_id, 0] = float(fields[col_index["x"]])
        positions[atom_id, 1] = float(fields[col_index["y"]])
        positions[atom_id, 2] = float(fields[col_index["z"]])
        if include_forces and forces is not None and {"fx", "fy", "fz"}.issubset(col_index):
            forces[atom_id, 0] = float(fields[col_index["fx"]])
            forces[atom_id, 1] = float(fields[col_index["fy"]])
            forces[atom_id, 2] = float(fields[col_index["fz"]])
        symbols_out[atom_id] = type_map[type_idx]

    atoms = Atoms(symbols=symbols_out, positions=positions, cell=cell, pbc=True)
    if include_forces and forces is not None:
        atoms.arrays["forces"] = forces
    atoms.info["md_step"] = int(step)
    return atoms


def parse_lammpstrj_all_frames(path: str | Path, *, type_map: list[str], include_forces: bool = True) -> dict[int, Atoms]:
    frames: dict[int, Atoms] = {}
    path = Path(path)
    if not path.exists():
        return frames
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    index = 0
    symbols = list(type_map)
    while index < len(lines):
        if not lines[index].startswith("ITEM: TIMESTEP"):
            index += 1
            continue
        step = int(lines[index + 1].strip())
        if not lines[index + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Unexpected lammpstrj format near timestep {step}")
        natoms = int(lines[index + 3].strip())
        cell, atom_header_index = _parse_lammpstrj_box(lines, index, step)
        atom_header = lines[index + 8].strip().split()[2:]
        atom_rows = lines[index + 9 : index + 9 + natoms]
        atoms = _build_atoms_from_lammpstrj_frame(
            atom_header=atom_header,
            atom_rows=atom_rows,
            natoms=natoms,
            type_map=symbols,
            cell=cell,
            step=step,
            include_forces=include_forces,
        )
        frames[int(step)] = atoms
        index = index + atom_header_index + natoms
    return frames


def extract_lammpstrj_selected_frames(
    path: str | Path,
    *,
    type_map: list[str],
    target_steps: set[int],
    include_forces: bool = False,
) -> dict[int, Atoms]:
    frames: dict[int, Atoms] = {}
    path = Path(path)
    if not path.exists() or not target_steps:
        return frames
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    index = 0
    symbols = list(type_map)
    while index < len(lines):
        if not lines[index].startswith("ITEM: TIMESTEP"):
            index += 1
            continue
        step = int(lines[index + 1].strip())
        if not lines[index + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Unexpected lammpstrj format near timestep {step}")
        natoms = int(lines[index + 3].strip())
        cell, atom_header_index = _parse_lammpstrj_box(lines, index, step)
        atom_header = lines[index + 8].strip().split()[2:]
        atom_rows = lines[index + 9 : index + 9 + natoms]
        if step in target_steps:
            atoms = _build_atoms_from_lammpstrj_frame(
                atom_header=atom_header,
                atom_rows=atom_rows,
                natoms=natoms,
                type_map=symbols,
                cell=cell,
                step=step,
                include_forces=include_forces,
            )
            frames[int(step)] = atoms
            if len(frames) == len(target_steps):
                break
        index = index + atom_header_index + natoms
    return frames
