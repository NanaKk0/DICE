from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io import read, write

from workflow_core.structures.canonical_identity import (
    copy_atoms_preserving_singlepoint_results,
    structure_id_for_atoms,
    validate_structure_source,
)


def read_extxyz_frame(path: str | Path, *, index: int = 0) -> Atoms:
    return read_extxyz_structures(path, allow_empty=False)[index]


def read_extxyz_structures(path: str | Path, *, allow_empty: bool = False) -> list[Atoms]:
    input_path = Path(path).resolve()
    if not input_path.exists() or not input_path.is_file():
        raise ValueError(f"missing extxyz file: {input_path}")
    if input_path.stat().st_size == 0:
        if allow_empty:
            return []
        raise ValueError(f"empty extxyz is not allowed: {input_path}")
    try:
        frames = list(read(str(input_path), index=":"))
    except Exception as exc:
        raise ValueError(f"invalid extxyz file: {input_path}") from exc
    if not frames and not allow_empty:
        raise ValueError(f"empty extxyz is not allowed: {input_path}")
    return frames


def write_extxyz_structures(path: str | Path, atoms_list: Iterable[Atoms], *, allow_empty: bool = True) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = list(atoms_list)
    if not frames:
        if not allow_empty:
            raise ValueError(f"empty extxyz is not allowed: {output}")
        output.write_text("", encoding="utf-8")
        return output
    write(output, frames, format="extxyz")
    return output


def concat_extxyz_files(
    sources: Iterable[str | Path],
    target: str | Path,
    *,
    allow_empty_sources: bool = True,
    allow_empty_output: bool = True,
) -> Path:
    frames: list[Atoms] = []
    for source in sources:
        source_path = Path(source)
        frames.extend(read_extxyz_structures(source_path, allow_empty=allow_empty_sources))
    return write_extxyz_structures(target, frames, allow_empty=allow_empty_output)


def copy_extxyz_file(
    source: str | Path,
    target: str | Path,
    *,
    allow_empty: bool = True,
) -> Path:
    frames = read_extxyz_structures(source, allow_empty=allow_empty)
    return write_extxyz_structures(target, frames, allow_empty=allow_empty)


def validate_extxyz_file(path: str | Path, *, allow_empty: bool = False) -> bool:
    try:
        read_extxyz_structures(path, allow_empty=allow_empty)
    except ValueError:
        return False
    return True


def annotate_canonical_structure(
    atoms: Atoms,
    *,
    force_std_max: float | None = None,
    route: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Atoms:
    item = copy_atoms_preserving_singlepoint_results(atoms)
    info = item.info or {}
    system_id = str(info.get("system_id") or item.get_chemical_formula(mode="hill"))
    times_used = int(info.get("times_used", info.get("seed_use_count", 0)) or 0)
    item.info["system_id"] = system_id
    item.info["structure_id"] = structure_id_for_atoms(item)
    item.info["structure_source"] = validate_structure_source(info.get("structure_source", "workflow"))
    for retired_alias in ("seed_system_id", "seed_structure_id", "source_system_id", "source_structure_id"):
        item.info.pop(retired_alias, None)
    item.info["times_used"] = max(0, times_used)
    item.info["seed_use_count"] = max(0, times_used)
    item.info.setdefault("force_std_max", float(info.get("force_std_max", info.get("decision_metric", 0.0)) or 0.0))
    if force_std_max is not None:
        item.info["force_std_max"] = float(force_std_max)
        item.info["student_uncertainty"] = float(force_std_max)
    if route is not None:
        item.info["route"] = str(route)
    for key, value in dict(metadata or {}).items():
        item.info[str(key)] = value
    return item


def active_extxyz_to_canonical_structures(
    active_xyz_path: str | Path,
    *,
    force_std_values: list[float] | None = None,
    routes: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> list[Atoms]:
    frames = read_extxyz_structures(active_xyz_path, allow_empty=True)
    result: list[Atoms] = []
    for index, atoms in enumerate(frames):
        force_std_max = None if force_std_values is None or index >= len(force_std_values) else force_std_values[index]
        route = None if routes is None or index >= len(routes) else routes[index]
        result.append(
            annotate_canonical_structure(
                atoms,
                force_std_max=force_std_max,
                route=route,
                metadata=metadata,
            )
        )
    return result


def active_extxyz_to_canonical_extxyz(
    active_xyz_path: str | Path,
    output_path: str | Path,
    *,
    force_std_values: list[float] | None = None,
    routes: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    frames = active_extxyz_to_canonical_structures(
        active_xyz_path,
        force_std_values=force_std_values,
        routes=routes,
        metadata=metadata,
    )
    return write_extxyz_structures(output_path, frames, allow_empty=True).resolve()
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
