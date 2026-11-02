from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from workflow_core.structures.validation import (
    validate_deepmd_export_inputs,
    validate_written_deepmd_dataset,
)


def write_deepmd_dataset(
    atoms_list: list[Any],
    output_dir: str | Path,
    element_order: list[str],
) -> Path:
    """Materialize one backend-native DeepMD dataset from canonical frames."""
    atoms_list = [_sort_atoms(atoms) for atoms in atoms_list]
    validation = validate_deepmd_export_inputs(atoms_list, element_order)
    stress_present = [has_reference_stress(atoms) for atoms in atoms_list]
    if any(stress_present) and not all(stress_present):
        raise ValueError("Split frames by reference stress availability before DeepMD export")
    require_virial = all(stress_present)
    resolved = Path(output_dir).resolve()
    set_dir = resolved / "set.000"
    if not require_virial and (set_dir / "virial.npy").exists():
        raise ValueError("Refusing stale virial labels in an energy/force-only dataset")
    set_dir.mkdir(parents=True, exist_ok=True)

    nframes = int(validation["nframes"])
    natoms = int(validation["natoms"])
    types = list(validation["types"])
    (resolved / "type_map.raw").write_text(" ".join(element_order) + "\n", encoding="utf-8")
    (resolved / "type.raw").write_text(" ".join(map(str, types)) + "\n", encoding="utf-8")

    coords = np.zeros((nframes, natoms, 3))
    forces = np.zeros((nframes, natoms, 3))
    energies = np.zeros(nframes)
    boxes = np.zeros((nframes, 3, 3))
    virials = np.zeros((nframes, 3, 3))
    for index, atoms in enumerate(atoms_list):
        coords[index] = atoms.positions
        force_array = atoms.arrays.get("forces")
        if force_array is None:
            try:
                force_array = atoms.get_forces()
            except Exception as exc:
                raise RuntimeError(
                    "Atoms object is missing reference forces; provide atoms.arrays['forces'] "
                    "or attach a calculator before DeepMD export."
                ) from exc
        forces[index] = np.asarray(force_array, dtype=float)
        energy_value = atoms.info.get("energy")
        if energy_value is None:
            try:
                energy_value = atoms.get_potential_energy()
            except Exception as exc:
                raise RuntimeError(
                    "Atoms object is missing reference energy; provide atoms.info['energy'] "
                    "or attach a calculator before DeepMD export."
                ) from exc
        energies[index] = float(energy_value)
        cell = atoms.get_cell()
        boxes[index] = np.eye(3) * 30.0 if cell.volume < 1e-6 else cell.array
        if require_virial:
            virials[index] = -_stress_matrix(atoms) * float(abs(cell.volume))

    np.save(set_dir / "coord.npy", coords.reshape(nframes, natoms * 3))
    np.save(set_dir / "force.npy", forces.reshape(nframes, natoms * 3))
    np.save(set_dir / "energy.npy", energies)
    np.save(set_dir / "box.npy", boxes)
    if require_virial:
        np.save(set_dir / "virial.npy", virials.reshape(nframes, 9))
    validate_written_deepmd_dataset(
        resolved,
        nframes=nframes,
        natoms=natoms,
        element_order=element_order,
        types=types,
        reference_boxes=boxes,
        require_virial=require_virial,
    )
    return resolved


def has_reference_stress(atoms: Any) -> bool:
    """Inspect stored labels only; never compute a missing reference label."""
    results = getattr(getattr(atoms, "calc", None), "results", {}) or {}
    return atoms.info.get("stress") is not None or results.get("stress") is not None


def _sort_atoms(atoms: Any) -> Any:
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    order = sorted(
        range(len(atoms)),
        key=lambda index: (
            symbols[index],
            positions[index][2],
            positions[index][1],
            positions[index][0],
        ),
    )
    sorted_atoms = atoms[order]
    results = getattr(getattr(atoms, "calc", None), "results", {}) or {}
    payload: dict[str, Any] = {}
    if "energy" in results:
        payload["energy"] = results["energy"]
    if "forces" in results:
        payload["forces"] = np.asarray(results["forces"], dtype=float)[order]
    if "stress" in results:
        payload["stress"] = results["stress"]
    if payload:
        sorted_atoms.calc = SinglePointCalculator(sorted_atoms, **payload)
    return sorted_atoms


def _stress_matrix(atoms: Any) -> np.ndarray:
    stress_value = atoms.info.get("stress")
    if stress_value is None and atoms.calc is not None:
        try:
            stress_value = atoms.get_stress()
        except Exception:
            pass
    if stress_value is None:
        raise RuntimeError(
            "Atoms object is missing reference stress; provide atoms.info['stress'] "
            "or attach a calculator that supports get_stress() before DeepMD export."
        )
    array = np.asarray(stress_value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite reference stress for DeepMD export")
    if array.shape == (3, 3):
        return array
    if array.shape == (9,):
        return array.reshape(3, 3)
    if array.shape == (6,):
        xx, yy, zz, yz, xz, xy = array.tolist()
        return np.array(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)), dtype=float)
    raise ValueError(f"Unsupported stress shape for DeepMD export: {array.shape!r}")
