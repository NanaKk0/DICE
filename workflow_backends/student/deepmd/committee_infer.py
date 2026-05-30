from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from deepmd.calculator import DP

from workflow_backends.student.deepmd.online_monitor import get_consistent_energy_forces


def evaluate_frames_with_committee(
    frames_path: str | Path,
    committee_graphs: list[str],
) -> dict[str, Any]:
    frames_path = Path(frames_path).resolve()
    if not frames_path.exists():
        raise FileNotFoundError(f"Dumped frames not found: {frames_path}")
    if len(committee_graphs) < 2:
        raise RuntimeError("At least two committee graphs are required for committee inference")

    atoms_list = list(read(str(frames_path), index=":"))
    calculators = [DP(str(Path(path).resolve())) for path in committee_graphs]
    n_models = len(calculators)

    forces_rows: list[np.ndarray] = []
    energy_rows: list[np.ndarray] = []
    frame_records: list[dict[str, Any]] = []

    for frame_index, atoms in enumerate(atoms_list):
        model_forces = []
        model_energies = []

        for calculator in calculators:
            frame_atoms = atoms.copy()
            frame_atoms.calc = calculator
            energy, forces, _ = get_consistent_energy_forces(frame_atoms)
            model_forces.append(np.asarray(forces, dtype=float))
            model_energies.append(float(energy))

        forces_rows.append(np.stack(model_forces, axis=0))
        energy_rows.append(np.asarray(model_energies, dtype=float))
        frame_records.append(
            {
                "frame_index": int(frame_index),
                "natoms": int(len(atoms)),
                "n_models": int(n_models),
            }
        )

    output_npz = frames_path.with_name("committee_forces.npz")
    np.savez_compressed(
        output_npz,
        forces=np.stack(forces_rows, axis=0) if forces_rows else np.empty((0, n_models, 0, 3)),
        energies=np.stack(energy_rows, axis=0) if energy_rows else np.empty((0, n_models)),
    )

    index_json = frames_path.with_name("committee_forces_index.json")
    with open(index_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "frames_path": str(frames_path),
                "committee_graphs": [str(Path(path).resolve()) for path in committee_graphs],
                "npz_path": str(output_npz.resolve()),
                "frames": frame_records,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "npz_path": str(output_npz.resolve()),
        "index_json": str(index_json.resolve()),
        "frames": frame_records,
    }

